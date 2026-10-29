// v4: t128x128x64 tile variant of the fused INT8 GEMM + per-row scale +
// per-col scale + residual (no bias) EVT epilogue. Used by the LLM L3+
// per-row plugins (PerRowInt8LinearResidual, FusedAttnCausalOprojLlm) which
// need the fp32 EVT residual epilogue (precision-preserving) AND want the
// t128x128 microbench winner.
//
// Only the residual-only variant is provided (no bias variants). Qwen3 Linear
// layers don't use bias, so this is sufficient. Bias variants would require
// duplicating the EVT chain plumbing and aren't needed by the LLM ladder.
//
// Math identical to dit_int8_rowwise_gemm_residual_bf16out (in
// dit_int8_rowwise_v2_fused_cuda.cu); only the threadblock/warp tile differ.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h"

#include "cute/tensor.hpp"

#include "dit_int8_rowwise_v2_fused.h"

namespace gr00t {
namespace dit_int8_rowwise_fused_t128 {

using namespace cute;

using ElementA = int8_t;
using LayoutA = cutlass::layout::RowMajor;
using ElementB = int8_t;
using LayoutB = cutlass::layout::ColumnMajor;
using ElementOutput = cutlass::bfloat16_t;
using ElementResidual = cutlass::bfloat16_t;
using LayoutC = cutlass::layout::RowMajor;
using ElementAccumulator = int32_t;
using ElementCompute = float;

constexpr int AlignmentA = 16;
constexpr int AlignmentB = 16;
constexpr int AlignmentC = 8;

using ArchTag = cutlass::arch::Sm80;
using OperatorClass = cutlass::arch::OpClassTensorOp;
// === t128x128x64 / warp 64x64x64 / 4-stage cp.async: LLM microbench winner. ===
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
constexpr int NumStages = 4;
constexpr int EVTEpilogueStages = 1;

using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
    ThreadblockShape, WarpShape, ElementOutput, AlignmentC, EVTEpilogueStages>;

using AccFetch = cutlass::epilogue::threadblock::VisitorAccFetch;
using ActScaleLoad = cutlass::epilogue::threadblock::VisitorColBroadcast<
    OutputTileThreadMap, float, Stride<_1, _0, _0>>;
using WtScaleLoad = cutlass::epilogue::threadblock::VisitorRowBroadcast<
    OutputTileThreadMap, float, Stride<_0, _1, int32_t>>;
using ResidualLoad = cutlass::epilogue::threadblock::VisitorAuxLoad<
    OutputTileThreadMap, ElementResidual, Stride<int64_t, _1, int64_t>>;

using MulActScale = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
using MulWtScale = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
using AddResidual = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::plus, float, float, cutlass::FloatRoundStyle::round_to_nearest>;

using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<
    OutputTileThreadMap, ElementOutput,
    cutlass::FloatRoundStyle::round_to_nearest,
    Stride<int64_t, _1, int64_t>>;

// EVT chain: (acc * act_scale * weight_scale) + residual → BF16.
using EVT_Scales = cutlass::epilogue::threadblock::Sm80EVT<
    MulWtScale,
    cutlass::epilogue::threadblock::Sm80EVT<MulActScale, AccFetch, ActScaleLoad>,
    WtScaleLoad>;
using EVT_ScalesResidual = cutlass::epilogue::threadblock::Sm80EVT<
    AddResidual, EVT_Scales, ResidualLoad>;
using EVT_StoreResidual = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT_ScalesResidual>;

using GemmKernel_Residual = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
    ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignmentA,
    ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignmentB,
    ElementOutput, LayoutC, AlignmentC,
    ElementAccumulator,
    ElementCompute,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EVT_StoreResidual,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    NumStages,
    cutlass::arch::OpMultiplyAddSaturate,
    EVTEpilogueStages
>::GemmKernel;

using GemmDevice_Residual = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_Residual>;

template <typename Gemm>
static void* get_ws(size_t needed) {
    static thread_local void* ws_ptr = nullptr;
    static thread_local size_t ws_cap = 0;
    if (needed > ws_cap) {
        if (ws_ptr) cudaFree(ws_ptr);
        if (cudaMalloc(&ws_ptr, needed) != cudaSuccess) {
            ws_ptr = nullptr; ws_cap = 0; return nullptr;
        }
        ws_cap = needed;
    }
    return ws_ptr;
}

static int run_residual_t128(
    void const* A, void const* B,
    void const* act_scale, void const* weight_scale,
    void const* residual,
    void* D,
    int M, int N, int K,
    cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    typename EVT_StoreResidual::Arguments evt_args{
        {  // EVT_ScalesResidual
            {  // EVT_Scales
                {  // MulActScale
                    {},
                    {reinterpret_cast<float const*>(act_scale), 1.0f, {}},
                    {}
                },
                {reinterpret_cast<float const*>(weight_scale), 1.0f, {_0{}, _1{}, int32_t(N)}},
                {}
            },
            {reinterpret_cast<ElementResidual*>(const_cast<void*>(residual)),
             ElementResidual(0), {static_cast<int64_t>(N), _1{}, static_cast<int64_t>(M) * N}},
            {}
        },
        {reinterpret_cast<ElementOutput*>(D),
         {static_cast<int64_t>(N), _1{}, static_cast<int64_t>(M) * N}}
    };

    typename GemmDevice_Residual::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size, 1, evt_args,
        reinterpret_cast<ElementA const*>(A),
        reinterpret_cast<ElementB const*>(B),
        nullptr, nullptr,
        static_cast<int64_t>(M) * K,
        static_cast<int64_t>(N) * K,
        0, 0, K, K, N, N);

    GemmDevice_Residual gemm;
    auto st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr,
            "[gemm_residual_t128] can_implement failed: M=%d N=%d K=%d code=%d\n",
            M, N, K, static_cast<int>(st));
        return static_cast<int>(st) | 0x10000;
    }
    size_t ws_sz = GemmDevice_Residual::get_workspace_size(args);
    void* ws = get_ws<GemmDevice_Residual>(ws_sz);
    if (ws_sz > 0 && ws == nullptr) return -1;
    st = gemm.initialize(args, ws, stream);
    if (st != cutlass::Status::kSuccess) return static_cast<int>(st) | 0x20000;
    st = gemm.run(stream);
    return (st == cutlass::Status::kSuccess) ? 0 : (static_cast<int>(st) | 0x30000);
}

}  // namespace dit_int8_rowwise_fused_t128
}  // namespace gr00t

extern "C" int dit_int8_rowwise_gemm_residual_bf16out_t128x128x64_w64x64x64_s4(
    void const* A, void const* B,
    void const* act_scale, void const* weight_scale,
    void const* residual,
    void* D, int M, int N, int K, cudaStream_t stream) {
    return gr00t::dit_int8_rowwise_fused_t128::run_residual_t128(
        A, B, act_scale, weight_scale, residual, D, M, N, K, stream);
}
