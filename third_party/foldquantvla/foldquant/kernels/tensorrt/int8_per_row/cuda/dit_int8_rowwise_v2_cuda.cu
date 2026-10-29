// Step B.1: DiT INT8 rowwise GEMM v2 (CUTLASS-wrapped, port of FlashRT-orin
// csrc/gemm/cutlass_sm80_int8_rowwise.cu).
//
// SM80/87 INT8 IMMA m16n8k32 via CUTLASS OpClassTensorOp.
// Tile 128×128×64, warp 64×64×64, 4-stage cp.async pipeline.
// Per-row act scale × per-col weight scale → BF16 output.
//
// This is a direct port of the FlashRT kernel; only the namespace and
// extern "C" entry name have been changed.

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

#include "dit_int8_rowwise_v2.h"

namespace gr00t {
namespace dit_int8_rowwise_v2 {

using namespace cute;

using ElementA = int8_t;
using LayoutA = cutlass::layout::RowMajor;
using ElementB = int8_t;
using LayoutB = cutlass::layout::ColumnMajor;
using ElementOutput = cutlass::bfloat16_t;
using LayoutC = cutlass::layout::RowMajor;
using ElementAccumulator = int32_t;
using ElementCompute = float;

constexpr int AlignmentA = 16;
constexpr int AlignmentB = 16;
constexpr int AlignmentC = 8;

using ArchTag = cutlass::arch::Sm80;
using OperatorClass = cutlass::arch::OpClassTensorOp;
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
using MulActScale = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
using MulWtScale = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<
    OutputTileThreadMap, ElementOutput,
    cutlass::FloatRoundStyle::round_to_nearest,
    Stride<int64_t, _1, int64_t>>;

using EVT_AccMulAct = cutlass::epilogue::threadblock::Sm80EVT<
    MulActScale, AccFetch, ActScaleLoad>;
using EVT_MulBoth = cutlass::epilogue::threadblock::Sm80EVT<
    MulWtScale, EVT_AccMulAct, WtScaleLoad>;
using EVT_NoBias = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT_MulBoth>;

using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
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
    EVT_NoBias,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    NumStages,
    cutlass::arch::OpMultiplyAddSaturate,
    EVTEpilogueStages
>::GemmKernel;

using GemmDevice = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

static int run(
    void const* A,
    void const* B,
    void const* act_scale,
    void const* weight_scale,
    void* D,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    typename EVT_NoBias::Arguments evt_args{
        {
            {
                {},
                {reinterpret_cast<float const*>(act_scale), 1.0f, {}},
                {}
            },
            {reinterpret_cast<float const*>(weight_scale), 1.0f, {_0{}, _1{}, int32_t(N)}},
            {}
        },
        {reinterpret_cast<ElementOutput*>(D),
         {static_cast<int64_t>(N), _1{}, static_cast<int64_t>(M) * N}}
    };

    typename GemmDevice::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        1,
        evt_args,
        reinterpret_cast<ElementA const*>(A),
        reinterpret_cast<ElementB const*>(B),
        nullptr,
        nullptr,
        static_cast<int64_t>(M) * K,
        static_cast<int64_t>(N) * K,
        0,
        0,
        K,
        K,
        N,
        N
    );

    GemmDevice gemm;
    auto st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr,
                     "[dit_int8_rowwise_v2] can_implement failed: M=%d N=%d K=%d code=%d\n",
                     M, N, K, static_cast<int>(st));
        return static_cast<int>(st) | 0x10000;
    }

    size_t ws_sz = GemmDevice::get_workspace_size(args);
    // thread_local like the sibling variants: a plain static shared across
    // host threads could be freed by one thread mid-read of another.
    static thread_local void* ws_ptr = nullptr;
    static thread_local size_t ws_cap = 0;
    if (ws_sz > ws_cap) {
        if (ws_ptr) cudaFree(ws_ptr);
        if (cudaMalloc(&ws_ptr, ws_sz) != cudaSuccess) {
            ws_ptr = nullptr;
            ws_cap = 0;
            return -1;
        }
        ws_cap = ws_sz;
    }

    st = gemm.initialize(args, ws_ptr, stream);
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr,
                     "[dit_int8_rowwise_v2] init failed: M=%d N=%d K=%d code=%d\n",
                     M, N, K, static_cast<int>(st));
        return static_cast<int>(st) | 0x20000;
    }

    st = gemm.run(stream);
    return (st == cutlass::Status::kSuccess) ? 0 : (static_cast<int>(st) | 0x30000);
}

}  // namespace dit_int8_rowwise_v2
}  // namespace gr00t

extern "C" int dit_int8_rowwise_gemm_bf16out(
    void const* A,
    void const* B,
    void const* act_scale,
    void const* weight_scale,
    void* D,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
    return gr00t::dit_int8_rowwise_v2::run(
        A, B, act_scale, weight_scale, D, M, N, K, stream);
}
