// Phase 2: DiT INT8 rowwise GEMM with fused bias + residual EVT epilogue.
//
// Sm80 CUTLASS EVT chain extension:
//
//   AccFetch (int32)
//     -> Mul(_, ActScale)   [col broadcast (M,)]  → fp32 (A@B)*act_scale
//     -> Mul(_, WtScale)    [row broadcast (N,)]  → fp32 (A@B)*act_scale*weight_scale
//    [-> Add(_, Bias)]      [row broadcast (N,)]  → fp32 + bias[n]
//    [-> Add(_, Residual)]  [aux load (M,N)]      → fp32 + residual[m,n]
//     -> StoreD                                   → bf16
//
// Tile: t64x64x64 / warp 32x32x64 / 4-stage (Phase 1 winner for M=51).

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
namespace dit_int8_rowwise_fused {

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
using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 64>;
using WarpShape = cutlass::gemm::GemmShape<32, 32, 64>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
constexpr int NumStages = 4;
constexpr int EVTEpilogueStages = 1;

using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
    ThreadblockShape, WarpShape, ElementOutput, AlignmentC, EVTEpilogueStages>;

// Shared building blocks.
using AccFetch = cutlass::epilogue::threadblock::VisitorAccFetch;
using ActScaleLoad = cutlass::epilogue::threadblock::VisitorColBroadcast<
    OutputTileThreadMap, float, Stride<_1, _0, _0>>;
using WtScaleLoad = cutlass::epilogue::threadblock::VisitorRowBroadcast<
    OutputTileThreadMap, float, Stride<_0, _1, int32_t>>;
using BiasLoad = cutlass::epilogue::threadblock::VisitorRowBroadcast<
    OutputTileThreadMap, float, Stride<_0, _1, int32_t>>;
using ResidualLoad = cutlass::epilogue::threadblock::VisitorAuxLoad<
    OutputTileThreadMap, ElementResidual, Stride<int64_t, _1, int64_t>>;

using MulActScale = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
using MulWtScale = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
using AddBias = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::plus, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
using AddResidual = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::plus, float, float, cutlass::FloatRoundStyle::round_to_nearest>;

using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<
    OutputTileThreadMap, ElementOutput,
    cutlass::FloatRoundStyle::round_to_nearest,
    Stride<int64_t, _1, int64_t>>;

// Common prefix: (acc * act_scale * weight_scale).
using EVT_Scales = cutlass::epilogue::threadblock::Sm80EVT<
    MulWtScale,
    cutlass::epilogue::threadblock::Sm80EVT<MulActScale, AccFetch, ActScaleLoad>,
    WtScaleLoad>;

// --- Variant 1: scales + bias + residual ---
using EVT_ScalesBias = cutlass::epilogue::threadblock::Sm80EVT<
    AddBias, EVT_Scales, BiasLoad>;
using EVT_ScalesBiasResidual = cutlass::epilogue::threadblock::Sm80EVT<
    AddResidual, EVT_ScalesBias, ResidualLoad>;
using EVT_StoreBiasResidual = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT_ScalesBiasResidual>;

// --- Variant 2: scales + bias only ---
using EVT_StoreBias = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT_ScalesBias>;

// --- Variant 3: scales + residual only ---
using EVT_ScalesResidual = cutlass::epilogue::threadblock::Sm80EVT<
    AddResidual, EVT_Scales, ResidualLoad>;
using EVT_StoreResidual = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT_ScalesResidual>;

template <typename EVT>
using GemmKernelT = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
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
    EVT,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    NumStages,
    cutlass::arch::OpMultiplyAddSaturate,
    EVTEpilogueStages
>::GemmKernel;

using GemmKernel_BiasResidual = GemmKernelT<EVT_StoreBiasResidual>;
using GemmKernel_Bias = GemmKernelT<EVT_StoreBias>;
using GemmKernel_Residual = GemmKernelT<EVT_StoreResidual>;

using GemmDevice_BiasResidual = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_BiasResidual>;
using GemmDevice_Bias = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_Bias>;
using GemmDevice_Residual = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_Residual>;

// Per-variant workspace cache. Workspace size is small (single-block swizzle counters);
// holding one global pointer per variant avoids cudaMalloc/Free in the steady state.
template <typename Gemm>
static void* get_ws(size_t needed) {
    static thread_local void* ws_ptr = nullptr;
    static thread_local size_t ws_cap = 0;
    if (needed > ws_cap) {
        if (ws_ptr) cudaFree(ws_ptr);
        if (cudaMalloc(&ws_ptr, needed) != cudaSuccess) {
            ws_ptr = nullptr;
            ws_cap = 0;
            return nullptr;
        }
        ws_cap = needed;
    }
    return ws_ptr;
}

static int run_bias_residual(
    void const* A, void const* B,
    void const* act_scale, void const* weight_scale,
    void const* bias, void const* residual,
    void* D,
    int M, int N, int K,
    cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    typename EVT_StoreBiasResidual::Arguments evt_args{
        {  // EVT_ScalesBiasResidual
            {  // EVT_ScalesBias
                {  // EVT_Scales = MulWtScale(MulActScale(Acc, ActScale), WtScale)
                    {  // MulActScale(Acc, ActScale)
                        {},  // AccFetch
                        {reinterpret_cast<float const*>(act_scale), 1.0f, {}},  // ActScaleLoad
                        {}  // MulActScale op
                    },
                    {reinterpret_cast<float const*>(weight_scale), 1.0f, {_0{}, _1{}, int32_t(N)}},  // WtScaleLoad
                    {}  // MulWtScale op
                },
                {reinterpret_cast<float const*>(bias), 0.0f, {_0{}, _1{}, int32_t(N)}},  // BiasLoad
                {}  // AddBias op
            },
            {reinterpret_cast<ElementResidual*>(const_cast<void*>(residual)),
             ElementResidual(0), {static_cast<int64_t>(N), _1{}, static_cast<int64_t>(M) * N}},  // ResidualLoad
            {}  // AddResidual op
        },
        {reinterpret_cast<ElementOutput*>(D),
         {static_cast<int64_t>(N), _1{}, static_cast<int64_t>(M) * N}}  // StoreD
    };

    typename GemmDevice_BiasResidual::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        1,
        evt_args,
        reinterpret_cast<ElementA const*>(A),
        reinterpret_cast<ElementB const*>(B),
        nullptr, nullptr,
        static_cast<int64_t>(M) * K,
        static_cast<int64_t>(N) * K,
        0, 0,
        K, K, N, N
    );

    GemmDevice_BiasResidual gemm;
    auto st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "[gemm_bias_residual] can_implement failed: M=%d N=%d K=%d code=%d\n",
                     M, N, K, static_cast<int>(st));
        return static_cast<int>(st) | 0x10000;
    }
    size_t ws_sz = GemmDevice_BiasResidual::get_workspace_size(args);
    void* ws = get_ws<GemmDevice_BiasResidual>(ws_sz);
    if (ws_sz > 0 && ws == nullptr) return -1;
    st = gemm.initialize(args, ws, stream);
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "[gemm_bias_residual] init failed: %d\n", static_cast<int>(st));
        return static_cast<int>(st) | 0x20000;
    }
    st = gemm.run(stream);
    return (st == cutlass::Status::kSuccess) ? 0 : (static_cast<int>(st) | 0x30000);
}

static int run_bias(
    void const* A, void const* B,
    void const* act_scale, void const* weight_scale,
    void const* bias,
    void* D,
    int M, int N, int K,
    cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    typename EVT_StoreBias::Arguments evt_args{
        {  // EVT_ScalesBias
            {  // EVT_Scales
                {  // MulActScale
                    {},
                    {reinterpret_cast<float const*>(act_scale), 1.0f, {}},
                    {}
                },
                {reinterpret_cast<float const*>(weight_scale), 1.0f, {_0{}, _1{}, int32_t(N)}},
                {}
            },
            {reinterpret_cast<float const*>(bias), 0.0f, {_0{}, _1{}, int32_t(N)}},
            {}
        },
        {reinterpret_cast<ElementOutput*>(D),
         {static_cast<int64_t>(N), _1{}, static_cast<int64_t>(M) * N}}
    };

    typename GemmDevice_Bias::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size, 1, evt_args,
        reinterpret_cast<ElementA const*>(A),
        reinterpret_cast<ElementB const*>(B),
        nullptr, nullptr,
        static_cast<int64_t>(M) * K,
        static_cast<int64_t>(N) * K,
        0, 0, K, K, N, N);

    GemmDevice_Bias gemm;
    auto st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) return static_cast<int>(st) | 0x10000;
    size_t ws_sz = GemmDevice_Bias::get_workspace_size(args);
    void* ws = get_ws<GemmDevice_Bias>(ws_sz);
    if (ws_sz > 0 && ws == nullptr) return -1;
    st = gemm.initialize(args, ws, stream);
    if (st != cutlass::Status::kSuccess) return static_cast<int>(st) | 0x20000;
    st = gemm.run(stream);
    return (st == cutlass::Status::kSuccess) ? 0 : (static_cast<int>(st) | 0x30000);
}

static int run_residual(
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
    if (st != cutlass::Status::kSuccess) return static_cast<int>(st) | 0x10000;
    size_t ws_sz = GemmDevice_Residual::get_workspace_size(args);
    void* ws = get_ws<GemmDevice_Residual>(ws_sz);
    if (ws_sz > 0 && ws == nullptr) return -1;
    st = gemm.initialize(args, ws, stream);
    if (st != cutlass::Status::kSuccess) return static_cast<int>(st) | 0x20000;
    st = gemm.run(stream);
    return (st == cutlass::Status::kSuccess) ? 0 : (static_cast<int>(st) | 0x30000);
}

}  // namespace dit_int8_rowwise_fused
}  // namespace gr00t

extern "C" int dit_int8_rowwise_gemm_bias_residual_bf16out(
    void const* A, void const* B,
    void const* act_scale, void const* weight_scale,
    void const* bias, void const* residual,
    void* D, int M, int N, int K, cudaStream_t stream) {
    return gr00t::dit_int8_rowwise_fused::run_bias_residual(
        A, B, act_scale, weight_scale, bias, residual, D, M, N, K, stream);
}

extern "C" int dit_int8_rowwise_gemm_bias_bf16out(
    void const* A, void const* B,
    void const* act_scale, void const* weight_scale,
    void const* bias,
    void* D, int M, int N, int K, cudaStream_t stream) {
    return gr00t::dit_int8_rowwise_fused::run_bias(
        A, B, act_scale, weight_scale, bias, D, M, N, K, stream);
}

extern "C" int dit_int8_rowwise_gemm_residual_bf16out(
    void const* A, void const* B,
    void const* act_scale, void const* weight_scale,
    void const* residual,
    void* D, int M, int N, int K, cudaStream_t stream) {
    return gr00t::dit_int8_rowwise_fused::run_residual(
        A, B, act_scale, weight_scale, residual, D, M, N, K, stream);
}
