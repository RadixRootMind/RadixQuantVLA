// Phase 1: Tile-size sweep for DiT INT8 rowwise GEMM (M=51 on Orin SM87).
//
// Templated CUTLASS Sm80 EVT-epilogue GEMM with INT8×INT8 → INT32 → fp32(per-row
// act_scale × per-col weight_scale) → BF16 output. Each tile variant is a
// separate template instantiation with its own extern "C" entry.
//
// Hardware target: Jetson AGX Orin 64GB, 16 SMs, SM87 (Ampere). The baseline
// 128x128x64 tile produces only 12 CTAs for shapes like (M=51, N=1536), leaving
// 4 SMs idle. Smaller tiles spread more CTAs across SMs at the cost of more
// global memory traffic per output element.

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

#include "dit_int8_rowwise_v2_tiles.h"

namespace gr00t {
namespace dit_int8_rowwise_tiles {

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
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
constexpr int EVTEpilogueStages = 1;

template <int TBM, int TBN, int TBK, int WM, int WN, int WK, int Stages>
struct GemmTile {
    using ThreadblockShape = cutlass::gemm::GemmShape<TBM, TBN, TBK>;
    using WarpShape = cutlass::gemm::GemmShape<WM, WN, WK>;

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
        Stages,
        cutlass::arch::OpMultiplyAddSaturate,
        EVTEpilogueStages
    >::GemmKernel;

    using GemmDevice = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename Tile>
static int run_impl(
    void const* A,
    void const* B,
    void const* act_scale,
    void const* weight_scale,
    void* D,
    int M,
    int N,
    int K,
    cudaStream_t stream,
    const char* tag) {
    using Gemm = typename Tile::GemmDevice;
    using EVT = typename Tile::EVT_NoBias;

    cutlass::gemm::GemmCoord problem_size(M, N, K);

    typename EVT::Arguments evt_args{
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

    typename Gemm::Arguments args(
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

    Gemm gemm;
    auto st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr,
                     "[%s] can_implement failed: M=%d N=%d K=%d code=%d\n",
                     tag, M, N, K, static_cast<int>(st));
        return static_cast<int>(st) | 0x10000;
    }

    size_t ws_sz = Gemm::get_workspace_size(args);
    // Per-variant workspace cache (static local). The size is variant-dependent
    // (different swizzles → potentially different workspace requirement).
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
                     "[%s] init failed: M=%d N=%d K=%d code=%d\n",
                     tag, M, N, K, static_cast<int>(st));
        return static_cast<int>(st) | 0x20000;
    }

    st = gemm.run(stream);
    return (st == cutlass::Status::kSuccess) ? 0 : (static_cast<int>(st) | 0x30000);
}

}  // namespace dit_int8_rowwise_tiles
}  // namespace gr00t

#define INSTANTIATE_TILE(suffix, TBM, TBN, TBK, WM, WN, WK, Stages)              \
    extern "C" int dit_int8_rowwise_gemm_bf16out_##suffix(                       \
        void const* A,                                                            \
        void const* B,                                                            \
        void const* act_scale,                                                    \
        void const* weight_scale,                                                 \
        void* D,                                                                  \
        int M,                                                                    \
        int N,                                                                    \
        int K,                                                                    \
        cudaStream_t stream) {                                                    \
        using Tile = gr00t::dit_int8_rowwise_tiles::GemmTile<                     \
            TBM, TBN, TBK, WM, WN, WK, Stages>;                                   \
        return gr00t::dit_int8_rowwise_tiles::run_impl<Tile>(                     \
            A, B, act_scale, weight_scale, D, M, N, K, stream, #suffix);          \
    }

INSTANTIATE_TILE(t128x128x64_w64x64x64_s4, 128, 128, 64, 64, 64, 64, 4)
INSTANTIATE_TILE(t64x128x64_w32x64x64_s4,   64, 128, 64, 32, 64, 64, 4)
INSTANTIATE_TILE(t64x64x64_w32x32x64_s4,    64,  64, 64, 32, 32, 64, 4)
INSTANTIATE_TILE(t32x128x64_w32x32x64_s4,   32, 128, 64, 32, 32, 64, 4)
INSTANTIATE_TILE(t128x64x64_w64x32x64_s4,  128,  64, 64, 64, 32, 64, 4)
INSTANTIATE_TILE(t64x256x64_w32x64x64_s4,   64, 256, 64, 32, 64, 64, 4)
INSTANTIATE_TILE(t64x128x64_w32x64x64_s3,   64, 128, 64, 32, 64, 64, 3)
INSTANTIATE_TILE(t64x64x64_w32x32x64_s3,    64,  64, 64, 32, 32, 64, 3)

#undef INSTANTIATE_TILE
