// Phase 1: Tile-size sweep for DiT INT8 rowwise GEMM.
// Each entry has the same signature as dit_int8_rowwise_gemm_bf16out but uses
// a different CUTLASS Threadblock/Warp tile shape. Naming convention:
//   dit_int8_rowwise_gemm_bf16out_t<TBM>x<TBN>x<TBK>_w<WM>x<WN>x<WK>_s<Stages>
//
// All entries share the same data-flow semantics (INT8×INT8 → INT32 → per-row
// act_scale × per-col weight_scale → BF16). Only tile shape differs.
#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DECL_TILE_VARIANT(name) \
    int name(                                   \
        void const* A,                          \
        void const* B,                          \
        void const* act_scale,                  \
        void const* weight_scale,               \
        void* D,                                \
        int M,                                  \
        int N,                                  \
        int K,                                  \
        cudaStream_t stream)

DECL_TILE_VARIANT(dit_int8_rowwise_gemm_bf16out_t128x128x64_w64x64x64_s4);
DECL_TILE_VARIANT(dit_int8_rowwise_gemm_bf16out_t64x128x64_w32x64x64_s4);
DECL_TILE_VARIANT(dit_int8_rowwise_gemm_bf16out_t64x64x64_w32x32x64_s4);
DECL_TILE_VARIANT(dit_int8_rowwise_gemm_bf16out_t32x128x64_w32x32x64_s4);
DECL_TILE_VARIANT(dit_int8_rowwise_gemm_bf16out_t128x64x64_w64x32x64_s4);
DECL_TILE_VARIANT(dit_int8_rowwise_gemm_bf16out_t64x256x64_w32x64x64_s4);
DECL_TILE_VARIANT(dit_int8_rowwise_gemm_bf16out_t64x128x64_w32x64x64_s3);
DECL_TILE_VARIANT(dit_int8_rowwise_gemm_bf16out_t64x64x64_w32x32x64_s3);

#undef DECL_TILE_VARIANT

#ifdef __cplusplus
}
#endif
