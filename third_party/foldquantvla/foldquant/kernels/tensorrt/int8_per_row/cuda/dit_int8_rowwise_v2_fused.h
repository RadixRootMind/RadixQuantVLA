// DiT INT8 rowwise GEMM with fused bias+residual epilogue.
//
// Extends the baseline INT8 GEMM chain with two extra EVT epilogue stages:
//   ((A @ B) * act_scale * weight_scale) + bias + residual → BF16 output
//
// Goal: collapse 3 separate kernels (GEMM, +bias, +residual) into 1 plugin call,
// avoiding intermediate DRAM round-trips and TRT Myelin fusion-boundary penalties.
//
// All shapes/dtypes match dit_int8_rowwise_v2_tiles.h. Tile = best from Phase 1
// (64x64x64, warp 32x32x64, 4-stage cp.async).
#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// GEMM + per-row act scale + per-col weight scale + bias (per-col) + residual (per-element) → BF16.
//
// Inputs:
//   A:             (M, K) INT8 row-major
//   B:             (K, N) INT8 column-major (== weight (N, K) row-major)
//   act_scale:     (M,)   FP32
//   weight_scale:  (N,)   FP32
//   bias:          (N,)   FP32      - added per output channel
//   residual:      (M, N) BF16 row-major - added per element
//
// Output:
//   D:             (M, N) BF16 row-major
//
// Tile: t64x64x64_w32x32x64_s4 (Phase 1 winner for M=51).
int dit_int8_rowwise_gemm_bias_residual_bf16out(
    void const* A,
    void const* B,
    void const* act_scale,
    void const* weight_scale,
    void const* bias,
    void const* residual,
    void* D,
    int M,
    int N,
    int K,
    cudaStream_t stream);

// Variant: GEMM + scales + bias only (no residual). Useful for layers without
// a residual connection (e.g. ff.net.0 gate projection).
int dit_int8_rowwise_gemm_bias_bf16out(
    void const* A,
    void const* B,
    void const* act_scale,
    void const* weight_scale,
    void const* bias,
    void* D,
    int M,
    int N,
    int K,
    cudaStream_t stream);

// Variant: GEMM + scales + residual only (no bias). Some DiT linear layers
// omit the bias term (matches Qwen3 / GR00T conventions).
int dit_int8_rowwise_gemm_residual_bf16out(
    void const* A,
    void const* B,
    void const* act_scale,
    void const* weight_scale,
    void const* residual,
    void* D,
    int M,
    int N,
    int K,
    cudaStream_t stream);

// t128x128x64 / warp 64x64x64 / 4-stage variant of the residual-only kernel:
// LLM ladder v4. Same math, same fp32 EVT epilogue precision, just a larger
// tile. Wins 8/8 LLM GEMM shapes per test_llm_tile_autotune.py.
int dit_int8_rowwise_gemm_residual_bf16out_t128x128x64_w64x64x64_s4(
    void const* A,
    void const* B,
    void const* act_scale,
    void const* weight_scale,
    void const* residual,
    void* D,
    int M,
    int N,
    int K,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
