// Step B.1: DiT INT8 rowwise GEMM v2 (CUTLASS-wrapped, FlashRT port).
//
// C interface for an INT8 W8A8 GEMM with per-row activation scales and
// per-column weight scales, producing BF16 output. Ported from FlashRT-orin
// csrc/gemm/cutlass_sm80_int8_rowwise.cu (verbatim) with an additional
// per-row dynamic activation quantization helper.
//
// Pipeline:
//   1. dit_int8_per_row_quant_bf16():   BF16 act (M, K) → INT8 (M, K) + scale (M,)
//   2. dit_int8_rowwise_gemm_bf16out(): INT8 A (M, K), INT8 B (K, N), scales → BF16 D (M, N)

#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Quantize a BF16 (M, K) tensor to INT8 row-by-row using dynamic per-row amax.
//
// Inputs:
//   in_bf16:    (M, K) BF16, row-major
//   M:          number of rows (typically batch_size * seq_len)
//   K:          number of columns (hidden dim)
//   stream:     CUDA stream
//
// Outputs:
//   out_i8:     (M, K) INT8, row-major
//   out_scale:  (M,)   FP32 - act_scale[m] = amax[m] / 127.0
//
// Each block processes one row; each thread handles K/blockDim.x elements.
// Returns 0 on success, non-zero CUDA error code otherwise.
int dit_int8_per_row_quant_bf16(
    void const* in_bf16,
    void* out_i8,
    void* out_scale,
    int M,
    int K,
    cudaStream_t stream);

// Rotation-enabled variant: applies a block-diagonal orthonormal Hadamard
// (FWHT, `rot_bs` per block) to the row BEFORE amax/quant, flattening the
// channel axis that a per-token scale cannot see.
//
// The caller MUST have folded the weight side offline (W' = W·Hᵀ, same
// orthonormal Sylvester Hadamard; see fwht.cuh) or the result is wrong.
//
//   rot_bs <= 1        → delegates to dit_int8_per_row_quant_bf16 (no shared mem)
//   rot_bs power of 2, K % rot_bs == 0 → rotation applied (costs K floats shared)
//   anything else      → cudaErrorInvalidValue (never a silent skip)
//   act_scale_pre / act_scale_ch: (K,) FP32 SmoothQuant vectors, or nullptr.
//   Pre divides the raw channel before the butterfly (SmoothRot order), post
//   divides the rotated channel after it. At most one is meaningful per site.
int dit_int8_per_row_quant_fwht_bf16(
    void const* in_bf16,
    void const* act_scale_pre,
    void const* act_scale_ch,
    void* out_i8,
    void* out_scale,
    int M,
    int K,
    int rot_bs,
    cudaStream_t stream);

// Static-per-row variant: caller supplies pre-computed per-row scale tensor.
int dit_int8_per_row_static_quant_bf16(
    void const* in_bf16,
    void const* static_scale,  // (M,) FP32
    void* out_i8,
    void* out_scale_copy,      // (M,) FP32, optional (NULL = skip copy)
    int M,
    int K,
    cudaStream_t stream);

// INT8 W8A8 GEMM with per-row act scale × per-col weight scale, BF16 output.
//
// Math: D[m,n] = SaturatingMatMul(A[m,:], B[:,n]) * act_scale[m] * weight_scale[n]
//
// Inputs:
//   A:             (M, K) INT8 row-major
//   B:             (K, N) INT8 column-major  (FlashRT layout - weight is N×K stored as K×N column-major)
//   act_scale:     (M,)   FP32 per-row
//   weight_scale:  (N,)   FP32 per-col
//   M, N, K:       dimensions
//   stream:        CUDA stream
//
// Output:
//   D:             (M, N) BF16 row-major
//
// Returns 0 on success, CUTLASS status code otherwise.
int dit_int8_rowwise_gemm_bf16out(
    void const* A,
    void const* B,
    void const* act_scale,
    void const* weight_scale,
    void* D,
    int M,
    int N,
    int K,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
