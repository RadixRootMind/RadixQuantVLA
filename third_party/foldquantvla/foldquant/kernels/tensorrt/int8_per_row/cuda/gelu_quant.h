// Phase 4: Fused GELU (tanh approximation) + per-row INT8 quantization.
//
// Replaces the FFN intermediate sequence (proj0 → gelu_tanh → per_row_quant)
// after-GEMM pieces into a single launch. Operates on the (M, N) BF16 output
// of the gate projection and produces (M, N) INT8 + (M,) per-row scale ready
// for the down-projection GEMM.
//
// Op contract:
//   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
//   amax[m] = max_k |gelu(x[m, k])|
//   scale[m] = amax[m] / 127
//   out_i8[m, k] = clamp(round(gelu(x[m, k]) / scale[m]), -127, 127)
#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

int dit_gelu_quant_bf16_to_int8(
    void const* in_bf16,     // (M, K) BF16, row-major
    void* out_i8,            // (M, K) INT8, row-major
    void* out_scale,         // (M,)   FP32 per-row quantization scale
    int M,
    int K,
    cudaStream_t stream);

// FoldQuant variant: SmoothQuant channel divide (act_scale_pre, the RAW-frame
// fold-before vector; pass NULL for the unfolded arm) then a block-diagonal
// Sylvester butterfly of width rot_bs, and only then the per-row amax; the
// quantizer must see the rotated row, since that is what the INT8 grid holds.
// Returns cudaErrorInvalidValue when rot_bs is not a usable block for K.
int dit_gelu_fwht_quant_bf16_to_int8(
    void const* in_bf16,       // (M, K) BF16, row-major
    void const* act_scale_pre, // (K,) FP32 or NULL
    void* out_i8,              // (M, K) INT8
    void* out_scale,           // (M,)   FP32 per-row scale
    int M,
    int K,
    int rot_bs,
    cudaStream_t stream);

// Static-per-row variant of dit_gelu_quant: skips dynamic amax, uses pre-
// computed static_scale[m] from offline calibration. out_scale_copy is the
// output scale tensor (filled with static_scale values for the downstream
// GEMM consumer); pass NULL to skip the copy.
int dit_gelu_static_quant_bf16_to_int8(
    void const* in_bf16,
    void const* static_scale,  // (M,) FP32
    void* out_i8,
    void* out_scale_copy,
    int M,
    int K,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
