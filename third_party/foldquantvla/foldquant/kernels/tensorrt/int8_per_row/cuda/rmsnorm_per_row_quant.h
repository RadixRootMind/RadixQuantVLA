// RMSNorm + per-row INT8 quantization fused kernel (LLM ladder L3+).
//
// Counterpart to fused_adaln_quant.h but with the Qwen3-style RMSNorm fused
// prologue:
//
//   normed[b, s, k] = x[b, s, k] * rsqrt(mean_k(x[b, s, :]^2) + eps)
//   y[b, s, k]      = normed * gamma[k]                     (per-channel affine)
//   amax[b, s]      = max_k |y[b, s, k]|
//   scale[b, s]     = amax[b, s] / 127
//   out_i8[b, s, k] = clamp(round(y[b, s, k] / scale[b, s]), -127, 127)
//
// Differences vs fused_adaln_quant:
//   - No mean subtract (RMS = sqrt(mean(x^2)), pure variance).
//   - Per-channel scaling tensor is the RMSNorm `weight` (gamma, shape (K,))
//     instead of a per-batch (B, K) AdaLN scale + shift.
//   - No additive shift (Qwen3 RMSNorm uses no bias).
//
// Math identity vs `nn.RMSNorm`:
//   normed = x * rsqrt(mean(x**2, -1, keepdim=True) + eps)
//   y      = gamma * normed
//
// Both dynamic-amax and static-amax variants are provided to match the
// per-row plugin patterns used elsewhere.
#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Dynamic per-row variant: compute amax at runtime from y.
//
// Inputs:
//   x_bf16:      (B, S, K) BF16 row-major (flattened (B*S, K))
//   gamma_bf16:  (K,)      BF16 per-channel scale (RMSNorm.weight)
//   B, S, K:     dims (M := B*S)
//   eps:         RMSNorm epsilon
//   stream:      CUDA stream
// Outputs:
//   out_i8:      (B, S, K) INT8 row-major
//   out_scale:   (B*S,)    FP32 - per-row quantization scale
//
// Returns 0 on success, CUDA error code otherwise.
int rmsnorm_per_row_quant_bf16_to_int8(
    void const* x_bf16,
    void const* gamma_bf16,
    void* out_i8,
    void* out_scale,
    int B,
    int S,
    int K,
    float eps,
    cudaStream_t stream);

// Rotation-enabled dynamic variant: applies a block-diagonal orthonormal
// Hadamard (FWHT, `rot_bs` per block) to y BEFORE amax/quant, so the per-token
// scale sees a channel-flattened row. Per-token quant is blind to the channel
// axis; on this LLM that blindness costs ~14% median per-channel error.
//
// The caller MUST have folded the weight side offline (W' = W·Hᵀ, same
// orthonormal Sylvester Hadamard; see fwht.cuh), otherwise the layer silently
// computes the wrong product.
//
//   rot_bs <= 1        → identical to rmsnorm_per_row_quant_bf16_to_int8
//   rot_bs power of 2, K % rot_bs == 0 → rotation applied
//   anything else      → cudaErrorInvalidValue (never a silent skip: the
//                        weights are already folded, so skipping is wrong)
int rmsnorm_fwht_per_row_quant_bf16_to_int8(
    void const* x_bf16,
    void const* gamma_bf16,
    void* out_i8,
    void* out_scale,
    int B,
    int S,
    int K,
    float eps,
    int rot_bs,
    cudaStream_t stream);

// Static per-row variant: caller supplies precomputed (B*S,) FP32 per-row scale.
// `out_scale_copy` is written-through with `static_scale` (downstream GEMM
// consumer expects a per-row scale tensor with the same layout as the dynamic
// path). Pass NULL for `out_scale_copy` to skip the copy.
int rmsnorm_per_row_static_quant_bf16_to_int8(
    void const* x_bf16,
    void const* gamma_bf16,
    void const* static_scale,
    void* out_i8,
    void* out_scale_copy,
    int B,
    int S,
    int K,
    float eps,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
