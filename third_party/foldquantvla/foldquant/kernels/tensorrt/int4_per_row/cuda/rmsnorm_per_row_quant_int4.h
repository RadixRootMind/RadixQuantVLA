// RMSNorm + per-row INT4 quantization fused kernel (LLM W4A4 prologue).
//
// INT4 counterpart of int8_per_row/rmsnorm_per_row_quant.h. Same RMSNorm math,
// same block layout; the quantizer range becomes [-7, 7] and the output is
// packed 2 nibbles per byte along K (the convention of dit_int4_rowwise.h and
// the build-time packer rotation.pack_int4_colmajor).
//
//   normed[b, s, k] = x[b, s, k] * rsqrt(mean_k(x[b, s, :]^2) + eps)
//   y[b, s, k]      = normed * gamma[k]                    (per-channel affine)
//   y               = block_FWHT(y, rot_bs)                (optional, see below)
//   amax[b, s]      = max_k |y[b, s, k]|
//   scale[b, s]     = amax[b, s] / 7
//   q[b, s, k]      = clamp(round(y[b, s, k] / scale[b, s]), -7, 7)
//   out[b, s, k/2] nibble(k&1) = q & 0xF    (even k → low, odd k → high)
//
// Only the DYNAMIC per-row variant exists, by design. A static per-row scale is
// calibrated on the UNrotated activation, and at 4 bits the rotation is not an
// optional refinement but the thing that makes the scheme work at all
// (simulated: cos 0.014 without it, 0.9995 with SQ+rotation), so the two are
// never combined and a static INT4 variant would have no caller.
#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// RMSNorm + optional block-diagonal orthonormal Hadamard (FWHT, `rot_bs` per
// block, applied to y BEFORE amax/quant) + per-row INT4 quant.
//
// Inputs:
//   x_bf16:      (B, S, K)  BF16 row-major
//   gamma_bf16:  (K,)       BF16 per-channel scale (RMSNorm.weight)
//   eps:         RMSNorm epsilon
// Outputs:
//   out_i4:      (B*S, K/2) bytes, packed INT4 row-major
//   out_scale:   (B*S,)     FP32 per-row quantization scale
//
// The caller MUST have folded the weight side offline (W' = W·Hᵀ, same
// orthonormal Sylvester Hadamard; see int8_per_row/fwht.cuh), otherwise the
// layer silently computes the wrong product.
//
//   rot_bs <= 1        → RMSNorm + per-row quant, no rotation
//   rot_bs power of 2, K % rot_bs == 0 → rotation applied
//   anything else      → cudaErrorInvalidValue (never a silent skip: the
//                        weights are already folded, so skipping is wrong)
int rmsnorm_fwht_per_row_quant_bf16_to_int4(
    void const* x_bf16,
    void const* gamma_bf16,
    void* out_i4,
    void* out_scale,
    int B,
    int S,
    int K,
    float eps,
    int rot_bs,
    float act_clip,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
