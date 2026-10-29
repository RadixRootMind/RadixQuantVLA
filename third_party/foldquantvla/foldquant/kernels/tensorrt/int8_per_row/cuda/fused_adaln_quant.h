// Phase 3: Fused LayerNorm + AdaLN modulation + per-row INT8 quantization.
//
// Replaces the Phase B prologue sequence (LayerNorm → mul(1+scale) → add(shift)
// → per_row_quant) with a single CUDA kernel. Each block handles one [b, s]
// token of width K; sum/sum-of-squares stats are computed in fp32 via
// warp-shuffle reductions, modulation is applied in fp32, then a second
// reduction finds the row amax and quantizes to INT8.
//
// Op contract (mathematically identical to AdaLayerNorm + per_row_quant):
//
//   normed[b, s, k] = (x[b, s, k] - mean(x[b, s, :])) * rstd(x[b, s, :])
//   y[b, s, k]      = normed * (1 + scale[b, k]) + shift[b, k]
//   amax[b, s]      = max_k |y[b, s, k]|
//   scale_a[b, s]   = amax[b, s] / 127
//   out_i8[b, s, k] = clamp(round(y[b, s, k] / scale_a[b, s]), -127, 127)
//
// LayerNorm here is elementwise_affine=False (matches gr00t.AdaLayerNorm).
// The modulation (scale, shift) is per-batch broadcast across the seq dim.
#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Inputs:
//   x_bf16:    (B, S, K) BF16 row-major (i.e. flattened (B*S, K))
//   scale_bf16:(B, K)    BF16 - per-batch broadcast across S
//   shift_bf16:(B, K)    BF16 - per-batch broadcast across S
//   B, S, K:   shape dimensions (M := B*S)
//   eps:       LayerNorm epsilon
//   stream:    CUDA stream
// Outputs:
//   out_i8:    (B, S, K) INT8 row-major
//   out_scale: (B*S,)    FP32 - per-row quantization scale
//
// Returns 0 on success, CUDA error code otherwise.
int fused_adaln_quant_bf16_to_int8(
    void const* x_bf16,
    void const* scale_bf16,
    void const* shift_bf16,
    void* out_i8,
    void* out_scale,
    int B,
    int S,
    int K,
    float eps,
    cudaStream_t stream);

// FoldQuant variant: adaLN modulation, then the RAW-frame SmoothQuant divide
// (act_scale_pre, NULL for the unfolded arm) and a block-diagonal Sylvester
// butterfly of width rot_bs, and only then the per-row amax; the quantizer must
// see the rotated row. Returns cudaErrorInvalidValue if rot_bs cannot tile K.
int fused_adaln_fwht_quant_bf16_to_int8(
    void const* x_bf16,
    void const* scale_bf16,
    void const* shift_bf16,
    void const* act_scale_pre,  // (K,) FP32 or NULL
    void* out_i8,
    void* out_scale,
    int B,
    int S,
    int K,
    float eps,
    int rot_bs,
    cudaStream_t stream);

// Static-per-row variant: caller supplies a precomputed (B*S,) FP32 scale tensor
// (e.g. from offline calibration). The kernel skips the dynamic amax reduction
// and quantizes each row using static_scale[row_idx] directly.
//
// out_scale here is OUTPUT and will be populated with the static_scale values
// copied through (so downstream GEMM sees a per-row scale tensor of the same
// layout as the dynamic path). Passing NULL is allowed if downstream consumer
// can read static_scale directly.
int fused_adaln_static_quant_bf16_to_int8(
    void const* x_bf16,
    void const* scale_bf16,
    void const* shift_bf16,
    void const* static_scale,  // (B*S,) FP32 - pre-computed per-row scale
    void* out_i8,
    void* out_scale_copy,      // (B*S,) FP32 - optional; writes static_scale through
    int B,
    int S,
    int K,
    float eps,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
