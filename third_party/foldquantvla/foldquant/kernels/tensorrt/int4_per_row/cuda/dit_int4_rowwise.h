// DiT INT4 rowwise GEMM (CUTLASS-wrapped, true W4A4 via Ampere s4 tensor cores).
//
// True 4-bit×4-bit GEMM for GR00T-N1.6 DiT on Jetson AGX Orin (sm_87). Uses the
// native CUTLASS int4×int4→int32 tensor-core path (mma.sync.m16n8k64.s4.s4.s32),
// cloned from the INT8 rowwise v2 kernel (int8_t → cutlass::int4b_t).
//
// Pipeline (mirrors dit_int8_rowwise_v2.h, but 4-bit):
//   1. dit_int4_per_row_quant_bf16():   BF16 act (M,K) → packed INT4 (M,K/2) + scale (M,)
//   2. dit_int4_rowwise_gemm_bf16out(): INT4 A (M,K), INT4 B (K,N), scales → BF16 D (M,N)
//
// INT4 packing convention (matches CUTLASS Array<int4b_t> sub-byte order):
//   element index i → byte i/2; even i → low nibble, odd i → high nibble;
//   each nibble holds a signed 4-bit two's-complement value in [-7, 7] (we use
//   symmetric range; -8 is unused). K must be even (DiT dims are multiples of 64).

#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Quantize a BF16 (M, K) tensor to packed INT4 row-by-row using dynamic per-row
// amax.  out_i4_packed is (M, K/2) bytes (2 nibbles per byte along K).
//   out_scale[m] = amax[m] / 7.0
// Returns 0 on success, non-zero CUDA error code otherwise.
int dit_int4_per_row_quant_bf16(
    void const* in_bf16,
    void* out_i4_packed,
    void* out_scale,
    int M,
    int K,
    float act_clip,
    cudaStream_t stream);

// Rotation-enabled variant of the above: applies a block-diagonal orthonormal
// Hadamard (FWHT, `rot_bs` per block) to the row BEFORE amax/quant, flattening
// the channel axis a per-token scale cannot see. This is the LLM W4A4 residual
// path (o_proj / down_proj); the DiT's FoldQuant path uses the dense composite
// SVD·Hadamard variant below instead.
//
// The caller MUST have folded the weight side offline (W' = W·Hᵀ, same
// orthonormal Sylvester Hadamard; see int8_per_row/fwht.cuh) or the result is
// wrong with no visible symptom.
//
//   rot_bs <= 1        → delegates to dit_int4_per_row_quant_bf16 (no shared mem,
//                        previous behaviour bit-for-bit)
//   rot_bs power of 2, K % rot_bs == 0 → rotation applied (costs K floats shared)
//   anything else      → cudaErrorInvalidValue (never a silent skip)
//   act_scale_pre: (K,) FP32 pre-rotation SmoothQuant scale, or nullptr. The
//                 SmoothRot / fold-before order, matching fold_rotation_sq_before.
//                 Note this order has no BF16 hazard here: the dense path bakes
//                 R/s as BF16 and loses mantissa past alpha 0.6, while a
//                 butterfly bakes no matrix at all.
//   act_scale_ch: (K,) FP32 post-rotation SmoothQuant scale, or nullptr. The
//                 rotated value is divided by it before amax/quant, the same
//                 effect the dense-rotation path gets from folding s_ch into its baked
//                 matrix (fold_rotation_sq). Requires rot_bs > 1.
int dit_int4_per_row_quant_fwht_bf16(
    void const* in_bf16,
    void const* act_scale_pre,
    void const* act_scale_ch,
    void* out_i4_packed,
    void* out_scale,
    int M,
    int K,
    int rot_bs,
    float act_clip,
    cudaStream_t stream);

// Fused permute + block-rotation + per-row INT4 quant (FoldQuant Stage 1). Consumes
// the RAW (un-rotated) activation and applies the composite rotation internally,
// so the rotation no longer needs to be materialized as BF16 ops in the graph.
//   in_bf16:   (M, K)        raw activation
//   perm:      (K,)          INT32 channel permutation (xp[k] = x[perm[k]])
//   rotation:  (nb, bs, bs)  FP32 per-block rotation, rx[c] = Σ_i xp[i]·R[i,c]
//   out_i4_packed: (M, K/2)  packed INT4 (rotated, quantized)
//   out_scale:     (M,)      FP32 per-row scale = amax/7
// block_size (bs) must be even and a multiple of 32; K = nb*bs.
// Returns 0 on success, non-zero otherwise.
int dit_int4_per_row_rotate_quant_bf16(
    void const* in_bf16,
    void const* perm,
    void const* rotation,
    void* out_i4_packed,
    void* out_scale,
    int M,
    int K,
    int block_size,
    cudaStream_t stream);

// INT4 W4A4 GEMM with per-row act scale × per-col weight scale, BF16 output.
//   A:            (M, K)   packed INT4 row-major     ((M, K/2) bytes)
//   B:            (K, N)   packed INT4 column-major  ((N, K/2) bytes; column n
//                          occupies K/2 contiguous bytes, nibbles along K)
//   act_scale:    (M,)     FP32 per-row
//   weight_scale: (N,)     FP32 per-col (output channel)
//   D:            (M, N)   BF16 row-major
// Returns 0 on success, CUTLASS status code otherwise.
int dit_int4_rowwise_gemm_bf16out(
    void const* A,
    void const* B,
    void const* act_scale,
    void const* weight_scale,
    void* D,
    int M,
    int N,
    int K,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// FoldQuant Stage 2 macro-fusion: fused int4 GEMM EVT variants (bias / residual)
// and rotation-augmented prologue kernels. Clones of the INT8 v2 fused kernels.
// ---------------------------------------------------------------------------

// INT4 GEMM + per-output-channel bias → BF16.
//   bias: (N,) FP32; otherwise same A/B/scale/D contract as dit_int4_rowwise_gemm_bf16out.
int dit_int4_rowwise_gemm_bias_bf16out(
    void const* A, void const* B,
    void const* act_scale, void const* weight_scale,
    void const* bias,
    void* D, int M, int N, int K, cudaStream_t stream);

// INT4 GEMM + bias + residual → BF16.  residual: (M, N) BF16 row-major.
int dit_int4_rowwise_gemm_bias_residual_bf16out(
    void const* A, void const* B,
    void const* act_scale, void const* weight_scale,
    void const* bias, void const* residual,
    void* D, int M, int N, int K, cudaStream_t stream);

// INT4 GEMM + residual (no bias) → BF16.
int dit_int4_rowwise_gemm_residual_bf16out(
    void const* A, void const* B,
    void const* act_scale, void const* weight_scale,
    void const* residual,
    void* D, int M, int N, int K, cudaStream_t stream);

// ---------------------------------------------------------------------------
// B1 rotation path: produce pre-rotation BF16 (permutation folded in) → cuBLAS
// strided-batched block rotation (R staged once, reused across rows) → cheap
// per-row INT4 quant (dit_int4_per_row_quant_bf16). Replaces the per-row R
// re-read in the fused rotate+quant kernels.
// ---------------------------------------------------------------------------

// AdaLN modulation + channel permutation → BF16 (out[j] = LN(x)[perm[j]]·(1+sc)+sh).
// scale/shift may be nullptr → plain LayerNorm (FFN proj0).
int dit_adaln_permute_bf16(
    void const* x, void const* scale, void const* shift, void const* perm,
    void* out, int B, int S, int K, float eps, cudaStream_t stream);

// GELU(tanh) + channel permutation → BF16 (out[j] = gelu(in[perm[j]])).
int dit_gelu_permute_bf16(
    void const* in, void const* perm, void* out, int M, int K, cudaStream_t stream);

// Channel permutation only → BF16 (out[j] = in[perm[j]]).  Post-SDPA attn_O input.
int dit_permute_bf16(
    void const* in, void const* perm, void* out, int M, int K, cudaStream_t stream);

// Dense block-diagonal rotation via cuBLAS strided-batched BF16 GEMM.
//   xr[m, blk*bs+c] = Σ_i xp[m, blk*bs+i] · R[blk,i,c]
// cublas_handle is a cublasHandle_t; R_bf16 is (nb,bs,bs) row-major BF16; K = nb*bs.
int dit_int4_block_rotate_bf16(
    void* cublas_handle,
    void const* xp_bf16, void const* R_bf16, void* xr_bf16,
    int M, int K, int block_size, cudaStream_t stream);

// AdaLN modulation GEMV: INT4 weight-only, BF16 activation (W4A16).
//   out[m,j] = scale[j] * Σ_i x[m,i]·int4(w[j,i]) + bias[j]
// w: row-major [out, in] INT4 packed 2/byte along in; x/scale/bias/out BF16.
int dit_adaln_gemv_int4_bf16(
    void const* x, void const* wq, void const* scale, void const* bias,
    void* out, int M, int in_dim, int out_dim, cudaStream_t stream);

// AdaLN modulation GEMV: INT4 weight + INT4 activation (W4A4). Same weight
// packing/scale as dit_adaln_gemv_int4_bf16; the BF16 input row is dynamically
// per-row int4-quantized (amax/7) internally.
//   out[m,j] = sa[m]·scale[j]·Σ_i q(x[m,i])·int4(w[j,i]) + bias[j], sa=amax/7.
int dit_adaln_gemv_int4a4_bf16(
    void const* x, void const* wq, void const* scale, void const* bias,
    void* out, int M, int in_dim, int out_dim, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
