// Fused permute + block-rotation + per-row INT4 quantization (packed 2 nibbles/byte).
//
// This is the FoldQuant "Stage 1" speed kernel: it folds the composite SVD·Hadamard
// activation rotation (previously ~5 BF16 ONNX nodes per Linear) into the int4
// quant prologue, so the DitInt4RowwiseGemm plugin can consume the RAW (un-rotated)
// activation and do perm → block-rotate → int4-quant in a single kernel launch.
//
// For each row m ∈ [0, M):
//   xp[k]          = x[m, perm[k]]                                  (permutation)
//   rx[blk*bs + c] = sum_i xp[blk*bs + i] * R[blk, i, c]            (block rotation)
//   amax[m]        = max_k |rx[k]|
//   scale[m]       = amax[m] / 7
//   q[k]           = clamp(round(rx[k] / scale[m]), -7, 7)
//   out_i4[m, k/2] nibble(k&1) = q[k] & 0xF                         (even→low, odd→high)
//
// R is the per-block rotation stack (nb, bs, bs) FP32, with rx[c] = Σ_i xp[i]·R[i,c]
// (matches apply_rotation / build_rotation in foldquant/rotation.py).
// bs (block_size) must be even and a multiple of 32
// (FoldQuant uses 64); K = nb*bs. Cloned from per_row_quant_int4_cuda.cu.
//
// One CUDA block per row, blockDim.x == block_size. The rotation is recomputed in a
// second pass (it is ~bs/N of the GEMM cost, negligible) to avoid buffering all K
// rotated values in shared memory for the large-K FFN layers.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "dit_int4_rowwise.h"

namespace {

// Block reduction of a max over blockDim.x threads (blockDim.x a multiple of 32).
__device__ inline float block_amax(float v)
{
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, v, offset);
        if (other > v) v = other;
    }
    __shared__ float warp_amaxes[32];
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    if (lane_id == 0) warp_amaxes[warp_id] = v;
    __syncthreads();
    if (warp_id == 0) {
        int n_warps = blockDim.x >> 5;
        float w = (lane_id < n_warps) ? warp_amaxes[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_xor_sync(0xffffffff, w, offset);
            if (other > w) w = other;
        }
        if (lane_id == 0) warp_amaxes[0] = w;
    }
    __syncthreads();
    return warp_amaxes[0];
}

// blockDim.x == bs; dynamic shared memory holds xpsh[bs] then rxsh[bs].
__global__ void per_row_rotate_quant_bf16_to_int4_kernel(
    const __nv_bfloat16* __restrict__ in,
    const int* __restrict__ perm,
    const float* __restrict__ R,   // (nb, bs, bs), rx[c] = Σ_i xp[i]·R[i,c]
    int8_t* __restrict__ out_i4,    // (M, K/2) packed
    float* __restrict__ out_scale,
    int M,
    int K,
    int bs)
{
    const int m = blockIdx.x;
    if (m >= M) return;

    const int nb = K / bs;
    const int t = threadIdx.x;                 // owns rotated column c == t within each block
    const __nv_bfloat16* row_in = in + (size_t)m * K;

    extern __shared__ float smem[];
    float* xpsh = smem;                         // [bs]
    float* rxsh = smem + bs;                    // [bs]

    // ---- Pass 1: global amax over the rotated activation (rx not stored) ----
    float t_amax = 0.0f;
    for (int blk = 0; blk < nb; ++blk) {
        xpsh[t] = __bfloat162float(row_in[perm[blk * bs + t]]);
        __syncthreads();
        const float* Rb = R + (size_t)blk * bs * bs;
        float acc = 0.0f;
        for (int i = 0; i < bs; ++i) {
            acc += xpsh[i] * Rb[i * bs + t];   // rx[blk*bs + t]
        }
        float a = fabsf(acc);
        if (a > t_amax) t_amax = a;
        __syncthreads();                        // protect xpsh before next blk
    }

    float row_amax = block_amax(t_amax);
    float scale = row_amax * (1.0f / 7.0f);
    float inv_scale = (scale > 1e-12f) ? (1.0f / scale) : 0.0f;
    if (t == 0) out_scale[m] = scale;

    // ---- Pass 2: recompute rx, quantize + pack (one byte per thread c<bs/2) ----
    int8_t* row_out = out_i4 + (size_t)m * (K / 2);
    const int half = bs / 2;
    for (int blk = 0; blk < nb; ++blk) {
        xpsh[t] = __bfloat162float(row_in[perm[blk * bs + t]]);
        __syncthreads();
        const float* Rb = R + (size_t)blk * bs * bs;
        float acc = 0.0f;
        for (int i = 0; i < bs; ++i) {
            acc += xpsh[i] * Rb[i * bs + t];
        }
        rxsh[t] = acc;
        __syncthreads();
        if (t < half) {
            float v0 = rxsh[2 * t];
            float v1 = rxsh[2 * t + 1];
            int q0 = (int)fmaxf(-7.0f, fminf(7.0f, rintf(v0 * inv_scale)));
            int q1 = (int)fmaxf(-7.0f, fminf(7.0f, rintf(v1 * inv_scale)));
            uint8_t packed = (uint8_t)((q0 & 0xF) | ((q1 & 0xF) << 4));
            row_out[blk * half + t] = (int8_t)packed;
        }
        __syncthreads();
    }
}

}  // namespace

extern "C" int dit_int4_per_row_rotate_quant_bf16(
    void const* in_bf16,
    void const* perm,
    void const* rotation,
    void* out_i4_packed,
    void* out_scale,
    int M,
    int K,
    int block_size,
    cudaStream_t stream)
{
    if (M <= 0 || K <= 0 || block_size <= 0) return 0;
    if (block_size & 31) return -2;          // require multiple of 32 (bs=64)
    dim3 grid(M, 1, 1);
    dim3 block(block_size, 1, 1);
    size_t shmem = (size_t)2 * block_size * sizeof(float);
    per_row_rotate_quant_bf16_to_int4_kernel<<<grid, block, shmem, stream>>>(
        static_cast<const __nv_bfloat16*>(in_bf16),
        static_cast<const int*>(perm),
        static_cast<const float*>(rotation),
        static_cast<int8_t*>(out_i4_packed),
        static_cast<float*>(out_scale),
        M, K, block_size);
    return static_cast<int>(cudaGetLastError());
}
