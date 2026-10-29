// LLM W4A4: RMSNorm + block-Hadamard + per-row INT4 quant prologue.
//
// INT4 clone of rmsnorm_per_row_quant_cuda.cu's dynamic kernel, feeding the
// FusedRmsNormLinearInt4 plugin (merged Q+K+V and merged gate+up GEMMs).
// Differences from the INT8 original, and only these:
//   - quant range 127 -> 7 (symmetric [-7, 7]; the -8 code is never emitted, so
//     the kernel matches the Python packer and the simulator's qmax_of(4));
//   - output is packed 2 nibbles/byte, (B*S, K/2) bytes, even k -> low nibble;
//   - the static-scale variant is deliberately absent. W4A4 is dynamic-only:
//     static per-tensor activation scale is what made `int8_w8a8` collapse to
//     SR 0.561, and static x rotation was already guard-rejected on the INT8 path.
//
// Algorithm per token (B*S blocks, blockDim.x = 256 threads = 8 warps):
//   Pass 1: read x_row, accumulate sumsq in fp32; reduce -> rstd.
//   Pass 2: y[k] = x[k] * rstd * gamma[k] into shared; optional FWHT; amax.
//   Pass 3: pack clamp(round(y[k] / (amax/7)), -7, 7) into nibbles.
//
// Shared mem: K floats (y buffer) + WarpsPerBlock floats (warp reduction).
// 8 KB at K=2048, 24 KB at K=6144, inside sm_87's 96 KB budget.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

#include "rmsnorm_per_row_quant_int4.h"
#include "fwht.cuh"

namespace gr00t {
namespace rmsnorm_per_row_quant_int4 {

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kWarpSize = 32;

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, off);
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, v, off);
        if (other > v) v = other;
    }
    return v;
}

// rot_bs <= 1 disables the rotation; the kernel then keeps the fused
// single-pass amax (no extra barrier, no re-read of y_buf) exactly like the
// INT8 original. That fused branch is not an optimisation detail: losing it
// was a measured regression on the INT8 path.
__global__ void rmsnorm_per_row_quant_int4_kernel(
    __nv_bfloat16 const* __restrict__ x,
    __nv_bfloat16 const* __restrict__ gamma,
    int8_t* __restrict__ out_i4,   // (B*S, K/2) packed
    float* __restrict__ out_scale,
    int B, int S, int K, float eps, int rot_bs, float act_clip) {
    const int token_id = blockIdx.x;
    if (token_id >= B * S) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    __nv_bfloat16 const* x_row = x + (size_t)token_id * K;
    int8_t* out_row = out_i4 + (size_t)token_id * (K / 2);

    extern __shared__ float smem[];
    float* y_buf = smem;
    float* warp_partials = smem + K;

    // --- Pass 1: sumsq for RMSNorm rstd ---
    // Stage x in y_buf while summing. The buffer is allocated either way, and
    // pass 2 would otherwise re-read the whole row from global memory: K bf16
    // loads per token, ~27 MB of redundant DRAM traffic per LLM forward on a
    // kernel that is bandwidth-bound. Each thread reads back only what it wrote
    // (identical stride), so this needs no extra barrier.
    float t_sumsq = 0.f;
    for (int k = tid; k < K; k += blockDim.x) {
        float v = __bfloat162float(x_row[k]);
        y_buf[k] = v;
        t_sumsq += v * v;
    }
    float w_sumsq = warp_reduce_sum(t_sumsq);
    const int kWarpsPerBlock = blockDim.x / kWarpSize;
    if (lane_id == 0) {
        warp_partials[warp_id] = w_sumsq;
    }
    __syncthreads();

    float rstd_v;
    if (warp_id == 0) {
        float v_sumsq = (lane_id < kWarpsPerBlock) ? warp_partials[lane_id] : 0.f;
        v_sumsq = warp_reduce_sum(v_sumsq);
        if (lane_id == 0) {
            float mean_sq = v_sumsq * (1.0f / (float)K);
            warp_partials[0] = rsqrtf(mean_sq + eps);
        }
    }
    __syncthreads();
    rstd_v = warp_partials[0];

    // --- Pass 2: y[k] = x[k] * rstd * gamma[k]; amax (after rotation) ---
    float t_amax = 0.f;
    if (gr00t::fwht::rot_bs_valid(rot_bs, K)) {
        for (int k = tid; k < K; k += blockDim.x) {
            y_buf[k] *= rstd_v * __bfloat162float(gamma[k]);
        }
        __syncthreads();
        // Block-diagonal Hadamard spreads the channel outliers a per-token scale
        // cannot see. Must run BEFORE amax. Weight side is folded offline
        // (W' = W.Hᵀ), so the product is unchanged.
        gr00t::fwht::block_fwht_smem(y_buf, K, rot_bs);  // syncs internally
        for (int k = tid; k < K; k += blockDim.x) {
            float a = fabsf(y_buf[k]);
            if (a > t_amax) t_amax = a;
        }
    } else {
        for (int k = tid; k < K; k += blockDim.x) {
            float y = y_buf[k] * rstd_v * __bfloat162float(gamma[k]);
            y_buf[k] = y;
            float a = fabsf(y);
            if (a > t_amax) t_amax = a;
        }
    }
    float w_amax = warp_reduce_max(t_amax);
    __syncthreads();
    if (lane_id == 0) {
        warp_partials[warp_id] = w_amax;
    }
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane_id < kWarpsPerBlock) ? warp_partials[lane_id] : 0.f;
        v = warp_reduce_max(v);
        if (lane_id == 0) {
            warp_partials[0] = v;
        }
    }
    __syncthreads();
    const float row_amax = warp_partials[0];

    const float q_scale = act_clip * row_amax * (1.0f / 7.0f);
    const float inv_q_scale = (q_scale > 1e-12f) ? (1.0f / q_scale) : 0.f;
    if (tid == 0) {
        out_scale[token_id] = q_scale;
    }

    // --- Pass 3: quantize and pack, one thread per output byte, so the two
    // nibbles of a byte are always written by the same thread. ---
    const int nbytes = K / 2;
    for (int b = tid; b < nbytes; b += blockDim.x) {
        int q0 = (int)fmaxf(-7.0f, fminf(7.0f, rintf(y_buf[2 * b]     * inv_q_scale)));
        int q1 = (int)fmaxf(-7.0f, fminf(7.0f, rintf(y_buf[2 * b + 1] * inv_q_scale)));
        uint8_t packed = (uint8_t)((q0 & 0xF) | ((q1 & 0xF) << 4));
        out_row[b] = (int8_t)packed;
    }
}

}  // anon

}  // namespace rmsnorm_per_row_quant_int4
}  // namespace gr00t

extern "C" int rmsnorm_fwht_per_row_quant_bf16_to_int4(
    void const* x_bf16,
    void const* gamma_bf16,
    void* out_i4_packed,
    void* out_scale,
    int B, int S, int K, float eps, int rot_bs, float act_clip,
    cudaStream_t stream) {
    if (B <= 0 || S <= 0 || K <= 0) return 0;
    if (K % 2 != 0) return static_cast<int>(cudaErrorInvalidValue);
    // A requested-but-unusable rot_bs is a build/config error, not something to
    // silently ignore: the weights are already folded with W.Hᵀ and the engine
    // would emit garbage with no visible symptom.
    if (rot_bs > 1 && !gr00t::fwht::rot_bs_valid(rot_bs, K)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    using namespace gr00t::rmsnorm_per_row_quant_int4;
    dim3 grid(B * S, 1, 1);
    dim3 block(kThreadsPerBlock, 1, 1);
    const int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
    size_t smem_bytes = sizeof(float) * ((size_t)K + kWarpsPerBlock);
    // Same wide-K guard as the INT8 twin (rmsnorm_per_row_quant_cuda.cu): the
    // row is staged in dynamic shared memory, and K > ~12K floats exceeds the
    // 48KB default per-block limit, so opt in explicitly. Gemma's 16384-wide
    // down_proj already tripped exactly this on the INT8 path.
    if (smem_bytes > 48 * 1024) {
        cudaError_t attr_rc = cudaFuncSetAttribute(
            rmsnorm_per_row_quant_int4_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes));
        if (attr_rc != cudaSuccess) return static_cast<int>(attr_rc);
    }
    rmsnorm_per_row_quant_int4_kernel<<<grid, block, smem_bytes, stream>>>(
        static_cast<__nv_bfloat16 const*>(x_bf16),
        static_cast<__nv_bfloat16 const*>(gamma_bf16),
        static_cast<int8_t*>(out_i4_packed),
        static_cast<float*>(out_scale),
        B, S, K, eps, rot_bs, act_clip);
    return static_cast<int>(cudaGetLastError());
}
