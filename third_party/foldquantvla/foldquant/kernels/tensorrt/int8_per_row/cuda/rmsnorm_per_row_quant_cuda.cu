// LLM ladder L3+: RMSNorm + per-row INT8 quant kernel.
//
// Ported from fused_adaln_quant_cuda.cu with:
//   - mean subtract removed (RMSNorm uses sumsq-only).
//   - per-batch (scale, shift) replaced by per-channel gamma (RMSNorm.weight).
//
// Algorithm per token (B*S blocks, blockDim.x = 256 threads = 8 warps):
//   Pass 1: read x_row, accumulate sumsq in fp32; reduce → rstd.
//   Pass 2: write y[k] = x[k] * rstd * gamma[k] to shared mem; find amax.
//   Pass 3: quant y[k] / (amax/127) → INT8 row-major.
//
// Shared mem: K floats (y buffer) + 2*WarpsPerBlock floats (warp reduction).
//
// The static-scale variant skips passes 2/3's amax reduction and just quantizes
// directly with the precomputed scale.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

#include "rmsnorm_per_row_quant.h"
#include "fwht.cuh"

namespace gr00t {
namespace rmsnorm_per_row_quant {

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

// rot_bs <= 1 disables the rotation and the kernel is bit-identical to the
// pre-rotation version (the FWHT block is simply skipped).
__global__ void rmsnorm_per_row_quant_kernel(
    __nv_bfloat16 const* __restrict__ x,
    __nv_bfloat16 const* __restrict__ gamma,
    int8_t* __restrict__ out_i8,
    float* __restrict__ out_scale,
    int B, int S, int K, float eps, int rot_bs) {
    const int token_id = blockIdx.x;
    if (token_id >= B * S) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    __nv_bfloat16 const* x_row = x + (size_t)token_id * K;
    int8_t* out_row = out_i8 + (size_t)token_id * K;

    extern __shared__ float smem[];
    float* y_buf = smem;
    float* warp_partials = smem + K;

    // --- Pass 1: sumsq for RMSNorm rstd ---
    float t_sumsq = 0.f;
    for (int k = tid; k < K; k += blockDim.x) {
        float v = __bfloat162float(x_row[k]);
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
            rstd_v = rsqrtf(mean_sq + eps);
            warp_partials[0] = rstd_v;
        }
    }
    __syncthreads();
    rstd_v = warp_partials[0];

    // --- Pass 2: y[k] = x[k] * rstd * gamma[k]; amax ---
    // Two shapes: the rotation needs the whole row staged before it can start,
    // so it writes y_buf, barriers, rotates, then reduces from shared. The far
    // more common no-rotation path (every DiT/INT8 caller, rot_bs<=1) keeps the
    // original fused single pass: amax straight from the register, no barrier,
    // no re-read of y_buf.
    float t_amax = 0.f;
    if (gr00t::fwht::rot_bs_valid(rot_bs, K)) {
        for (int k = tid; k < K; k += blockDim.x) {
            float xv = __bfloat162float(x_row[k]);
            float g  = __bfloat162float(gamma[k]);
            y_buf[k] = xv * rstd_v * g;
        }
        __syncthreads();
        // Block-diagonal Hadamard spreads the channel outliers a per-token scale
        // cannot see. Must run BEFORE amax. Weight side is folded offline
        // (W' = W·Hᵀ), so the product is unchanged.
        gr00t::fwht::block_fwht_smem(y_buf, K, rot_bs);  // syncs internally
        for (int k = tid; k < K; k += blockDim.x) {
            float a = fabsf(y_buf[k]);
            if (a > t_amax) t_amax = a;
        }
    } else {
        for (int k = tid; k < K; k += blockDim.x) {
            float xv = __bfloat162float(x_row[k]);
            float g  = __bfloat162float(gamma[k]);
            float y  = xv * rstd_v * g;
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
    float row_amax;
    if (warp_id == 0) {
        float v = (lane_id < kWarpsPerBlock) ? warp_partials[lane_id] : 0.f;
        v = warp_reduce_max(v);
        if (lane_id == 0) {
            warp_partials[0] = v;
        }
    }
    __syncthreads();
    row_amax = warp_partials[0];

    float q_scale = row_amax * (1.0f / 127.0f);
    float inv_q_scale = (q_scale > 1e-12f) ? (1.0f / q_scale) : 0.f;
    if (tid == 0) {
        out_scale[token_id] = q_scale;
    }

    // --- Pass 3: quantize and write INT8 ---
    for (int k = tid; k < K; k += blockDim.x) {
        float q = y_buf[k] * inv_q_scale;
        q = fmaxf(-127.0f, fminf(127.0f, rintf(q)));
        out_row[k] = static_cast<int8_t>(q);
    }
}

__global__ void rmsnorm_per_row_static_quant_kernel(
    __nv_bfloat16 const* __restrict__ x,
    __nv_bfloat16 const* __restrict__ gamma,
    float const* __restrict__ static_scale,
    int8_t* __restrict__ out_i8,
    float* __restrict__ out_scale_copy,
    int B, int S, int K, float eps) {
    const int token_id = blockIdx.x;
    if (token_id >= B * S) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    __nv_bfloat16 const* x_row = x + (size_t)token_id * K;
    int8_t* out_row = out_i8 + (size_t)token_id * K;

    extern __shared__ float smem[];
    float* warp_partials = smem;  // small region

    // Pass 1: sumsq.
    float t_sumsq = 0.f;
    for (int k = tid; k < K; k += blockDim.x) {
        float v = __bfloat162float(x_row[k]);
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
            rstd_v = rsqrtf(mean_sq + eps);
            warp_partials[0] = rstd_v;
        }
    }
    __syncthreads();
    rstd_v = warp_partials[0];

    float q_scale = static_scale[token_id];
    float inv_q_scale = (q_scale > 1e-12f) ? (1.0f / q_scale) : 0.f;
    if (tid == 0 && out_scale_copy != nullptr) {
        out_scale_copy[token_id] = q_scale;
    }

    // Pass 2: directly quantize y[k] = x[k] * rstd * gamma[k] (no amax reduction).
    for (int k = tid; k < K; k += blockDim.x) {
        float xv = __bfloat162float(x_row[k]);
        float g  = __bfloat162float(gamma[k]);
        float y  = xv * rstd_v * g;
        float q  = y * inv_q_scale;
        q = fmaxf(-127.0f, fminf(127.0f, rintf(q)));
        out_row[k] = static_cast<int8_t>(q);
    }
}

}  // anon

static int launch_dynamic(
    void const* x,
    void const* gamma,
    void* out_i8,
    void* out_scale,
    int B, int S, int K, float eps, int rot_bs,
    cudaStream_t stream) {
    if (B <= 0 || S <= 0 || K <= 0) return 0;
    // A requested-but-unusable rot_bs is a build/config error, not something to
    // silently ignore: the weights would already be folded with W·Hᵀ and the
    // engine would emit garbage with no visible symptom.
    if (rot_bs > 1 && !gr00t::fwht::rot_bs_valid(rot_bs, K)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    dim3 grid(B * S, 1, 1);
    dim3 block(kThreadsPerBlock, 1, 1);
    const int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
    size_t smem_bytes = sizeof(float) * (K + kWarpsPerBlock);
    // Same wide-K guard as dit_int8_per_row_quant_fwht_bf16: the row is staged
    // in dynamic shared memory, and K > ~12K floats exceeds the 48KB default
    // per-block limit, so opt in explicitly (latent here: no current model norms
    // a 16K-wide activation, but the failure would be the same silent -1).
    if (smem_bytes > 48 * 1024) {
        cudaError_t attr_rc = cudaFuncSetAttribute(
            rmsnorm_per_row_quant_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes));
        if (attr_rc != cudaSuccess) return static_cast<int>(attr_rc);
    }
    rmsnorm_per_row_quant_kernel<<<grid, block, smem_bytes, stream>>>(
        static_cast<__nv_bfloat16 const*>(x),
        static_cast<__nv_bfloat16 const*>(gamma),
        static_cast<int8_t*>(out_i8),
        static_cast<float*>(out_scale),
        B, S, K, eps, rot_bs);
    return static_cast<int>(cudaGetLastError());
}

static int launch_static(
    void const* x,
    void const* gamma,
    void const* static_scale,
    void* out_i8,
    void* out_scale_copy,
    int B, int S, int K, float eps,
    cudaStream_t stream) {
    if (B <= 0 || S <= 0 || K <= 0) return 0;
    dim3 grid(B * S, 1, 1);
    dim3 block(kThreadsPerBlock, 1, 1);
    const int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
    size_t smem_bytes = sizeof(float) * kWarpsPerBlock;
    rmsnorm_per_row_static_quant_kernel<<<grid, block, smem_bytes, stream>>>(
        static_cast<__nv_bfloat16 const*>(x),
        static_cast<__nv_bfloat16 const*>(gamma),
        static_cast<float const*>(static_scale),
        static_cast<int8_t*>(out_i8),
        static_cast<float*>(out_scale_copy),
        B, S, K, eps);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace rmsnorm_per_row_quant
}  // namespace gr00t

// Legacy entry point, preserved verbatim for existing callers (rot disabled).
extern "C" int rmsnorm_per_row_quant_bf16_to_int8(
    void const* x_bf16,
    void const* gamma_bf16,
    void* out_i8,
    void* out_scale,
    int B, int S, int K, float eps,
    cudaStream_t stream) {
    return gr00t::rmsnorm_per_row_quant::launch_dynamic(
        x_bf16, gamma_bf16, out_i8, out_scale, B, S, K, eps, /*rot_bs=*/0, stream);
}

// Rotation-enabled entry point. rot_bs must be a power of two dividing K;
// rot_bs <= 1 behaves exactly like the legacy call above.
extern "C" int rmsnorm_fwht_per_row_quant_bf16_to_int8(
    void const* x_bf16,
    void const* gamma_bf16,
    void* out_i8,
    void* out_scale,
    int B, int S, int K, float eps, int rot_bs,
    cudaStream_t stream) {
    return gr00t::rmsnorm_per_row_quant::launch_dynamic(
        x_bf16, gamma_bf16, out_i8, out_scale, B, S, K, eps, rot_bs, stream);
}

extern "C" int rmsnorm_per_row_static_quant_bf16_to_int8(
    void const* x_bf16,
    void const* gamma_bf16,
    void const* static_scale,
    void* out_i8,
    void* out_scale_copy,
    int B, int S, int K, float eps,
    cudaStream_t stream) {
    return gr00t::rmsnorm_per_row_quant::launch_static(
        x_bf16, gamma_bf16, static_scale, out_i8, out_scale_copy,
        B, S, K, eps, stream);
}
