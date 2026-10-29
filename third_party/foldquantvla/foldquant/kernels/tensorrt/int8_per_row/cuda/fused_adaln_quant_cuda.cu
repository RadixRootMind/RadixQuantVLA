// Phase 3: Fused LayerNorm + AdaLN + per-row INT8 quantization kernel.
//
// Single-pass-ish algorithm per token:
//   Pass 1: read x_row, accumulate sum and sum-of-squares in fp32.
//   Reduce: warp-shuffle → block reduction in shared memory → mean, rstd.
//   Pass 2: compute y[k] = (x[k]-mean)*rstd * (1+scale[k]) + shift[k] in fp32,
//           store in shared mem, find amax via warp reduction.
//   Pass 3: write int8 = clamp(round(y[k] / quant_scale), -127, 127).
//
// Shared mem layout (dynamic): K * float (post-modulation buffer)
//                            + 32 floats (warp reduction scratch).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

#include "fwht.cuh"
#include "fused_adaln_quant.h"

namespace gr00t {
namespace fused_adaln_quant {

namespace {

constexpr int kThreadsPerBlock = 256;  // 8 warps
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

__global__ void fused_adaln_quant_kernel(
    __nv_bfloat16 const* __restrict__ x,
    __nv_bfloat16 const* __restrict__ scale,
    __nv_bfloat16 const* __restrict__ shift,
    int8_t* __restrict__ out_i8,
    float* __restrict__ out_scale,
    int B,
    int S,
    int K,
    float eps) {
    const int token_id = blockIdx.x;
    if (token_id >= B * S) return;

    const int b = token_id / S;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    __nv_bfloat16 const* x_row = x + (size_t)token_id * K;
    __nv_bfloat16 const* scale_row = scale + (size_t)b * K;
    __nv_bfloat16 const* shift_row = shift + (size_t)b * K;
    int8_t* out_row = out_i8 + (size_t)token_id * K;

    // Dynamic shared memory layout:
    //   y_buf  : K floats (post-modulation values to be quantized)
    //   warp_partials : kWarpsPerBlock floats (used as scratch for warp reductions)
    extern __shared__ float smem[];
    float* y_buf = smem;
    float* warp_partials = smem + K;  // up to kThreadsPerBlock/32 = 8 floats

    // --- Pass 1: compute sum and sum-of-squares ---
    float t_sum = 0.f, t_sumsq = 0.f;
    for (int k = tid; k < K; k += blockDim.x) {
        float v = __bfloat162float(x_row[k]);
        t_sum += v;
        t_sumsq += v * v;
    }
    float w_sum = warp_reduce_sum(t_sum);
    float w_sumsq = warp_reduce_sum(t_sumsq);

    // Store warp partials to shared, then reduce across warps in warp 0.
    // Pack: warp_partials[w] = sum_w, warp_partials[w + kWarpsPerBlock] = sumsq_w
    const int kWarpsPerBlock = blockDim.x / kWarpSize;
    if (lane_id == 0) {
        warp_partials[warp_id] = w_sum;
        warp_partials[warp_id + kWarpsPerBlock] = w_sumsq;
    }
    __syncthreads();

    float mean_v, rstd_v;
    if (warp_id == 0) {
        float v_sum   = (lane_id < kWarpsPerBlock) ? warp_partials[lane_id] : 0.f;
        float v_sumsq = (lane_id < kWarpsPerBlock) ? warp_partials[lane_id + kWarpsPerBlock] : 0.f;
        v_sum   = warp_reduce_sum(v_sum);
        v_sumsq = warp_reduce_sum(v_sumsq);
        if (lane_id == 0) {
            float inv_K = 1.0f / (float)K;
            mean_v = v_sum * inv_K;
            float var = v_sumsq * inv_K - mean_v * mean_v;
            if (var < 0.f) var = 0.f;  // numerical guard
            rstd_v = rsqrtf(var + eps);
            warp_partials[0] = mean_v;
            warp_partials[1] = rstd_v;
        }
    }
    __syncthreads();
    mean_v = warp_partials[0];
    rstd_v = warp_partials[1];

    // --- Pass 2: modulate, store to shared, find row amax ---
    float t_amax = 0.f;
    for (int k = tid; k < K; k += blockDim.x) {
        float xv = __bfloat162float(x_row[k]);
        float sv = __bfloat162float(scale_row[k]);
        float shv = __bfloat162float(shift_row[k]);
        float y  = (xv - mean_v) * rstd_v * (1.0f + sv) + shv;
        y_buf[k] = y;
        float a = fabsf(y);
        if (a > t_amax) t_amax = a;
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


// FoldQuant variant: adaLN modulation, then the SmoothQuant channel divide and the
// block-diagonal butterfly, and only then the per-row amax.
//
// adaLN's (1+scale)/shift are elementwise, so they commute with nothing the
// rotation does: the rotation mixes channels and must sit between the
// modulation and the quantizer. The amax therefore has to be taken on the
// rotated row: it is the row the INT8 grid stores, and the offline weight fold
// inverts the rotation on the other side of the GEMM.
__global__ void fused_adaln_fwht_quant_kernel(
    __nv_bfloat16 const* __restrict__ x,
    __nv_bfloat16 const* __restrict__ scale,
    __nv_bfloat16 const* __restrict__ shift,
    float const* __restrict__ act_scale_pre,   // (K,) RAW-frame vector or nullptr
    int8_t* __restrict__ out_i8,
    float* __restrict__ out_scale,
    int B,
    int S,
    int K,
    float eps,
    int rot_bs) {
    const int token_id = blockIdx.x;
    if (token_id >= B * S) return;

    const int b = token_id / S;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int kWarpsPerBlock = blockDim.x / kWarpSize;

    __nv_bfloat16 const* x_row = x + (size_t)token_id * K;
    __nv_bfloat16 const* scale_row = scale + (size_t)b * K;
    __nv_bfloat16 const* shift_row = shift + (size_t)b * K;
    int8_t* out_row = out_i8 + (size_t)token_id * K;

    extern __shared__ float smem[];
    float* y_buf = smem;
    float* warp_partials = smem + K;

    // --- Pass 1: LayerNorm statistics (identical to the plain kernel) ---
    float t_sum = 0.f, t_sumsq = 0.f;
    for (int k = tid; k < K; k += blockDim.x) {
        const float xv = __bfloat162float(x_row[k]);
        t_sum += xv;
        t_sumsq += xv * xv;
    }
    float w_sum = warp_reduce_sum(t_sum);
    float w_sumsq = warp_reduce_sum(t_sumsq);
    if (lane_id == 0) {
        warp_partials[warp_id] = w_sum;
        warp_partials[warp_id + kWarpsPerBlock] = w_sumsq;
    }
    __syncthreads();
    if (warp_id == 0) {
        float v_sum = (lane_id < kWarpsPerBlock) ? warp_partials[lane_id] : 0.f;
        float v_sumsq = (lane_id < kWarpsPerBlock) ? warp_partials[lane_id + kWarpsPerBlock] : 0.f;
        v_sum = warp_reduce_sum(v_sum);
        v_sumsq = warp_reduce_sum(v_sumsq);
        if (lane_id == 0) {
            const float mean = v_sum / (float)K;
            warp_partials[0] = mean;
            warp_partials[1] = rsqrtf(v_sumsq / (float)K - mean * mean + eps);
        }
    }
    __syncthreads();
    const float mean_v = warp_partials[0];
    const float rstd_v = warp_partials[1];
    __syncthreads();

    // --- Pass 2: modulate, raw-frame divide, stage for the rotation ---
    for (int k = tid; k < K; k += blockDim.x) {
        const float xv = __bfloat162float(x_row[k]);
        const float sv = __bfloat162float(scale_row[k]);
        const float shv = __bfloat162float(shift_row[k]);
        float y = (xv - mean_v) * rstd_v * (1.0f + sv) + shv;
        if (act_scale_pre != nullptr) {
            const float sc = act_scale_pre[k];
            y = (sc > 1e-12f) ? (y / sc) : 0.f;
        }
        y_buf[k] = y;
    }
    __syncthreads();

    // --- Pass 3: the rotation (syncs internally) ---
    gr00t::fwht::block_fwht_smem(y_buf, K, rot_bs);

    // --- Pass 4: amax of the ROTATED row, then quantize ---
    float t_amax = 0.f;
    for (int k = tid; k < K; k += blockDim.x) {
        const float a = fabsf(y_buf[k]);
        if (a > t_amax) t_amax = a;
    }
    float w_amax = warp_reduce_max(t_amax);
    if (lane_id == 0) warp_partials[warp_id] = w_amax;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane_id < kWarpsPerBlock) ? warp_partials[lane_id] : 0.f;
        v = warp_reduce_max(v);
        if (lane_id == 0) warp_partials[0] = v;
    }
    __syncthreads();

    const float q_scale = warp_partials[0] * (1.0f / 127.0f);
    const float inv_q_scale = (q_scale > 1e-12f) ? (1.0f / q_scale) : 0.f;
    if (tid == 0) out_scale[token_id] = q_scale;

    for (int k = tid; k < K; k += blockDim.x) {
        float q = y_buf[k] * inv_q_scale;
        q = fmaxf(-127.0f, fminf(127.0f, rintf(q)));
        out_row[k] = static_cast<int8_t>(q);
    }
}

// Static-scale variant: skips dynamic amax reduction in favor of a precomputed
// per-row scale from offline calibration.
__global__ void fused_adaln_static_quant_kernel(
    __nv_bfloat16 const* __restrict__ x,
    __nv_bfloat16 const* __restrict__ scale,
    __nv_bfloat16 const* __restrict__ shift,
    float const* __restrict__ static_scale,
    int8_t* __restrict__ out_i8,
    float* __restrict__ out_scale_copy,
    int B, int S, int K, float eps) {
    const int token_id = blockIdx.x;
    if (token_id >= B * S) return;
    const int b = token_id / S;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    __nv_bfloat16 const* x_row     = x     + (size_t)token_id * K;
    __nv_bfloat16 const* scale_row = scale + (size_t)b * K;
    __nv_bfloat16 const* shift_row = shift + (size_t)b * K;
    int8_t* out_row = out_i8 + (size_t)token_id * K;

    extern __shared__ float smem[];
    float* warp_partials = smem;  // small region

    // Pass 1: sum + sumsq for LN stats.
    float t_sum = 0.f, t_sumsq = 0.f;
    for (int k = tid; k < K; k += blockDim.x) {
        float v = __bfloat162float(x_row[k]);
        t_sum += v;
        t_sumsq += v * v;
    }
    float w_sum = warp_reduce_sum(t_sum);
    float w_sumsq = warp_reduce_sum(t_sumsq);
    const int kWarpsPerBlock = blockDim.x / kWarpSize;
    if (lane_id == 0) {
        warp_partials[warp_id] = w_sum;
        warp_partials[warp_id + kWarpsPerBlock] = w_sumsq;
    }
    __syncthreads();

    float mean_v, rstd_v;
    if (warp_id == 0) {
        float v_sum   = (lane_id < kWarpsPerBlock) ? warp_partials[lane_id] : 0.f;
        float v_sumsq = (lane_id < kWarpsPerBlock) ? warp_partials[lane_id + kWarpsPerBlock] : 0.f;
        v_sum   = warp_reduce_sum(v_sum);
        v_sumsq = warp_reduce_sum(v_sumsq);
        if (lane_id == 0) {
            float inv_K = 1.0f / (float)K;
            mean_v = v_sum * inv_K;
            float var = v_sumsq * inv_K - mean_v * mean_v;
            if (var < 0.f) var = 0.f;
            rstd_v = rsqrtf(var + eps);
            warp_partials[0] = mean_v;
            warp_partials[1] = rstd_v;
        }
    }
    __syncthreads();
    mean_v = warp_partials[0];
    rstd_v = warp_partials[1];

    float q_scale = static_scale[token_id];
    float inv_q_scale = (q_scale > 1e-12f) ? (1.0f / q_scale) : 0.f;
    if (tid == 0 && out_scale_copy != nullptr) {
        out_scale_copy[token_id] = q_scale;
    }

    // Pass 2: modulate + quantize directly (no amax reduction needed).
    for (int k = tid; k < K; k += blockDim.x) {
        float xv  = __bfloat162float(x_row[k]);
        float sv  = __bfloat162float(scale_row[k]);
        float shv = __bfloat162float(shift_row[k]);
        float y   = (xv - mean_v) * rstd_v * (1.0f + sv) + shv;
        float q   = y * inv_q_scale;
        q = fmaxf(-127.0f, fminf(127.0f, rintf(q)));
        out_row[k] = static_cast<int8_t>(q);
    }
}

}  // anon

static int launch_static(
    void const* x,
    void const* scale,
    void const* shift,
    void const* static_scale,
    void* out_i8,
    void* out_scale_copy,
    int B, int S, int K, float eps,
    cudaStream_t stream) {
    if (B <= 0 || S <= 0 || K <= 0) return 0;
    dim3 grid(B * S, 1, 1);
    dim3 block(kThreadsPerBlock, 1, 1);
    const int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
    // Need 2*kWarpsPerBlock scratch (sum + sumsq partials).
    size_t smem_bytes = sizeof(float) * (2 * kWarpsPerBlock);
    fused_adaln_static_quant_kernel<<<grid, block, smem_bytes, stream>>>(
        static_cast<__nv_bfloat16 const*>(x),
        static_cast<__nv_bfloat16 const*>(scale),
        static_cast<__nv_bfloat16 const*>(shift),
        static_cast<float const*>(static_scale),
        static_cast<int8_t*>(out_i8),
        static_cast<float*>(out_scale_copy),
        B, S, K, eps);
    return static_cast<int>(cudaGetLastError());
}

static int launch_fwht(
    void const* x,
    void const* scale,
    void const* shift,
    void const* act_scale_pre,
    void* out_i8,
    void* out_scale,
    int B, int S, int K,
    float eps,
    int rot_bs,
    cudaStream_t stream) {
    if (B <= 0 || S <= 0 || K <= 0) return 0;
    // Fail closed rather than scramble: an unusable block size would make the
    // butterfly non-orthogonal, and the offline weight fold would not invert it.
    if (!gr00t::fwht::rot_bs_valid(rot_bs, K)) return static_cast<int>(cudaErrorInvalidValue);

    dim3 grid(B * S, 1, 1);
    dim3 block(kThreadsPerBlock, 1, 1);
    const int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
    size_t smem_bytes = sizeof(float) * (K + 2 * kWarpsPerBlock);

    fused_adaln_fwht_quant_kernel<<<grid, block, smem_bytes, stream>>>(
        static_cast<__nv_bfloat16 const*>(x),
        static_cast<__nv_bfloat16 const*>(scale),
        static_cast<__nv_bfloat16 const*>(shift),
        static_cast<float const*>(act_scale_pre),
        static_cast<int8_t*>(out_i8),
        static_cast<float*>(out_scale),
        B, S, K, eps, rot_bs);
    return static_cast<int>(cudaGetLastError());
}

static int launch(
    void const* x,
    void const* scale,
    void const* shift,
    void* out_i8,
    void* out_scale,
    int B,
    int S,
    int K,
    float eps,
    cudaStream_t stream) {
    if (B <= 0 || S <= 0 || K <= 0) return 0;

    dim3 grid(B * S, 1, 1);
    dim3 block(kThreadsPerBlock, 1, 1);

    const int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
    size_t smem_bytes = sizeof(float) * (K + 2 * kWarpsPerBlock);

    fused_adaln_quant_kernel<<<grid, block, smem_bytes, stream>>>(
        static_cast<__nv_bfloat16 const*>(x),
        static_cast<__nv_bfloat16 const*>(scale),
        static_cast<__nv_bfloat16 const*>(shift),
        static_cast<int8_t*>(out_i8),
        static_cast<float*>(out_scale),
        B, S, K, eps);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace fused_adaln_quant
}  // namespace gr00t

extern "C" int fused_adaln_quant_bf16_to_int8(
    void const* x_bf16,
    void const* scale_bf16,
    void const* shift_bf16,
    void* out_i8,
    void* out_scale,
    int B,
    int S,
    int K,
    float eps,
    cudaStream_t stream) {
    return gr00t::fused_adaln_quant::launch(
        x_bf16, scale_bf16, shift_bf16, out_i8, out_scale, B, S, K, eps, stream);
}

extern "C" int fused_adaln_fwht_quant_bf16_to_int8(
    void const* x_bf16,
    void const* scale_bf16,
    void const* shift_bf16,
    void const* act_scale_pre,
    void* out_i8,
    void* out_scale,
    int B, int S, int K,
    float eps,
    int rot_bs,
    cudaStream_t stream) {
    return gr00t::fused_adaln_quant::launch_fwht(
        x_bf16, scale_bf16, shift_bf16, act_scale_pre, out_i8, out_scale, B, S, K, eps, rot_bs, stream);
}

extern "C" int fused_adaln_static_quant_bf16_to_int8(
    void const* x_bf16,
    void const* scale_bf16,
    void const* shift_bf16,
    void const* static_scale,
    void* out_i8,
    void* out_scale_copy,
    int B,
    int S,
    int K,
    float eps,
    cudaStream_t stream) {
    return gr00t::fused_adaln_quant::launch_static(
        x_bf16, scale_bf16, shift_bf16, static_scale,
        out_i8, out_scale_copy, B, S, K, eps, stream);
}
