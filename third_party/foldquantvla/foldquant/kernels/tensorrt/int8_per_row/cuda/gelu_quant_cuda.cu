// Phase 4: Fused GELU(tanh) + per-row INT8 quantization kernel.
//
// Two-pass algorithm per row:
//   Pass 1: read x_row, apply GELU(tanh) in fp32, store fp32 in shared mem,
//           accumulate per-row amax via warp/block reductions.
//   Pass 2: quantize: int8 = clamp(round(gelu_val / (amax/127)), -127, 127).
//
// Same structure as fused_adaln_quant_cuda but without LayerNorm/AdaLN.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

#include "fwht.cuh"
#include "gelu_quant.h"

namespace gr00t {
namespace gelu_quant {

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kWarpSize = 32;

// sqrt(2/pi) constant used by the tanh GELU approximation.
__device__ __forceinline__ float gelu_tanh(float x) {
    const float c = 0.7978845608f;       // sqrt(2/pi)
    const float k = 0.044715f;
    float inner = c * (x + k * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, v, off);
        if (other > v) v = other;
    }
    return v;
}

__global__ void gelu_quant_kernel(
    __nv_bfloat16 const* __restrict__ in,
    int8_t* __restrict__ out_i8,
    float* __restrict__ out_scale,
    int M,
    int K) {
    const int m = blockIdx.x;
    if (m >= M) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int kWarpsPerBlock = blockDim.x / kWarpSize;

    __nv_bfloat16 const* in_row = in + (size_t)m * K;
    int8_t* out_row = out_i8 + (size_t)m * K;

    // Dynamic shared memory:
    //   y_buf : K floats (GELU outputs)
    //   warp_partials : kWarpsPerBlock floats (warp max scratch)
    extern __shared__ float smem[];
    float* y_buf = smem;
    float* warp_partials = smem + K;

    // Pass 1: GELU + amax accumulation.
    float t_amax = 0.f;
    for (int k = tid; k < K; k += blockDim.x) {
        float v = __bfloat162float(in_row[k]);
        float y = gelu_tanh(v);
        y_buf[k] = y;
        float a = fabsf(y);
        if (a > t_amax) t_amax = a;
    }
    float w_amax = warp_reduce_max(t_amax);
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
    float row_amax = warp_partials[0];

    float q_scale = row_amax * (1.0f / 127.0f);
    float inv_q_scale = (q_scale > 1e-12f) ? (1.0f / q_scale) : 0.f;

    if (tid == 0) {
        out_scale[m] = q_scale;
    }

    // Pass 2: quantize.
    for (int k = tid; k < K; k += blockDim.x) {
        float q = y_buf[k] * inv_q_scale;
        q = fmaxf(-127.0f, fminf(127.0f, rintf(q)));
        out_row[k] = static_cast<int8_t>(q);
    }
}


// FoldQuant variant: SmoothQuant channel divide + block-diagonal butterfly, then the
// per-row amax.
//
// The order is the whole point. The rotation mixes channels, so it cannot be
// folded into anything elementwise that precedes it: not GELU, not a norm's
// gamma. It has to run between the activation and the quantizer, and the amax
// has to be taken AFTER it, on the rotated values the INT8 grid will actually
// hold. Measuring before the rotation and quantizing after is the silent-garbage
// case: every channel is scaled by a number derived from a different basis.
//
// `act_scale_pre` is the fold-before (SmoothRot) vector: the scale lands on the
// RAW channel, so it divides here, before the butterfly, and the matching weight
// fold multiplied the weight's raw columns offline. Pass nullptr for the
// unfolded arm; the butterfly alone is still a valid rotation.
__global__ void gelu_fwht_quant_kernel(
    __nv_bfloat16 const* __restrict__ in,
    float const* __restrict__ act_scale_pre,   // (K,) or nullptr
    int8_t* __restrict__ out_i8,
    float* __restrict__ out_scale,
    int M,
    int K,
    int rot_bs) {
    const int m = blockIdx.x;
    if (m >= M) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int kWarpsPerBlock = blockDim.x / kWarpSize;

    __nv_bfloat16 const* in_row = in + (size_t)m * K;
    int8_t* out_row = out_i8 + (size_t)m * K;

    extern __shared__ float smem[];
    float* y_buf = smem;
    float* warp_partials = smem + K;

    // Pass 1: GELU, then the raw-frame channel divide. No amax yet; it belongs
    // after the rotation.
    for (int k = tid; k < K; k += blockDim.x) {
        float y = gelu_tanh(__bfloat162float(in_row[k]));
        if (act_scale_pre != nullptr) {
            const float s = act_scale_pre[k];
            y = (s > 1e-12f) ? (y / s) : 0.f;
        }
        y_buf[k] = y;
    }
    __syncthreads();

    // Pass 2: the rotation itself (syncs internally; every thread participates).
    gr00t::fwht::block_fwht_smem(y_buf, K, rot_bs);

    // Pass 3: per-row amax of the ROTATED row.
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
    if (tid == 0) out_scale[m] = q_scale;

    for (int k = tid; k < K; k += blockDim.x) {
        float q = y_buf[k] * inv_q_scale;
        q = fmaxf(-127.0f, fminf(127.0f, rintf(q)));
        out_row[k] = static_cast<int8_t>(q);
    }
}

// Static-scale variant.
__global__ void gelu_static_quant_kernel(
    __nv_bfloat16 const* __restrict__ in,
    float const* __restrict__ static_scale,
    int8_t* __restrict__ out_i8,
    float* __restrict__ out_scale_copy,
    int M, int K) {
    const int m = blockIdx.x;
    if (m >= M) return;

    const int tid = threadIdx.x;

    __nv_bfloat16 const* in_row = in + (size_t)m * K;
    int8_t* out_row = out_i8 + (size_t)m * K;

    float q_scale = static_scale[m];
    float inv_q_scale = (q_scale > 1e-12f) ? (1.0f / q_scale) : 0.f;
    if (tid == 0 && out_scale_copy != nullptr) {
        out_scale_copy[m] = q_scale;
    }

    for (int k = tid; k < K; k += blockDim.x) {
        float v = __bfloat162float(in_row[k]);
        float y = gelu_tanh(v);
        float q = y * inv_q_scale;
        q = fmaxf(-127.0f, fminf(127.0f, rintf(q)));
        out_row[k] = static_cast<int8_t>(q);
    }
}

}  // anon

static int launch_fwht(
    void const* in,
    void const* act_scale_pre,
    void* out_i8,
    void* out_scale,
    int M,
    int K,
    int rot_bs,
    cudaStream_t stream) {
    if (M <= 0 || K <= 0) return 0;
    // Fail closed: an invalid block would make the butterfly a non-orthogonal
    // scramble that the offline weight fold does not invert. The caller must
    // route an unrotatable width to the plain kernel, not silently get garbage.
    if (!gr00t::fwht::rot_bs_valid(rot_bs, K)) return static_cast<int>(cudaErrorInvalidValue);

    dim3 grid(M, 1, 1);
    dim3 block(kThreadsPerBlock, 1, 1);
    const int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
    size_t smem_bytes = sizeof(float) * (K + kWarpsPerBlock);

    gelu_fwht_quant_kernel<<<grid, block, smem_bytes, stream>>>(
        static_cast<__nv_bfloat16 const*>(in),
        static_cast<float const*>(act_scale_pre),
        static_cast<int8_t*>(out_i8),
        static_cast<float*>(out_scale),
        M, K, rot_bs);
    return static_cast<int>(cudaGetLastError());
}

static int launch_static(
    void const* in,
    void const* static_scale,
    void* out_i8,
    void* out_scale_copy,
    int M,
    int K,
    cudaStream_t stream) {
    if (M <= 0 || K <= 0) return 0;
    dim3 grid(M, 1, 1);
    dim3 block(kThreadsPerBlock, 1, 1);
    gelu_static_quant_kernel<<<grid, block, 0, stream>>>(
        static_cast<__nv_bfloat16 const*>(in),
        static_cast<float const*>(static_scale),
        static_cast<int8_t*>(out_i8),
        static_cast<float*>(out_scale_copy),
        M, K);
    return static_cast<int>(cudaGetLastError());
}

static int launch(
    void const* in,
    void* out_i8,
    void* out_scale,
    int M,
    int K,
    cudaStream_t stream) {
    if (M <= 0 || K <= 0) return 0;

    dim3 grid(M, 1, 1);
    dim3 block(kThreadsPerBlock, 1, 1);
    const int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
    size_t smem_bytes = sizeof(float) * (K + kWarpsPerBlock);

    gelu_quant_kernel<<<grid, block, smem_bytes, stream>>>(
        static_cast<__nv_bfloat16 const*>(in),
        static_cast<int8_t*>(out_i8),
        static_cast<float*>(out_scale),
        M, K);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace gelu_quant
}  // namespace gr00t

extern "C" int dit_gelu_quant_bf16_to_int8(
    void const* in_bf16,
    void* out_i8,
    void* out_scale,
    int M,
    int K,
    cudaStream_t stream) {
    return gr00t::gelu_quant::launch(in_bf16, out_i8, out_scale, M, K, stream);
}

extern "C" int dit_gelu_fwht_quant_bf16_to_int8(
    void const* in_bf16,
    void const* act_scale_pre,
    void* out_i8,
    void* out_scale,
    int M,
    int K,
    int rot_bs,
    cudaStream_t stream) {
    return gr00t::gelu_quant::launch_fwht(
        in_bf16, act_scale_pre, out_i8, out_scale, M, K, rot_bs, stream);
}

extern "C" int dit_gelu_static_quant_bf16_to_int8(
    void const* in_bf16,
    void const* static_scale,
    void* out_i8,
    void* out_scale_copy,
    int M,
    int K,
    cudaStream_t stream) {
    return gr00t::gelu_quant::launch_static(
        in_bf16, static_scale, out_i8, out_scale_copy, M, K, stream);
}
