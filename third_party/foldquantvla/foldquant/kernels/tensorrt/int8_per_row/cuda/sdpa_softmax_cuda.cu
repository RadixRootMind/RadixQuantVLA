// Row-wise softmax kernel for the SDPA scores tensor (in-place).
//
// One block per row of length S. Block size = 64 threads (2 warps), each
// thread handles ceil(S/64) elements. For S=51 each thread does ~1 element,
// reduction via warp shuffle.
//
// Numerically stable: max → exp(x-max) → sum → divide.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

#include "sdpa_softmax.h"

namespace gr00t {
namespace sdpa_softmax {

namespace {

constexpr int kThreadsPerBlock = 64;
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

// Mask-aware kernel. `mask` may be nullptr (no mask) or a BF16 row of length S
// shared by `rows_per_mask` consecutive io rows. The mask value is added to the
// FP32 score (additive convention; -1e4 for mask-out, 0 for attend), then the
// standard 3-pass softmax runs over the masked scores.
__global__ void softmax_inplace_kernel(__nv_bfloat16* __restrict__ io, int S,
                                        __nv_bfloat16 const* __restrict__ mask,
                                        int rows_per_mask) {
    const int row = blockIdx.x;
    __nv_bfloat16* row_ptr = io + (size_t)row * S;
    __nv_bfloat16 const* mask_row = (mask != nullptr)
        ? (mask + (size_t)(row / rows_per_mask) * S) : nullptr;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    extern __shared__ float smem[];
    // Layout: row_vals[S] (fp32) + warp_partials[2] (max, sum)
    float* row_vals = smem;
    float* warp_partials = smem + S;

    // Pass 1: load BF16 → FP32 (+ optional mask), compute local max.
    float t_max = -INFINITY;
    for (int k = tid; k < S; k += blockDim.x) {
        float v = __bfloat162float(row_ptr[k]);
        if (mask_row != nullptr) v += __bfloat162float(mask_row[k]);
        row_vals[k] = v;
        if (v > t_max) t_max = v;
    }
    float w_max = warp_reduce_max(t_max);

    const int kWarpsPerBlock = blockDim.x / kWarpSize;
    if (lane_id == 0) warp_partials[warp_id] = w_max;
    __syncthreads();

    float row_max;
    if (warp_id == 0) {
        float v = (lane_id < kWarpsPerBlock) ? warp_partials[lane_id] : -INFINITY;
        v = warp_reduce_max(v);
        if (lane_id == 0) warp_partials[0] = v;
    }
    __syncthreads();
    row_max = warp_partials[0];

    // Pass 2: exp(x - max), accumulate sum.
    float t_sum = 0.f;
    for (int k = tid; k < S; k += blockDim.x) {
        float e = expf(row_vals[k] - row_max);
        row_vals[k] = e;
        t_sum += e;
    }
    float w_sum = warp_reduce_sum(t_sum);
    if (lane_id == 0) warp_partials[warp_id] = w_sum;
    __syncthreads();

    float row_sum;
    if (warp_id == 0) {
        float v = (lane_id < kWarpsPerBlock) ? warp_partials[lane_id] : 0.f;
        v = warp_reduce_sum(v);
        if (lane_id == 0) warp_partials[0] = v;
    }
    __syncthreads();
    row_sum = warp_partials[0];

    float inv_sum = (row_sum > 1e-12f) ? (1.0f / row_sum) : 0.f;

    // Pass 3: write back BF16 (e * inv_sum).
    for (int k = tid; k < S; k += blockDim.x) {
        row_ptr[k] = __float2bfloat16(row_vals[k] * inv_sum);
    }
}

}  // anon

static int launch_masked(void* io_bf16, void const* mask_bf16, int rows_per_mask,
                          int HS, int S, cudaStream_t stream) {
    if (HS <= 0 || S <= 0) return 0;
    dim3 grid(HS, 1, 1);
    dim3 block(kThreadsPerBlock, 1, 1);
    size_t smem_bytes = sizeof(float) * (S + 2 * (kThreadsPerBlock / kWarpSize));
    softmax_inplace_kernel<<<grid, block, smem_bytes, stream>>>(
        static_cast<__nv_bfloat16*>(io_bf16), S,
        static_cast<__nv_bfloat16 const*>(mask_bf16),
        rows_per_mask > 0 ? rows_per_mask : 1);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace sdpa_softmax
}  // namespace gr00t

extern "C" int sdpa_softmax_bf16_inplace(
    void* io_bf16, int HS, int S, cudaStream_t stream) {
    return gr00t::sdpa_softmax::launch_masked(io_bf16, nullptr, 1, HS, S, stream);
}

extern "C" int sdpa_softmax_bf16_inplace_masked(
    void* io_bf16, void const* mask_bf16, int rows_per_mask,
    int HS, int S, cudaStream_t stream) {
    return gr00t::sdpa_softmax::launch_masked(io_bf16, mask_bf16, rows_per_mask,
                                               HS, S, stream);
}
