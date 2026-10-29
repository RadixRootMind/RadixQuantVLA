// Per-row dynamic INT8 activation quantization kernel.
//
// For each row m ∈ [0, M):
//   amax[m] = max over k ∈ [0,K) of |in_bf16[m,k]|
//   scale[m] = amax[m] / 127.0  (guarded against zero)
//   out_i8[m,k] = clamp(round(in_bf16[m,k] / scale[m]), -127, 127)
//
// Single-pass: each block handles one row, threads stride over K to compute
// amax via warp reduction, then write quantized output.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>

#include "dit_int8_rowwise_v2.h"
#include "fwht.cuh"

namespace cg = cooperative_groups;

namespace {

// One block per row. Block dim = 256 threads = 8 warps.
// Each thread reads K/256 elements (e.g. K=1536 → 6 elements per thread).
__global__ void per_row_quant_bf16_to_int8_kernel(
    const __nv_bfloat16* __restrict__ in,
    int8_t* __restrict__ out_i8,
    float* __restrict__ out_scale,
    int M,
    int K)
{
    const int m = blockIdx.x;
    if (m >= M) return;

    const __nv_bfloat16* row_in = in + (size_t)m * K;
    int8_t* row_out = out_i8 + (size_t)m * K;

    // Phase 1: thread-local amax over assigned columns
    float t_amax = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float v = __bfloat162float(row_in[k]);
        v = fabsf(v);
        if (v > t_amax) t_amax = v;
    }

    // Phase 2: warp-level reduction (32 threads per warp)
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, t_amax, offset);
        if (other > t_amax) t_amax = other;
    }

    // Phase 3: block-level reduction across warps via shared memory
    __shared__ float warp_amaxes[32];  // up to 32 warps (we use 8)
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    if (lane_id == 0) {
        warp_amaxes[warp_id] = t_amax;
    }
    __syncthreads();

    if (warp_id == 0) {
        int n_warps = blockDim.x >> 5;
        float v = (lane_id < n_warps) ? warp_amaxes[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_xor_sync(0xffffffff, v, offset);
            if (other > v) v = other;
        }
        if (lane_id == 0) {
            warp_amaxes[0] = v;
        }
    }
    __syncthreads();

    float row_amax = warp_amaxes[0];
    // Guard against zero to avoid div-by-zero. INT8 representable range:
    // [-127, 127]. We map amax -> 127. Scale = amax / 127.
    float scale = row_amax * (1.0f / 127.0f);
    float inv_scale = (scale > 1e-12f) ? (1.0f / scale) : 0.0f;

    if (threadIdx.x == 0) {
        out_scale[m] = scale;
    }

    // Phase 4: quantize each element using inv_scale
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float v = __bfloat162float(row_in[k]);
        float q = v * inv_scale;
        // round-half-to-even-ish via rintf; clamp to [-127, 127]
        q = fmaxf(-127.0f, fminf(127.0f, rintf(q)));
        row_out[k] = static_cast<int8_t>(q);
    }
}

// Rotation variant: block-diagonal Hadamard between the input and the
// quantizer, so the per-token scale sees a channel-flattened row.
//
// Unlike the plain kernel above this one MUST stage the row in shared memory:
// the FWHT is an in-place multi-pass butterfly, and its output (not the input)
// is what amax and the quantizer need. Shared cost is K floats: 8 KB at K=2048
// (o_proj), 24 KB at K=6144 (down_proj), both inside sm_87's 96 KB budget.
//
// The plain kernel is left untouched on purpose: `dit_int8_per_row_quant_bf16`
// is shared with the DiT's INT8 v2 plugins, which must not change behaviour.
__global__ void per_row_fwht_quant_bf16_to_int8_kernel(
    const __nv_bfloat16* __restrict__ in,
    const float* __restrict__ act_scale_pre,  // (K,) or nullptr - fold-before
    const float* __restrict__ act_scale_ch,   // (K,) or nullptr - fold-after
    int8_t* __restrict__ out_i8,
    float* __restrict__ out_scale,
    int M, int K, int rot_bs)
{
    const int m = blockIdx.x;
    if (m >= M) return;

    const __nv_bfloat16* row_in = in + (size_t)m * K;
    int8_t* row_out = out_i8 + (size_t)m * K;

    extern __shared__ float y_buf[];

    // SmoothQuant, the same two fold orders the INT4 twin carries: fold-before
    // divides the RAW channel on the way in (fold_rotation_sq_before scales the
    // rotation's input axis), fold-after divides the ROTATED channel once the
    // butterfly has run (fold_rotation_sq scales its output axis). A dense
    // matrix absorbs either; a fixed butterfly absorbs neither, so both arrive
    // as vectors and are applied on their own side of the transform.
    if (act_scale_pre != nullptr) {
        for (int k = threadIdx.x; k < K; k += blockDim.x) {
            y_buf[k] = __bfloat162float(row_in[k]) / act_scale_pre[k];
        }
    } else {
        for (int k = threadIdx.x; k < K; k += blockDim.x) {
            y_buf[k] = __bfloat162float(row_in[k]);
        }
    }
    __syncthreads();

    if (gr00t::fwht::rot_bs_valid(rot_bs, K)) {
        gr00t::fwht::block_fwht_smem(y_buf, K, rot_bs);  // syncs internally
    }

    if (act_scale_ch != nullptr) {
        for (int k = threadIdx.x; k < K; k += blockDim.x) {
            y_buf[k] /= act_scale_ch[k];
        }
        __syncthreads();
    }

    float t_amax = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float v = fabsf(y_buf[k]);
        if (v > t_amax) t_amax = v;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, t_amax, offset);
        if (other > t_amax) t_amax = other;
    }
    __shared__ float warp_amaxes[32];
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    if (lane_id == 0) warp_amaxes[warp_id] = t_amax;
    __syncthreads();
    if (warp_id == 0) {
        int n_warps = blockDim.x >> 5;
        float v = (lane_id < n_warps) ? warp_amaxes[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_xor_sync(0xffffffff, v, offset);
            if (other > v) v = other;
        }
        if (lane_id == 0) warp_amaxes[0] = v;
    }
    __syncthreads();

    const float scale = warp_amaxes[0] * (1.0f / 127.0f);
    const float inv_scale = (scale > 1e-12f) ? (1.0f / scale) : 0.0f;
    if (threadIdx.x == 0) out_scale[m] = scale;

    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float q = y_buf[k] * inv_scale;
        q = fmaxf(-127.0f, fminf(127.0f, rintf(q)));
        row_out[k] = static_cast<int8_t>(q);
    }
}

// Static-per-row variant: uses precomputed scale instead of dynamic amax.
__global__ void per_row_static_quant_bf16_to_int8_kernel(
    const __nv_bfloat16* __restrict__ in,
    const float* __restrict__ static_scale,
    int8_t* __restrict__ out_i8,
    float* __restrict__ out_scale_copy,
    int M, int K) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const __nv_bfloat16* row_in = in + (size_t)m * K;
    int8_t* row_out = out_i8 + (size_t)m * K;

    float q_scale = static_scale[m];
    float inv_q_scale = (q_scale > 1e-12f) ? (1.0f / q_scale) : 0.f;
    if (threadIdx.x == 0 && out_scale_copy != nullptr) {
        out_scale_copy[m] = q_scale;
    }
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float v = __bfloat162float(row_in[k]);
        float q = v * inv_q_scale;
        q = fmaxf(-127.0f, fminf(127.0f, rintf(q)));
        row_out[k] = static_cast<int8_t>(q);
    }
}

}  // anon namespace


extern "C" int dit_int8_per_row_quant_bf16(
    void const* in_bf16,
    void* out_i8,
    void* out_scale,
    int M,
    int K,
    cudaStream_t stream)
{
    if (M <= 0 || K <= 0) return 0;
    const int threads = 256;
    dim3 grid(M, 1, 1);
    dim3 block(threads, 1, 1);
    per_row_quant_bf16_to_int8_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(in_bf16),
        static_cast<int8_t*>(out_i8),
        static_cast<float*>(out_scale),
        M, K);
    cudaError_t e = cudaGetLastError();
    return static_cast<int>(e);
}

// Rotation-enabled per-row quant. rot_bs must be a power of two dividing K;
// rot_bs <= 1 falls back to the plain (unstaged) kernel so the no-rotation path
// keeps its exact previous behaviour and costs no shared memory.
extern "C" int dit_int8_per_row_quant_fwht_bf16(
    void const* in_bf16,
    void const* act_scale_pre,
    void const* act_scale_ch,
    void* out_i8,
    void* out_scale,
    int M,
    int K,
    int rot_bs,
    cudaStream_t stream)
{
    if (M <= 0 || K <= 0) return 0;
    if (rot_bs <= 1) {
        return dit_int8_per_row_quant_bf16(in_bf16, out_i8, out_scale, M, K, stream);
    }
    if (!gr00t::fwht::rot_bs_valid(rot_bs, K)) {
        // Weights are already folded with W·Hᵀ upstream, so silently skipping
        // the rotation here would produce a wrong result with no symptom.
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const int threads = 256;
    dim3 grid(M, 1, 1);
    dim3 block(threads, 1, 1);
    size_t smem_bytes = sizeof(float) * (size_t)K;
    // The row is staged in dynamic shared memory, so K > 12288 floats exceeds
    // the 48KB default per-block limit (Gemma's 16384-wide down_proj is the
    // first shape to hit it; every other family's intermediate is <= 6144).
    // Opting in is a per-function attribute; sm90 allows up to ~227KB.
    if (smem_bytes > 48 * 1024) {
        cudaError_t attr_rc = cudaFuncSetAttribute(
            per_row_fwht_quant_bf16_to_int8_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes));
        if (attr_rc != cudaSuccess) return static_cast<int>(attr_rc);
    }
    per_row_fwht_quant_bf16_to_int8_kernel<<<grid, block, smem_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(in_bf16),
        static_cast<const float*>(act_scale_pre),
        static_cast<const float*>(act_scale_ch),
        static_cast<int8_t*>(out_i8),
        static_cast<float*>(out_scale),
        M, K, rot_bs);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int dit_int8_per_row_static_quant_bf16(
    void const* in_bf16,
    void const* static_scale,
    void* out_i8,
    void* out_scale_copy,
    int M,
    int K,
    cudaStream_t stream)
{
    if (M <= 0 || K <= 0) return 0;
    const int threads = 256;
    dim3 grid(M, 1, 1);
    dim3 block(threads, 1, 1);
    per_row_static_quant_bf16_to_int8_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(in_bf16),
        static_cast<const float*>(static_scale),
        static_cast<int8_t*>(out_i8),
        static_cast<float*>(out_scale_copy),
        M, K);
    return static_cast<int>(cudaGetLastError());
}
