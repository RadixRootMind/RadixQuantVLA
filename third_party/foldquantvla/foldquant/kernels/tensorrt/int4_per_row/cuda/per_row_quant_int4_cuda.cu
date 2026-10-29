// Per-row dynamic INT4 activation quantization kernel (packed 2 nibbles/byte).
//
// For each row m ∈ [0, M):
//   amax[m]  = max over k ∈ [0,K) of |in_bf16[m,k]|
//   scale[m] = amax[m] / 7.0                       (guarded against zero)
//   q[m,k]   = clamp(round(in_bf16[m,k] / scale[m]), -7, 7)
//   out_i4_packed[m, k/2] nibble(k&1) = q[m,k] & 0xF   (even k → low, odd k → high)
//
// Cloned from per_row_quant_cuda.cu (INT8) with 127→7 range and nibble packing.
// One block per row; the amax reduction strides over all K, the pack loop strides
// over K/2 bytes so each thread writes one full byte (no cross-thread nibble race).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "dit_int4_rowwise.h"
#include "fwht.cuh"

namespace {

__global__ void per_row_quant_bf16_to_int4_kernel(
    const __nv_bfloat16* __restrict__ in,
    int8_t* __restrict__ out_i4,   // (M, K/2) packed
    float* __restrict__ out_scale,
    int M,
    int K,
    float act_clip)
{
    const int m = blockIdx.x;
    if (m >= M) return;

    const __nv_bfloat16* row_in = in + (size_t)m * K;
    int8_t* row_out = out_i4 + (size_t)m * (K / 2);

    // Phase 1: thread-local amax over assigned columns
    float t_amax = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float v = fabsf(__bfloat162float(row_in[k]));
        if (v > t_amax) t_amax = v;
    }

    // Phase 2: warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, t_amax, offset);
        if (other > t_amax) t_amax = other;
    }

    // Phase 3: block reduction across warps
    __shared__ float warp_amaxes[32];
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
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

    float row_amax = warp_amaxes[0];
    // INT4 symmetric range [-7, 7]. Scale = amax / 7.
    float scale = act_clip * row_amax * (1.0f / 7.0f);
    float inv_scale = (scale > 1e-12f) ? (1.0f / scale) : 0.0f;
    if (threadIdx.x == 0) out_scale[m] = scale;

    // Phase 4: quantize + pack, one thread per output byte (2 elements).
    const int nbytes = K / 2;
    for (int b = threadIdx.x; b < nbytes; b += blockDim.x) {
        float v0 = __bfloat162float(row_in[2 * b]);
        float v1 = __bfloat162float(row_in[2 * b + 1]);
        int q0 = (int)fmaxf(-7.0f, fminf(7.0f, rintf(v0 * inv_scale)));
        int q1 = (int)fmaxf(-7.0f, fminf(7.0f, rintf(v1 * inv_scale)));
        // low nibble = even index (2b), high nibble = odd index (2b+1)
        uint8_t packed = (uint8_t)((q0 & 0xF) | ((q1 & 0xF) << 4));
        row_out[b] = (int8_t)packed;
    }
}

// Rotation variant: block-diagonal Hadamard between the input and the
// quantizer. INT4 clone of per_row_fwht_quant_bf16_to_int8_kernel.
//
// Unlike the plain kernel this MUST stage the row in shared memory: the FWHT is
// an in-place multi-pass butterfly and its OUTPUT is what amax and the packer
// need. Shared cost is K floats: 8 KB at K=2048 (o_proj), 24 KB at K=6144
// (down_proj), both inside sm_87's 96 KB budget.
//
// The plain kernel above is left untouched on purpose: it is shared with the
// DiT int4 macro plugins, which must not change behaviour.
__global__ void per_row_fwht_quant_bf16_to_int4_kernel(
    const __nv_bfloat16* __restrict__ in,
    const float* __restrict__ act_scale_pre, // (K,) or nullptr - pre-rotation SmoothQuant
    const float* __restrict__ act_scale_ch,  // (K,) or nullptr - post-rotation SmoothQuant
    int8_t* __restrict__ out_i4,   // (M, K/2) packed
    float* __restrict__ out_scale,
    int M, int K, int rot_bs, float act_clip)
{
    const int m = blockIdx.x;
    if (m >= M) return;

    const __nv_bfloat16* row_in = in + (size_t)m * K;
    int8_t* row_out = out_i4 + (size_t)m * (K / 2);

    extern __shared__ float y_buf[];

    // Two SmoothQuant orders exist in the folds this kernel serves, and they
    // divide on opposite axes: fold-before scales the RAW channel and then
    // rotates (rotation.fold_rotation_sq_before divides the rotation's
    // INPUT axis), fold-after scales the ROTATED channel (fold_rotation_sq
    // divides the OUTPUT axis). A dense matrix can absorb either; a fixed
    // butterfly absorbs neither, so both arrive as vectors and are applied on
    // their own side of the transform.
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

    // SmoothQuant, applied AFTER the rotation: the expert's per-channel scale is
    // the amax of the *rotated* activation, which is why the dense-rotation path folds
    // it into the baked matrix (rotation.fold_rotation_sq divides R[:,c] by
    // s_ch[c], i.e. it scales the rotation's OUTPUT channel). A fixed butterfly
    // has no coefficients to absorb it, so it arrives as its own vector and is
    // applied here: same arithmetic, one extra pass over shared memory.
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

    // INT4 symmetric range [-7, 7]. The -8 code is never emitted, matching the
    // Python packer and the simulator's qmax_of(4).
    const float scale = act_clip * warp_amaxes[0] * (1.0f / 7.0f);
    const float inv_scale = (scale > 1e-12f) ? (1.0f / scale) : 0.0f;
    if (threadIdx.x == 0) out_scale[m] = scale;

    // One thread per output byte (2 elements), so no cross-thread nibble race.
    const int nbytes = K / 2;
    for (int b = threadIdx.x; b < nbytes; b += blockDim.x) {
        int q0 = (int)fmaxf(-7.0f, fminf(7.0f, rintf(y_buf[2 * b]     * inv_scale)));
        int q1 = (int)fmaxf(-7.0f, fminf(7.0f, rintf(y_buf[2 * b + 1] * inv_scale)));
        uint8_t packed = (uint8_t)((q0 & 0xF) | ((q1 & 0xF) << 4));
        row_out[b] = (int8_t)packed;
    }
}

// Register-resident FWHT + per-row INT4 quant, for rows too wide to stage in
// shared memory cheaply.
//
// The smem variant holds the whole row (K floats) so the row-wide amax can be
// taken before quantizing; at K=6144 that is 24 KB per block and the occupancy
// cap costs more than the rotation saves when this runs inside a fused macro
// plugin (measured on the GR00T DiT: FusedFfnBlockInt4 1.594 -> 1.696 ms).
//
// The Hadamard is block-diagonal with bs=64, so a 64-element block needs nothing
// from any other block. One warp owns one block, two elements per lane: every
// butterfly stage becomes a __shfl_xor_sync, the values stay in registers, and
// shared memory holds only the amax reduction. The two elements a lane owns are
// also exactly one packed INT4 byte, so the store needs no cross-lane traffic.
//
// Element e of a block lives in lane e>>1, slot e&1. For stage h >= 2 the partner
// of e is e^h == lane ^ (h>>1) in the SAME slot; stage h == 1 pairs the two slots
// inside one lane. Requires bs == 64 and K % 64 == 0; the launcher checks both.
constexpr int kRegsMaxPairs = 32;   // 32 pairs/lane * 8 warps * 64 = K <= 16384

__global__ __launch_bounds__(256) void per_row_fwht_quant_regs_bf16_to_int4_kernel(
    const __nv_bfloat16* __restrict__ in,
    const float* __restrict__ act_scale_pre,
    const float* __restrict__ act_scale_ch,
    int8_t* __restrict__ out_i4,
    float* __restrict__ out_scale,
    int M, int K, float act_clip)
{
    const int m = blockIdx.x;
    if (m >= M) return;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;      // 8
    const int nblk = K >> 6;                 // 64-wide Hadamard blocks in this row

    const __nv_bfloat16* row_in = in + (size_t)m * K;
    int8_t* row_out = out_i4 + (size_t)m * (K / 2);

    float v0[kRegsMaxPairs], v1[kRegsMaxPairs];
    int held = 0;
    float t_amax = 0.0f;

    for (int blk = warp; blk < nblk; blk += nwarps, ++held) {
        const int base = (blk << 6) + (lane << 1);
        float a = __bfloat162float(row_in[base]);
        float b = __bfloat162float(row_in[base + 1]);
        if (act_scale_pre != nullptr) {           // fold-before: raw channel
            a /= act_scale_pre[base];
            b /= act_scale_pre[base + 1];
        }
        // stage h = 1 is the two slots this lane already owns
        float na = a + b, nb = a - b;
        a = na; b = nb;
        for (int h = 2; h <= 32; h <<= 1) {
            const float pa = __shfl_xor_sync(0xffffffffu, a, h >> 1);
            const float pb = __shfl_xor_sync(0xffffffffu, b, h >> 1);
            const int e0 = (lane << 1);
            a = (e0 & h) ? (pa - a) : (a + pa);
            b = ((e0 + 1) & h) ? (pb - b) : (b + pb);
        }
        const float inv = rsqrtf(64.0f);          // orthonormal, matches the fold
        a *= inv; b *= inv;
        if (act_scale_ch != nullptr) {            // fold-after: rotated channel
            a /= act_scale_ch[base];
            b /= act_scale_ch[base + 1];
        }
        v0[held] = a; v1[held] = b;
        t_amax = fmaxf(t_amax, fmaxf(fabsf(a), fabsf(b)));
    }

    for (int off = 16; off > 0; off >>= 1) {
        t_amax = fmaxf(t_amax, __shfl_xor_sync(0xffffffffu, t_amax, off));
    }
    __shared__ float warp_amax[32];
    if (lane == 0) warp_amax[warp] = t_amax;
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = 0.0f;
        for (int i = 0; i < nwarps; ++i) v = fmaxf(v, warp_amax[i]);
        warp_amax[0] = v;
    }
    __syncthreads();

    const float scale = act_clip * warp_amax[0] * (1.0f / 7.0f);
    const float inv_scale = (scale > 1e-12f) ? (1.0f / scale) : 0.0f;
    if (threadIdx.x == 0) out_scale[m] = scale;

    int idx = 0;
    for (int blk = warp; blk < nblk; blk += nwarps, ++idx) {
        const int q0 = (int)fmaxf(-7.0f, fminf(7.0f, rintf(v0[idx] * inv_scale)));
        const int q1 = (int)fmaxf(-7.0f, fminf(7.0f, rintf(v1[idx] * inv_scale)));
        row_out[(blk << 5) + lane] = (int8_t)(uint8_t)((q0 & 0xF) | ((q1 & 0xF) << 4));
    }
}

}  // namespace

extern "C" int dit_int4_per_row_quant_bf16(
    void const* in_bf16,
    void* out_i4_packed,
    void* out_scale,
    int M,
    int K,
    float act_clip,
    cudaStream_t stream)
{
    if (M <= 0 || K <= 0) return 0;
    const int threads = 256;
    dim3 grid(M, 1, 1);
    dim3 block(threads, 1, 1);
    per_row_quant_bf16_to_int4_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(in_bf16),
        static_cast<int8_t*>(out_i4_packed),
        static_cast<float*>(out_scale),
        M, K, act_clip);
    return static_cast<int>(cudaGetLastError());
}

// Rotation-enabled per-row INT4 quant. rot_bs must be a power of two dividing K;
// rot_bs <= 1 falls back to the plain (unstaged) kernel so the no-rotation path
// keeps its exact previous behaviour and costs no shared memory.
extern "C" int dit_int4_per_row_quant_fwht_bf16(
    void const* in_bf16,
    void const* act_scale_pre,
    void const* act_scale_ch,
    void* out_i4_packed,
    void* out_scale,
    int M,
    int K,
    int rot_bs,
    float act_clip,
    cudaStream_t stream)
{
    if (M <= 0 || K <= 0) return 0;
    // Odd K would floor nbytes = K/2 and silently drop the last channel in
    // the plain kernel this delegates to (reachable via the rotation-off
    // gptq_sq ablation); the rotated branch is implicitly even (power-of-2
    // rot_bs divides K) but the check belongs to the launcher, not luck.
    if (K % 2 != 0) return static_cast<int>(cudaErrorInvalidValue);
    if (rot_bs <= 1) {
        // No rotation means no rotated channel axis for a POST-rotation scale to
        // live on; a caller passing one has a folding bug, so refuse it. A
        // pre-rotation scale is still meaningful: it divides the raw channel,
        // which exists with or without a rotation (that is plain SmoothQuant,
        // and it is what the rotation-off ablation arms need).
        if (act_scale_ch != nullptr) return static_cast<int>(cudaErrorInvalidValue);
        if (act_scale_pre == nullptr) {
            return dit_int4_per_row_quant_bf16(in_bf16, out_i4_packed, out_scale,
                                                M, K, act_clip, stream);
        }
        // fall through: the kernel below skips the butterfly when rot_bs is
        // invalid but still applies the scale it was handed.
    }
    if (rot_bs > 1 && !gr00t::fwht::rot_bs_valid(rot_bs, K)) {
        // Weights are already folded with W·Hᵀ upstream, so silently skipping
        // the rotation here would produce a wrong result with no symptom.
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const int threads = 256;
    dim3 grid(M, 1, 1);
    dim3 block(threads, 1, 1);

    // Wide rows go to the register-resident variant: the smem kernel would stage
    // K floats per block (24 KB at K=6144), and that occupancy cap costs more
    // than the rotation saves when this runs beside the rest of a fused macro
    // plugin. The register path needs bs == 64 (one warp per Hadamard block, two
    // elements per lane) and enough registers to hold the row.
    const int pairs_per_lane = (K / 64 + (threads / 32) - 1) / (threads / 32);
    if (rot_bs == 64 && (K % 64) == 0 && pairs_per_lane <= 32 && K > 2048) {
        per_row_fwht_quant_regs_bf16_to_int4_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(in_bf16),
            static_cast<const float*>(act_scale_pre),
            static_cast<const float*>(act_scale_ch),
            static_cast<int8_t*>(out_i4_packed),
            static_cast<float*>(out_scale),
            M, K, act_clip);
        return static_cast<int>(cudaGetLastError());
    }

    size_t smem_bytes = sizeof(float) * (size_t)K;
    // Same wide-K guard as the INT8 twin: K > ~12K floats exceeds the 48KB
    // default dynamic-smem limit (Gemma's 16384-wide down_proj is exactly
    // that case on the INT8 path), so opt in explicitly.
    if (smem_bytes > 48 * 1024) {
        cudaError_t attr_rc = cudaFuncSetAttribute(
            per_row_fwht_quant_bf16_to_int4_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes));
        if (attr_rc != cudaSuccess) return static_cast<int>(attr_rc);
    }
    per_row_fwht_quant_bf16_to_int4_kernel<<<grid, block, smem_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(in_bf16),
        static_cast<const float*>(act_scale_pre),
        static_cast<const float*>(act_scale_ch),
        static_cast<int8_t*>(out_i4_packed),
        static_cast<float*>(out_scale),
        M, K, rot_bs, act_clip);
    return static_cast<int>(cudaGetLastError());
}
