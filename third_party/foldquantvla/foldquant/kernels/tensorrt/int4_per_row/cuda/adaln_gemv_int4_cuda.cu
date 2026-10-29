// AdaLN modulation GEMV with INT4 weight-only, BF16 activation (W4A16).
// Fused dequant: out[m,j] = scale[j] * Σ_i x[m,i]·int4(w[j,i]) + bias[j].
// Weight w is row-major [out, in], INT4 packed 2/byte along `in` (i even = low
// nibble, i odd = high nibble), signed (-7..7). Activation/scale/bias are BF16.
//
// The AdaLN modulation runs at M = B (one conditioning vector per sample, B=1
// in the action head) so this is a GEMV: grid = M*out threads, each reads its
// weight row ONCE (no reuse problem) and reads HALF the bytes of the BF16 weight
// → never slower than the BF16 MatMul it replaces, while storing 4-bit weights.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include "dit_int4_rowwise.h"

namespace {
// One WARP per output channel: the 32 lanes split the `in` reduction, then a
// warp shuffle-reduce. M*out warps → 32× the threads of a thread-per-output map,
// restoring occupancy for the M=1 (B=1) AdaLN GEMV.
__global__ void adaln_gemv_int4_warp_kernel(
    const __nv_bfloat16* __restrict__ x,    // [M, in]
    const uint8_t*       __restrict__ wq,   // [out, (in+1)/2] packed int4
    const __nv_bfloat16* __restrict__ scale,// [out]
    const __nv_bfloat16* __restrict__ bias, // [out]
    __nv_bfloat16*       __restrict__ out,   // [M, out]
    int M, int in_dim, int out_dim) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= M * out_dim) return;
    int m = warp / out_dim;
    int j = warp % out_dim;
    const __nv_bfloat16* xrow = x + (size_t)m * in_dim;
    const uint8_t* wrow = wq + (size_t)j * ((in_dim + 1) / 2);
    float acc = 0.f;
    for (int i = lane; i < in_dim; i += 32) {
        uint8_t byte = wrow[i >> 1];
        int nib = (i & 1) ? (byte >> 4) : (byte & 0x0F);
        int v = (nib < 8) ? nib : nib - 16;          // signed int4
        acc += __bfloat162float(xrow[i]) * (float)v;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) {
        float s = __bfloat162float(scale[j]);
        float b = __bfloat162float(bias[j]);
        out[(size_t)m * out_dim + j] = __float2bfloat16(acc * s + b);
    }
}
}  // anon

extern "C" int dit_adaln_gemv_int4_bf16(
    void const* x, void const* wq, void const* scale, void const* bias,
    void* out, int M, int in_dim, int out_dim, cudaStream_t stream) {
    if (M * out_dim <= 0) return 0;
    int tpb = 256;                                   // 8 warps/block
    long threads = (long)M * out_dim * 32;
    int blocks = (int)((threads + tpb - 1) / tpb);
    adaln_gemv_int4_warp_kernel<<<blocks, tpb, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<const uint8_t*>(wq),
        static_cast<const __nv_bfloat16*>(scale),
        static_cast<const __nv_bfloat16*>(bias),
        static_cast<__nv_bfloat16*>(out),
        M, in_dim, out_dim);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// ---------------------------------------------------------------------------
// W4A4 variant: identical warp-per-output GEMV, but the BF16 activation row is
// dynamically quantized to INT4 (per-row amax, qmax=7) so the reduction is an
// int4×int4 product. out[m,j] = sa[m]·scale[j]·Σ_i q(x[m,i])·int4(w[j,i]) + b[j],
// with sa[m] = amax_i|x[m,i]| / 7. M = B is tiny; each warp recomputes its row's
// amax (cheap, K reads) rather than staging a shared per-row scale.
namespace {
__global__ void adaln_gemv_int4a4_warp_kernel(
    const __nv_bfloat16* __restrict__ x,    // [M, in]
    const uint8_t*       __restrict__ wq,   // [out, (in+1)/2] packed int4
    const __nv_bfloat16* __restrict__ scale,// [out]
    const __nv_bfloat16* __restrict__ bias, // [out]
    __nv_bfloat16*       __restrict__ out,   // [M, out]
    int M, int in_dim, int out_dim) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= M * out_dim) return;
    int m = warp / out_dim;
    int j = warp % out_dim;
    const __nv_bfloat16* xrow = x + (size_t)m * in_dim;
    const uint8_t* wrow = wq + (size_t)j * ((in_dim + 1) / 2);
    // Per-row amax over this warp's lanes, then warp-reduce → sa = amax/7.
    float amax = 0.f;
    for (int i = lane; i < in_dim; i += 32)
        amax = fmaxf(amax, fabsf(__bfloat162float(xrow[i])));
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, o));
    amax = __shfl_sync(0xffffffffu, amax, 0);        // broadcast lane0 → all lanes
    float sa = amax / 7.f;
    float inv_sa = (sa > 0.f) ? (1.f / sa) : 0.f;
    // int4×int4 accumulate.
    float acc = 0.f;
    for (int i = lane; i < in_dim; i += 32) {
        float xf = __bfloat162float(xrow[i]) * inv_sa;
        int xq = (int)rintf(xf);
        xq = xq < -7 ? -7 : (xq > 7 ? 7 : xq);       // symmetric int4
        uint8_t byte = wrow[i >> 1];
        int nib = (i & 1) ? (byte >> 4) : (byte & 0x0F);
        int v = (nib < 8) ? nib : nib - 16;          // signed int4
        acc += (float)xq * (float)v;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) {
        float s = __bfloat162float(scale[j]);
        float b = __bfloat162float(bias[j]);
        out[(size_t)m * out_dim + j] = __float2bfloat16(sa * s * acc + b);
    }
}
}  // anon

extern "C" int dit_adaln_gemv_int4a4_bf16(
    void const* x, void const* wq, void const* scale, void const* bias,
    void* out, int M, int in_dim, int out_dim, cudaStream_t stream) {
    if (M * out_dim <= 0) return 0;
    int tpb = 256;                                   // 8 warps/block
    long threads = (long)M * out_dim * 32;
    int blocks = (int)((threads + tpb - 1) / tpb);
    adaln_gemv_int4a4_warp_kernel<<<blocks, tpb, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<const uint8_t*>(wq),
        static_cast<const __nv_bfloat16*>(scale),
        static_cast<const __nv_bfloat16*>(bias),
        static_cast<__nv_bfloat16*>(out),
        M, in_dim, out_dim);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
