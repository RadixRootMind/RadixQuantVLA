// FoldQuant Stage 2 / B1 prologues: produce the PRE-rotation activation in BF16 with
// the channel permutation folded into the store (yp[j] = f(x)[perm[j]]). The dense
// block rotation that follows is then a cuBLAS strided-batched GEMM (R staged once,
// reused across all rows) instead of a per-row global re-read of R. Cheaper and
// avoids the 74%-of-runtime bottleneck of the fused rotate+quant kernels.
//
//   dit_adaln_permute_bf16 : y[k] = LN(x)[k]*(1+scale[k])+shift[k];  out[j] = y[perm[j]]
//                            (scale/shift may be null → plain LayerNorm, FFN proj0)
//   dit_gelu_permute_bf16  : out[j] = gelu_tanh(in[perm[j]])
//   dit_permute_bf16       : out[j] = in[perm[j]]   (post-SDPA attn_O input gather)

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

#include "dit_int4_rowwise.h"

namespace {

constexpr int kWarp = 32;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = kWarp / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float gelu_tanh(float x) {
    const float c = 0.7978845608f, k = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(c * (x + k * x * x * x)));
}

// One CTA per token. Pass 1: LN stats over x (original order). Pass 2: permuted store.
__global__ void adaln_permute_bf16_kernel(
    __nv_bfloat16 const* __restrict__ x,
    __nv_bfloat16 const* __restrict__ scale,   // may be nullptr
    __nv_bfloat16 const* __restrict__ shift,   // may be nullptr
    int const* __restrict__ perm,
    __nv_bfloat16* __restrict__ out,           // (B*S, K) permuted
    int B, int S, int K, float eps) {
    const int tok = blockIdx.x;
    if (tok >= B * S) return;
    const int b = tok / S;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5, lane = tid & 31;
    const int nwarps = blockDim.x / kWarp;

    __nv_bfloat16 const* x_row = x + (size_t)tok * K;
    __nv_bfloat16 const* sc_row = scale ? scale + (size_t)b * K : nullptr;
    __nv_bfloat16 const* sh_row = shift ? shift + (size_t)b * K : nullptr;
    __nv_bfloat16* o_row = out + (size_t)tok * K;

    extern __shared__ float sm[];   // 2*nwarps scratch
    float t_sum = 0.f, t_sq = 0.f;
    for (int k = tid; k < K; k += blockDim.x) {
        float v = __bfloat162float(x_row[k]);
        t_sum += v; t_sq += v * v;
    }
    float ws = warp_sum(t_sum), wq = warp_sum(t_sq);
    if (lane == 0) { sm[warp_id] = ws; sm[warp_id + nwarps] = wq; }
    __syncthreads();
    __shared__ float mean_s, rstd_s;
    if (warp_id == 0) {
        float vs = (lane < nwarps) ? sm[lane] : 0.f;
        float vq = (lane < nwarps) ? sm[lane + nwarps] : 0.f;
        vs = warp_sum(vs); vq = warp_sum(vq);
        if (lane == 0) {
            float m = vs / K, var = vq / K - m * m;
            if (var < 0.f) var = 0.f;
            mean_s = m; rstd_s = rsqrtf(var + eps);
        }
    }
    __syncthreads();
    float mean = mean_s, rstd = rstd_s;
    // Permuted store: out[j] = modulate(x[perm[j]]).
    for (int j = tid; j < K; j += blockDim.x) {
        int c = perm[j];
        float xv = __bfloat162float(x_row[c]);
        float sv = sc_row ? __bfloat162float(sc_row[c]) : 0.f;
        float hv = sh_row ? __bfloat162float(sh_row[c]) : 0.f;
        float y = (xv - mean) * rstd * (1.f + sv) + hv;
        o_row[j] = __float2bfloat16(y);
    }
}

__global__ void gelu_permute_bf16_kernel(
    __nv_bfloat16 const* __restrict__ in,
    int const* __restrict__ perm,
    __nv_bfloat16* __restrict__ out,
    int M, int K) {
    const int m = blockIdx.x;
    if (m >= M) return;
    __nv_bfloat16 const* in_row = in + (size_t)m * K;
    __nv_bfloat16* o_row = out + (size_t)m * K;
    for (int j = threadIdx.x; j < K; j += blockDim.x)
        o_row[j] = __float2bfloat16(gelu_tanh(__bfloat162float(in_row[perm[j]])));
}

__global__ void permute_bf16_kernel(
    __nv_bfloat16 const* __restrict__ in,
    int const* __restrict__ perm,
    __nv_bfloat16* __restrict__ out,
    int M, int K) {
    const int m = blockIdx.x;
    if (m >= M) return;
    __nv_bfloat16 const* in_row = in + (size_t)m * K;
    __nv_bfloat16* o_row = out + (size_t)m * K;
    for (int j = threadIdx.x; j < K; j += blockDim.x)
        o_row[j] = in_row[perm[j]];
}

}  // namespace

extern "C" int dit_adaln_permute_bf16(
    void const* x, void const* scale, void const* shift, void const* perm,
    void* out, int B, int S, int K, float eps, cudaStream_t stream) {
    if (B <= 0 || S <= 0 || K <= 0) return 0;
    dim3 grid(B * S), block(256);
    size_t sm = sizeof(float) * 2 * (256 / 32);
    adaln_permute_bf16_kernel<<<grid, block, sm, stream>>>(
        static_cast<__nv_bfloat16 const*>(x),
        static_cast<__nv_bfloat16 const*>(scale),
        static_cast<__nv_bfloat16 const*>(shift),
        static_cast<int const*>(perm),
        static_cast<__nv_bfloat16*>(out), B, S, K, eps);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int dit_gelu_permute_bf16(
    void const* in, void const* perm, void* out, int M, int K, cudaStream_t stream) {
    if (M <= 0 || K <= 0) return 0;
    dim3 grid(M), block(256);
    gelu_permute_bf16_kernel<<<grid, block, 0, stream>>>(
        static_cast<__nv_bfloat16 const*>(in), static_cast<int const*>(perm),
        static_cast<__nv_bfloat16*>(out), M, K);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int dit_permute_bf16(
    void const* in, void const* perm, void* out, int M, int K, cudaStream_t stream) {
    if (M <= 0 || K <= 0) return 0;
    dim3 grid(M), block(256);
    permute_bf16_kernel<<<grid, block, 0, stream>>>(
        static_cast<__nv_bfloat16 const*>(in), static_cast<int const*>(perm),
        static_cast<__nv_bfloat16*>(out), M, K);
    return static_cast<int>(cudaGetLastError());
}
