// Block-diagonal fast Walsh-Hadamard transform, in shared memory.
//
// Why this exists: per-row (per-token) INT8 handles the TOKEN axis and is blind
// to the CHANNEL axis. On the GR00T N1.6 Qwen3 LLM that blindness costs ~14%
// median per-channel relative error. A SmoothQuant channel fold recovers ~31% of
// it for free (pure offline weight prep), but the residual channel spread needs
// an actual rotation, and a rotation cannot be folded anywhere: it mixes
// channels, so it survives neither RMSNorm's elementwise gamma nor SiLU. It has
// to run between the norm/SiLU and the quantizer, hence this kernel.
// Simulated payoff (validated against measured engines to within 2%):
//   per-row 0.1446 → +SQ 0.0999 → +SQ+rot(bs=64) 0.0451.
//
// Why Hadamard and not the DiT's SVD·Hadamard: the Sylvester Hadamard is a FIXED
// matrix, so the kernel needs only an integer block size: no R tensor to bake,
// upload, or serialize (the DiT's rotation field cost 252 MB of engine until it
// was folded to BF16). And it is O(bs log bs) in-register butterflies instead of
// a cuBLAS batched GEMM.
//
// Ordering note: the iterative butterfly below computes the transform of the
// Sylvester/Kronecker Hadamard in natural order, i.e. exactly
//   H_1 = [1];  H_2m = [[H_m, H_m], [H_m, -H_m]]
// which is the construction `hadamard()` in sim_llm_rotation.py and
// `_hadamard()` in llm_plugin_scheme_helpers.py use. The Python weight fold and
// this kernel MUST agree on that convention or the rotation stops being an
// identity and the layer silently produces garbage.

#pragma once

#include <cuda_runtime.h>

namespace gr00t {
namespace fwht {

// In-place block-diagonal FWHT over `y[0..K)`, viewed as K/bs contiguous blocks
// of `bs` elements each. Orthonormal (scaled by 1/sqrt(bs)), so the transform is
// its own inverse and W' = W·Hᵀ offline restores the exact product.
//
// Requires: bs a power of two, bs >= 2, K % bs == 0, `y` in shared memory,
// and ALL threads of the block participating (it __syncthreads internally).
__device__ __forceinline__ void block_fwht_smem(float* y, int K, int bs) {
    const int half = bs >> 1;
    const int npairs = K >> 1;   // every stage touches K/2 disjoint pairs
    // bs, half, and each stage width h are powers of two, so every div/mod in
    // the index math is a shift/mask; do it explicitly (runtime operands defeat
    // the compiler's strength reduction, leaving true IDIVs on the hot path).
    const int log_bs = __ffs(bs) - 1;
    const int log_half = __ffs(half) - 1;   // half >= 1 (bs >= 2)
    int log_h = 0;                          // log2(h), h starts at 1
    for (int h = 1; h < bs; h <<= 1) {
        for (int p = threadIdx.x; p < npairs; p += blockDim.x) {
            const int blk = p >> log_half;          // which bs-sized block
            const int q = p & (half - 1);           // pair index within the block
            const int group = q >> log_h;
            const int offset = q & (h - 1);
            const int i0 = (blk << log_bs) + (group << (log_h + 1)) + offset;
            const int i1 = i0 + h;
            const float a = y[i0];
            const float b = y[i1];
            y[i0] = a + b;
            y[i1] = a - b;
        }
        __syncthreads();
        ++log_h;
    }
    // Normalize so H is orthonormal, matching the Python-side fold exactly.
    const float inv = rsqrtf((float)bs);
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        y[k] *= inv;
    }
    __syncthreads();
}

// True when `bs` is a usable rotation block size for width `K`.
// bs <= 1 means "rotation disabled".
__host__ __device__ __forceinline__ bool rot_bs_valid(int bs, int K) {
    if (bs <= 1) return false;
    if ((bs & (bs - 1)) != 0) return false;   // power of two
    if (K % bs != 0) return false;
    return true;
}

}  // namespace fwht
}  // namespace gr00t
