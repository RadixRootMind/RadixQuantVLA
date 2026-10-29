// In-place softmax for the SDPA scores tensor inside the v2 full-attn plugins.
//
// Input shape: (H, S, S) BF16 row-major, where:
//   H = number of heads × batch
//   S = sequence length (the softmax axis is the last S)
//
// Each block handles one row (h, s1); threads cooperate via warp shuffle to
// compute max → exp → sum, then write normalized values back in-place.
//
// The math (numerically stable):
//   m  = max_{s2} x[h, s1, s2]
//   z  = sum_{s2} exp(x[h, s1, s2] - m)
//   y[h, s1, s2] = exp(x[h, s1, s2] - m) / z
#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// In-place row-wise softmax on a (HS, S) BF16 tensor, where HS = num_heads × batch
// and S = sequence length. Each row of length S is softmaxed independently.
//
//   io_bf16: (HS, S) BF16 row-major, modified in place.
//   HS:      number of softmax rows
//   S:       row length
//   stream:  CUDA stream
//
// Returns 0 on success, CUDA error code otherwise.
int sdpa_softmax_bf16_inplace(
    void* io_bf16,
    int HS,
    int S,
    cudaStream_t stream);

// Masked variant: same as above, but adds an additive BF16 mask before exp.
// The mask is broadcast: every `rows_per_mask` consecutive io rows share one
// mask row of length `S`. Pass mask_bf16=nullptr for the un-masked behaviour.
// This lets one softmax kernel serve both self-attn (no mask) and masked
// cross-attn with the broadcast pattern (B, 1, 1, S_kv) → (B*H*S, S_kv).
int sdpa_softmax_bf16_inplace_masked(
    void* io_bf16,
    void const* mask_bf16,    // (num_masks, S) BF16 - broadcast row. nullptr → no mask.
    int rows_per_mask,         // how many io rows share one mask row
    int HS,
    int S,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
