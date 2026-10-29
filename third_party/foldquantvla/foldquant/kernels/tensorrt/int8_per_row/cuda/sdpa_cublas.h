// Copyright (c) 2026 The FoldQuant Authors.
// Licensed under the Apache License, Version 2.0; see LICENSE.
//
// Batched BF16 scaled-dot-product attention on cuBLAS, shared by the fused
// self-/cross-attention plugins (INT8 and INT4 macro libs).
//
// The plugins keep Q, K and V as head slices inside one row-major activation
// buffer per sample: head h of sample b starts at
//     ptr + b * sample_stride + h * D
// with leading dimension `ld`. Two strides (head = D, sample = sample_stride) do
// not fit one cuBLAS strided-batched call (the per-sample offset is not affine
// in the flat index b*H + h), so this helper issues one strided-batched GEMM per
// sample with batchCount = H. B is the number of parallel environments (<= 8),
// so the extra launches are negligible next to the GEMMs themselves.
//
// scores is (B, H, S_q, S_kv) contiguous; attn is row-major (B*S_q, ld_attn)
// with head h of sample b at attn + b*attn_sample_stride + h*D.
#pragma once

#include <cmath>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "sdpa_softmax.h"

namespace gr00t {
namespace v1 {
namespace plugins {

// One row-major BF16 buffer holding H head slices (width D) per row.
struct SdpaOperand {
    __nv_bfloat16 const* ptr;   // head 0 of sample 0
    int ld;                     // elements per row
    long long sample_stride;    // elements between consecutive samples
};

// Error codes: cuBLAS status | 0x10000 (Q·Kᵀ) or | 0x20000 (P·V), the softmax's
// CUDA error, or -2 for an unsupported mask row count.
inline int sdpa_bf16_cublas(cublasHandle_t handle,
                            SdpaOperand q, SdpaOperand k, SdpaOperand v,
                            __nv_bfloat16* scores,
                            __nv_bfloat16* attn, int ld_attn, long long attn_sample_stride,
                            void const* mask_bf16, int mask_rows,
                            int B, int H, int S_q, int S_kv, int D,
                            cudaStream_t stream) noexcept {
    // mask_bf16: additive (mask_rows, S_kv) rows; mask_rows == B gives one row per
    // sample, mask_rows == 1 broadcasts one row to every sample. nullptr = no mask.
    if (mask_bf16 != nullptr && mask_rows != 1 && mask_rows != B) return -2;

    const float alpha = 1.0f / sqrtf(static_cast<float>(D));
    const float beta = 0.0f;
    const float one = 1.0f;
    const long long scores_sample = static_cast<long long>(H) * S_q * S_kv;
    const long long scores_head = static_cast<long long>(S_q) * S_kv;

    // Row-major C(S_q, S_kv) = Q(S_q, D) @ K(S_kv, D)ᵀ. cuBLAS is column-major, so
    // compute Cᵀ(S_kv, S_q) = K @ Qᵀ: gemm(M_c=S_kv, N_c=S_q, K_c=D, A=K op T, B=Q op N).
    for (int b = 0; b < B; ++b) {
        cublasStatus_t cu = cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, S_kv, S_q, D, &alpha,
            k.ptr + b * k.sample_stride, CUDA_R_16BF, k.ld, static_cast<long long>(D),
            q.ptr + b * q.sample_stride, CUDA_R_16BF, q.ld, static_cast<long long>(D),
            &beta, scores + b * scores_sample, CUDA_R_16BF, S_kv, scores_head,
            H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (cu != CUBLAS_STATUS_SUCCESS) return static_cast<int>(cu) | 0x10000;
    }

    // Softmax over the last axis of every (b, h, s_q) row, all samples in one launch.
    const int rows = B * H * S_q;
    int rc;
    if (mask_bf16 != nullptr) {
        const int rows_per_mask = (mask_rows == 1) ? rows : H * S_q;
        rc = sdpa_softmax_bf16_inplace_masked(scores, mask_bf16, rows_per_mask, rows, S_kv, stream);
    } else {
        rc = sdpa_softmax_bf16_inplace(scores, rows, S_kv, stream);
    }
    if (rc != 0) return rc;

    // Row-major C(S_q, D) = P(S_q, S_kv) @ V(S_kv, D):
    // Cᵀ(D, S_q) = Vᵀ(D, S_kv) @ Pᵀ(S_kv, S_q): gemm(M_c=D, N_c=S_q, K_c=S_kv, A=V op N, B=P op N).
    for (int b = 0; b < B; ++b) {
        cublasStatus_t cu = cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, D, S_q, S_kv, &one,
            v.ptr + b * v.sample_stride, CUDA_R_16BF, v.ld, static_cast<long long>(D),
            scores + b * scores_sample, CUDA_R_16BF, S_kv, scores_head,
            &beta, attn + b * attn_sample_stride, CUDA_R_16BF, ld_attn, static_cast<long long>(D),
            H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (cu != CUBLAS_STATUS_SUCCESS) return static_cast<int>(cu) | 0x20000;
    }
    return 0;
}

// Sample count from a (…, S, K) activation: rows must tile S exactly.
// Returns -1 when they do not (a hand-built network with a mismatched rank).
inline int sdpa_sample_count(long long rows, int S) noexcept {
    if (S <= 0 || rows <= 0 || rows % S != 0) return -1;
    return static_cast<int>(rows / S);
}

}  // namespace plugins
}  // namespace v1
}  // namespace gr00t
