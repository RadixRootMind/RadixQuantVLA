// Dense block-diagonal rotation as a cuBLAS strided-batched BF16 GEMM.
//
// xr[m, blk*bs + c] = Σ_i xp[m, blk*bs + i] · R[blk, i, c]      (per block blk)
//
// This is `nb` batches of (M, bs)·(bs, bs). cuBLAS stages each R block once into
// shared memory and reuses it across all M rows, replacing the naive per-row
// global re-read of R (which was 74% of the W4A4 runtime). R is BF16 (nb,bs,bs)
// row-major; xp/xr are BF16 (M, K) row-major; accumulation is FP32.
//
// Column-major derivation (cuBLAS computes C_cm = op(A)·op(B), C is M_×N_):
//   want C_cm[c, m] = xr[m, blk*bs+c] = Σ_i R[blk,i,c]·xp[m, blk*bs+i]
//   ⇒ A_cm[c,i] = R[blk,i,c]  → A=R, lda=bs, strideA=bs*bs  (transA=N)
//     B_cm[i,m] = xp[m, blk*bs+i] → B=xp(+blk*bs), ldb=K, strideB=bs  (transB=N)
//     C_cm[c,m] = xr[m, blk*bs+c] → C=xr(+blk*bs), ldc=K, strideC=bs
//     M_=bs, N_=M(tokens), K_=bs, batch=nb.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdio>

#include "dit_int4_rowwise.h"

extern "C" int dit_int4_block_rotate_bf16(
    void* cublas_handle,
    void const* xp_bf16,
    void const* R_bf16,
    void* xr_bf16,
    int M, int K, int block_size,
    cudaStream_t stream) {
    if (M <= 0 || K <= 0 || block_size <= 0) return 0;
    const int bs = block_size;
    const int nb = K / bs;
    auto handle = reinterpret_cast<cublasHandle_t>(cublas_handle);
    cublasSetStream(handle, stream);
    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st = cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        /*M_=*/bs, /*N_=*/M, /*K_=*/bs,
        &alpha,
        /*A=*/R_bf16,  CUDA_R_16BF, /*lda=*/bs, /*strideA=*/(long long)bs * bs,
        /*B=*/xp_bf16, CUDA_R_16BF, /*ldb=*/K,  /*strideB=*/bs,
        &beta,
        /*C=*/xr_bf16, CUDA_R_16BF, /*ldc=*/K,  /*strideC=*/bs,
        /*batch=*/nb,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "[dit_int4_block_rotate] cublas failed: M=%d K=%d bs=%d code=%d\n",
                     M, K, bs, static_cast<int>(st));
        return static_cast<int>(st) | 0x40000;
    }
    return 0;
}
