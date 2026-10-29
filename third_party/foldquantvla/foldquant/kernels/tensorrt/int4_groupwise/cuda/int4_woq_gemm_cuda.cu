/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Copyright (c) 2023 MIT HAN Lab
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * reference: https://github.com/mit-han-lab/llm-awq/blob/main/awq/kernels/csrc/quantization_new/gemm/gemm_cuda.cu
 */

#include "dequantize.cuh"
#include <cuda_bf16.h>
#include <cuda_pipeline_primitives.h>
#include <stdexcept>

namespace trt_edgellm
{
namespace kernel
{

#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 4)
#define L2_CACHEHINT(size) ".L2::" #size "B"
#else
#define L2_CACHEHINT(size)
#endif

#define kInterleave 4
#define OP_M 16
#define OP_N 8
#define OP_K 16
#define INTRIN_M 16
#define INTRIN_N 16
#define INTRIN_K 16
#define WARP_SIZE 32
#define SMEM_PAD_A 0
#define SMEM_PAD_B 0
#define PACK_SIZE 8

template <int N>
__inline__ __host__ __device__ int get_log_tile(int n)
{
    if (N >= 8 && n >= 6)
        return 3;
    else if (N >= 4 && n >= 3)
        return 2;
    else if (N >= 2 && n >= 2)
        return 1;
    else
        return 0;
}

__inline__ __device__ uint2 get_block_idx_mapping(int blockIdx_x, int blockIdx_y, int log_tile)
{
    return make_uint2((blockIdx_x >> log_tile), (blockIdx_y << log_tile) + ((blockIdx_x) & ((1 << (log_tile)) - 1)));
}

__inline__ __device__ uint32_t cast_smem_ptr_to_uint(void const* const ptr)
{
    uint32_t smem_int_ptr;

    asm("{.reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
        : "=r"(smem_int_ptr)
        : "l"(ptr));

    return smem_int_ptr;
}

__inline__ __device__ void ldmatrix_m8n8_x4_b16(half* shared_warp, int ax0_0, uint32_t addr)
{
    __asm__ __volatile__(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(((unsigned*) (shared_warp + (ax0_0 * 8)))[0]), "=r"(((unsigned*) (shared_warp + (ax0_0 * 8)))[1]),
        "=r"(((unsigned*) (shared_warp + (ax0_0 * 8)))[2]), "=r"(((unsigned*) (shared_warp + (ax0_0 * 8)))[3])
        : "r"(addr));
}

__inline__ __device__ void cp_async_cg_A(uint32_t smem_int_ptr, uint4 const* __restrict__ src, bool mask)
{
    int const cp_size = 16;
    asm volatile("{"
               "  .reg .pred p;"
               "  setp.ne.b32 p, %0, 0;"
               "  @p cp.async.cg.shared.global" L2_CACHEHINT(128) " [%1], [%2], %3;"
                                                                  "}" ::"r"((int)mask),
               "r"(smem_int_ptr),
               "l"(src),
               "n"(cp_size));
}

// Full BF16 path: D and C are 4 FP32 registers, A and B are BF16.
// BF16 inputs avoid the FP16 cascade NaN: when an upstream layer's BF16 output
// exceeds FP16 range (65504), Cast(BF16→FP16) at this layer's input would
// produce Inf and propagate NaN. Native BF16 MMA preserves the full FP32
// exponent range end-to-end.
//
// NOTE: A_shared_warp and B_shared_warp are still typed `half*` (the underlying
// shared memory is just 16-bit byte storage). The stored bit patterns are BF16
// however (see share_to_reg_one_stage_B_T2 for the dequant→BF16 conversion),
// and the surgeried ONNX feeds BF16 activation directly to the plugin (no Cast).
__device__ __inline__ void mma_m16n8k16(float* C_warp, half* A_shared_warp, half* B_shared_warp)
{
    __asm__ __volatile__(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
        : "=f"(C_warp[0]), "=f"(C_warp[1]), "=f"(C_warp[2]), "=f"(C_warp[3])
        : "r"(((unsigned*) A_shared_warp)[0]), "r"(((unsigned*) A_shared_warp)[1]), "r"(((unsigned*) A_shared_warp)[2]),
        "r"(((unsigned*) A_shared_warp)[3]), "r"(((unsigned*) B_shared_warp)[0]), "r"(((unsigned*) B_shared_warp)[1]),
        "f"(C_warp[0]), "f"(C_warp[1]), "f"(C_warp[2]), "f"(C_warp[3]));
}

template <int CTA_M, int CTA_N, int CTA_K, int CTA_SIZE, int SHARED_K_ITERS, int STAGES>
__device__ __inline__ void global_to_share_one_stage_A_T2(half const* src, half* dst, int global_nrows,
    int global_ncols, int cta_offset_m, int cta_offset_n, int global_iter_k, int shared_iter_k, bool mask)
{
    constexpr int threads_needed = (CTA_M * CTA_K) / PACK_SIZE / SHARED_K_ITERS;
    constexpr int threads_used = threads_needed < CTA_SIZE ? threads_needed : CTA_SIZE;
    constexpr int total_global_iters = (CTA_M * CTA_K) / PACK_SIZE / threads_used;
    constexpr int partial_global_iters = (total_global_iters + SHARED_K_ITERS - 1) / SHARED_K_ITERS;
    constexpr int cta_step_m_or_n = (threads_used * PACK_SIZE) / CTA_K;
    constexpr int warp_step_m_or_n = (WARP_SIZE * PACK_SIZE) / CTA_K;
    constexpr int threads_per_row = CTA_K / PACK_SIZE;
    constexpr int kSmemCol = CTA_K + SMEM_PAD_A;
    bool local_mask = mask & (threadIdx.y * WARP_SIZE + threadIdx.x < threads_used);
    int ld_col = (threadIdx.x % threads_per_row);
#pragma unroll
    for (int _global_iter = 0; _global_iter < partial_global_iters; ++_global_iter)
    {
        int global_iter = shared_iter_k * partial_global_iters + _global_iter;
        int ld_row = global_iter * cta_step_m_or_n + threadIdx.y * warp_step_m_or_n + (threadIdx.x / threads_per_row);
        int ld_col_swizzled = (ld_col ^ (ld_row) & 7) * PACK_SIZE;
        void* dst_ptr = (void*) (dst + ld_row * kSmemCol + ld_col_swizzled);
        uint4* src_ptr = (uint4*) (src + (ld_row + cta_offset_m) * global_ncols + ld_col * PACK_SIZE
            + global_iter_k
                * CTA_K); // cta_offset_m * global_ncols + global_iter * cta_step_m_or_n * global_ncols + threadIdx.y *
                          // warp_step_m_or_n * global_ncols + (threadIdx.x / threads_per_row) * global_ncols +
                          // global_iter_k * CTA_K + (threadIdx.x % threads_per_row) * PACK_SIZE);
        if constexpr (STAGES > 1)
        {
            uint32_t addr = cast_smem_ptr_to_uint(dst_ptr);
            cp_async_cg_A(addr, src_ptr,
                local_mask & (ld_row + cta_offset_m < global_nrows)
                    & (reinterpret_cast<half*>(src_ptr) < (src + global_nrows * global_ncols)));
        }
        else
        {
            if (local_mask & (ld_row + cta_offset_m < global_nrows))
                *(uint4*) dst_ptr = *src_ptr;
        }
    }
}

template <int CTA_M, int CTA_N, int CTA_K, int CTA_SIZE, int SHARED_K_ITERS, int STAGES>
__device__ __inline__ void global_to_share_one_stage_B_T2(half const* src, half* dst, int src_size, int global_ncols,
    int cta_offset_m, int cta_offset_n, int global_iter_k, int shared_iter_k, bool mask)
{
    constexpr int threads_needed = (CTA_N / kInterleave * CTA_K) / PACK_SIZE / SHARED_K_ITERS;
    constexpr int threads_used = threads_needed < CTA_SIZE ? threads_needed : CTA_SIZE;
    constexpr int total_global_iters = (CTA_N / kInterleave * CTA_K) / PACK_SIZE / threads_used;
    constexpr int partial_global_iters = (total_global_iters + SHARED_K_ITERS - 1) / SHARED_K_ITERS;
    constexpr int cta_step_m_or_n = (threads_used * PACK_SIZE) / CTA_K;
    constexpr int warp_step_m_or_n = (WARP_SIZE * PACK_SIZE) / CTA_K;
    constexpr int threads_per_row = CTA_K / PACK_SIZE;
    constexpr int kSmemCol = CTA_K + SMEM_PAD_B;
    bool local_mask = mask & (threadIdx.y * WARP_SIZE + threadIdx.x < threads_used);
#pragma unroll
    for (int _global_iter = 0; _global_iter < partial_global_iters; ++_global_iter)
    {
        int global_iter = shared_iter_k * partial_global_iters + _global_iter;

        int ld_row = global_iter * cta_step_m_or_n + threadIdx.y * warp_step_m_or_n + (threadIdx.x / threads_per_row);
        int ld_col = (threadIdx.x % threads_per_row);
        int ld_col_swizzled = ld_col ^ (ld_row % 2) & 7;
        void* dst_ptr = (void*) (dst + (ld_row * kSmemCol + ld_col_swizzled * PACK_SIZE));
        uint4* src_ptr = (uint4*) (src + global_iter_k * CTA_K + cta_offset_n / kInterleave * global_ncols
            + ld_row * global_ncols + ld_col * PACK_SIZE);
        if constexpr (STAGES > 1)
        {
            uint32_t addr = cast_smem_ptr_to_uint(dst_ptr);
            cp_async_cg_A(addr, src_ptr, local_mask & (reinterpret_cast<half*>(src_ptr) < (src + src_size)));
        }
        else
        {
            if (local_mask)
                *(uint4*) dst_ptr = *src_ptr;
        }
    }
}

template <int CTA_M, int CTA_N, int CTA_K, int CTA_SIZE, int STAGES, int G>
__device__ __inline__ void global_to_share_one_stage_scales_T2(half const* src, half* dst, int src_size,
    int global_ncols, int cta_offset_m, int cta_offset_n, int global_iter_k, int shared_iter_k, bool mask)
{
    constexpr int threads_needed = CTA_N / PACK_SIZE / 1;
    constexpr int threads_used = threads_needed < CTA_SIZE ? threads_needed : CTA_SIZE;
    constexpr int threads_per_row = CTA_N / PACK_SIZE;
    bool local_mask = mask & (threadIdx.y * WARP_SIZE + threadIdx.x < threads_used);
    int g_idx = global_iter_k * CTA_K / G;

    void* dst_ptr = (void*) (dst + (threadIdx.x % threads_per_row) * PACK_SIZE);
    uint4* src_ptr = (uint4*) (src + g_idx * global_ncols + cta_offset_n + (threadIdx.x % threads_per_row) * PACK_SIZE);
    if (STAGES > 1)
    {
        uint32_t addr = cast_smem_ptr_to_uint(dst_ptr);
        cp_async_cg_A(addr, src_ptr, local_mask & (reinterpret_cast<half*>(src_ptr) < (src + src_size)));
    }
    else
    {
        if (local_mask)
        {
            *(uint4*) dst_ptr = *src_ptr;
        }
    }
}

template <int CTA_M, int CTA_N, int CTA_K, int STAGES, int shared_iters>
__device__ __inline__ void share_to_reg_one_stage_A_T2(
    half const* src, half* dst, int warp_offset_m, int warp_offset_n, int k_0_1)
{
    constexpr int kSmemCol = CTA_K + SMEM_PAD_A;

    for (int shared_iter = 0; shared_iter < shared_iters; ++shared_iter)
    {

        int ld_row = warp_offset_m + shared_iter * OP_M + (threadIdx.x % 16);
        int ld_col = k_0_1 * 16 + (threadIdx.x / 16) * 8;
        int ld_col_swizzled = ((ld_col / PACK_SIZE) ^ (ld_row) & 7) * PACK_SIZE;
        void* addr_ptr = (void*) (src + ld_row * kSmemCol + ld_col_swizzled);

        uint32_t addr = cast_smem_ptr_to_uint(addr_ptr);
        ldmatrix_m8n8_x4_b16(dst, shared_iter, addr);
    }
}

template <int CTA_M, int CTA_N, int CTA_K, int STAGES, bool ldmatrix, int shared_iters, int G>
__device__ __inline__ void share_to_reg_one_stage_B_T2(
    half const* src, half* src_scales, half* dst, half* dst_fp16, int warp_offset_m, int warp_offset_n, int k_0_1)
{
    [[maybe_unused]] constexpr int kSmemCol = CTA_K + SMEM_PAD_B;
    int r0 = ((threadIdx.x / 8 / 2) * 8 + threadIdx.x % 8);
    int c0 = ((threadIdx.x / 8) % 2) * 8;
    int r = r0 / 4;
    int c = (r0 % 4) * 16 + c0;
    [[maybe_unused]] int c_swizzled = ((c / PACK_SIZE) ^ (r % 2) & 7) * PACK_SIZE;

    if constexpr (ldmatrix)
    {
#pragma unroll
        for (int shared_iter = 0; shared_iter < shared_iters; ++shared_iter)
        {
            void* addr_ptr = (void*) (src + warp_offset_n / kInterleave * kSmemCol
                + shared_iter * 16 / kInterleave * kSmemCol + k_0_1 * 16 + r * kSmemCol + c_swizzled);
            uint32_t addr = cast_smem_ptr_to_uint(addr_ptr);
            ldmatrix_m8n8_x4_b16(dst, shared_iter, addr);
        }
    }

    // Dequantize INT4 → FP16 (via magic-number PTX), multiply by FP16 scale, then
    // convert FP16 result → BF16 for storage. The BF16 patterns are read by the
    // BF16 MMA instruction. Dequant×scale fits FP16 (max 7×scale_max), so the
    // intermediate FP16 multiply is safe; the BF16 conversion only loses a few
    // mantissa bits (BF16 7-bit vs FP16 10-bit), benign for this magnitude.
#pragma unroll
    for (int shared_iter = 0; shared_iter < shared_iters; ++shared_iter)
    {
        half scale = src_scales[warp_offset_n + 16 * shared_iter + 8 * (k_0_1 % 2) + threadIdx.x / 4];
        half2 scale2 = make_half2(scale, scale);
        half2 loaded[4];
        dequantize_s4_to_fp16x2(*reinterpret_cast<half2*>(dst + (k_0_1 % 2) * 4 + (k_0_1 / 2 * 2) + shared_iter * 8),
            reinterpret_cast<uint4*>(loaded));
        __nv_bfloat162 loaded_bf[4];
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            loaded[i] = __hmul2(loaded[i], scale2);
            loaded_bf[i] = __floats2bfloat162_rn(__half2float(loaded[i].x), __half2float(loaded[i].y));
        }
        *reinterpret_cast<uint4*>(dst_fp16 + shared_iter * 16 + 8 * (k_0_1 % 2))
            = *reinterpret_cast<uint4*>(loaded_bf);
    }
}

template <int CTA_M, int CTA_N, int CTA_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, int G>
__global__ void gemm_w4a16_T2(__nv_bfloat16 const* __restrict__ A_bf, half const* __restrict__ B,
    half const* __restrict__ scales, __nv_bfloat16* __restrict__ C, int M, int N, int K)
{
    // A is BF16 from upstream (no Cast(BF16→FP16) before the plugin). Loaders
    // do byte-level cp.async into shared memory; pointer reinterpret as half is
    // safe here because it's only used for memcpy + bounds checks (FP16 and
    // BF16 are both 16-bit; the BF16 bit pattern is only re-interpreted by
    // the BF16 MMA instruction at the consumer side).
    half const* A = reinterpret_cast<half const*>(A_bf);
    constexpr int NUM_WARPS = CTA_M / WARP_M * CTA_N / WARP_N;
    constexpr int CTA_SIZE = NUM_WARPS * WARP_SIZE;
    int num_blocks_n = (N + CTA_N - 1) / CTA_N;
    int num_blocks_m = (M + CTA_M - 1) / CTA_M;
    int blockIdx_y = blockIdx.x % (num_blocks_m * num_blocks_n);
    int const log_tile = get_log_tile<1>((N + CTA_N - 1) / CTA_N);
    int blockIdx_m = blockIdx_y / (num_blocks_n >> log_tile);
    int blockIdx_n = blockIdx_y % (num_blocks_n >> log_tile);
    uint2 const block_idx_mapping = get_block_idx_mapping(blockIdx_m, blockIdx_n, log_tile);
    blockIdx_m = block_idx_mapping.x;
    blockIdx_n = block_idx_mapping.y;

    float C_warp[CTA_M * CTA_N / CTA_SIZE];
    constexpr int kSmemPadKA = CTA_K + SMEM_PAD_A;
    constexpr int kSmemPadKB = CTA_K + SMEM_PAD_B;
    constexpr int kSmemSizeAPerStage = CTA_M * kSmemPadKA;
    constexpr int kSmemSizeBPerStage = CTA_N / kInterleave * kSmemPadKB;
    constexpr int kSmemSizeA = kSmemSizeAPerStage * STAGES;
    constexpr int kSmemSizeB = kSmemSizeBPerStage * STAGES;
    constexpr int scales_load_interval = G / CTA_K;
    extern __shared__ half mem_shared[];
    half* A_shared = mem_shared;
    half* B_shared = mem_shared + kSmemSizeA;
    half* scales_shared = mem_shared + kSmemSizeA + kSmemSizeB;
    half A_shared_warp_[2][WARP_M * INTRIN_K / WARP_SIZE];
    half B_shared_warp_[2][WARP_N * 32 / WARP_SIZE];
    half B_shared_warp_tmp_[2][WARP_N * 16 / WARP_SIZE];
    int cta_offset_m = blockIdx_m * CTA_M;
    int cta_offset_n = blockIdx_n * CTA_N;
    int warp_offset_m = (threadIdx.y % (CTA_M / WARP_M)) * WARP_M;
    int warp_offset_n = (threadIdx.y / (CTA_M / WARP_M)) * WARP_N;

    for (int i = 0; i < CTA_M * CTA_N / CTA_SIZE; i++)
        C_warp[i] = 0.f;

    int gemm_iters = (K + CTA_K - 1) / CTA_K;
    int k_0_0_ld = 0;
    int k_0_0 = 0;
    constexpr int prologue_stages = STAGES == 1 ? 1 : STAGES - 1;
#pragma unroll
    for (k_0_0_ld = 0; k_0_0_ld < prologue_stages; ++k_0_0_ld)
    {
        global_to_share_one_stage_A_T2<CTA_M, CTA_N, CTA_K, CTA_SIZE, 1, STAGES>(
            A, A_shared + k_0_0_ld * kSmemSizeAPerStage, M, K, cta_offset_m, cta_offset_n, k_0_0_ld, 0, true);
        global_to_share_one_stage_B_T2<CTA_M, CTA_N, CTA_K, CTA_SIZE, 1, STAGES>(
            B, B_shared + k_0_0_ld * kSmemSizeBPerStage, N / 4 * K, K, cta_offset_m, cta_offset_n, k_0_0_ld, 0, true);
        global_to_share_one_stage_scales_T2<CTA_M, CTA_N, CTA_K, CTA_SIZE, STAGES, G>(scales,
            scales_shared + (k_0_0_ld / scales_load_interval) * CTA_N, K / G * N, N, cta_offset_m, cta_offset_n,
            k_0_0_ld, 0, k_0_0_ld < gemm_iters && k_0_0_ld % scales_load_interval == 0);
        if constexpr (STAGES > 1)
            __pipeline_commit();
    }
    if constexpr (STAGES > 1)
        __pipeline_wait_prior(STAGES - 2);
    __syncthreads();

    share_to_reg_one_stage_A_T2<CTA_M, CTA_N, CTA_K, STAGES, WARP_M / INTRIN_M>(
        A_shared, A_shared_warp_[0], warp_offset_m, warp_offset_n, 0);
    share_to_reg_one_stage_B_T2<CTA_M, CTA_N, CTA_K, STAGES, true, WARP_N / INTRIN_N, G>(
        B_shared, scales_shared, B_shared_warp_tmp_[0], B_shared_warp_[0], warp_offset_m, warp_offset_n, 0);
    constexpr int SHARED_K_ITERS = WARP_K / INTRIN_K;

    for (; k_0_0 < gemm_iters; ++k_0_0, ++k_0_0_ld)
    {
        int ld_stage = k_0_0_ld % STAGES;
        int compute_stage = k_0_0 % STAGES;
        half* A_shared_this_compute_stage;
        half* B_shared_this_compute_stage;
        half* scales_shared_this_compute_stage;

        for (int iter_k = 0; iter_k < SHARED_K_ITERS; ++iter_k)
        {
            A_shared_this_compute_stage = A_shared + compute_stage * kSmemSizeAPerStage;
            B_shared_this_compute_stage = B_shared + compute_stage * kSmemSizeBPerStage;
            scales_shared_this_compute_stage = scales_shared + (compute_stage / scales_load_interval) * CTA_N;
            share_to_reg_one_stage_A_T2<CTA_M, CTA_N, CTA_K, STAGES, WARP_M / INTRIN_M>(A_shared_this_compute_stage,
                A_shared_warp_[(iter_k + 1) % 2], warp_offset_m, warp_offset_n, (iter_k + 1) % SHARED_K_ITERS);
            if ((iter_k + 1) % kInterleave == 0)
            {
                if (compute_stage % 2 == 1)
                {
                    share_to_reg_one_stage_B_T2<CTA_M, CTA_N, CTA_K, STAGES, true, WARP_N / INTRIN_N, G>(
                        B_shared_this_compute_stage, scales_shared_this_compute_stage, B_shared_warp_tmp_[1],
                        B_shared_warp_[((iter_k + 1) / 2) % 2], warp_offset_m, warp_offset_n,
                        (iter_k + 1) % SHARED_K_ITERS);
                }
                else
                {
                    share_to_reg_one_stage_B_T2<CTA_M, CTA_N, CTA_K, STAGES, true, WARP_N / INTRIN_N, G>(
                        B_shared_this_compute_stage, scales_shared_this_compute_stage, B_shared_warp_tmp_[0],
                        B_shared_warp_[((iter_k + 1) / 2) % 2], warp_offset_m, warp_offset_n,
                        (iter_k + 1) % SHARED_K_ITERS);
                }
            }
            else
            {
                if (compute_stage % 2 == 1)
                {
                    share_to_reg_one_stage_B_T2<CTA_M, CTA_N, CTA_K, STAGES, false, WARP_N / INTRIN_N, G>(
                        B_shared_this_compute_stage, scales_shared_this_compute_stage, B_shared_warp_tmp_[1],
                        B_shared_warp_[((iter_k + 1) / 2) % 2], warp_offset_m, warp_offset_n,
                        (iter_k + 1) % SHARED_K_ITERS);
                }
                else
                {
                    share_to_reg_one_stage_B_T2<CTA_M, CTA_N, CTA_K, STAGES, false, WARP_N / INTRIN_N, G>(
                        B_shared_this_compute_stage, scales_shared_this_compute_stage, B_shared_warp_tmp_[0],
                        B_shared_warp_[((iter_k + 1) / 2) % 2], warp_offset_m, warp_offset_n,
                        (iter_k + 1) % SHARED_K_ITERS);
                }
            }
            __syncthreads();
            half* A_shared_warp = A_shared_warp_[iter_k % 2];
            half* B_shared_warp = B_shared_warp_[(iter_k / 2) % 2];
            for (int i_0_3 = 0; i_0_3 < WARP_M / INTRIN_M; ++i_0_3)
            {
                for (int j_0_4 = 0; j_0_4 < WARP_N / INTRIN_N; ++j_0_4)
                {
                    mma_m16n8k16(C_warp + i_0_3 * WARP_N / INTRIN_N * 8 + j_0_4 * 8, A_shared_warp + i_0_3 * 8,
                        B_shared_warp + j_0_4 * 16 + (iter_k % 2) * 4);
                    mma_m16n8k16(C_warp + i_0_3 * WARP_N / INTRIN_N * 8 + j_0_4 * 8 + 4, A_shared_warp + i_0_3 * 8,
                        B_shared_warp + j_0_4 * 16 + (iter_k % 2) * 4 + 8);
                }
            }

            if (iter_k < WARP_K / INTRIN_K - 1)
            {
                if constexpr (STAGES == 1)
                    __syncthreads();
                global_to_share_one_stage_A_T2<CTA_M, CTA_N, CTA_K, CTA_SIZE, WARP_K / INTRIN_K, STAGES>(A,
                    A_shared + ld_stage * kSmemSizeAPerStage, M, K, cta_offset_m, cta_offset_n, k_0_0_ld, iter_k,
                    k_0_0_ld < gemm_iters);
                global_to_share_one_stage_B_T2<CTA_M, CTA_N, CTA_K, CTA_SIZE, WARP_K / INTRIN_K, STAGES>(B,
                    B_shared + ld_stage * kSmemSizeBPerStage, N / 4 * K, K, cta_offset_m, cta_offset_n, k_0_0_ld,
                    iter_k, k_0_0_ld < gemm_iters);
            }

            if (iter_k == WARP_K / INTRIN_K - 2)
            {
                if constexpr (STAGES == 1 && WARP_K / INTRIN_K > 2)
                {
                    __syncthreads();
                }
                global_to_share_one_stage_A_T2<CTA_M, CTA_N, CTA_K, CTA_SIZE, WARP_K / INTRIN_K, STAGES>(A,
                    A_shared + ld_stage * kSmemSizeAPerStage, M, K, cta_offset_m, cta_offset_n, k_0_0_ld, iter_k + 1,
                    k_0_0_ld < gemm_iters);
                global_to_share_one_stage_B_T2<CTA_M, CTA_N, CTA_K, CTA_SIZE, WARP_K / INTRIN_K, STAGES>(B,
                    B_shared + ld_stage * kSmemSizeBPerStage, N / 4 * K, K, cta_offset_m, cta_offset_n, k_0_0_ld,
                    iter_k + 1, k_0_0_ld < gemm_iters);
                global_to_share_one_stage_scales_T2<CTA_M, CTA_N, CTA_K, CTA_SIZE, STAGES, G>(scales,
                    scales_shared + (ld_stage / scales_load_interval) * CTA_N, K / G * N, N, cta_offset_m, cta_offset_n,
                    k_0_0_ld, iter_k, k_0_0_ld < gemm_iters && k_0_0_ld % scales_load_interval == 0);
                if constexpr (STAGES > 1)
                {
                    __pipeline_commit();
                    __pipeline_wait_prior(STAGES - 2);
                }
                compute_stage = (k_0_0 + 1) % STAGES;
                __syncthreads();
            }
        }
    }
    for (int ax0_0_1 = 0; ax0_0_1 < WARP_M / INTRIN_M; ++ax0_0_1)
    {
        for (int ax1_0_1 = 0; ax1_0_1 < WARP_N / INTRIN_N; ++ax1_0_1)
        {
            for (int local_id = 0; local_id < OP_M * 16 / WARP_SIZE; local_id += 2)
            {
                int write_row
                    = cta_offset_m + warp_offset_m + ax0_0_1 * OP_M + ((local_id % 4) / 2 * 8 + (threadIdx.x / 4));
                if (write_row < M)
                {
                    // FP32 accumulator → BF16 output. BF16 has same exponent range
                    // as FP32 (±3.4e38), so no overflow/clamp needed even for outlier
                    // weight scales (e.g. ALOHA mlp.down_proj scale=1176, accumulator
                    // can reach ~10^6). Previous FP16 output truncated these values
                    // → cascaded NaN through softmax / accuracy collapse.
                    int const c_off = ax0_0_1 * WARP_N / INTRIN_N * 8 + ax1_0_1 * 8 + local_id;
                    __nv_bfloat162 const out_b2 = __float22bfloat162_rn(make_float2(C_warp[c_off], C_warp[c_off + 1]));
                    *reinterpret_cast<__nv_bfloat162*>(C + write_row * N + cta_offset_n + warp_offset_n + ax1_0_1 * 16
                        + (local_id / 4) * 8 + (local_id % 2) + (threadIdx.x % 4) * 2)
                        = out_b2;
                }
            };
        }
    }
}

void gemm_forward_cuda_new(__nv_bfloat16 const* in_feats, int8_t const* weights_device, half const* scaling_factors,
    __nv_bfloat16* out_feats, int m, int n, int k, int group_size, cudaStream_t stream)
{
    // The GEMM kernel will load packed int4 weights as fp16 data tensor.
    half const* kernel = reinterpret_cast<half const*>(weights_device);

    // The kernel template is instantiated with G = 128 only; any other group
    // size would silently dequantize with wrong scales (the GEMV path has the
    // same guard).
    if (group_size != 128)
    {
        throw std::runtime_error("Unsupported group size for gemm kernel (only 128 supported).\n");
    }

    constexpr int G = 128;
    constexpr int CTA_M = 64;
    constexpr int CTA_N = 128;
    constexpr int CTA_K = 64;
    constexpr int WARP_M = 64;
    constexpr int WARP_N = 32;
    constexpr int WARP_K = 64;
    constexpr int STAGES = 4;

    constexpr int NUM_WARPS = (CTA_M / WARP_M) * (CTA_N / WARP_N);
    constexpr int kSmemByteSize
        = (CTA_M * (CTA_K + SMEM_PAD_A) + CTA_N * (CTA_K + SMEM_PAD_B) / kInterleave + CTA_N) * STAGES * sizeof(half);
    static_assert(kSmemByteSize < 99 * 1024, "Shared Memory exceeds device limit.");

    int j_factors1 = n / CTA_N / 1;
    dim3 num_blocks((m + CTA_M - 1) / CTA_M * j_factors1);
    dim3 threads_per_block(WARP_SIZE, NUM_WARPS);
    auto kernel_func = gemm_w4a16_T2<CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, WARP_K, STAGES, G>;
    cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemByteSize);
    kernel_func<<<num_blocks, threads_per_block, kSmemByteSize, stream>>>(
        in_feats, kernel, scaling_factors, out_feats, m, n, k);
}

} // namespace kernel
} // namespace trt_edgellm
