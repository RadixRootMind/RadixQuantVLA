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
 *
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
 * reference: https://github.com/mit-han-lab/llm-awq/blob/main/awq/kernels/csrc/quantization_new/gemv/gemv_cuda.cu
 */

#include "dequantize.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>

#define PACK_FACTOR 8
#define WARP_SIZE 32
#define MEM_ACCESS_SIZE 128

namespace trt_edgellm
{
namespace kernel
{

// Reduce sum within the warp using the tree reduction algorithm.
// psum is FP32 accumulator (changed from FP16 to handle outlier weight scales
// observed in finetuned LLMs; products can exceed FP16 65504 max).
template <int Num, int WarpSize>
__device__ __forceinline__ static void warp_reduce(float* psum, float (*out_smem)[Num * 4])
{
    // kInterleave = 4
    float fpsum[Num];
#pragma unroll
    for (int i = 0; i < Num; ++i)
    {
        fpsum[i] = psum[i];
    }

#pragma unroll
    for (int i = 0; i < Num; ++i)
    {
        // T0 + T1 + T8 + T9 + T16 + T17 + T24 + T25 (kInterleave = 4)
        fpsum[i] += __shfl_xor_sync(~0, fpsum[i], 16);
        fpsum[i] += __shfl_xor_sync(~0, fpsum[i], 8);
        fpsum[i] += __shfl_xor_sync(~0, fpsum[i], 1);
    }
    __syncthreads();
    int warp = threadIdx.x / WarpSize, lane = threadIdx.x % WarpSize;
    if (lane == 0 || lane == 2 || lane == 4 || lane == 6)
    {
#pragma unroll
        for (int i = 0; i < Num; ++i)
        {
            out_smem[warp][i * 4 + lane / 2] = fpsum[i];
        }
    }
    __syncthreads();
};

__device__ __forceinline__ int make_divisible(int c, int divisor)
{
    return (c + divisor - 1) / divisor;
}

template <int NPerBlock, int Batch, int BlockSize, int GroupSize>
__global__ void gemv_kernel(
    __nv_bfloat16 const* inputs, uint32_t const* weight, half const* scales, __nv_bfloat16* outputs, int const IC, int const OC)
{
    int const kStride = 64;
    int const kElemsPerThread = MEM_ACCESS_SIZE / 4;
    int const kThreadsNumPerTile = kStride / kElemsPerThread;
    // assert(MEM_ACCESS_SIZE == 128);

    static constexpr int kShuffleBasicTile = 2;
    static constexpr int kShuffleContinous = 4;
    static constexpr int kShuffleStrided = 4;

    constexpr int Num = NPerBlock * Batch;
    constexpr int kInterleave = 4;

    __nv_bfloat16 local_inputs[kElemsPerThread];
    uint32_t local_qweights[MEM_ACCESS_SIZE / 32];
    half half_weight_buffer[kElemsPerThread];
    half dequantized_weight[kElemsPerThread * NPerBlock];
    half local_scale[NPerBlock];

    float psum[Num];
    for (int i = 0; i < Num; ++i)
        psum[i] = 0.f;

    // extern __shared__ uint8_t shmem[];
    // float(*out_smem)[Num * kInterleave] = reinterpret_cast<float(*)[Num * kInterleave]>(shmem);
    __shared__ float out_smem[BlockSize / WARP_SIZE * 2][Num * kInterleave];

    int const blk_row_offset = blockIdx.x * NPerBlock * kInterleave;
    int const thd_row_offset = (threadIdx.x / kThreadsNumPerTile) % kInterleave;
    int const act_k_offset = threadIdx.x / (kThreadsNumPerTile * kInterleave) * kStride
        + (threadIdx.x % kThreadsNumPerTile) * kElemsPerThread;
    int const group_offset = act_k_offset / GroupSize;
    // TODO: use make_divisible
    uint32_t const* blk_weight_ptr = weight + blk_row_offset * IC / PACK_FACTOR;
    half const* scale_ptr = scales + blk_row_offset + thd_row_offset + group_offset * OC;
    __nv_bfloat16 const* inputs_ptr = inputs + act_k_offset;

    int const act_forward_step = BlockSize * kElemsPerThread / kInterleave;
    int const scale_forward_step = act_forward_step / GroupSize * OC;

    // Main loop iteration, each block completes the outputs for several OCs
    for (int kk = threadIdx.x * kElemsPerThread; kk < IC * kInterleave; kk += BlockSize * kElemsPerThread)
    {
// Load qweight, scales and scaled_zeros
#pragma unroll
        for (int idx = 0; idx < NPerBlock; ++idx)
        {
            // use float4 to load weights, each thread load 32 int4 numbers (1 x float4, 128 bit)
            *((float4*) (local_qweights)) = *((float4*) (blk_weight_ptr + (idx * kInterleave * IC + kk) / PACK_FACTOR));
            local_scale[idx] = *(scale_ptr + idx * kInterleave);

// Map int4 qweight to fp format
#pragma unroll
            for (int i = 0; i < MEM_ACCESS_SIZE / 32; ++i)
            {
                // Converts 32 bits (8 x int4) to 8 fp16
                dequantize_s4_to_fp16x2(*reinterpret_cast<half2*>(local_qweights + i),
                    reinterpret_cast<uint4*>(half_weight_buffer + i * PACK_FACTOR));
            }

// Dequantize (apply s/z) and shuffle elements to match the weight packing format
#pragma unroll
            for (int i = 0; i < kShuffleContinous; ++i)
            {
#pragma unroll
                for (int j = 0; j < kShuffleStrided; ++j)
                {
                    half2 w = *reinterpret_cast<half2*>(
                        half_weight_buffer + (i + j * kShuffleContinous) * kShuffleBasicTile);
                    w = __hmul2(w, __half2half2(local_scale[idx]));
                    dequantized_weight[((i * kShuffleStrided + j) * kShuffleBasicTile + 0) * NPerBlock + idx] = w.x;
                    dequantized_weight[((i * kShuffleStrided + j) * kShuffleBasicTile + 1) * NPerBlock + idx] = w.y;
                }
            }
        }
#pragma unroll
        for (int batch_idx = 0; batch_idx < Batch; ++batch_idx)
        {
            __nv_bfloat16 const* local_inputs_ptr = inputs_ptr + batch_idx * IC;
#pragma unroll
            for (int idx = 0; idx < kElemsPerThread / 8; ++idx)
            {
                // load activation, 8 BF16 (128 bits) / step. BF16 same byte-width as FP16.
                *((float4*) (local_inputs + idx * 8)) = *((float4*) (local_inputs_ptr + idx * 8));
            }
// Perform the MACs (FP32 accumulator)
#pragma unroll
            for (int x = 0; x < NPerBlock / 2; ++x)
            {
#pragma unroll
                for (int y = 0; y < kElemsPerThread; ++y)
                {
                    half2 const w_h2 = *reinterpret_cast<half2*>(dequantized_weight + y * NPerBlock + x * 2);
                    float const a_f = __bfloat162float(local_inputs[y]);
                    int const off = batch_idx * NPerBlock + x * 2;
                    psum[off + 0] = fmaf(__half2float(w_h2.x), a_f, psum[off + 0]);
                    psum[off + 1] = fmaf(__half2float(w_h2.y), a_f, psum[off + 1]);
                }
            }
        }
        inputs_ptr += act_forward_step;
        scale_ptr += scale_forward_step;
    }

    warp_reduce<Num, WARP_SIZE>(psum, out_smem);

    // Num * Interleave = batch * NPerBlock * Interleave -> 1 thread_block write back num
    // BF16 output: full FP32 exponent range, no overflow/clamp needed.
    for (int i = threadIdx.x; i < Num * kInterleave; i += BlockSize)
    {
        int batch_idx = i / (NPerBlock * kInterleave);
        int oc_idx = i % (NPerBlock * kInterleave);
        float acc = 0.f;
        for (int j = 0; j < BlockSize / WARP_SIZE; ++j)
        {
            acc += out_smem[j][i];
        }
        outputs[batch_idx * OC + blk_row_offset + oc_idx] = __float2bfloat16_rn(acc);
    }
}

void gemv_forward_cuda_new(__nv_bfloat16 const* in_feats, int8_t const* weights_device, half const* scaling_factors,
    __nv_bfloat16* out_feats, int m, int n, int k, int group_size, cudaStream_t stream)
{
    auto kernel = reinterpret_cast<uint32_t const*>(weights_device);
    static constexpr int N_PER_BLOCK = 2;
    static constexpr int K_INTERLEAVE = 4;
    static constexpr int BLOCK_SIZE = 256;

    dim3 num_blocks(n / N_PER_BLOCK / K_INTERLEAVE);
    dim3 num_threads(BLOCK_SIZE);

    if (group_size != 128)
    {
        throw std::runtime_error("Unsupported group size for gemv kernel.\n");
    }
    switch (m)
    {
    case 1:
        gemv_kernel<N_PER_BLOCK, 1, BLOCK_SIZE, 128>
            <<<num_blocks, num_threads, 0, stream>>>(in_feats, kernel, scaling_factors, out_feats, k, n);
        break;
    case 2:
        gemv_kernel<N_PER_BLOCK, 2, BLOCK_SIZE, 128>
            <<<num_blocks, num_threads, 0, stream>>>(in_feats, kernel, scaling_factors, out_feats, k, n);
        break;
    case 3:
        gemv_kernel<N_PER_BLOCK, 3, BLOCK_SIZE, 128>
            <<<num_blocks, num_threads, 0, stream>>>(in_feats, kernel, scaling_factors, out_feats, k, n);
        break;
    case 4:
        gemv_kernel<N_PER_BLOCK, 4, BLOCK_SIZE, 128>
            <<<num_blocks, num_threads, 0, stream>>>(in_feats, kernel, scaling_factors, out_feats, k, n);
        break;
    case 5:
        gemv_kernel<N_PER_BLOCK, 5, BLOCK_SIZE, 128>
            <<<num_blocks, num_threads, 0, stream>>>(in_feats, kernel, scaling_factors, out_feats, k, n);
        break;
    case 6:
        gemv_kernel<N_PER_BLOCK, 6, BLOCK_SIZE, 128>
            <<<num_blocks, num_threads, 0, stream>>>(in_feats, kernel, scaling_factors, out_feats, k, n);
        break;
    default: throw std::runtime_error("Unsupported batch size for gemv kernel.\n");
    }
}

} // namespace kernel
} // namespace trt_edgellm
