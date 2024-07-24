#pragma once
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

// 使用Warp-Level原语，执行tree-reduction，求和
// __shfl_down_sync(mask, var, delta)：前面的thread向后面的thread要数据，mask指示活跃线程
// __shfl_up_sync: 后面的thread向前面的thread要数据
// 1. 返回前面的thread向后面的thread要的数据，比如__shfl_down_sync(0xffffffff, sum, 16)那就是返回16号线程，17号线程的数据
// 2. 使用warp shuffle指令的数据交换不会出现warp在shared memory上交换数据时的不一致现象，这一点是由GPU driver完成，故无需任何sync, 比如syncwarp
// 3. if存在的必要性: block Size为人为指定，那么有可能位于以下5个if的区间，所以需要这些if根据实际分配的block size来过滤操作
template <class value_t>
__device__ __forceinline__ value_t warpReduceSum(value_t sum, int WarpSize)
{
    if (WarpSize >= 32)
        sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (WarpSize >= 16)
        sum += __shfl_down_sync(0xffffffff, sum, 8); // 0-8, 1-9, 2-10, etc.
    if (WarpSize >= 8)
        sum += __shfl_down_sync(0xffffffff, sum, 4); // 0-4, 1-5, 2-6, etc.
    if (WarpSize >= 4)
        sum += __shfl_down_sync(0xffffffff, sum, 2); // 0-2, 1-3, 4-6, 5-7, etc.
    if (WarpSize >= 2)
        sum += __shfl_down_sync(0xffffffff, sum, 1); // 0-1, 2-3, 4-5, etc.
    return sum;
}

// __shfl_xor_sync：通过XOR操作实现跨线程的值交换，在某些情况下提供更好的性能，特别是在非连续的索引操作中
template <class value_t>
__device__ __forceinline__ value_t warpReduceSum_v2(value_t sum, int WarpSize)
{
    #pragma unroll
    for (int mask = kWarpSize; mask >= 2; mask >>= 1) 
    {
        sum += __shfl_xor_sync(0xffffffff, val, mask); // 使用__shfl_xor_sync与相隔mask距离的线程交换数据
    }
    return sum;
}

template <class value_t>
__device__ __forceinline__ value_t warpReduceMax(value_t max, int WarpSize)
{
    if (WarpSize >= 32)
        max = maxf(max, __shfl_down_sync(0xffffffff, max, 16));
    if (WarpSize >= 16)
        max = maxf(max, __shfl_down_sync(0xffffffff, max, 8));
    if (WarpSize >= 8)
        max = maxf(max, __shfl_down_sync(0xffffffff, max, 4));
    if (WarpSize >= 4)
        max = maxf(max, __shfl_down_sync(0xffffffff, max, 2));
    if (WarpSize >= 2)
        max = maxf(max, __shfl_down_sync(0xffffffff, max, 1));
    return max;
}

// Block reduce sum/max/min device helper for Layer/RMS Norm/Softmax etc.
// grid 1D block 1D, grid(N/128), block(128)
template <class value_t, const int BLOCK_SIZE = 128>
__device__ __forceinline__ value_t blockReduceSum(value_t val)
{
    // always <= 32 warps per block (limited by 1024 threads per block)
    constexpr int WARP_NUM = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    static __shared__ value_t shared[WARP_NUM];

    val = warpReduceSum<value_t>(val, WARP_SIZE);
    if (lane == 0)
        shared[warp] = val;
    __syncthreads();
    val = (lane < WARP_NUM) ? shared[lane] : 0.0f;
    val = warpReduceSum<value_t>(val, WARP_SIZE);
    return val;
}

template <class value_t, const int BLOCK_SIZE = 128>
__device__ __forceinline__ value_t blockReduceMax(value_t val)
{
    // always <= 32 warps per block (limited by 1024 threads per block)
    constexpr int WARP_NUM = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    static __shared__ value_t shared[WARP_NUM];

    val = warpReduceMax<value_t>(val, WARP_SIZE);
    if (lane == 0)
        shared[warp] = val;
    __syncthreads();
    val = (lane < WARP_NUM) ? shared[lane] : 0.0f;
    val = warpReduceMax<value_t>(val, WARP_SIZE);
    return val;
}