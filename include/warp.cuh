#pragma once
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

// 使用Warp-Level原语，执行tree-reduction，求和
// __shfl_down_sync：前面的thread向后面的thread要数据
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