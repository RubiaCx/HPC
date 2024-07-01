#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <random>
#include <math.h>
#include <chrono>
#include "../include/utils.h"
#include "../include/warp.cuh"
#include "../include/error.cuh"
// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// cal offset from row col and ld , in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// transfer float4
#define FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

// Dot Product
// grid(N/128), block(128)
// a: Nx1, b: Nx1, y=sum(elementwise_mul(a,b))
// 每个线程计算一个元素的乘积，然后进行warp级别的reduce，最后一个warp进行block级别的reduce
template <const int BLOCK_SIZE = 128, const int WARP_SIZE = 32>
__global__ void dot(float *a, float *b, float *y, int N)
{
    int tid = threadIdx.x;
    int idx = blockIdx.x * BLOCK_SIZE + tid;
    constexpr int NUM_WARPS = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    float prod = (idx < N) ? a[idx] * b[idx] : 0.0f;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    // perform warp sync reduce.
    prod = warp_reduce_sum<float>(prod, NUM_WARPS);
    if (lane == 0)
        reduce_smem[warp] = prod;
    __syncthreads(); 
    // the first warp compute the final sum.
    prod = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0)  // 第一个warp（warp == 0）将warp级别的reduce结果进行block级别的reduce
        prod = warp_reduce_sum<float>(prod, NUM_WARPS);
    if (tid == 0)   // 第一个线程（tid == 0）将最终的点积结果通过原子加操作累加到全局内存的 y
        atomicAdd(y, prod);
}

// Dot Product + Vec4
// grid(N/128), block(128/4)
// a: Nx1, b: Nx1, y=sum(elementwise_mul(a,b))
template <const int BLOCK_SIZE = 128 / 4, const int WARP_SIZE = 32>
__global__ void dot_vec4(float *a, float *b, float *y, int N)
{
    int tid = threadIdx.x;
    int idx = (blockIdx.x * BLOCK_SIZE + tid) * 4;
    constexpr int NUM_WARPS = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    float4 reg_a = FLOAT4(a[idx]);
    float4 reg_b = FLOAT4(b[idx]);
    float prod = (idx < N) ? (reg_a.x * reg_b.x + reg_a.y * reg_b.y + reg_a.z * reg_b.z + reg_a.w * reg_b.w) : 0.0f;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    prod = warp_reduce_sum<WARP_SIZE>(prod);
    if (lane == 0)
        reduce_smem[warp] = prod;
    __syncthreads(); 
    prod = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0)
        prod = warp_reduce_sum<NUM_WARPS>(prod);
    if (tid == 0)
        atomicAdd(y, prod);
}