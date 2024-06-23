// optimize sgemm
#include <stdio.h>
#include <stdlib.h>
#include "assert.h"

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../include/error.cuh"

// cal offset from row col and ld , in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// transfer float4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])


// 同一行的 thread 总是会同样的读取 A 矩阵的同一行数据；同一列的 thread 总是会读取 B 矩阵的同一列数据
// 对于每一个 Block，将数据移动到这个 Block 共享的一块高速存储区 shared memory 上，从而减少与全局内存交互的次数
// 考虑到 shared memory 的容量有限，因此可以一次只取一部分 k，然后通过循环迭代的方式完成这个 Block 所负责的矩阵乘区域
template <
    const int BLOCK_SIZE_M,         // height of block of C that each thread block calculate
    const int BLOCK_SIZE_K,         // width of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,         // width of block of C that each thread block calculate
    const int THREAD_SIZE_Y,        // height of block of C that each thread calculate
    const int THREAD_SIZE_X,        // width of block of C that each thread calculate
    const bool ENABLE_DOUBLE_BUFFER // whether enable double buffering or not
    >
__global__ void Sgemm_kernel_tiling(
    float *__restrict__ A, float *__restrict__ B, float *__restrict__ C,
    const int M, const int N, const int K,
    float alpha, float beta)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;
    // Thread index
    int tx = threadIdx.x % BLOCK_SIZE_N;
    int ty = threadIdx.x / BLOCK_SIZE_N;

     // 申请共享内存空间
    __shared__ float As[BLOCK_SIZE_M * BLOCK_SIZE_K];
    __shared__ float Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];

     // 移动到当前block
    A = &A[by * BLOCK_SIZE_M * K];
    B = &B[bx * BLOCK_SIZE_N];
    C = &C[by * BLOCK_SIZE_M * N + bx * BLOCK_SIZE_N];

    float tmp = 0.;
    for (int k = 0; k < K; k += BLOCK_SIZE_K) {
        // 缓存A_tile和B_tile
        As[ty * BLOCK_SIZE_K + tx] = A[ty * K + tx];
        Bs[ty * BLOCK_SIZE_N + tx] = B[ty * N + tx];
        // 同步所有线程缓存完成
        __syncthreads();
        A += BLOCK_SIZE_K;
        B += BLOCK_SIZE_K * N;
        for (int i = 0; i < BLOCK_SIZE_K; i++) 
        {
            tmp += As[ty * BLOCK_SIZE_K + i] * Bs[i * BLOCK_SIZE_N + tx];
        }
        // FMA计算需要读取缓存数据，在新一轮写入缓存前进行同步，确保所有线程计算完成
        __syncthreads();
    }
    C[ty * N + tx] = alpha * tmp + beta * C[ty * N + tx];
}

double gemm_tiling(float *h_A, float *h_B, float *h_C, 
                  size_t M, size_t K, size_t N)
{
    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_K = 8;
    const int BLOCK_SIZE_N = 128;
    const int THREAD_SIZE_X = 8;
    const int THREAD_SIZE_Y = 8;
    const bool ENABLE_DOUBLE_BUFFER = false;

    float *d_A;
    float *d_B;
    float *d_C;
    float alpha = 1.0;
    float beta = 1.0;
    size_t bytes_A = sizeof(float) * M * K;
    size_t bytes_B = sizeof(float) * K * N;
    size_t bytes_C = sizeof(float) * M * N;
    CHECK_CUDA(cudaMalloc(&d_A, bytes_A));
    CHECK_CUDA(cudaMalloc(&d_B, bytes_B));
    CHECK_CUDA(cudaMalloc(&d_C, bytes_C));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyHostToDevice));

    double msecPerMatrixMul = 0.0;
    double gigaFlops = 0.0;
    // double flopsPerMatrixMul = 2.0 * M * N * K;
    double flopsPerMatrixMul = 2.0 * M * N * K - M * N;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 100;

    CHECK_CUDA(cudaEventRecord(start));
    for (int run = 0; run < nIter; run++)
    {
        dim3 dimBlock(32, 32);
        dim3 dimGrid((M + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y); // CEIL_DIV(M, 32), CEIL_DIV(N, 32)
        Sgemm_kernel_tiling<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER>
                           <<<dimGrid, dimBlock>>>(d_A, d_B, d_C, alpha, beta, M, N, K);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        } 
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&msecTotal, start, stop));

    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    msecPerMatrixMul = msecTotal / nIter;
    gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf("My Tiling GEMM Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
           gigaFlops, msecPerMatrixMul, flopsPerMatrixMul);

    // for(int i = 0; i < M; i++)
    // {
    //     for(int j = 0; j < N; j++) {
    //         printf("h_C[%d][%d] = %f\n", i, j, h_C[i * N + j]);
    //     }
    //     printf("\n");
    // }

    // Free Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return gigaFlops;
}