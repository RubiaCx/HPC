// optimize sgemm
#include <stdio.h>
#include <stdlib.h>
#include "assert.h"

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// cal offset from row col and ld , in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// transfer float4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

#define CHECK_CUDA(func)                                               \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess)                                     \
        {                                                              \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
            return EXIT_FAILURE;                                       \
        }                                                              \
    }
// 每个线程计算对应C矩阵中多个元素
    // 通过共享内存缓存减少了全局内存访存量和FMA乘累加的访存延迟，但计算访存比没有得到改善，每次迭代计算都需要两个访存指令和一个计算指令
    // 引入thread tile，即一个线程负责block中多个元素的计算
// 可以通过增加block大小（BM，BN）值，进一步降低全局内存的访存量，但是不能无限增大
    // 一方面，block分块矩阵尺寸过大，block数量减少，这样会造成大量 SM（Streaming Multiprocessor）的闲置浪费
    // 另一方面，BN和BM的增加，需要申请更多的共享内存，单线程内共享内存占用越多，活跃线程束越少，不利于隐藏指令延迟
    // 在增加BM和BN值的同时，为了减少共享内存占用，减小BK值

// 访存量：每个block需要从global memory中读取(K/BK)*(BM*BK+BK*BN)个单精度浮点数，整个C存在(M/BM)*(N/BN)个block，因此完成C中所有元素计算需要读取(M/BM)*(N/BN)*(K/BK)*(BM*BK+BK*BN)个单精度浮点数
    // 优化前全局访存量为2*K*M*N
    // 共享内存缓存优化后，访存量减少为原来的1/2*(1/BN)*(1/BM)，当BN=BM=32时，访存减少至1/32
template <
    const int BLOCK_SIZE_M,         // height of block of C that each thread block calculate
    const int BLOCK_SIZE_K,         // width of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,         // width of block of C that each thread block calculate
    const int THREAD_SIZE>
__global__ void Sgemm_kernel_tiling_thread_2d(float * A, float * B, float * C,
                                           size_t M, size_t K, size_t N,
                                           float alpha, float beta)
{
    // Block index
    int by = blockIdx.y; // 行
    int bx = blockIdx.x; // 列
    // Thread index
    int tx = threadIdx.x * THREAD_SIZE; // Correct calculation for tx within block
    int ty = threadIdx.y * THREAD_SIZE; // Correctly map ty to cover THREAD_SIZE elements

    __shared__ float As[BLOCK_SIZE_M * BLOCK_SIZE_K];
    __shared__ float Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];

    // Move to the current block's top left corner in global memory
    A += by * BLOCK_SIZE_M * K;
    B += bx * BLOCK_SIZE_N;
    C += by * BLOCK_SIZE_M * N + bx * BLOCK_SIZE_N;

    // Thread workload: each thread handles THREAD_SIZE elements of C
    float acc[THREAD_SIZE][THREAD_SIZE] = {0.0}; 
    for (int k = 0; k < K; k += BLOCK_SIZE_K) 
    {
        if (ty < BLOCK_SIZE_M && tx < BLOCK_SIZE_K) 
        {
            As[ty * BLOCK_SIZE_K + tx] = A[ty * K + tx];
        }
        if (ty < BLOCK_SIZE_K && tx < BLOCK_SIZE_N) 
        {
            Bs[ty * BLOCK_SIZE_N + tx] = B[ty * N + tx];
        }
        __syncthreads(); // Synchronize to make sure the tiles are loaded

        // Perform the matrix multiplication for the BLOCK_SIZE_K
        for (int i = 0; i < BLOCK_SIZE_K; ++i) 
        {
            for (int j = 0; j < THREAD_SIZE; ++j) 
                for (int k = 0; k < THREAD_SIZE; ++k) 
                {
                    //内层循环中 As[(ty + j) * BK + i] 重复访问TN次
                    acc[j][k] += As[(ty + j) * BLOCK_SIZE_K + i] * Bs[i * BLOCK_SIZE_N + tx + k];
                }
        }
        __syncthreads(); // Synchronize before loading new tiles

        // Move the A and B pointers to the next tiles
        A += BLOCK_SIZE_K;
        B += BLOCK_SIZE_K * N;
    }

    // Write the results to the C matrix
    for (int j = 0; j < THREAD_SIZE; ++j) 
    for (int k = 0; k < THREAD_SIZE; ++k) 
        {
            C[(ty + j) * N + tx + k] = alpha * acc[j][k] + beta * C[(ty + j) * N + tx + k];
        }
}

double gemm_tiling_thread_2d(float *h_A, float *h_B, float *h_C, 
                  size_t M, size_t K, size_t N)
{
    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_N = 128;
    const int BLOCK_SIZE_K = 8;
    const int THREAD_SIZE = 8;

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
    double flopsPerMatrixMul = 2.0 * M * K * N;
    // double flopsPerMatrixMul = 2.0 * M * N * K - M * N;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 1000;

    dim3 dimBlock(16, 16);
    dim3 dimGrid((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N, (M  + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M); // CEIL_DIV(M, 32), CEIL_DIV(N, 32)
    Sgemm_kernel_tiling_thread_2d<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE><<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, K, N, alpha, beta);
    CHECK_CUDA(cudaEventRecord(start));
    for (int run = 0; run < nIter; run++)
    {
        Sgemm_kernel_tiling_thread_2d<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE><<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, K, N, alpha, beta);
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
    printf("My Tiling Thread 2D GEMM Performance = %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
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