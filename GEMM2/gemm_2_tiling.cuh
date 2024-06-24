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
// 每个线程计算对应C矩阵中一个元素
// 同一行的 thread 总是会同样的读取 A 矩阵的同一行数据；同一列的 thread 总是会读取 B 矩阵的同一列数据，对于每一个 Block，将数据移动到这个 Block 共享的一块高速存储区 shared memory 上，从而减少与全局内存交互的次数
// 访存量：每个block需要从global memory中读取(K/BK)*(BM*BK+BK*BN)个单精度浮点数，整个C存在(M/BM)*(N/BN)个block，因此完成C中所有元素计算需要读取(M/BM)*(N/BN)*(K/BK)*(BM*BK+BK*BN)个单精度浮点数
    // 优化前全局访存量为2*K*M*N
    // 共享内存缓存优化后，访存量减少为原来的1/2*(1/BN)*(1/BM)，当BN=BM=32时，访存减少至1/32
template <const int TILE_SIZE>
__global__ void Sgemm_kernel_tiling(float * A, float * B, float * C,
                                    size_t M, size_t K, size_t N,
                                    float alpha, float beta)
{
    // Block index
    int by = blockIdx.y; // 行
    int bx = blockIdx.x; // 列
    // Thread index
    int tx = threadIdx.x;   // 线程块内的行
    int ty = threadIdx.y;   // 线程块内的列

     // 申请共享内存空间
    __shared__ float As[TILE_SIZE * TILE_SIZE];
    __shared__ float Bs[TILE_SIZE * TILE_SIZE];

     // 移动到当前block
    A += by * TILE_SIZE * K;
    B += bx * TILE_SIZE; // B = &B[bx * TILE_SIZE];
    C += by * TILE_SIZE * N + bx * TILE_SIZE;

    float tmp = 0.;
    for (int k = 0; k < K; k += TILE_SIZE) 
    {
        // 缓存A_tile和B_tile
        As[ty * TILE_SIZE + tx] = A[ty * K + tx];   // 从 A 子块的起始位置开始，每个线程加载一个元素到共享内存数组 As
        Bs[ty * TILE_SIZE + tx] = B[ty * N + tx];
        // 同步所有线程缓存完成
        __syncthreads();
        A += TILE_SIZE;
        B += TILE_SIZE * N;
        for (int i = 0; i < TILE_SIZE; i++) 
        {
            tmp += As[ty * TILE_SIZE + i] * Bs[i * TILE_SIZE + tx];
        }
        // FMA计算需要读取缓存数据，在新一轮写入缓存前进行同步，确保所有线程计算完成
        __syncthreads();
    }
    C[ty * N + tx] = alpha * tmp + beta * C[ty * N + tx];
}

double gemm_tiling(float *h_A, float *h_B, float *h_C, 
                  size_t M, size_t K, size_t N)
{
    const int TILE_SIZE = 32;

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

    dim3 dimBlock(32, 32);
    dim3 dimGrid((M + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y); // CEIL_DIV(M, 32), CEIL_DIV(N, 32)
    Sgemm_kernel_tiling<TILE_SIZE><<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, K, N, alpha, beta);
    CHECK_CUDA(cudaEventRecord(start));
    for (int run = 0; run < nIter; run++)
    {
        Sgemm_kernel_tiling<TILE_SIZE><<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, K, N, alpha, beta);
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