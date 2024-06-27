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

/* 一共开启 M * N个thread，每个thread负责C中一个元素的计算
 * 每一个乘法运算需要读两次内存和一次FMA
 *** 计算访存比：每次迭代需要进行一次FMA（乘累加）和两次全局内存读取，计算访存比1/2
    - 较低的计算访存比无法有效隐藏访存延迟
 *** 访存量：访问全局内存，C矩阵每个元素计算需要访问2K个单精度浮点数，完成全部计算需要 2*K*M*N
    - 全局内存访问延迟高（几百cycle）
    - 相同位置元素被重复读取（C中同一行元素计算共享A中同一行元素，C中同一列元素计算共享B中同一列元素）
*/
__global__ void Sgemm_kernel_global(
    float * A, float * B, float * C,
    size_t M, size_t K, size_t N,
    float alpha, float beta)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(ty < M && tx < N) 
    {
        float c = 0.0;
        for(int i = 0; i < K; ++i){
            c += A[ty * K + i] * B[i * N + tx]; // PROBLEM
        }
        C[ty * N + tx] = beta * C[ty * N + tx] + alpha * c;
    }
}

double gemm_global(float *h_A, float *h_B, float *h_C, 
                  size_t M, size_t K, size_t N)
{
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
    dim3 dimGrid((M + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);
    Sgemm_kernel_global<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, K, N, alpha, beta);
    CHECK_CUDA(cudaEventRecord(start));
    for (int run = 0; run < nIter; run++)
    {
        Sgemm_kernel_global<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, K, N, alpha, beta);
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
    printf("My Naive GEMM Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
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