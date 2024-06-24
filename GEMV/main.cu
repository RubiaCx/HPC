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
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

void cpuSgemv(float *A, float *x, float *y, int M, int N)
{
    for (int i = 0; i < M; i++)
    {
        float sum = 0.0;
        for (int j = 0; j < N; j++)
        {
            sum += A[i * N + j] * x[j];
        }
        y[i] = sum;
    }
}

// x = blockIdx.x * blockDim.x + threadIdx.x;
// y = blockIdx.y * blockDim.y + threadIdx.y
// N = 16 每个warp处理2行
template <const int WARP_SIZE, const int ROW_PER_WARP, class value_t>
__global__ void Sgemv_kernel_n16(value_t *__restrict__ A, value_t *__restrict__ x, value_t *__restrict__ y,
                                 const int M, const int N)
{
    int tx = threadIdx.x;  // 每个线程
    int warpRowBase = blockIdx.x * ROW_PER_WARP;  // 每个block处理ROW_PER_WARP行，blockIdx.x * ROW_PER_WARP 是第一行的索引

    // 处理第一行和第二行
    for (int row = warpRowBase; row < min(warpRowBase + ROW_PER_WARP, M); ++row) {
        if (tx < N) 
        {
            float res = A[row * N + tx] * x[tx];
            // 在warp内进行归约求和
            res = warpReduceSum<value_t>(res, WARP_SIZE);
            // 只有第一个线程写回结果到全局内存
            if (tx == 0)
                y[row] = res;
        }
    }

}

// N = 32，每个warp处理1行
template <const int WARP_SIZE, class value_t>
__global__ void Sgemv_kernel_n32(value_t *__restrict__ A, value_t *__restrict__ x, value_t *__restrict__ y,
                                 const int M, const int N)
{
    int tx = threadIdx.x;  // 每个线程处理一列
    int row = blockIdx.x;  // 每个块处理一行

    if (row < M)
    {
        value_t res = 0;
        // 每个线程处理一列中的元素
        res += A[row * N + tx] * x[tx];
        
        // 对 warp 内的结果进行归约求和
        res = warpReduceSum<value_t>(res, WARP_SIZE);

        // 只有第一个线程写回结果到全局内存
        if (tx == 0)
            y[row] = res;
    }
}

// N = 128，每个warp处理1行，向量化指令，即一个指令处理4个元素
template <const int WARP_SIZE, class value_t>
__global__ void Sgemv_kernel_n128(value_t *__restrict__ A, value_t *__restrict__ x, value_t *__restrict__ y,
                                  const int M, const int N)
{
    int tx = threadIdx.x;  
    int row = blockIdx.x;  // 每个块处理一行

    if (row < M)
    {
        value_t res = 0;
        int numVectorPerRow = N / 4; // 因为使用 float4，每行有 N/4 个向量
        int numVectorsPerThread = numVectorPerRow / WARP_SIZE;  // 计算每个线程应处理多少向量
        int vectorIndexOffset = tx * numVectorsPerThread;       // 每个线程处理向量的起始索引

        // 确保不越界
        int maxVectorIndex = min((tx + 1) * numVectorsPerThread, numVectorPerRow);

        for (int i = vectorIndexOffset; i < maxVectorIndex; i++)
        {
            int col = i * 4;  // 每个向量对应四个列
            float4 vecA = reinterpret_cast<float4*>(&A[row * N + col])[0];
            float4 vecX = reinterpret_cast<float4*>(&x[col])[0];
            res += vecA.x * vecX.x + vecA.y * vecX.y + vecA.z * vecX.z + vecA.w * vecX.w;
        }

        res = warpReduceSum<value_t>(res, WARP_SIZE);

        // Write the result to global memory
        if (tx == 0)
            y[row] = res;
    }
}


template <class value_t>
__device__ value_t kahanSum(value_t *data, int N)
{
    value_t sum = 0.0;
    value_t c = 0.0; // 一个运行时小的误差补偿变量
    for (int i = 0; i < N; ++i)
    {
        value_t y = data[i] - c;
        value_t t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

template <const int WARP_SIZE, const int ROW_PER_WARP, class value_t>
__global__ void Sgemv_kernel_Kahan(value_t *__restrict__ A, value_t *__restrict__ x, value_t *__restrict__ y,
                                   const int M, const int N)
{
    extern __shared__ value_t shared_data[];

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int laneId = tx % WARP_SIZE;
    int current_warp_row = (blockDim.y * bx + ty) * ROW_PER_WARP;
    const int sub_WARP_SIZE = WARP_SIZE / ROW_PER_WARP;
    int kLaneId = laneId % sub_WARP_SIZE;
    int current_thread_row = current_warp_row + laneId / sub_WARP_SIZE;

    if (current_thread_row < M)
    {
        int col = kLaneId;
        while (col < N)
        {
            shared_data[threadIdx.x * N + col] = A[current_thread_row * N + col] * x[col];
            col += sub_WARP_SIZE;
        }
        __syncthreads(); // 确保所有数据都写入shared memory

        if (kLaneId == 0)
        { // 使用第一个线程来执行Kahan求和
            y[current_thread_row] = kahanSum<value_t>(&shared_data[threadIdx.x * N], N);
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("usage: ./main [M] [N]\n");
        exit(0);
    }
    size_t M = atoi(argv[1]);
    size_t N = atoi(argv[2]);

    size_t bytes_A = sizeof(float) * M * N;
    size_t bytes_x = sizeof(float) * N;
    size_t bytes_y = sizeof(float) * M;
    float *h_A = (float *)malloc(bytes_A);
    float *h_x = (float *)malloc(bytes_x);
    float *h_y = (float *)malloc(bytes_y);
    float *h_y_gpu = (float *)malloc(bytes_y);
    float *h_y_cpu = (float *)malloc(bytes_y);
    float *h_y_api = (float *)malloc(bytes_y);
    float *d_A;
    float *d_x;
    float *d_y;
    generate_random_value_float(h_A, M * N, 0.0, 1.0);
    generate_random_value_float(h_x, N, 0.0, 1.0);
    memset(h_y, 0, M * sizeof(float));
    memset(h_y_gpu, 0, M * sizeof(float));
    memset(h_y_cpu, 0, M * sizeof(float));
    memset(h_y_api, 0, M * sizeof(float));

    // // 输出前10个元素以验证
    // for (int i = 0; i < 10; ++i) {
    //     std::cout << h_A[i] << " ";
    // }
    // std::cout << std::endl;

    int nIter = 1;
    cpuSgemv(h_A, h_x, h_y_cpu, M, N);
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (int run = 0; run < nIter; run++)
    {
        cpuSgemv(h_A, h_x, h_y_cpu, M, N);
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = end_cpu - start_cpu;
    printf("CPU time: %.5f seconds\n", cpu_time.count() / nIter);
    // for (int i = 0; i < M; ++i) {
    //     std::cout << h_y_cpu[i] << " ";
    // }
    // std::cout << std::endl;
    const int WARP_SIZE = 32;
    const int ROW_PER_WARP = 2;

    float gpu_time = 0;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaMalloc(&d_A, bytes_A));
    CHECK_CUDA(cudaMalloc(&d_x, bytes_x));
    CHECK_CUDA(cudaMalloc(&d_y, bytes_y));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, bytes_x, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, bytes_y, cudaMemcpyHostToDevice));
    if(N <= 16) {
        dim3 dimGrid((M + ROW_PER_WARP - 1) / ROW_PER_WARP);  // 确保所有行都被处理，每个块处理 ROW_PER_WARP 行
        dim3 dimBlock(WARP_SIZE);  // 每行分配 WARP_SIZE / ROW_PER_WARP 个线程，每块处理 ROW_PER_WARP 行
        Sgemv_kernel_n16<WARP_SIZE, ROW_PER_WARP, float><<<dimGrid, dimBlock>>>(d_A, d_x, d_y, M, N);
        CHECK_CUDA(cudaEventRecord(start));
        for (int run = 0; run < nIter; run++)
        {
            Sgemv_kernel_n16<WARP_SIZE, ROW_PER_WARP, float><<<dimGrid, dimBlock>>>(d_A, d_x, d_y, M, N);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                printf("CUDA Error: %s\n", cudaGetErrorString(err));
            }
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&gpu_time, start, stop));
        CHECK_CUDA(cudaMemcpy(h_y_gpu, d_y, bytes_y, cudaMemcpyDeviceToHost));
        printf("GPU time: %.5f seconds\n", gpu_time / nIter);
    }
    else if(N == 32) {
        dim3 dimGrid(M);  
        dim3 dimBlock(WARP_SIZE); 
        Sgemv_kernel_n32<WARP_SIZE, float><<<dimGrid, dimBlock>>>(d_A, d_x, d_y, M, N);
        CHECK_CUDA(cudaEventRecord(start));
        for (int run = 0; run < nIter; run++)
        {
            Sgemv_kernel_n32<WARP_SIZE, float><<<dimGrid, dimBlock>>>(d_A, d_x, d_y, M, N);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                printf("CUDA Error: %s\n", cudaGetErrorString(err));
            }
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&gpu_time, start, stop));
        CHECK_CUDA(cudaMemcpy(h_y_gpu, d_y, bytes_y, cudaMemcpyDeviceToHost));
        printf("GPU time: %.5f seconds\n", gpu_time / nIter);
    }
    else if(N >= 128) {
        dim3 dimGrid(M);  
        dim3 dimBlock(WARP_SIZE); 
        Sgemv_kernel_n128<WARP_SIZE, float><<<dimGrid, dimBlock>>>(d_A, d_x, d_y, M, N);
        CHECK_CUDA(cudaEventRecord(start));
        for (int run = 0; run < nIter; run++)
        {
            Sgemv_kernel_n128<WARP_SIZE, float><<<dimGrid, dimBlock>>>(d_A, d_x, d_y, M, N);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                printf("CUDA Error: %s\n", cudaGetErrorString(err));
            }
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&gpu_time, start, stop));
        CHECK_CUDA(cudaMemcpy(h_y_gpu, d_y, bytes_y, cudaMemcpyDeviceToHost));
        printf("GPU time: %.5f seconds\n", gpu_time / nIter);
    }
   
    // for (int i = 0; i < M; ++i) {
    //     std::cout << h_y_gpu[i] << " ";
    // }
    // std::cout << std::endl;
    // cublas
    cublasHandle_t blas_handle;
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;
    float api_time = 0;
    CHECK_CUDA(cudaMemcpy(d_y, h_y, bytes_y, cudaMemcpyHostToDevice));
    cublasSgemv(blas_handle, CUBLAS_OP_T, N, M, &alpha, d_A, N, d_x, 1, &beta, d_y, 1);
    CHECK_CUDA(cudaEventRecord(start));
    for (int run = 0; run < nIter; run++)
    {
        cublasSgemv(blas_handle, CUBLAS_OP_T, N, M, &alpha, d_A, N, d_x, 1, &beta, d_y, 1);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&api_time, start, stop));
    printf("API time: %.5f seconds\n", api_time / nIter);
    CHECK_CUDA(cudaMemcpy(h_y_api, d_y, bytes_y, cudaMemcpyDeviceToHost));
    cublasDestroy(blas_handle);
    // for (int i = 0; i < M; ++i) {
    //     std::cout << h_y_api[i] << " ";
    // }
    // std::cout << std::endl;
    bool correct = true;
    correct = checkAnswer(h_y_api, h_y_cpu, M, 1);
    printf("API vs CPU %s\n", correct ? "Result= PASS" : "Result= FAIL");
    correct = checkAnswer(h_y_gpu, h_y_cpu, M, 1);
    printf("GPU vs CPU %s\n", correct ? "Result= PASS" : "Result= FAIL");
    // Free Memory
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);

    free(h_A);
    free(h_x);
    free(h_y);
    free(h_y_api);
    free(h_y_cpu);
    free(h_y_gpu);
}