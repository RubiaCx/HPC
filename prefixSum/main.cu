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

void cpuPrefixSum(float *x, float *y, int N)
{
    y[0] = x[0];
    for (int i = 1; i < N; i++)
    {
        y[i] = y[i - 1] + x[i];
    }
}

__global__ void prefixSum_Gmem_kernel(float * in, float * out, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N)
    {
        out[idx] = in[idx];
        __syncthreads();
        for(int i = 1; i < N; i++)
        {
            if(idx >= i)
            {
                out[idx] = out[idx] + in[idx-i];
            }
            __syncthreads();
        }
    }
}
// x = blockIdx.x * blockDim.x + threadIdx.x;
// y = blockIdx.y * blockDim.y + threadIdx.y;
template <int BLOCK_DIM>
__global__ void prefixSum_Smem_kernel(float *X, float *Y, int N)
{
    __shared__ float shared_X[BLOCK_DIM];
    int tid = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int tx = threadIdx.x; // 线程在块内的索引

    if (tid < N)
    {
        shared_X[tx] = X[tid];
    }
    else
    {
        shared_X[tx] = 0;
    }

    __syncthreads();

    for (unsigned int stride = 1; stride < BLOCK_DIM; stride *= 2)
    {
        float temp = 0;
        if (tx >= stride)
        { 
            temp = shared_X[tx - stride];
        }
        __syncthreads(); // 确保所有的temp都被正确赋值
        if (tx >= stride)
        {
            shared_X[tx] += temp;
        }
        __syncthreads(); // 确保所有的写都已经写入
    }

    if (tid < N)
    {
        Y[tid] = shared_X[tx];
    }
}

// https://www.nowcoder.com/discuss/392602151581753344
// brent-kung分段前缀和，同时处理双倍的数据
template <int BLOCK_DIM>
__global__ void brentkung_presum_intrablock(float *X, float *Y, int N)
{
    __shared__ float shared_X[BLOCK_DIM];

    // 将数组加载到共享寄存器，一个线程加载两个元素。
    int tid = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x; // 线程在块内的索引
    if (tid < N)
        shared_X[tx] = X[tid];
    else
        shared_X[tx] = 0.0f;

    if ((tid + blockDim.x) < N)
        shared_X[tx + blockDim.x] = X[tid + blockDim.x];
    else
        shared_X[tx + blockDim.x] = 0.0f;

    // 不带控制流的归约求和
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        __syncthreads();
        int index = (tx + 1) * stride * 2 - 1; // 每个线程更新两个间隔stride的元素
        if (index < BLOCK_DIM)
            shared_X[index] += shared_X[index - stride];
    }

    // 分发部分和
    for (unsigned int stride = BLOCK_DIM / 4; stride > 0; stride /= 2)
    {
        __syncthreads();
        int index = (tx + 1) * stride * 2 - 1;
        if (index + stride < BLOCK_DIM)
            shared_X[index + stride] += shared_X[index];
    }
    __syncthreads();
    if (tid < N)
        X[tid] = shared_X[tx];
    if (tid + blockDim.x < N)
        X[tid + blockDim.x] = shared_X[tx + blockDim.x];

    __syncthreads();
    if (tx == 0)
    {
        Y[blockIdx.x] = shared_X[BLOCK_DIM - 1];
    }
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("usage: ./main [N]\n");
        exit(0);
    }
    size_t N = atoi(argv[1]);

    size_t bytes_x = sizeof(float) * N;
    size_t bytes_y = sizeof(float) * N;
    float *h_x = (float *)malloc(bytes_x);
    float *h_y = (float *)malloc(bytes_y);
    float *h_y_gpu = (float *)malloc(bytes_y);
    float *h_y_gpu1 = (float *)malloc(bytes_y);
    float *h_y_gpu2 = (float *)malloc(bytes_y);
    float *h_y_cpu = (float *)malloc(bytes_y);
    float *d_x;
    float *d_y;
    generate_random_value_float(h_x, N, 0.0, 2.0);
    memset(h_y, 0, N * sizeof(float));
    memset(h_y_gpu, 0, N * sizeof(float));
    memset(h_y_gpu1, 0, N * sizeof(float));
    memset(h_y_gpu2, 0, N * sizeof(float));
    memset(h_y_cpu, 0, N * sizeof(float));

    // // 输出前10个元素以验证
    // for (int i = 0; i < 10; ++i) {
    //     std::cout << h_x[i] << " ";
    // }
    // std::cout << std::endl;

    int nIter = 100;
    cpuPrefixSum(h_x, h_y_cpu, N);
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (int run = 0; run < nIter; run++)
    {
        cpuPrefixSum(h_x, h_y_cpu, N);
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = end_cpu - start_cpu;
    printf("CPU time: %.5f seconds\n", cpu_time.count() / nIter);
    for (int i = 0; i < N; ++i)
    {
        std::cout << h_y_cpu[i] << " ";
    }
    std::cout << std::endl;

    const int blockSize = 1024;

    int numBlocks = (N + blockSize - 1) / blockSize;
    dim3 dimGrid(numBlocks);  
    dim3 dimBlock(blockSize); 
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaMalloc(&d_x, bytes_x));
    CHECK_CUDA(cudaMalloc(&d_y, bytes_y));

    float gpu_time = 0.0;
    CHECK_CUDA(cudaMemcpy(d_x, h_x, bytes_x, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, bytes_y, cudaMemcpyHostToDevice));
    prefixSum_Gmem_kernel<<<dimGrid, dimBlock>>>(d_x, d_y, N);
    CHECK_CUDA(cudaEventRecord(start));
    for (int run = 0; run < nIter; run++)
    {
        prefixSum_Gmem_kernel<<<dimGrid, dimBlock>>>(d_x, d_y, N);
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
    printf("GPU Gmem time: %.5f seconds\n", gpu_time / nIter);
    for (int i = 0; i < N; ++i) {
        std::cout << h_y_gpu[i] << " ";
    }
    std::cout << std::endl;

    // ***** shared memory ***** //
    float gpu_time1 = 0.0;
    CHECK_CUDA(cudaMemcpy(d_x, h_x, bytes_x, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, bytes_y, cudaMemcpyHostToDevice));
    prefixSum_Smem_kernel<blockSize><<<dimGrid, dimBlock>>>(d_x, d_y, N);
    CHECK_CUDA(cudaEventRecord(start));
    for (int run = 0; run < nIter; run++)
    {
        prefixSum_Smem_kernel<blockSize><<<dimGrid, dimBlock>>>(d_x, d_y, N);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        }
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&gpu_time1, start, stop));
    CHECK_CUDA(cudaMemcpy(h_y_gpu1, d_y, bytes_y, cudaMemcpyDeviceToHost));
    printf("GPU Smem time: %.5f seconds\n", gpu_time1 / nIter);
    for (int i = 0; i < N; ++i) {
        std::cout << h_y_gpu1[i] << " ";
    }
    std::cout << std::endl;

    float gpu_time2 = 0.0;
    CHECK_CUDA(cudaMemcpy(d_x, h_x, bytes_x, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, bytes_y, cudaMemcpyHostToDevice));
    brentkung_presum_intrablock<blockSize><<<dimGrid, dimBlock>>>(d_x, d_y, N);
    CHECK_CUDA(cudaEventRecord(start));
    for (int run = 0; run < nIter; run++)
    {
        brentkung_presum_intrablock<blockSize><<<dimGrid, dimBlock>>>(d_x, d_y, N);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        }
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&gpu_time2, start, stop));
    CHECK_CUDA(cudaMemcpy(h_y_gpu2, d_y, bytes_y, cudaMemcpyDeviceToHost));
    printf("GPU2 time: %.5f seconds\n", gpu_time2 / nIter);
    bool correct = checkAnswer(h_y_gpu2, h_y_cpu, N, 1);
    printf("GPU vs CPU %s\n", correct ? "Result= PASS" : "Result= FAIL");
    for (int i = 0; i < N; ++i) {
        std::cout << h_y_gpu2[i] << " ";
    }
    std::cout << std::endl;
    // Free Memory
    cudaFree(d_x);
    cudaFree(d_y);

    free(h_x);
    free(h_y);
    free(h_y_cpu);
    free(h_y_gpu);
    free(h_y_gpu2);
}