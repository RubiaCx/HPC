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
#include <cuda_fp16.h>  

void cpuPrefixSum(half *x, float *y, int N) {
    y[0] = __half2float(x[0]);

    for (int i = 1; i < N; i++) {
        y[i] = y[i - 1] + __half2float(x[i]);
    }
}

__global__ void prefixSum_Smem_half2float_kernel(half *input, float *output, int N)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 将输入的half数据加载到共享内存的float数组中
    if (i < N) {
        sdata[tid] = __half2float(input[i]);
    } else {
        sdata[tid] = 0.0f;
    }

    __syncthreads();

    // 进行并行前缀和的计算
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x)
            sdata[index] += sdata[index - stride];
        __syncthreads();
    }

    // 逆向传播以填充前缀和
    for (int stride = blockDim.x / 4; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (tid + 1) * stride * 2 - 1;
        if (index + stride < blockDim.x)
            sdata[index + stride] += sdata[index];
    }
    __syncthreads();

    // 将结果写回全局内存
    if (i < N)
        output[i] = sdata[tid];

}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("usage: ./main [N]\n");
        exit(0);
    }
    size_t N = atoi(argv[1]);

    size_t bytes_x = sizeof(half) * N;
    size_t bytes_y = sizeof(float) * N;

    half *h_x = (half *)malloc(bytes_x);
    float *h_y = (float *)malloc(bytes_y);
    float *h_y_cpu = (float *)malloc(bytes_y);
    float *h_y_gpu = (float *)malloc(bytes_y);
    
    generate_random_value_half(h_x, N, 0.0, 2.0);

    memset(h_y, 0, N * sizeof(float));
    memset(h_y_gpu, 0, N * sizeof(float));
    memset(h_y_cpu, 0, N * sizeof(float));

    // // 输出前10个元素以验证
    // for (int i = 0; i < 10; ++i) {
    //     std::cout << __half2float(h_x[i]) << " ";
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
    // for (int i = 0; i < N; ++i)
    // {
    //     std::cout << h_y_cpu[i] << " ";
    // }
    // std::cout << std::endl;

    half *d_x;
    float *d_y;
    cudaMalloc(&d_x, bytes_x);
    cudaMalloc(&d_y, bytes_y);

    const int BLOCK_SIZE = 256;
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    size_t sharedMemorySize = BLOCK_SIZE * sizeof(float);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaMalloc(&d_x, bytes_x));
    CHECK_CUDA(cudaMalloc(&d_y, bytes_y));

    float gpu_time = 0.0;
    CHECK_CUDA(cudaMemcpy(d_x, h_x, bytes_x, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, bytes_y, cudaMemcpyHostToDevice));
    prefixSum_Smem_half2float_kernel<<<gridSize, blockSize, sharedMemorySize>>>(d_x, d_y, N);
    CHECK_CUDA(cudaEventRecord(start));
    for (int run = 0; run < nIter; run++)
    {
        prefixSum_Smem_half2float_kernel<<<gridSize, blockSize, sharedMemorySize>>>(d_x, d_y, N);
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
    // for (int i = 0; i < N; ++i) {
    //     std::cout << h_y_gpu[i] << " ";
    // }
    // std::cout << std::endl;

    // Free Memory
    cudaFree(d_x);
    cudaFree(d_y);

    free(h_x);
    free(h_y);
    free(h_y_cpu);
    free(h_y_gpu);
}