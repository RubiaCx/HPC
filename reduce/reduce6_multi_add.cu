#include <stdio.h>
#include <assert.h>
#include <cublas_v2.h>
#include <iostream>
#include <random>

void generate_random_value_float(float *result, int size, float lower_bound, float upper_bound)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(lower_bound, upper_bound);

    for (int i = 0; i < size; ++i)
    {
        result[i] = dist(gen);
    }
}

template <unsigned int BLOCK_SIZE>
__device__ void warpReduce(volatile float *cache, int tid)
{
    if (BLOCK_SIZE >= 64) // 其实走不到这里
        cache[tid] += cache[tid + 32];
    if (BLOCK_SIZE >= 32)
        cache[tid] += cache[tid + 16];
    if (BLOCK_SIZE >= 16)
        cache[tid] += cache[tid + 8];
    if (BLOCK_SIZE >= 8)
        cache[tid] += cache[tid + 4];
    if (BLOCK_SIZE >= 4)
        cache[tid] += cache[tid + 2];
    if (BLOCK_SIZE >= 2)
        cache[tid] += cache[tid + 1];
}

template <int BLOCK_SIZE, int NUM_PER_THREAD>
__global__ void reduce(float *d_in, float *d_out)
{
    __shared__ float sdata[BLOCK_SIZE];
    // each thread loads one element from global memory to shared mem
    unsigned int i = blockIdx.x * blockDim.x * NUM_PER_THREAD + threadIdx.x;
    unsigned int tid = threadIdx.x;
    // HERE
    sdata[tid] = 0;
#pragma unroll
    for (int iter = 0; iter < NUM_PER_THREAD; iter++)
    {
        sdata[tid] += d_in[i + iter * BLOCK_SIZE];
    }
    __syncthreads();

    // do reduction in shared mem
    if (BLOCK_SIZE >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128)
    {
        if (tid < 64)
        {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }
    // Final warp reduction
    if (tid < 32) warpReduce<BLOCK_SIZE>(sdata, tid);

    // Write result for this block to global memory
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

bool check(float *out, float *res, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (out[i] != res[i])
            return false;
    }
    return true;
}

int main(int argc, char **argv)
{
    const int TEST_TIMES = 10;
    const int N = 32 * 1024 * 1024;
    const int BLOCK_SIZE = 256;
    const int BLOCK_NUM = 2048 ; // (N + BLOCK_SIZE - 1) / BLOCK_SIZE / 2;
    const int NUM_PER_BLOCK = N / BLOCK_NUM;
    const int NUM_PER_THREAD = NUM_PER_BLOCK / BLOCK_SIZE;
    dim3 grid(BLOCK_NUM, 1);
    dim3 block(BLOCK_SIZE, 1);

    float *h_i_data = (float *)malloc(N * sizeof(float));
    generate_random_value_float(h_i_data, N, 1.0, 2.0);
    float *h_o_data = (float *)malloc(BLOCK_NUM * sizeof(float));
    float *answer = (float *)malloc(BLOCK_NUM * sizeof(float));
    float *d_i_data, *d_o_data;
    cudaMalloc(&d_i_data, N * sizeof(float));
    cudaMalloc(&d_o_data, BLOCK_NUM * sizeof(float));

    cudaMemcpy(d_i_data, h_i_data, N * sizeof(float), cudaMemcpyHostToDevice);

    // events for timing
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float ms;

   // ------------
    // generate answer
    // ------------
    for (int i = 0; i < BLOCK_NUM; i++)
    {
        float cur = 0;
        for (int j = 0; j < NUM_PER_BLOCK; j++)
        {
            if (i * NUM_PER_BLOCK + j < N)
            {
                cur += h_i_data[i * NUM_PER_BLOCK + j];
            }
        }
        answer[i] = cur;
    }

    // ------------
    // time kernels
    // ------------

    //  warm up
    reduce<BLOCK_SIZE, NUM_PER_THREAD><<<grid, block>>>(d_i_data, d_o_data);

    cudaEventRecord(startEvent, 0);
    for (int i = 0; i < TEST_TIMES; i++)
        reduce<BLOCK_SIZE, NUM_PER_THREAD><<<grid, block>>>(d_i_data, d_o_data);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);

    cudaMemcpy(h_o_data, d_o_data, BLOCK_NUM * sizeof(float), cudaMemcpyDeviceToHost);

    if (check(h_o_data, answer, BLOCK_NUM))
    {
        printf("Time = %.6lf ms\n", ms / TEST_TIMES); // Time = 0.790528 ms
    }
    else
    {
        printf("Time = %.6lf ms\n", ms / TEST_TIMES); // Time = 0.790528 ms
    }

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(d_i_data);
    cudaFree(d_o_data);
    free(h_i_data);
    free(h_o_data);
    free(answer);   
    return 0;
}