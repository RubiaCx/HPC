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
__device__ __forceinline__ float warpReduceSum(float sum)
{
    if (BLOCK_SIZE >= 32)
        sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (BLOCK_SIZE >= 16)
        sum += __shfl_down_sync(0xffffffff, sum, 8); // 0-8, 1-9, 2-10, etc.
    if (BLOCK_SIZE >= 8)
        sum += __shfl_down_sync(0xffffffff, sum, 4); // 0-4, 1-5, 2-6, etc.
    if (BLOCK_SIZE >= 4)
        sum += __shfl_down_sync(0xffffffff, sum, 2); // 0-2, 1-3, 4-6, 5-7, etc.
    if (BLOCK_SIZE >= 2)
        sum += __shfl_down_sync(0xffffffff, sum, 1); // 0-1, 2-3, 4-5, etc.
    return sum;
}

template <int BLOCK_SIZE, int NUM_PER_THREAD, int WARP_SIZE>
__global__ void reduce(float *d_in, float *d_out)
{
    float sum = 0;
    __shared__ float sdata[BLOCK_SIZE];
    // each thread loads one element from global memory to shared mem
    unsigned int i = blockIdx.x * BLOCK_SIZE * NUM_PER_THREAD + threadIdx.x;
    // unsigned int i = blockIdx.x * blockDim.x * NUM_PER_THREAD + threadIdx.x;
    unsigned int tid = threadIdx.x;

#pragma unroll
    for (int iter = 0; iter < NUM_PER_THREAD; iter++)
    {
        sdata[tid] += d_in[i + iter * BLOCK_SIZE];
    }
    __syncthreads();

    // Shared mem for partial sums (one per warp in the block)
    static __shared__ float warpLevelSums[WARP_SIZE];
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;

    sum = warpReduceSum<BLOCK_SIZE>(sum);

    if (laneId == 0)
        warpLevelSums[warpId] = sum;
    __syncthreads();
    // read from shared memory only if that warp existed
    sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelSums[laneId] : 0;
    // Final reduce using first warp
    if (warpId == 0)
        sum = warpReduceSum<BLOCK_SIZE / WARP_SIZE>(sum);
    // write result for this block to global mem
    if (tid == 0)
        d_out[blockIdx.x] = sum;
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
    const int WARP_SIZE = 32;
    const int BLOCK_SIZE = 256;
    const int BLOCK_NUM = 1024;
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
    reduce<BLOCK_SIZE, NUM_PER_THREAD, WARP_SIZE><<<grid, block>>>(d_i_data, d_o_data);

    cudaEventRecord(startEvent, 0);
    for (int i = 0; i < TEST_TIMES; i++)
        reduce<BLOCK_SIZE, NUM_PER_THREAD, WARP_SIZE><<<grid, block>>>(d_i_data, d_o_data);
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