#include <stdio.h>
#include <assert.h>
#include <cublas_v2.h>
#include <iostream>
#include <random>

void generate_random_value_float(float * result, int size, float lower_bound, float upper_bound) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(lower_bound, upper_bound);
    
    for(int i = 0; i < size; ++i) {
        result[i] = dist(gen);
    }
}

template <int BLOCK_SIZE>
__global__ void reduce(float *d_in, float *d_out)
{
    __shared__ float sdata[BLOCK_SIZE];
    // each thread loads one element from global memory to shared mem
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    sdata[tid] = d_in[i];
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        // if (tid % (2 * s) == 0) // 对每个线程进行分支判断，以确定是否执行加法操作
        // {
        //     sdata[tid] += sdata[tid + s];
        // }
        int index = 2 * s * tid;    // 交错地址访问（Interleaved Addressing）
        if (index < blockDim.x)     // 在修改后的代码中，只需要进行一次分支判断来计算操作的索引
        {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0)
        d_out[blockIdx.x] = sdata[tid];
}

int main(int argc, char **argv)
{
    const int TEST_TIMES = 10;
    const int N = 32 * 1024 * 1024;
    const int BLOCK_SIZE = 256;

    int32_t BLOCK_NUM = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(BLOCK_NUM, 1);
    dim3 block(BLOCK_SIZE, 1);

    float *h_i_data = (float *)malloc(N * sizeof(float));
    generate_random_value_float(h_i_data, N, 1.0, 2.0);
    float *h_o_data = (float *)malloc(N / BLOCK_SIZE * sizeof(float));
    float *d_i_data, *d_o_data;
    cudaMalloc(&d_i_data, N * sizeof(float));
    cudaMalloc(&d_o_data, N / BLOCK_SIZE * sizeof(float));

    cudaMemcpy(d_i_data, h_i_data, N * sizeof(float), cudaMemcpyHostToDevice);

    // events for timing
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float ms;

    // ------------
    // generate answer
    // ------------
    float answer = 0.0f;
    for (int i = 0; i < N; ++i)
    {
        answer += h_i_data[i];
    }

    // ------------
    // time kernels
    // ------------

    //  warm up
    reduce<BLOCK_SIZE><<<grid, block>>>(d_i_data, d_o_data);

    cudaEventRecord(startEvent, 0);
    for (int i = 0; i < TEST_TIMES; i++)
        reduce<BLOCK_SIZE><<<grid, block>>>(d_i_data, d_o_data);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    cudaMemcpy(h_o_data, d_o_data, BLOCK_NUM * sizeof(float), cudaMemcpyDeviceToHost);

    float my_answer = 0;
    for(int i = 0; i < BLOCK_NUM; i++)
    {
        my_answer += h_o_data[i * BLOCK_SIZE];
    }
    if ((my_answer - answer) < 1e-6) // 允许小于1e-6的误差 if (result[i] != answer[i])
    {
        printf("Time = %.6lf ms\n", ms / TEST_TIMES); // Time = 0.790528 ms
    }

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(d_i_data);
    cudaFree(d_o_data);
    free(h_i_data);
    free(h_o_data);

    return 0;
}