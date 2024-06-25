#include <stdio.h>
#include <assert.h>
#include <cublas_v2.h>
#include "utils.hpp"

__global__ void sequential_scan_kernel(int *data, int *prefix_sum, int N)
{
    prefix_sum[0] = 0;
    for (int i = 1; i < N; i++)
    {
        prefix_sum[i] = prefix_sum[i - 1] + data[i - 1];
    }
}

__global__ void stone_scan(float *input, float *output, int len)
{
    __shared__ float XY[BLOCK_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len)
        XY[threadIdx.x] = input[i];
    __syncthreads();
    float temp = XY[0];
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        __syncthreads();
        if (threadIdx.x >= stride)
            temp = XY[threadIdx.x] + XY[threadIdx.x - stride];

        __syncthreads();
        XY[threadIdx.x] = temp;
    }
    if (i < len)
    {
        output[i] = XY[threadIdx.x];
    }
}

__global__ void stone_gen_Last_Item(float *output, float *last_list, int num_blocks)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // get those last items into the list;
    if ((threadIdx.x == blockDim.x - 1) && (blockIdx.x != (num_blocks - 1)))
    {
        last_list[blockIdx.x] = output[i];
    }
}

__global__ void stone_block_addition(float *output, float *last_list, int num_blocks, int len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = output[i];
    for (unsigned int j = 0; j < num_blocks - 1; j++)
    {
        if (blockIdx.x > j)
        {
            temp += last_list[j];
        }
        else
        {
            break;
        }
    }
    if (i < len)
    {
        output[i] = temp;
    }
}

int main(int argc, char **argv)
{
    const int numElements = 16384;
    int BLOCK_SIZE = 
    int numblocks = numElements / BLOCK_SIZE + 1;
    const int TILE_DIM = 32;
    const int BLOCK_ROWS = 8;

    dim3 dimGrid(num_blocks, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    int devId = 0;
    if (argc > 1)
        devId = atoi(argv[1]);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, devId);
    printf("\nDevice : %s\n", prop.name);
    printf("Matrix size: %d %d, Block size: %d %d, Tile size: %d %d\n",
           nx, ny, TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);
    printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n",
           dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);

    cudaSetDevice(devId);
    // some previous code ....
    int numblocks = numElements / BLOCK_SIZE + 1;
    
    stone_scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numElements);
    stone_gen_Last_Item<<<dimGrid, dimBlock>>>(deviceOutput, last_list, num_blocks);
    stone_block_addition<<<dimGrid, dimBlock>>>(deviceOutput, last_list, num_blocks, numElements);
    // some end code ...
}