#include <iostream>
using namespace std;

__device__ int reduceCountMask(int sum, int * mask, int warp_size) 
{
    #pragma unroll
    for(int m = warp_size / 2; m >= 1; m >>=1)
    {
        sum += __shrl_xor_sync(0xffffffff, mask, m);
    }
    return sum;
}

// 获得mask中为true的数量，即reduce sum，结果为y的长度
__global__ void countMask(int * count, int * mask, int warp_size, int n) 
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x % warp_size;
    count = (idx < n) ? mask[idx] : 0;
    int sum = reduceCountMask(0, mask, warp_size);
    if(idx < n && lane == 0)
    {
        atomicAdd(count, sum);
    }
}


__global__ void mask_vector(int * x, int * mask, int n, int *y, int * pos)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n && mask[idx])
    {
        int curr = atomicAdd(pos, 1);
        y[curr] = x[idx];
    }
}

int main()
{
    int x[] = {1,3,2,4,5};
    int mask[] = {1,0,0,0,1};
    int size = sizeof(x) / sizeof(x[0]);
    int *d_x, *d_mask, *d_y, *d_size;

    cudaMalloc(&d_x, sizeof(int) * size);
    cudaMalloc(&d_mask, sizeof(int) * size);
    cudaMalloc(&d_y, sizeof(int) * size);
    cudaMalloc(&d_size, sizeof(int) * size);

    cudaMemcpy(d_x, x, size * sizeof(int), Host2Device);//忘了
    cudaMemcpy(d_mask, mask, size * sizeof(int), Host2Device);//忘了

    CountMask<<<1, size>>>(d_size, d_mask, 32, size); 
   

}
