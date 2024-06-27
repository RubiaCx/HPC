# GEMM

- **HPC优化的核心思想：怎么样让数据放在更近的存储上来掩盖计算的延时，从而减少存储墙的影响**
- **单算子的优化思路**
    - **带宽**：尽可能提高单条指令的效率
    - **延迟**：通过流水尽可能掩盖指令间的延迟

nvcc main.cu -o a1 -L /usr/local/cuda/lib64 -lcudart -lcuda -lcublas

https://rubiablue.notion.site/SGEMM-ce45cdf477dd4d87a866927c08997452?pvs=4

## Global memory
- 在GPU中，一共开启$m\times n$个thread，每个thread需要读取矩阵A的一行与矩阵B的一列，而后将计算结果写回至矩阵C中
    - 即每个thread负责C矩阵中的一个元素的计算
- 一共需要从**global memory**中进行$2\times m\times n\times k$次读操作和$m\times n$次写操作

- 问题：主题循环由2条load和1条FMA组成，计算访存指令比1/3，导致了访存延迟不能被隐藏，从而性能不理想

- 计算访存比：每次迭代需要进行一次FMA（乘累加）和两次全局内存读取，计算访存比1/2
    - 较低的计算访存比无法有效隐藏访存延迟

- 访存量：访问全局内存，C矩阵每个元素计算需要访问$2K$个单精度浮点数，完成全部计算需要 $2*K*M*N$
    - 全局内存访问延迟高（几百cycle）
    - 相同位置元素被重复读取（C中同一行元素计算共享A中同一行元素，C中同一列元素计算共享B中同一列元素）

```python
__global__ void Sgemm_kernel_global(
    float * A, float * B, float * C,
    const int M, const int N, const int K,
    float alpha, float beta)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if(ty < M && tx < N) {
        float c = 0;
        for(int i = 0; i < K; ++i){
            c += A[ty * K + i] * B[i * N + tx]; // PROBLEM
        }
        C[ty * N + tx] = beta * C[ty * N + tx] + alpha * c;
    }

}

```

## Tiling: use the Smem


shared memory

# 参考
- https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE