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
#define FETCH_FLOAT4(pointer) (*(reinterpret_cast<float4*>(&(pointer))))

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
// 每个线程计算对应C矩阵中多个元素
    // 通过共享内存缓存减少了全局内存访存量和FMA乘累加的访存延迟，但计算访存比没有得到改善，每次迭代计算都需要两个访存指令和一个计算指令
    // 引入thread tile，即一个线程负责block中多个元素的计算
// 可以通过增加block大小（BM，BN）值，进一步降低全局内存的访存量，但是不能无限增大
    // 一方面，block分块矩阵尺寸过大，block数量减少，这样会造成大量 SM（Streaming Multiprocessor）的闲置浪费
    // 另一方面，BN和BM的增加，需要申请更多的共享内存，单线程内共享内存占用越多，活跃线程束越少，不利于隐藏指令延迟
    // 在增加BM和BN值的同时，为了减少共享内存占用，减小BK值

// 访存量：每个block需要从global memory中读取(K/BK)*(BM*BK+BK*BN)个单精度浮点数，整个C存在(M/BM)*(N/BN)个block，因此完成C中所有元素计算需要读取(M/BM)*(N/BN)*(K/BK)*(BM*BK+BK*BN)个单精度浮点数
    // 优化前全局访存量为2*K*M*N
    // 共享内存缓存优化后，访存量减少为原来的1/2*(1/BN)*(1/BM)，当BN=BM=32时，访存减少至1/32
template <
    const int BLOCK_SIZE_M,  // 每个线程块计算C的块高度
    const int BLOCK_SIZE_K,  // 每个线程块加载到共享内存的A的块宽度
    const int BLOCK_SIZE_N   // 每个线程块计算C的块宽度
>
__global__ void SgemmKernelTiling(float* a, float* b, float* c, int M, int N, int K) {
    // 定义共享内存以存储A和B的子块
    __shared__ float shared_a[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ float shared_b[BLOCK_SIZE_K][BLOCK_SIZE_N];

    int bx = blockIdx.x;  // 块索引X
    int by = blockIdx.y;  // 块索引Y
    int tx = threadIdx.x; // 线程索引X
    int ty = threadIdx.y; // 线程索引Y

    // 计算此线程负责的C矩阵的行和列
    int row = by * BLOCK_SIZE_M + ty;
    int col = bx * BLOCK_SIZE_N + tx;

    float sum = 0.0f; // 累加器，用于计算点乘结果

    // 遍历K维度上的所有子块
    for (int k = 0; k < (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K; ++k) {
        // 加载A和B的子块到共享内存
        if (k * BLOCK_SIZE_K + tx < K && row < M)
            shared_a[ty][tx] = a[row * K + k * BLOCK_SIZE_K + tx];
        else
            shared_a[ty][tx] = 0.0;

        if (k * BLOCK_SIZE_K + ty < K && col < N)
            shared_b[ty][tx] = b[(k * BLOCK_SIZE_K + ty) * N + col];
        else
            shared_b[ty][tx] = 0.0;

        __syncthreads();  // 确保所有数据加载到共享内存

        // 计算子块的乘积并累加到sum
        for (int i = 0; i < BLOCK_SIZE_K; ++i) {
            sum += shared_a[ty][i] * shared_b[i][tx];
        }

        __syncthreads();  // 确保所有线程完成计算前不开始下一轮加载
    }

    // 将计算结果写回全局内存
    if (row < M && col < N) {
        c[row * N + col] = sum;
    }
}

template <
    const int BLOCK_SIZE_M = 128,  // 输出块的行数
    const int BLOCK_SIZE_N = 128,  // 输出块的列数
    const int TILE_K = 8,          // K维度的分块大小
    const int THREAD_TILE_M = 8,   // 每个线程处理的行数
    const int THREAD_TILE_N = 8    // 每个线程处理的列数
>
__global__ void sgemm_thread_tile_vec4(float* a, float* b, float* c, int M, int N, int K) {
    __shared__ float s_a[BLOCK_SIZE_M][TILE_K];
    __shared__ float s_b[TILE_K][BLOCK_SIZE_N];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int start_row_c = by * BLOCK_SIZE_M + ty * THREAD_TILE_M;
    int start_col_c = bx * BLOCK_SIZE_N + tx * THREAD_TILE_N;

    float r_c[THREAD_TILE_M][THREAD_TILE_N] = {0.0};

    for (int bk = 0; bk < (K + TILE_K - 1) / TILE_K; ++bk) {
        for (int i = 0; i < THREAD_TILE_M; i++) {
            for (int j = 0; j < THREAD_TILE_N; j += 4) {
                int idx_a = (start_row_c + i) * K + bk * TILE_K + j;
                int idx_b = (bk * TILE_K + i) * N + start_col_c + j;
                FETCH_FLOAT4(s_a[ty * THREAD_TILE_M + i][tx * THREAD_TILE_N + j]) = FETCH_FLOAT4(a[idx_a]);
                FETCH_FLOAT4(s_b[ty * THREAD_TILE_M + i][tx * THREAD_TILE_N + j]) = FETCH_FLOAT4(b[idx_b]);
            }
        }
        __syncthreads();

        for (int k = 0; k < TILE_K; k++) {
            for (int i = 0; i < THREAD_TILE_M; i++) {
                for (int j = 0; j < THREAD_TILE_N; j++) {
                    r_c[i][j] += s_a[ty * THREAD_TILE_M + i][k] * s_b[k][tx * THREAD_TILE_N + j];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < THREAD_TILE_M; i++) {
        for (int j = 0; j < THREAD_TILE_N; j += 4) {
            FETCH_FLOAT4(c[(start_row_c + i) * N + start_col_c + j]) = FETCH_FLOAT4(r_c[i][j]);
        }
    }
}


