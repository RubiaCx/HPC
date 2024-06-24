#include "spmm_opt.h"

static constexpr int WARP_SIZE = 32;

// block.x = 1;block.y = WARP_SIZE; 所以使用threadIdx.y
    
/* 计算稀疏矩阵A和密集矩阵B的乘积，结果存储在密集矩阵C中，A以CSR（Compressed Sparse Row）格式存储，大小NxN，B和C的大小是NxK
 * 使用共享内存来存储每个warp内部的索引和值
 * 每个线程处理矩阵A的一行与矩阵B的一列的乘积
 * 对于每个warp，线程们读取其负责的A矩阵元素的索引和值，存入共享内存
*/
__global__ void spmm_kernel_optimized(int *csrRowPtr, int *csrColIdx, float *csrValues,
                                      float *denseMatrix, float *resultMatrix, int N, int K) {
    // 在共享内存中声明索引和值的数组，大小为一个warp的大小
    __shared__ int sharedColIdx[WARP_SIZE];
    __shared__ float sharedValues[WARP_SIZE];

    // 计算当前线程负责的A矩阵的行号和B矩阵的列号
    int idx = threadIdx.y;  
    int rowIdx = blockIdx.x; // A的行号 = 第 blockIdx.x 行
    int colIdx = blockIdx.y * WARP_SIZE + idx; // B的列号，由blockIdx.y和threadIdx.y计算得到

    // 如果当前行号超出矩阵A的行数，则直接返回
    if (rowIdx >= N) { return; }

    // 从A矩阵的CSR格式指针中获取当前行的开始和结束位置
    int rowStart = csrRowPtr[rowIdx];
    int rowEnd   = csrRowPtr[rowIdx + 1];

    float sum = 0; // 初始化该线程的求和结果为0

    // 遍历当前行的所有元素
    for (int elementIdx = rowStart; elementIdx < rowEnd; elementIdx += WARP_SIZE) {
        int currentElement = elementIdx + idx; // 计算该线程负责的元素索引，elementIdx 是当前warp开始处理的A矩阵非零元素的索引
        if (currentElement < rowEnd) { // 如果索引有效，则从A中读取索引和值，存入共享内存
            sharedColIdx[idx] = csrColIdx[currentElement] * K; // 计算B矩阵中的对应行索引
            sharedValues[idx] = csrValues[currentElement];
        }
        __syncwarp(); // 同步warp内的所有线程，确保所有数据都被读取和存储

        for (int i = 0; i < min(WARP_SIZE, rowEnd - elementIdx); i++) {
            // 计算对应B矩阵的索引，并进行乘法累加
            sum += sharedValues[i] * denseMatrix[sharedColIdx[i] + colIdx];
        }
        __syncwarp(); // 再次同步，准备下一轮计算
    }
    // C[rowIdx][colIdx] = A[rowIdx][sharedColIdx[i]] * B[sharedColIdx[i]][colIdx]
    resultMatrix[rowIdx * K + colIdx] = sum;
}



__global__ void spmm_kernel_opt2(int *a_ptr, int *a_idx, float *a_val,
                                 float *b_val, float *c_val, int a_size, int b_cols)
{
    __shared__ int sm_idxs[WARP_SIZE];
    __shared__ float sm_vals[WARP_SIZE];

    int row_id = blockIdx.x;
    int col_id = blockIdx.y * WARP_SIZE * 2 + threadIdx.y; /// here

    if (row_id >= a_size)
    {
        return;
    }

    int begin_ptr = a_ptr[row_id];
    int end_ptr = a_ptr[row_id + 1];

    float sum0 = 0;
    float sum1 = 0;

    for (int base_ptr = begin_ptr; base_ptr < end_ptr; base_ptr += WARP_SIZE)
    {
        int now_ptr = base_ptr + threadIdx.y;
        if (now_ptr < end_ptr)
        {
            // pre-compute row start
            sm_idxs[threadIdx.y] = a_idx[now_ptr] * b_cols;
            sm_vals[threadIdx.y] = a_val[now_ptr];
        }
        __syncwarp();
        int sm_end = min(WARP_SIZE, end_ptr - base_ptr);
        for (int i = 0; i < sm_end; i++)
        {
            int b_idx = sm_idxs[i] + col_id;
            float val = sm_vals[i];
            sum0 += val * b_val[b_idx];
            sum1 += val * b_val[b_idx + WARP_SIZE]; /// here
        }
        __syncwarp();
    }
    int c_idx = row_id * b_cols + col_id;
    c_val[c_idx] = sum0;
    c_val[c_idx + WARP_SIZE] = sum1; /// here
}

static inline int ceiling(int a, int b)
{
    return (a + b - 1) / b;
}

void SpMMOpt::preprocess(float *vin, float *vout)
{
    if (feat_in % WARP_SIZE != 0)
    {
        fprintf(stderr, "Error: K must be a multiple of %d, got %d\n",WARP_SIZE, feat_in);
    }

    block.x = 1;
    block.y = WARP_SIZE;
    grid.x = num_v;

    if (feat_in <= WARP_SIZE)
    {
        grid.y = ceiling(feat_in, WARP_SIZE);
    }
    else
    {
        grid.y = ceiling(feat_in, WARP_SIZE * 2);
    }
}

void SpMMOpt::run(float *vin, float *vout)
{
    if (feat_in <= WARP_SIZE)
    {
        spmm_kernel_opt<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
    }
    else
    {
        spmm_kernel_opt2<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
    }
}