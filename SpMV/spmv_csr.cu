#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMV
#include <bits/stdc++.h>
#include <sys/time.h>
#include "utils/load.hpp"
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
#define CHECK_CUSPARSE(func)                                               \
    {                                                                      \
        cusparseStatus_t status = (func);                                  \
        if (status != CUSPARSE_STATUS_SUCCESS)                             \
        {                                                                  \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cusparseGetErrorString(status), status);      \
            return EXIT_FAILURE;                                           \
        }                                                                  \
    }

using index_t = int;
using value_t = float;

template <typename index_t, typename value_t>
void csr_SpMV_CPU(index_t n_rows, index_t n_cols, index_t num_nnz,
                  const index_t *row_oft, const index_t *col_ids, 
                  const value_t *Ax, const value_t *x, value_t *y) 
{
    for (index_t row = 0; row < n_rows; ++row) 
    {
        value_t sum = value_t(0);
        for (index_t col_off = row_oft[row]; col_off < row_oft[row+1]; ++col_off) 
        {
            sum += Ax[col_off] * x[col_ids[col_off]];
        }
        y[row] = sum;
    }
}

/* 1 SpMV 朴素实现
 * 让CUDA 的每个线程负责 CSR 矩阵中某一行的计算，每个计算结果是一个标量
 */
template <typename index_t, typename value_t>
__global__ void csr_SpMV_scalar_kernel(index_t n_rows, index_t n_cols, index_t num_nnz,
                                       const index_t *row_oft, const index_t * col_ids,
                                       const value_t *data, const value_t *x,
                                       value_t *y) 
{
    int row = blockIdx.x * blockDim.x + threadIdx.x; // 计算当前线程对应的行号
    if (row < n_rows)
    {
        // 计算当前行的非零元素范围
        value_t sum = 0;
        // 遍历当前行的所有非零元素
        for (int element = row_oft[row]; element < row_oft[row + 1]; element++)
        {
            sum += data[element] * x[col_ids[element]];
        }
        y[row] = sum;
    }
}

template <class value_t>
__device__ __forceinline__ value_t warpReduceSum(value_t sum, int WarpSize)
{
    if (WarpSize >= 32)
        sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (WarpSize >= 16)
        sum += __shfl_down_sync(0xffffffff, sum, 8); // 0-8, 1-9, 2-10, etc.
    if (WarpSize >= 8)
        sum += __shfl_down_sync(0xffffffff, sum, 4); // 0-4, 1-5, 2-6, etc.
    if (WarpSize >= 4)
        sum += __shfl_down_sync(0xffffffff, sum, 2); // 0-2, 1-3, 4-6, 5-7, etc.
    if (WarpSize >= 2)
        sum += __shfl_down_sync(0xffffffff, sum, 1); // 0-1, 2-3, 4-5, etc.
    return sum;
}

/* 1 SpMV 向量化实现
 * 每个 warp 负责 CSR 矩阵每一行的计算,
 * 由于 warp 中有 32 个线程, 因此计算结束后每行是一个 32 维的向量, 需要通过 warpReduceSum 得到最终的结果, 并由 lane 0 线程写回结果向量
 */
template <typename index_t, typename value_t>
__global__ void csr_SpMV_vector_kernel(index_t n_rows, index_t n_cols, index_t num_nnz,
                                       const index_t *row_oft, const index_t * col_ids,
                                       const value_t *data, const value_t *x, value_t *y)
{
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = thread_id / 32;  // Assuming 32 threads per warp
    // const int lane = thread_id % 32;
    const int lane = thread_id & (32 - 1);
    const index_t row = warp_id;  // One warp per row
    value_t sum = 0;
    if (row < n_rows)
    {
        const index_t row_start = row_oft[row];
        const index_t row_end   = row_oft[row + 1];
        for (index_t element = row_start + lane; element < row_end; element += 32)  // Stride of 32 for coalesced access
        {
            sum += data[element] * x[col_ids[element]];
        }
        sum = warpReduceSum<value_t>(sum, 32);  
        if (lane == 0 && row < n_rows)
        {
            y[row] = sum;
        }
    }

}

// THREADS_PER_VECTOR 是 32 时效率最高
template <typename index_t, typename value_t>
__global__ void csr_SpMV_my_kernel(index_t n_rows, index_t n_cols, index_t num_nnz,
                                   const index_t *row_oft, const index_t * col_ids,
                                   const value_t *data, const value_t *x, value_t *y,
                                   int VECTORS_PER_BLOCK, int THREADS_PER_VECTOR)
{
    const index_t THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;
    const index_t thread_id = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    
    const index_t thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);  // thread index within the vector
    const index_t row_id = thread_id / THREADS_PER_VECTOR;  // global vector index

    if(row_id < n_rows)
    {
        const index_t row_start = row_oft[row_id];     
        const index_t row_end   = row_oft[row_id+1];

        value_t sum = 0;
        // accumulate local sums
        for (index_t element = row_start + thread_lane; element < row_end; element += THREADS_PER_VECTOR) // vs 32
        {
            sum += data[element] * x[col_ids[element]];
        }
        sum = warpReduceSum<value_t>(sum, THREADS_PER_VECTOR);  
        if (thread_lane == 0)
        {
            y[row_id] = sum;
        }   
    }
}
/* 分析：Scalar 与 Vector 在线程的负载均衡上不同
 * 由于稀疏矩阵的特性, 矩阵的每一行非零元个数会各有不同, 尤其是对一些幂律性比较突出的稀疏矩阵, 行中的非零元可能差别很大
 * 因此, 在 Scalar 中不同线程之间的工作量可能会相差很大, 从而导致 warp divergence 等问题, 大幅影响性能
 *       在 Vector 将一个 warp 负责矩阵的一行, 一定程度上减轻了 warp 内线程的负载不均与 warp divergence, 但有时矩阵行中的非零元较少, 32 个线程就会存在线程空闲的问题, 同时 warp 间的负载不均依旧存在
 */

int main(int argc, char** argv)
{
    //// Timer
    cudaEvent_t start, stop;
    struct timeval begin, end;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //// Read host problem definition from file 
    if (argc < 2) 
    {
        std::cerr << "usage: ./bin/<program-name>  <filename.mtx>..." << std::endl;
        exit(1);        
    }
    csr_t<index_t, index_t, value_t> csr = ToCsr(LoadCoo<index_t, index_t, value_t>(argv[1]));

    index_t num_rows, num_cols;
    index_t num_nnz;
    num_rows = csr.number_of_rows;
    num_cols = csr.number_of_columns;
    num_nnz = csr.number_of_nonzeros;
    std::vector<value_t> hX(num_cols, 1);
    std::vector<value_t> hY(num_rows, value_t(0));
    std::vector<value_t> hY_scalar(num_rows, value_t(0));
    std::vector<value_t> hY_vector(num_rows, value_t(0));
    std::vector<value_t> hY_my(num_rows, value_t(0));
    std::cout << "num_rows: " << num_rows << "  num_cols: " << num_cols << "  num_nnz: " << num_nnz << std::endl; 
    int iter = 2000;

    //// Device memory management
    index_t *dA_row_offset, *dA_col_indices;
    value_t *dA_value, *dX, *dY;
    CHECK_CUDA(cudaMalloc((void **)&dA_row_offset, (num_rows + 1) * sizeof(index_t)))
    CHECK_CUDA(cudaMalloc((void **)&dA_col_indices, num_nnz * sizeof(index_t)))
    CHECK_CUDA(cudaMalloc((void **)&dA_value, num_nnz * sizeof(value_t)))
    CHECK_CUDA(cudaMalloc((void **)&dX, num_cols * sizeof(value_t)))
    CHECK_CUDA(cudaMalloc((void **)&dY, num_rows * sizeof(value_t)))

    CHECK_CUDA(cudaMemcpy(dA_row_offset, csr.row_offsets.data(), (num_rows + 1) * sizeof(index_t), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dA_col_indices, csr.column_indices.data(), num_nnz * sizeof(index_t), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dA_value, csr.nonzero_values.data(), num_nnz * sizeof(value_t), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dX, hX.data(), num_cols * sizeof(value_t), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dY, hY.data(), num_rows * sizeof(value_t), cudaMemcpyHostToDevice))

    //// warp(32 thread) for a row
    index_t mean_col_num = (num_nnz + (num_rows - 1)) / num_rows;
    printf("The average col num is: %d\n", mean_col_num);
    int THREADS_PER_VECTOR;
    unsigned int VECTORS_PER_BLOCK;
    unsigned int NUM_BLOCKS;
    if (mean_col_num <= 2) 
    {
        THREADS_PER_VECTOR = 2;
        VECTORS_PER_BLOCK = 128;
    } 
    else if (mean_col_num > 2 && mean_col_num <= 4) 
    {
        THREADS_PER_VECTOR = 4;
        VECTORS_PER_BLOCK = 64;
    } 
    else if (mean_col_num > 4 && mean_col_num <= 8) 
    {
        THREADS_PER_VECTOR = 8;
        VECTORS_PER_BLOCK = 32;
    } 
    else if (mean_col_num > 8 && mean_col_num <= 16) 
    {
        THREADS_PER_VECTOR = 16;
        VECTORS_PER_BLOCK = 16;
    } else if (mean_col_num > 16) 
    {
        THREADS_PER_VECTOR = 32;
        VECTORS_PER_BLOCK = 8;
    }
    NUM_BLOCKS = static_cast<unsigned int>((num_rows + (VECTORS_PER_BLOCK - 1)) / VECTORS_PER_BLOCK);
    printf("THREADS_PER_VECTOR: %d, VECTORS_PER_BLOCK: %d, NUM_BLOCKS: %d\n", THREADS_PER_VECTOR, VECTORS_PER_BLOCK, NUM_BLOCKS);
    //// CPU
    std::vector<value_t> hY_CPU(num_rows, 0);
    gettimeofday(&begin, NULL);
    for(int i = 0; i < iter; i++) 
        csr_SpMV_CPU<index_t, value_t>(num_rows, num_cols, num_nnz,
                                    csr.row_offsets.data(),csr.column_indices.data(), csr.nonzero_values.data(),
                                    hX.data(), hY_CPU.data());
    gettimeofday(&end, NULL);
    double time_use = ((end.tv_sec - begin.tv_sec) * 1000000 + (end.tv_usec - begin.tv_usec)) / 1000.0;
    printf("Time for CPU computation: %f ms\n", time_use/1000);

    //// CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    value_t alpha = 1.0f;
    value_t beta = 0.0f;
    void *dBuffer = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle))
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, num_rows, num_cols, num_nnz,
                                     dA_row_offset, dA_col_indices, dA_value,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, num_cols, dX, CUDA_R_32F)) // Create dense vector X
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, num_rows, dY, CUDA_R_32F)) // Create dense vector y
    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, matA, vecX, &beta, vecY,
                                           CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize))
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))
    // execute SpMV API
    cudaEventRecord(start, 0);
    for(int i = 0; i < iter; i++) 
        CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, vecX, &beta, vecY,
                                CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer))
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float timeCompute = 0.0;
    cudaEventElapsedTime(&timeCompute, start, stop);
    printf("Time for API computation: %f ms\n", timeCompute/iter);
    // device result check
    CHECK_CUDA(cudaMemcpy(hY.data(), dY, num_rows * sizeof(value_t), cudaMemcpyDeviceToHost))
    double delta = 0.0;
    for (int i = 0; i < num_rows; i++)
    {
        delta += abs(hY[i] - hY_CPU[i]);
    }
    printf("CUSPARSE vs CPU: sum: %12lf  avg: %12lf\n", delta, delta / num_rows);
    
    //// execute SpMV-scalar
    // Define the execution configuration
    dim3 blocks(NUM_BLOCKS);
    dim3 threads(THREADS_PER_VECTOR * VECTORS_PER_BLOCK);
    // Record start time
    cudaEventRecord(start, 0);
    // Launch the kernel
    for(int i = 0; i < iter; i++) 
        csr_SpMV_scalar_kernel<index_t, value_t><<<blocks, threads>>>
                              (num_rows, num_cols, num_nnz, dA_row_offset, dA_col_indices, dA_value, dX, dY);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // Calculate elapsed time
    float timeScalarCompute = 0.0;
    cudaEventElapsedTime(&timeScalarCompute, start, stop);
    printf("Time for scalar computation: %f ms\n", timeScalarCompute/iter);
    CHECK_CUDA(cudaMemcpy(hY_scalar.data(), dY, num_rows * sizeof(value_t), cudaMemcpyDeviceToHost));
    delta = 0.0;
    for (int i = 0; i < num_rows; i++)
    {
        delta += abs(hY_scalar[i] - hY_CPU[i]);
    }
    (delta == 0) ? printf("GPU scalar is SAME with CPU\n") : printf("GPU scalar vs CPU: sum: %12lf  avg: %12lf\n", delta, delta / num_rows);
    delta = 0.0;
    for (int i = 0; i < num_rows; i++)
    {
        delta += abs(hY[i] - hY_scalar[i]);
    }
    (delta == 0) ? printf("GPU scalar is SAME with CUSPARSE\n") : printf("GPU scalar vs CUSPARSE: sum: %12lf  avg: %12lf\n", delta, delta / num_rows);

    //// execute SpMV-vector
    // Define the execution configuration
    cudaEventRecord(start, 0);
    // Launch the kernel
    for(int i=0; i < iter; i++) 
        csr_SpMV_vector_kernel<index_t, value_t><<<blocks, threads>>>
                              (num_rows, num_cols, num_nnz, dA_row_offset, dA_col_indices, dA_value, dX, dY);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // Calculate elapsed time
    float timeVectorCompute = 0.0;
    cudaEventElapsedTime(&timeVectorCompute, start, stop);
    printf("Time for vector computation: %f ms\n", timeVectorCompute/iter);
    CHECK_CUDA(cudaMemcpy(hY_vector.data(), dY, num_rows * sizeof(value_t), cudaMemcpyDeviceToHost));
    delta = 0.0;
    for (int i = 0; i < num_rows; i++)
    {
        delta += abs(hY_vector[i] - hY_CPU[i]);
    }
    (delta == 0) ? printf("GPU vector is SAME with CPU\n") : printf("GPU vector vs CPU: sum: %12lf  avg: %12lf\n", delta, delta / num_rows);
    delta = 0.0;
    for (int i = 0; i < num_rows; i++)
    {
        delta += abs(hY[i] - hY_vector[i]);
    }
    (delta == 0) ? printf("GPU vector is SAME with CUSPARSE\n") : printf("GPU vector vs CUSPARSE: sum: %12lf  avg: %12lf\n", delta, delta / num_rows);

    //// execute SpMV-vector
    // Define the execution configuration
    cudaEventRecord(start, 0);
    // Launch the kernel
    for(int i=0; i < iter; i++) 
        csr_SpMV_my_kernel<index_t, value_t><<<blocks, threads>>>
                         (num_rows, num_cols, num_nnz, dA_row_offset, dA_col_indices, dA_value, dX, dY, VECTORS_PER_BLOCK, THREADS_PER_VECTOR);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // Calculate elapsed time
    float timeMyCompute = 0.0;
    cudaEventElapsedTime(&timeMyCompute, start, stop);
    printf("Time for My computation: %f ms\n", timeMyCompute/iter);
    CHECK_CUDA(cudaMemcpy(hY_my.data(), dY, num_rows * sizeof(value_t), cudaMemcpyDeviceToHost));
    delta = 0.0;
    for (int i = 0; i < num_rows; i++)
    {
        delta += abs(hY_my[i] - hY_CPU[i]);
    }
    (delta == 0) ? printf("GPU My is SAME with CPU\n") : printf("GPU My vs CPU: sum: %12lf  avg: %12lf\n", delta, delta / num_rows);
    delta = 0.0;
    for (int i = 0; i < num_rows; i++)
    {
        delta += abs(hY[i] - hY_my[i]);
    }
    (delta == 0) ? printf("GPU My is SAME with CUSPARSE\n") : printf("GPU My vs CUSPARSE: sum: %12lf  avg: %12lf\n", delta, delta / num_rows);


    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(matA))
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX))
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY))
    CHECK_CUSPARSE(cusparseDestroy(handle))

    // device memory deallocation
    CHECK_CUDA(cudaFree(dBuffer))
    CHECK_CUDA(cudaFree(dA_row_offset))
    CHECK_CUDA(cudaFree(dA_col_indices))
    CHECK_CUDA(cudaFree(dA_value))
    CHECK_CUDA(cudaFree(dX))
    CHECK_CUDA(cudaFree(dY))

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return EXIT_SUCCESS;
}