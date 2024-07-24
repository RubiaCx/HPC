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
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

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

// K: ldA
// N: ldB
template <
    const int BLOCK_SIZE_M,         // height of block of C that each thread block calculate
    const int BLOCK_SIZE_K,         // width of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,         // width of block of C that each thread block calculate
    const int THREAD_SIZE_Y,        // height of block of C that each thread calculate
    const int THREAD_SIZE_X,        // width of block of C that each thread calculate
    const bool ENABLE_DOUBLE_BUFFER // whether enable double buffering or not
    >
__global__ void Sgemm_kernel(
    float *__restrict__ A,
    float *__restrict__ B,
    float *__restrict__ C,
    const int M,
    const int N,
    const int K)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // the threads number in Block of X,Y
    const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

    // thread id in cur Block
    const int tid = ty * THREAD_X_PER_BLOCK + tx;

    // shared memory
    __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];
    // registers for C
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
    // registers for A and B
    float frag_a[2][THREAD_SIZE_Y];
    float frag_b[2][THREAD_SIZE_X];
    // registers load global memory
    const int ldg_num_a = BLOCK_SIZE_M * BLOCK_SIZE_K / (THREAD_NUM_PER_BLOCK * 4);
    const int ldg_num_b = BLOCK_SIZE_K * BLOCK_SIZE_N / (THREAD_NUM_PER_BLOCK * 4);
    float ldg_a_reg[4 * ldg_num_a];
    float ldg_b_reg[4 * ldg_num_b];

    // threads number in one row
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

    // row number and col number that needs to be loaded by this thread
    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4;
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

    // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    A = &A[(BLOCK_SIZE_M * by) * K];
    B = &B[BLOCK_SIZE_N * bx];

// transfer first tile from global mem to shared mem
//  load A from global memory to shared memory
#pragma unroll
    for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE)
    {
        int ldg_index = i / A_TILE_ROW_STRIDE * 4;
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
            A_TILE_ROW_START + i, // row
            A_TILE_COL,           // col
            K)]);
        As[0][A_TILE_COL][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index];
        As[0][A_TILE_COL + 1][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 1];
        As[0][A_TILE_COL + 2][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 2];
        As[0][A_TILE_COL + 3][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 3];
    }
// load B from global memory to shared memory
#pragma unroll
    for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE)
    {
        FETCH_FLOAT4(Bs[0][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
            B_TILE_ROW_START + i, // row
            B_TILE_COL,           // col
            N)]);
    }
    __syncthreads();
// load A from shared memory to register
#pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4)
    {
        FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(As[0][0][THREAD_SIZE_Y * ty + thread_y]);
    }
// load B from shared memory to register
#pragma unroll
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4)
    {
        FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(Bs[0][0][THREAD_SIZE_X * tx + thread_x]);
    }

    int write_stage_idx = 1;
    int tile_idx = 0;
    do
    {
        tile_idx += BLOCK_SIZE_K;
        // load next tile from global mem
        if (tile_idx < K)
        {
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE)
            {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
                    A_TILE_ROW_START + i,  // row
                    A_TILE_COL + tile_idx, // col
                    K)]);
            }
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE)
            {
                int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_b_reg[ldg_index]) = FETCH_FLOAT4(B[OFFSET(
                    tile_idx + B_TILE_ROW_START + i, // row
                    B_TILE_COL,                      // col
                    N)]);
            }
        }

        int load_stage_idx = write_stage_idx ^ 1;

#pragma unroll
        for (int j = 0; j < BLOCK_SIZE_K - 1; ++j)
        {
// load next tile from shared mem to register
// load A from shared memory to register
#pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4)
            {
                FETCH_FLOAT4(frag_a[(j + 1) % 2][thread_y]) = FETCH_FLOAT4(As[load_stage_idx][j + 1][THREAD_SIZE_Y * ty + thread_y]);
            }
// load B from shared memory to register
#pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4)
            {
                FETCH_FLOAT4(frag_b[(j + 1) % 2][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx][j + 1][THREAD_SIZE_X * tx + thread_x]);
            }
// compute C THREAD_SIZE_X x THREAD_SIZE_Y
#pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y)
            {
#pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x)
                {
                    accum[thread_y][thread_x] += frag_a[j % 2][thread_y] * frag_b[j % 2][thread_x];
                }
            }
        }

        if (tile_idx < K)
        {
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE)
            {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                As[write_stage_idx][A_TILE_COL][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index];
                As[write_stage_idx][A_TILE_COL + 1][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 1];
                As[write_stage_idx][A_TILE_COL + 2][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 2];
                As[write_stage_idx][A_TILE_COL + 3][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 3];
            }
// load B from global memory to shared memory
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE)
            {
                int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(ldg_b_reg[ldg_index]);
            }
            // use double buffer, only need one sync
            __syncthreads();
            // switch
            write_stage_idx ^= 1;
        }

// load first tile from shared mem to register of next iter
// load A from shared memory to register
#pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4)
        {
            FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(As[load_stage_idx ^ 1][0][THREAD_SIZE_Y * ty + thread_y]);
        }
// load B from shared memory to register
#pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4)
        {
            FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx ^ 1][0][THREAD_SIZE_X * tx + thread_x]);
        }
// compute last tile mma THREAD_SIZE_X x THREAD_SIZE_Y
#pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y)
        {
#pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x)
            {
                accum[thread_y][thread_x] += frag_a[1][thread_y] * frag_b[1][thread_x];
            }
        }
    } while (tile_idx < K);

// store back to C
#pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y)
    {
#pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4)
        {
            FETCH_FLOAT4(C[OFFSET(
                BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
                BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x,
                N)]) = FETCH_FLOAT4(accum[thread_y][thread_x]);
        }
    }
}

double gemm_v1(float *h_A, float *h_B, float *h_C, 
             size_t M, size_t K, size_t N)
{
    float *d_A;
    float *d_B;
    float *d_C;
    size_t bytes_A = sizeof(float) * M * K;
    size_t bytes_B = sizeof(float) * K * N;
    size_t bytes_C = sizeof(float) * M * N;
    CHECK_CUDA(cudaMalloc(&d_A, bytes_A));
    CHECK_CUDA(cudaMalloc(&d_B, bytes_B));
    CHECK_CUDA(cudaMalloc(&d_C, bytes_C));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyHostToDevice));

    double msecPerMatrixMul = 0.0;
    double gigaFlops = 0.0;
    double flopsPerMatrixMul = 2.0 * M * N * K;

    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_N = 128;
    const int BLOCK_SIZE_K = 8;
    const int THREAD_SIZE_X = 8;
    const int THREAD_SIZE_Y = 8;
    const bool ENABLE_DOUBLE_BUFFER = false;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 1000;
    
    dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
    Sgemm_kernel<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER>
            <<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaEventRecord(start));
    for (int run = 0; run < nIter; run++)
    {
        Sgemm_kernel<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER>
            <<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        } 
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&msecTotal, start, stop));

    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    msecPerMatrixMul = msecTotal / nIter;
    gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf("My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
           gigaFlops, msecPerMatrixMul, flopsPerMatrixMul);

    // for(int i = 0; i < M; i++)
    // {
    //     for(int j = 0; j < N; j++) {
    //         printf("h_C[%d][%d] = %f\n", i, j, h_C[i * N + j]);
    //     }
    //     printf("\n");
    // }

    // Free Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return gigaFlops;
}