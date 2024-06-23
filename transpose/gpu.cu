/*
 * wsl中跑不出来
 */
#include <stdio.h>
#include <assert.h>
#include <cublas_v2.h>

const int TEST_TIMES = 100; // 测试一百次

// Check errors and print GB/s
void checkAndPrint(const float *answer, const float *result, int n, float ms)
{
    bool passed = true;
    for (int i = 0; i < n; i++)
        if (fabs(result[i] - answer[i]) > 1e-6) // 允许小于1e-6的误差 if (result[i] != answer[i])
        {
            printf("FAILED because i = %d, answer = %f but result = %f\n", i, result[i], answer[i]);
            passed = false;
            break;
        }
    if (passed)
        printf("%20.2f%20.6f\n", 2 * n * sizeof(float) * 1e-6 * TEST_TIMES / ms, ms / TEST_TIMES);
}

// simple copy kernel
// Used as answererence case represultenting best effective bandwidth.
template <int TILE_DIM, int BLOCK_ROWS>
__global__ void copy(float *odata, const float *idata)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) // 4
        odata[(y + j) * width + x] = idata[(y + j) * width + x];
}

// copy kernel using shared memory
// Also used as answererence case, demonstrating effect of using shared memory.
template <int TILE_DIM, int BLOCK_ROWS>
__global__ void copySharedMem(float *odata, const float *idata)
{
    __shared__ float tile[TILE_DIM * TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[(threadIdx.y + j) * TILE_DIM + threadIdx.x] = idata[(y + j) * width + x];

    __syncthreads();

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(y + j) * width + x] = tile[(threadIdx.y + j) * TILE_DIM + threadIdx.x];
}

// naive transpose
// Simplest transpose; doesn't use shared memory.
// Global memory reads are coalesced but writes are not.
template <int TILE_DIM, int BLOCK_ROWS>
__global__ void transposeNaive(float *odata, const float *idata)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[x * width + (y + j)] = idata[(y + j) * width + x];
}

// coalesced transpose
// Uses shared memory to achieve coalesing in both reads and writes
// Tile width == #banks causes shared memory bank conflicts.
template <int TILE_DIM, int BLOCK_ROWS>
__global__ void transposeCoalesced(float *odata, const float *idata)
{
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x; // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
}

// No bank-conflict transpose
// Same as transposeCoalesced except the first tile dimension is padded
// to avoid shared memory bank conflicts.
template <int TILE_DIM, int BLOCK_ROWS>
__global__ void transposeNoBankConflicts(float *odata, const float *idata)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x; // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
}

template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void transpose_v5_64x32(int M, int N,                  //
                                   float *__restrict__ odata, const float *__restrict__ idata) {
  __shared__ float smem[64][32];
  // Global Memory -> Shared Memory
  {
    constexpr const int ITER_Y = 32 / BLOCK_DIM_Y;
    constexpr const int ITER_X = 64 / BLOCK_DIM_X;
    static_assert(ITER_Y * BLOCK_DIM_Y == 32);
    static_assert(ITER_X * BLOCK_DIM_X == 64);

#pragma unroll
    for (int iy = 0; iy < ITER_Y; iy++) {
      const int ly = iy * BLOCK_DIM_Y + threadIdx.x / BLOCK_DIM_X;
      const int gy = blockIdx.x * 32 + ly;
#pragma unroll
      for (int ix = 0; ix < ITER_X; ix++) {
        const int lx = ix * BLOCK_DIM_X + threadIdx.x % BLOCK_DIM_X;
        const int gx = blockIdx.y * 64 + lx;
        if (gy < M && gx < N) {
          smem[lx][(lx + ly) % 32] = idata[gy * N + gx];
        }
      }
    }
  }
  __syncthreads();

  // Shared Memory -> Global Memory
  {
    constexpr const auto ITER_Y = 64 / BLOCK_DIM_Y;
    constexpr const auto ITER_X = 32 / BLOCK_DIM_X;
    static_assert(ITER_Y * BLOCK_DIM_Y == 64);
    static_assert(ITER_X * BLOCK_DIM_X == 32);

#pragma unroll
    for (int iy = 0; iy < ITER_Y; iy++) {
      const int ly = iy * BLOCK_DIM_Y + threadIdx.x / BLOCK_DIM_X;
      const int gy = blockIdx.y * 64 + ly;
#pragma unroll
      for (int ix = 0; ix < ITER_X; ix++) {
        const int lx = ix * BLOCK_DIM_X + threadIdx.x % BLOCK_DIM_X;
        const int gx = blockIdx.x * 32 + lx;
        if (gy < N && gx < M) {
          odata[gy * M + gx] = smem[ly][(lx + ly) % 32];
        }
      }
    }
  }
}

int main(int argc, char **argv)
{
    const int nx = 16384;
    const int ny = 16384;
    const int mem_size = nx * ny * sizeof(float);
    const int TILE_DIM = 32;
    const int BLOCK_ROWS = 8;

    dim3 dimGrid(nx / TILE_DIM, ny / TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

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

    float *h_idata = (float *)malloc(mem_size);
    float *h_cdata = (float *)malloc(mem_size);
    float *h_tdata = (float *)malloc(mem_size);
    float *answer = (float *)malloc(mem_size);

    float *d_idata, *d_cdata, *d_tdata;
    cudaMalloc(&d_idata, mem_size);
    cudaMalloc(&d_cdata, mem_size);
    cudaMalloc(&d_tdata, mem_size);

    // check parameters and calculate execution configuration
    if (nx % TILE_DIM || ny % TILE_DIM)
    {
        printf("nx and ny must be a multiple of TILE_DIM\n");
        goto error_exit;
    }

    if (TILE_DIM % BLOCK_ROWS)
    {
        printf("TILE_DIM must be a multiple of BLOCK_ROWS\n");
        goto error_exit;
    }

    // host
    for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
            h_idata[j * nx + i] = j * nx + i;

    // correct resultult for error checking
    for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
            answer[j * nx + i] = h_idata[i * nx + j];

    // device
    cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice);

    // events for timing
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float ms;

    // ------------
    // time kernels
    // ------------
    printf("%25s%25s%25s\n", "Routine", "Bandwidth (GB/s)", "Time (ms)");

    // ----
    // copy
    // ----
    printf("%25s", "copy");
    cudaMemset(d_cdata, 0, mem_size);
    // warm up
    copy<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock>>>(d_cdata, d_idata);
    cudaEventRecord(startEvent, 0);
    for (int i = 0; i < TEST_TIMES; i++)
        copy<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock>>>(d_cdata, d_idata);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost);
    checkAndPrint(h_idata, h_cdata, nx * ny, ms);

    // -------------
    // copySharedMem
    // -------------
    printf("%25s", "shared memory copy");
    cudaMemset(d_cdata, 0, mem_size);
    // warm up
    copySharedMem<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock>>>(d_cdata, d_idata);
    cudaEventRecord(startEvent, 0);
    for (int i = 0; i < TEST_TIMES; i++)
        copySharedMem<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock>>>(d_cdata, d_idata);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost);
    checkAndPrint(h_idata, h_cdata, nx * ny, ms);

    // --------------
    // transposeNaive
    // --------------
    printf("%25s", "naive transpose");
    cudaMemset(d_tdata, 0, mem_size);
    // warmup
    transposeNaive<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    cudaEventRecord(startEvent, 0);
    for (int i = 0; i < TEST_TIMES; i++)
        transposeNaive<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost);
    checkAndPrint(answer, h_tdata, nx * ny, ms);

    // ------------------
    // transposeCoalesced
    // ------------------
    printf("%25s", "coalesced transpose");
    cudaMemset(d_tdata, 0, mem_size);
    // warmup
    transposeCoalesced<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    cudaEventRecord(startEvent, 0);
    for (int i = 0; i < TEST_TIMES; i++)
        transposeCoalesced<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost);
    checkAndPrint(answer, h_tdata, nx * ny, ms);

    // ------------------------
    // transposeNoBankConflicts
    // ------------------------
    printf("%25s", "conflict-free transpose");
    cudaMemset(d_tdata, 0, mem_size);
    // warmup
    transposeNoBankConflicts<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    cudaEventRecord(startEvent, 0);
    for (int i = 0; i < TEST_TIMES; i++)
        transposeNoBankConflicts<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost);
    checkAndPrint(answer, h_tdata, nx * ny, ms);


error_exit:
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(d_tdata);
    cudaFree(d_cdata);
    cudaFree(d_idata);
    free(h_idata);
    free(h_tdata);
    free(h_cdata);
    free(answer);
}