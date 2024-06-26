void cpu_prefix_sum(float * in, float * out, int N)
{
    out[0] = in[0];
    for(int i = 1; i < N; i++)
    {
        out[i] = out[i-1] + in[i];
    }
}

__global__ void gpu_prefix_sum(float * in, float * out, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N)
    {
        out[idx] = in[idx];
        __syncthreads();
        for(int i = 1; i < N; i++)
        {
            if(idx >= i)
            {
                out[idx] = out[idx] + in[idx-i];
            }
            __syncthreads();
        }
    }
}

__global__ void gpu_prefix_sum_shared(float * in, float * out, int N)
{
    extern __shared__ float sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N)
    {
        sdata[threadIdx.x] = in[idx];
    }
    else
    {
        sdata[threadIdx.x] = 0;
    }
    __syncthreads();
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        float temp = 0.0;
        if(threadIdx.x >= stride) // 安全左移
        {
            temp = sdata[threadIdx.x - stride];
        }
        __syncthreads();

        if(threadIdx.x >= stride)
        {
            sdata[threadIdx.x] += temp;
        }
        __syncthreads();
    }
    if (threadIdx.x < N)
    {
        out[idx] = sdata[threadIdx.x];
    }
    
}

__global__ void gpu_gemm_global(float * A, float * B, float * C, int M, int K, int N)
{
    int c_col = blockIdx.x * blockDim.x + threadIdx.x;
    int c_row = blockIdx.y * blockDim.y + threadIdx.y;
    if(c_col < N && c_row < M)
    {
        float sum = 0.0;
        for(int i = 0; i < K; i++)
        {
            sum += A[c_row * K + i] * B[i * N + c_col];
        }
        C[c_row * N + c_col] = sum;
    }
}

// 沿K维度分块
template<const int TILE_WIDTH>
__global__ void matrixMulKernel(float* A, float* B, float* C, int M, int K, int N) {
    // 声明共享内存，用于存储子矩阵
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];
    // 每个block计算C的一个子块，每个thread计算一个元素
    int bx = blockIdx.x;  // Block index
    int by = blockIdx.y;
    int tx = threadIdx.x; // Thread index
    int ty = threadIdx.y;

    // 计算行列索引
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float sum = 0.0f;
    // 遍历矩阵A和B的子块
    for (int m = 0; m < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        // 每个线程加载A、B矩阵的一个元素到共享内存
        if (m * TILE_WIDTH + tx < K && row < M)
            tileA[ty][tx] = A[row * K + m * TILE_WIDTH + tx];
        else
            tileA[ty][tx] = 0.0;

        if (m * TILE_WIDTH + ty < K && col < N)
            tileB[ty][tx] = B[(m * TILE_WIDTH + ty) * N + col];
        else
            tileB[ty][tx] = 0.0;

        __syncthreads(); // 确保数据加载到共享内存完成

        // 计算子矩阵乘积并累加到sum中
        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += tileA[ty][k] * tileB[k][tx];

        __syncthreads(); // 确保所有计算完成再加载下一轮数据
    }

    if (row < M && col < N)
        C[row * N + col] = sum; // 写回结果矩阵
}
