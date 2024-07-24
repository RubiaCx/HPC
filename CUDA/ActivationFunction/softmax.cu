
// Softmax x: N, y: N
// grid(N/128), block(K=128)
template <const int BLOCK_SIZE = 128>
__global__ void softmax(float *x, float *y, float *total, int N)
{
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    constexpr int WARP_NUM = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[WARP_NUM];

    float sum = (idx < N) ? expf(x[idx]) : 0.0f;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    sum = warpReduceSum<float>(sum, WARP_SIZE);
    if (lane == 0)
        reduce_smem[warp] = sum;
    __syncthreads();
    // compute the final sum in each warp
    sum = (lane < WARP_NUM) ? reduce_smem[lane] : 0.0f;
    sum = warpReduceSum<float>(sum, WARP_SIZE); // sum(e^x_0,...,e^x_n-1)
    // get the total sum of all blocks.
    if (tid == 0)
        atomicAdd(total, sum);
    __threadfence(); // grid level memory fence 注意这里需要网格级别的内存同步
    // e^x_i / sum(e^x_0,...,e^x_n-1)
    if (idx < N)
        y[idx] = reduce_smem[tid] / (*total);
}

// Softmax x: N, y: N
// grid(N/128), block(K=128)
template <const int BLOCK_SIZE = 128>
__global__ void softmax_v2(float *x, float *y, float *total, int N)
{
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;

    float exp_val = (idx < N) ? expf(x[idx]) : 0.0f;
    float sum = block_reduce_sum<BLOCK_SIZE>(exp_val);
    // get the total sum of all blocks.
    if (tid == 0)
        atomicAdd(total, sum);
    __threadfence(); // grid level memory fence  注意这里需要网格级别的内存同步
    // e^x_i/sum(e^x_0,...,e^x_n-1)
    if (idx < N)
        y[idx] = exp_val / (*total);
}

// Softmax Vec4 x: N, y: N
// grid(N/128), block(128/4)
template <const int BLOCK_SIZE = 128 / 4>
__global__ void softmax_v2_vec4(float *x, float *y, float *total, int N)
{
    const int tid = threadIdx.x;
    const int idx = (blockIdx.x * blockDim.x + tid) * 4;

    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_exp;
    reg_exp.x = (idx < N) ? expf(reg_x.x) : 0.0f;
    reg_exp.y = (idx < N) ? expf(reg_x.y) : 0.0f;
    reg_exp.z = (idx < N) ? expf(reg_x.z) : 0.0f;
    reg_exp.w = (idx < N) ? expf(reg_x.w) : 0.0f;
    float exp_val = (reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w);
    float sum = block_reduce_sum<BLOCK_SIZE>(exp_val);
    // get the total sum of all blocks.
    if (tid == 0)
        atomicAdd(total, sum);
    __threadfence(); // grid level memory fence  注意这里需要网格级别的内存同步
    // e^x_i/sum(e^x_0,...,e^x_n-1)
    if (idx < N)
    {
        float4 reg_y;
        reg_y.x = reg_exp.x / (*total);
        reg_y.y = reg_exp.y / (*total);
        reg_y.z = reg_exp.z / (*total);
        reg_y.w = reg_exp.w / (*total);
        FLOAT4(y[idx]) = reg_y;
    }
}