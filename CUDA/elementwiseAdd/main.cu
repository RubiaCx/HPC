// transfer float4
#define FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

// ElementWise Add
// grid(N/128), block(128)
// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add(float *a, float *b, float *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        c[idx] = a[idx] + b[idx];
}

// ElementWise Add + Vec4
// grid(N/128), block(128/4)
// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_vec4(float *a, float *b, float *c, int N)
{
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N)
    {
        float4 reg_a = FLOAT4(a[idx]);
        float4 reg_b = FLOAT4(b[idx]);
        float4 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;
        FLOAT4(c[idx]) = reg_c;
    }
}