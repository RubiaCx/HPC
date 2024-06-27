// MY Kernel
#include "gemm_v1.cuh"
#include "cuBLAS.cuh"
#include "gemm_1_global.cuh"
#include "gemm_2_tiling.cuh"
#include "gemm_3_tiling.cuh"
#include "gemm_4_tiling.cuh"
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

bool checkAnswer(const float *answer, const float *result, int M, int N)
{
    double eps = 1.e-6; // machine zero
    double dot_length = M;
    for (int i = 0; i < M * N; i++)
    {
        int row = i / N;
        int col = i % N;
        double abs_err = fabs(answer[i] - result[col * M + row]);
        double abs_val = fabs(answer[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps)
        {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, answer[i], result[col * M + row], eps);
            return false;
        }
    }
    return true;
}


int main(int argc, char **argv)
{
    if (argc != 4)
    {
        printf("usage: ./main [M] [K] [N]\n");
        exit(0);
    }
    size_t M = atoi(argv[1]);
    size_t K = atoi(argv[2]);
    size_t N = atoi(argv[3]);
    assert(M % 8 == 0);
    assert(N % 8 == 0);
    assert(K % 8 == 0);
    size_t bytes_A = sizeof(float) * M * K;
    size_t bytes_B = sizeof(float) * K * N;
    size_t bytes_C = sizeof(float) * M * N;

    float *h_A = (float *)malloc(bytes_A);
    float *h_B = (float *)malloc(bytes_B);
    float *h_C = (float *)malloc(bytes_C);
    float *h_C_gemm1  = (float *)malloc(bytes_C);
    float *h_C_gemm2  = (float *)malloc(bytes_C);
    float *h_C_gemm3  = (float *)malloc(bytes_C);
    float *h_C_gemm4  = (float *)malloc(bytes_C);
    float *h_C_CUBLAS = (float *)malloc(bytes_C);
    // generate A
    for (int i = 0; i < M * K; i++)
    {
        h_A[i] = i / 12;
    }
    // generate B
    for (int i = 0; i < K * N; i++)
    {
        h_B[i] = i % 12;
    }

    double gigaFlops_cublas = cuBLAS(h_A, h_B, h_C_CUBLAS, M, K, N);
    double gigaFlops_gemm1 = gemm_v1(h_A, h_B, h_C_gemm1, M, K, N);
    double gigaFlops_global = gemm_global(h_A, h_B, h_C_gemm1, M, K, N);
    double gigaFlops_tiling = gemm_tiling(h_A, h_B, h_C_gemm2, M, K, N);
    double gigaFlops_tiling_thread = gemm_tiling_thread(h_A, h_B, h_C_gemm4, M, K, N);
    double gigaFlops_tiling_thread_2d = gemm_tiling_thread_2d(h_A, h_B, h_C_gemm4, M, K, N);
    //  for(int i = 0; i < M; i++)
    // {
    //     for(int j = 0; j < N; j++) {
    //         printf("h_C_gemm1[%d][%d] = %f\n", i, j, h_C_gemm1[i * N + j]);
    //     }
    //     printf("\n");
    // }
    // for(int i = 0; i < M; i++)
    // {
    //     for(int j = 0; j < N; j++) {
    //         printf("h_C_CUBLAS[%d][%d] = %f\n", i, j, h_C_CUBLAS[i * N + j]);
    //     }
    //     printf("\n");
    // }
    bool correct = checkAnswer(h_C_gemm2, h_C_CUBLAS, M, N);
    
    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    printf("ratio = %f\n", gigaFlops_gemm1 / gigaFlops_cublas);
    printf("ratio global = %f\n", gigaFlops_global / gigaFlops_cublas);
    printf("ratio tiling = %f\n", gigaFlops_tiling / gigaFlops_cublas);
    printf("ratio tiling thread = %f\n", gigaFlops_tiling_thread / gigaFlops_cublas);
    printf("ratio tiling thread 2D = %f\n", gigaFlops_tiling_thread_2d / gigaFlops_cublas);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_CUBLAS);
    free(h_C_gemm1);
    free(h_C_gemm2);
    free(h_C_gemm3);

}