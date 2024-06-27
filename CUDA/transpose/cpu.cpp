#include <stdio.h>
#include <stdlib.h>
#include <chrono>

const int N = 1024; // matrix size is NxN
const int TEST_TIMES = 100; // 测试一百次

// Generate a random float value between 0 and 1
void fill_matrix(float in[])
{
    srand(time(0));
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            in[i * N + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

void transpose_CPU(float in[], float out[])
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            out[j * N + i] = in[i * N + j];
        }
    }
}

int main(int argc, char **argv)
{
    int numbytes = N * N * sizeof(float);
    float *in = (float *)malloc(numbytes);
    float *out = (float *)malloc(numbytes);

    fill_matrix(in);

    auto begin = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < TEST_TIMES; i++)
        transpose_CPU(in, out);

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    
    printf("The time by host:\t%f(ms)\n", elapsed.count() * 1e-6);

    free(in);
    free(out);
    return 0;
}