#pragma once
#include <iostream>
#include <vector>
#include <random>

// Generate an array of random integers within the specified range
// std::vector<int> int_array = generate_random_array_int(10, 0, 100);
std::vector<int> generate_random_array_int(int size, int lower_bound, int upper_bound) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(lower_bound, upper_bound);
    
    std::vector<int> result(size);
    for(int i = 0; i < size; ++i) {
        result[i] = dist(gen);
    }
    return result;
}

// Generate an array of random floating-point numbers within the specified range
// std::vector<float> float_array = generate_random_array_float(10, 0.0f, 100.0f);
std::vector<float> generate_random_array_float(int size, float lower_bound, float upper_bound) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(lower_bound, upper_bound);
    
    std::vector<float> result(size);
    for(int i = 0; i < size; ++i) {
        result[i] = dist(gen);
    }
    return result;
}
void generate_random_value_float(float * result, int size, float lower_bound, float upper_bound) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(lower_bound, upper_bound);
    
    for(int i = 0; i < size; ++i) {
        result[i] = dist(gen);
    }
}

// Check errors and print GB/s
template <int TEST_TIMES>
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
