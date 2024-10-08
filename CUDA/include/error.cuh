#pragma once
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#define CHECK(call)                                                     \
do {                                                                    \
    const cudaError_t error_code = call;                                \
    if (error_code != cudaSuccess)                                      \
    {                                                                   \
        printf("CUDA ERROR: \n");                                       \
        printf("    FILE: %s\n", __FILE__);                             \
        printf("    LINE: %d\n", __LINE__);                             \
        printf("    ERROR CODE: %d\n", error_code);                     \
        printf("    ERROR TEXT: %s\n", cudaGetErrorString(error_code)); \
        exit(1);                                                        \
    }                                                                   \
}while(0);                                                              \

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