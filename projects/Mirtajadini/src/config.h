#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

//hyperparameters 
#define IMAGE_H 96
#define IMAGE_W 96
#define IMAGE_C 3
#define INPUT_DIM (IMAGE_H * IMAGE_W * IMAGE_C) // 27648 = 96*96*3
#define HIDDEN_DIM 8192


//Mapping: 0=Airplane, 1=Bird, 2=Car, 3=Dog, 4=Cat
#define OUTPUT_DIM 5 

#define TILE_SIZE 32

extern const char* STL10_CLASS_NAMES[5];

//Error Handling
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS Error: %d at %s:%d\n", status, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)