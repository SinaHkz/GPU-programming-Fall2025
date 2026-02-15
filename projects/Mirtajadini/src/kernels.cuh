#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

//preprocessing
__global__ void preprocessNormalizeFP32(const unsigned char* __restrict__ input, float* __restrict__ output, int num_elements);
__global__ void preprocessNormalizeFP16(const unsigned char* __restrict__ input, half* __restrict__ output, int num_elements);


//bias & activation 
// add bias vector (size N) to matrix (M x N)
__global__ void addBias(float* data, const float* bias, int M, int N);
__global__ void reluKernel(float* data, int size);


//compute kernels
//FP32 baseline (FP32 Storage, FP32 Math)
__global__ void matrixMulTiled(const float* A, const float* B, float* C, int M, int N, int K);


//Mixed Precision CUDA Cores (FP16 Storage, FP32 Math)
__global__ void matrixMulTiledFP16(const half* A, const half* B, float* C, int M, int N, int K);

__global__ void floatToHalfKernel(const float* input, half* output, int size);