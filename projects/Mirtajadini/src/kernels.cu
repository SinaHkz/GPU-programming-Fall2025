#include "kernels.cuh"
#include "config.h"
#include <device_launch_parameters.h> // REQUIRED for __syncthreads to be visible in IntelliSense

__global__ void preprocessNormalizeFP32(const unsigned char* __restrict__ input, float* __restrict__ output, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float val = static_cast<float>(input[idx]);
        // Normalize: (val / 255.0 - 0.5) / 0.5
        output[idx] = (val * (2.0f / 255.0f)) - 1.5f;
    }
}

__global__ void preprocessNormalizeFP16(const unsigned char* __restrict__ input, half* __restrict__ output, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float val = static_cast<float>(input[idx]); //1.5
        float normalized = (val * (2.0f / 255.0f)) - 1.5f;
        output[idx] = __float2half(normalized);
    }
}

//adding bias & activation
__global__ void addBias(float* data, const float* bias, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int col = idx % N;
        data[idx] += bias[col];
    }
}

__global__ void reluKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

__global__ void floatToHalfKernel(const float* input, half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output[idx] = __float2half(input[idx]);
}

//FP32 
__global__ void matrixMulTiled(const float* A, const float* B, float* C, int M, int N, int K) {
    int by = blockIdx.y; int bx = blockIdx.x;
    int ty = threadIdx.y; int tx = threadIdx.x;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float val = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        __shared__ float As[TILE_SIZE][TILE_SIZE];
        __shared__ float Bs[TILE_SIZE][TILE_SIZE];

        if (row < M && (t * TILE_SIZE + tx) < K)
            As[ty][tx] = A[row * K + (t * TILE_SIZE + tx)];
        else As[ty][tx] = 0.0f;

        if ((t * TILE_SIZE + ty) < K && col < N)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            val += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = val;
}

//Mixed Precision (CUDA Cores)
//reads Half and converts to float for FMA, writes float
__global__ void matrixMulTiledFP16(const half* A, const half* B, float* C, int M, int N, int K) {
    int by = blockIdx.y; int bx = blockIdx.x;
    int ty = threadIdx.y; int tx = threadIdx.x;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float val = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        __shared__ half As[TILE_SIZE][TILE_SIZE];
        __shared__ half Bs[TILE_SIZE][TILE_SIZE];

        if (row < M && (t * TILE_SIZE + tx) < K)
            As[ty][tx] = A[row * K + (t * TILE_SIZE + tx)];
        else As[ty][tx] = __float2half(0.0f);

        if ((t * TILE_SIZE + ty) < K && col < N)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else Bs[ty][tx] = __float2half(0.0f);

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            //conversion to float happens here (using CUDA Cores)
            float a_float = __half2float(As[ty][k]);
            float b_float = __half2float(Bs[k][tx]);
            val += a_float * b_float;
        }
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = val;
}