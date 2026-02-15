#ifndef POOL_LAYER_H
#define POOL_LAYER_H

#pragma once
#include "tensor.h"

// argmax is int array with size = B*C*Hout*Wout
// It stores the winning location inside the pooling window (offset 0..poolH*poolW-1).
void maxpool_backward(const Tensor& dY,
        const int* d_argmax,
        Tensor& dX,
        int poolH, int poolW,
        int strideH, int strideW,
        int blocksize,
        cudaStream_t stream = 0
);

// Forward Declaration
void maxpool_forward(
    const Tensor* d_input,
    Tensor* d_output,
    int* d_argmax, // Added this parameter
    int pool_size,
    int stride,
    int blocksize,
    cudaStream_t stream = 0
);
#endif