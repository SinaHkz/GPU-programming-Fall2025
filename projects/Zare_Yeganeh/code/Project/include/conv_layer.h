#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include "tensor.h"

void conv_forward(
    const Tensor* d_input,
    const Tensor* d_weights,
    const float* d_bias,
    Tensor* d_output,
    int pad,
    int stride,
    int  blocksize,
    cudaStream_t stream = 0
);

void conv_backward(
    const Tensor& X,
    const Tensor& W,
    const Tensor& dY,
    Tensor& dX,
    Tensor& dW,
    Tensor& db,
    int K,
    int stride, int pad,
    int blocksize ,
    cudaStream_t stream = 0
);

#endif