#ifndef FC_LAYER_H
#define FC_LAYER_H

#include "tensor.h"

// Forward Pass
void fc_forward(
    const Tensor* d_input,
    const Tensor* d_weights,
    const float* d_bias,
    Tensor* d_output,
    int blocksize,
    cudaStream_t stream = 0
);

// Backward Pass
void fc_backward(
    const Tensor* d_input,
    const Tensor* d_weights,
    const Tensor* d_dY,
    Tensor* d_dX,
    Tensor* d_dW,
    float* d_db,
    int blocksize,
    cudaStream_t stream = 0
);
//
#endif // FC_LAYER_H