
#ifndef PROJECT_CONV_TILE_H
#define PROJECT_CONV_TILE_H

#include "tensor.h"

void conv_forward(
        const Tensor* d_input,
        const Tensor* d_weights,
        const float* d_bias,
        Tensor* d_output,
        int pad,
        int stride,
        int  blocksize ,
        cudaStream_t stream = 0
);

#endif