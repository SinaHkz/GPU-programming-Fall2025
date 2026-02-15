#pragma once
#include <cuda_runtime.h>
#include "tensor.h"

// Fused Conv + Bias + ReLU (tiled, shared-memory)
// stream-aware wrapper
void conv_forward_fused_bias_relu(
        const Tensor* d_input,
        const Tensor* d_weights,
        const float*  d_bias,
        Tensor*       d_output_relu,
        int pad,
        int stride,
        int blocksize,
        cudaStream_t stream = 0
);
