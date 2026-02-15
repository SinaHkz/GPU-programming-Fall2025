#include <cuda_runtime.h>
#include <stdio.h>
#include "conv_layer.h"
#include "tensor.h"
//#include "/include/utils.h"

__global__ void conv_forward_naive_kernel(
    const Tensor input,
    const Tensor weights,
    const float* __restrict__ bias,
    Tensor output,
    int pad,
    int stride,
    int total_elements
) {
    // 1. Calculate Global Thread Index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        // 2. Decompose Index into (b, oc, oh, ow)
        int temp_idx = idx;
        int ow = temp_idx % output.W;
        temp_idx /= output.W;
        int oh = temp_idx % output.H;
        temp_idx /= output.H;
        int oc = temp_idx % output.C;
        int b  = temp_idx / output.C;

        float sum = 0.0f;

        // 3. Convolution Loops
        // Loop over input channels (input.C)
        for (int ic = 0; ic < input.C; ++ic) {
            // Loop over kernel spatial dimensions (weights.H x weights.W)
            for (int kh = 0; kh < weights.H; ++kh) {
                for (int kw = 0; kw < weights.W; ++kw) {
                    
                    int ih_in = oh * stride - pad + kh;
                    int iw_in = ow * stride - pad + kw;

                    // Boundary Check for Padding
                    if (ih_in >= 0 && ih_in < input.H && iw_in >= 0 && iw_in < input.W) {
                        
                        // Indexing using NCHW layout
                        int input_idx = ((b * input.C + ic) * input.H + ih_in) * input.W + iw_in;
                        int weight_idx = ((oc * weights.C + ic) * weights.H + kh) * weights.W + kw;

                        sum += input.data[input_idx] * weights.data[weight_idx];
                    }
                }
            }
        }

        // 4. Add Bias (Bias is per output channel)
        if (bias != nullptr) {
            sum += bias[oc];
        }

        // 5. Write to Output
        output.data[idx] = sum;
    }
}

void conv_forward(
        const Tensor* d_input,
        const Tensor* d_weights,
        const float* d_bias,
        Tensor* d_output,
        int pad,
        int stride,
        int blocksize,
        cudaStream_t s
) {
    int total_output_elements = d_output->N * d_output->C * d_output->H * d_output->W;

    int threadsPerBlock = blocksize;
    if (threadsPerBlock <= 0 || threadsPerBlock > 1024) threadsPerBlock = 256;

    int blocksPerGrid = (total_output_elements + threadsPerBlock - 1) / threadsPerBlock;

    conv_forward_naive_kernel<<<blocksPerGrid, threadsPerBlock ,0, s>>>(
            *d_input, *d_weights, d_bias, *d_output, pad, stride, total_output_elements
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("conv_forward_naive_kernel Launch Error: %s\n", cudaGetErrorString(err));
        return;
    }
}
