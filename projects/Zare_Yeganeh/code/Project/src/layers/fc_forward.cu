#include <cuda_runtime.h>
#include <stdio.h>
#include "fc_layer.h"
#include "tensor.h"
//#include "/include/utils.h"

__global__ void fc_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features
) {
    // We map the 1D index to the 2D output matrix (Batch, Out_Features)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;

    if (idx < total_elements) {

        int out_idx = idx % out_features;
        int b = idx / out_features;

        float sum = 0.0f;

        // Dot Product
        int input_row_start = b * in_features;
        int weight_row_start = out_idx * in_features;

        for (int i = 0; i < in_features; ++i) {
            sum += input[input_row_start + i] * weights[weight_row_start + i];
        }

        // Add Bias
        if (bias != nullptr) {
            sum += bias[out_idx];
        }

        // Store Result
        output[idx] = sum;
    }
}

void fc_forward(
    const Tensor* d_input,
    const Tensor* d_weights,
    const float* d_bias,
    Tensor* d_output,
    int blocksize,
    cudaStream_t s
) {
    int batch_size = d_input->N;
    
    // Calculate total input features (flatten C, H, W)
    int in_features = d_input->C * d_input->H * d_input->W;
    int out_features = d_weights->N;

    // Validate dimensions to prevent mismatches
    int weight_in_features = d_weights->C * d_weights->H * d_weights->W;
    if (in_features != weight_in_features) {
        fprintf(stderr, "FC Layer Error: Input features (%d) != Weight input cols (%d)\n",
            in_features, weight_in_features);
        exit(EXIT_FAILURE);
    }

    int total_elements = batch_size * out_features;
    int threadsPerBlock = blocksize;
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

    fc_forward_kernel<<<blocksPerGrid, threadsPerBlock , 0, s>>>(
        d_input->data,
        d_weights->data,
        d_bias,
        d_output->data,
        batch_size,
        in_features,
        out_features
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("fc_forward_kernel Launch Error: %s\n", cudaGetErrorString(err));
    }

}