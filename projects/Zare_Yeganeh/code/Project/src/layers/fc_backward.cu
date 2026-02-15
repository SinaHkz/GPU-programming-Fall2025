#include <cuda_runtime.h>
#include "fc_layer.h"
#include "tensor.h"
#include <stdio.h>

//#include "/include/utils.h"

__global__ void fc_backward_input_kernel(
    const float* __restrict__ dY,
    const float* __restrict__ W,
    float* __restrict__ dX,
    int B, int Cin, int Cout
) {
    // One thread per input element (b, ci)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * Cin;

    if (idx < total) {
        int ci = idx % Cin;
        int b  = idx / Cin;

        float acc = 0.0f;
        
        // Dot product: row 'b' of dY dot column 'ci' of W

        for (int co = 0; co < Cout; ++co) {
            acc += dY[b * Cout + co] * W[co * Cin + ci];
        }

        dX[idx] = acc;
    }
}

__global__ void fc_backward_weight_kernel(
    const float* __restrict__ dY,
    const float* __restrict__ X,
    float* __restrict__ dW,
    int B, int Cin, int Cout
) {
    // One thread per weight element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Cout * Cin;

    if (idx < total) {
        int ci = idx % Cin;
        int co = idx / Cin;

        float acc = 0.0f;

        // Sum over batch dimension

        for (int b = 0; b < B; ++b) {
            acc += dY[b * Cout + co] * X[b * Cin + ci];
        }

        dW[idx] = acc;
    }
}

__global__ void fc_backward_bias_kernel(
    const float* __restrict__ dY,
    float* __restrict__ db,
    int B, int Cout
) {
    // One thread per output neuron
    int co = blockIdx.x * blockDim.x + threadIdx.x;

    if (co < Cout) {
        float acc = 0.0f;
        
        // Sum the gradients for this neuron across all batch items
        for (int b = 0; b < B; ++b) {
            acc += dY[b * Cout + co];
        }

        db[co] = acc;
    }
}

void fc_backward(
        const Tensor* d_input,
        const Tensor* d_weights,
        const Tensor* d_dY,
        Tensor* d_dX,
        Tensor* d_dW,
        float* d_db,
        int blocksize,
        cudaStream_t s
) {
    int B = d_input->N;

    int Cin  = d_weights->C * d_weights->H * d_weights->W;
    int Cout = d_weights->N;

    int threads = blocksize;

    if (d_dX != nullptr) {
        int total_dX = B * Cin;
        int blocks_dX = (total_dX + threads - 1) / threads;
        fc_backward_input_kernel<<<blocks_dX, threads , 0, s>>>(
                d_dY->data, d_weights->data, d_dX->data, B, Cin, Cout
        );
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("fc_backward_input_kernel Launch Error: %s\n", cudaGetErrorString(err));
    }

    if (d_dW != nullptr) {
        int total_dW = Cout * Cin;
        int blocks_dW = (total_dW + threads - 1) / threads;
        fc_backward_weight_kernel<<<blocks_dW, threads , 0, s>>>(
                d_dY->data, d_input->data, d_dW->data, B, Cin, Cout
        );
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("fc_backward_weight_kernel Launch Error: %s\n", cudaGetErrorString(err));
    }

    if (d_db != nullptr) {
        int total_db = Cout;
        int blocks_db = (total_db + threads - 1) / threads;
        fc_backward_bias_kernel<<<blocks_db, threads , 0, s>>>(
                d_dY->data, d_db, B, Cout
        );
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("fc_backward_bias_kernel Launch Error: %s\n", cudaGetErrorString(err));
    }
}
