#include <cuda_runtime.h>
#include <float.h>
#include "pool_layer.h"
#include "tensor.h"
#include <stdio.h>

//#include "/include/utils.h"

__device__ __forceinline__ int idx4_nchw(int n, int c, int h, int w, int C, int H, int W) {
    return ((n * C + c) * H + h) * W + w;
}

__global__ void maxpool_forward_kernel(
    const float* __restrict__ X,
    float* __restrict__ Y,
    int* __restrict__ argmax,
    int B, int C, int Hin, int Win,
    int Hout, int Wout,
    int poolH, int poolW,
    int strideH, int strideW
) {
    // One thread per output pixel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * C * Hout * Wout;

    if (idx < total_elements) {
        int temp_idx = idx;
        int wo = temp_idx % Wout; temp_idx /= Wout;
        int ho = temp_idx % Hout; temp_idx /= Hout;
        int c  = temp_idx % C;    temp_idx /= C;
        int b  = temp_idx;

        int h_start = ho * strideH;
        int w_start = wo * strideW;
        int h_end = h_start + poolH;
        int w_end = w_start + poolW;

        // Find Max in Window
        float max_val = -FLT_MAX; 
        int max_idx_offset = 0; 

        // Iterate over the pooling window
        for (int i = h_start; i < h_end; ++i) {
            for (int j = w_start; j < w_end; ++j) {
                // Check Bounds
                if (i < Hin && j < Win) {
                    int input_idx = idx4_nchw(b, c, i, j, C, Hin, Win);
                    float val = X[input_idx];

                    if (val > max_val) {
                        max_val = val;
                        // Calculate offset relative to window start for argmax
                        // Offset = (dy * poolW) + dx
                        int dy = i - h_start;
                        int dx = j - w_start;
                        max_idx_offset = dy * poolW + dx;
                    }
                }
            }
        }

        // Store Result and Index, idx corresponds to the output flattened index (b, c, ho, wo)
        Y[idx] = max_val;
        argmax[idx] = max_idx_offset; 
    }
}

void maxpool_forward(
    const Tensor* d_input,
    Tensor* d_output,
    int* d_argmax,
    int pool_size,
    int stride,
    int blocksize,
    cudaStream_t s
) {
    int B = d_input->N;
    int C = d_input->C;
    int Hin = d_input->H;
    int Win = d_input->W;
    
    int Hout = d_output->H; 
    int Wout = d_output->W;

    int total_output = B * C * Hout * Wout;

    int threadsPerBlock = blocksize;
    int blocksPerGrid = (total_output + threadsPerBlock - 1) / threadsPerBlock;

    maxpool_forward_kernel<<<blocksPerGrid, threadsPerBlock , 0, s>>>(
        d_input->data,
        d_output->data,
        d_argmax,
        B, C, Hin, Win,
        Hout, Wout,
        pool_size, pool_size,
        stride, stride
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("maxpool_forward_kernel Launch Error: %s\n", cudaGetErrorString(err));
    }

}