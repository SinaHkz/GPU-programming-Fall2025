#include <cuda_runtime.h>
#include <stdio.h>
#include "pool_layer.h"
#include "tensor.h"

__device__ __forceinline__ int idx4_nchw(int n,int c,int h,int w,int C,int H,int W){
    return ((n*C + c)*H + h)*W + w;
}

__global__ void zero_kernel(float* __restrict__ x, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] = 0.0f;
}

// For each output grad element dY[b,c,ho,wo], send it to the winner input location in dX.
__global__ void maxpool_backward_kernel(const float* __restrict__ dY,
                                        const int* __restrict__ argmax,
                                        float* __restrict__ dX,
                                        int B, int C,
                                        int Hin, int Win,
                                        int Hout, int Wout,
                                        int poolH, int poolW,
                                        int strideH, int strideW)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * Hout * Wout;
    if (out_idx >= total) return;

    // Decode out_idx -> (b, c, ho, wo)
    int wo = out_idx % Wout; out_idx /= Wout;
    int ho = out_idx % Hout; out_idx /= Hout;
    int c  = out_idx % C;    out_idx /= C;
    int b  = out_idx;

    // Base top-left of pooling window in input
    int in_y0 = ho * strideH;
    int in_x0 = wo * strideW;

    // Winner offset inside pool window
    int off = argmax[idx4_nchw(b, c, ho, wo, C, Hout, Wout)];
    int ky  = off / poolW;
    int kx  = off - ky * poolW;

    int iy = in_y0 + ky;
    int ix = in_x0 + kx;

    if ((unsigned)iy < (unsigned)Hin && (unsigned)ix < (unsigned)Win) {
        int dx_idx = idx4_nchw(b, c, iy, ix, C, Hin, Win);
        float g    = dY[idx4_nchw(b, c, ho, wo, C, Hout, Wout)];

        // If pooling windows overlap (stride < pool), multiple outputs may write same input -> atomicAdd.
        atomicAdd(&dX[dx_idx], g);
    }
}

void maxpool_backward(const Tensor& dY,
                      const int* d_argmax,
                      Tensor& dX,
                      int poolH, int poolW,
                      int strideH, int strideW,
                      int blocksize,
                      cudaStream_t s)
{
    int B = dY.N;
    int C = dY.C;
    int Hout = dY.H;
    int Wout = dY.W;

    int Hin = dX.H;
    int Win = dX.W;

    int threads = blocksize;
    if (threads <= 0 || threads > 1024) threads = 256;

    // zero dX first (because only max locations receive gradients)
    int Ndx = dX.N * dX.C * dX.H * dX.W;
    int zblocks = (Ndx + threads - 1) / threads;
    zero_kernel<<<zblocks, threads ,  0, s>>>(dX.data, Ndx);

    // scatter dY gradients back to dX at argmax locations
    int total = B * C * Hout * Wout;
    int blocks = (total + threads - 1) / threads;

    maxpool_backward_kernel<<<blocks, threads , 0, s>>>(
            dY.data, d_argmax, dX.data,
            B, C,
            Hin, Win,
            Hout, Wout,
            poolH, poolW,
            strideH, strideW
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf(" maxpool_backward_kernel Launch Error: %s\n", cudaGetErrorString(err));
    }
}
