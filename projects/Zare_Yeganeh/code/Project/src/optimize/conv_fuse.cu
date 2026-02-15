#include <cuda_runtime.h>
#include <cstdio>
#include "conv_fused.h"
#include "tensor.h"

#ifndef TILE_W
#define TILE_W 16
#endif
#ifndef TILE_H
#define TILE_H 16
#endif

__device__ __forceinline__ int idx4(int n,int c,int h,int w,int C,int H,int W){
    return ((n*C + c)*H + h)*W + w;
}
__device__ __forceinline__ int widx(int oc,int ic,int kh,int kw,int Cin,int K){
    return ((oc*Cin + ic)*K + kh)*K + kw;
}

__device__ __forceinline__ float relu(float x){ return x > 0.0f ? x : 0.0f; }

// Fused: Conv + Bias + ReLU
__global__ void conv_forward_tiled_fused_kernel(
        Tensor input,
        Tensor weights,
        const float* __restrict__ bias,
        Tensor output,            // output is AFTER ReLU
        int K, int pad, int stride
){
    int oc = blockIdx.x;

    int outH = output.H;
    int outW = output.W;

    int tilesW = (outW + TILE_W - 1) / TILE_W;
    int tilesH = (outH + TILE_H - 1) / TILE_H;
    int tilesPerImg = tilesW * tilesH;

    int tile_linear = blockIdx.y;
    int n  = tile_linear / tilesPerImg;
    int t  = tile_linear % tilesPerImg;
    int th = t / tilesW;
    int tw = t % tilesW;

    int oh0 = th * TILE_H;
    int ow0 = tw * TILE_W;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int oh = oh0 + ty;
    int ow = ow0 + tx;

    const int shH = TILE_H * stride + (K - 1);
    const int shW = TILE_W * stride + (K - 1);

    extern __shared__ float shmem[];

    float acc = 0.0f;

    for (int ic = 0; ic < input.C; ++ic) {

        for (int s = ty * TILE_W + tx; s < shH * shW; s += TILE_H * TILE_W) {
            int sh_y = s / shW;
            int sh_x = s % shW;

            int ih = (oh0 * stride - pad) + sh_y;
            int iw = (ow0 * stride - pad) + sh_x;

            float v = 0.0f;
            if (ih >= 0 && ih < input.H && iw >= 0 && iw < input.W) {
                v = input.data[idx4(n, ic, ih, iw, input.C, input.H, input.W)];
            }
            shmem[sh_y * shW + sh_x] = v;
        }

        __syncthreads();

        if (oh < outH && ow < outW) {
            int sh_oh = (oh - oh0) * stride;
            int sh_ow = (ow - ow0) * stride;

            float sum = 0.0f;
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    float x = shmem[(sh_oh + kh) * shW + (sh_ow + kw)];
                    float w = weights.data[widx(oc, ic, kh, kw, input.C, K)];
                    sum += x * w;
                }
            }
            acc += sum;
        }

        __syncthreads();
    }

    if (oh < outH && ow < outW) {
        if (bias) acc += bias[oc];
        acc = relu(acc);
        output.data[idx4(n, oc, oh, ow, output.C, outH, outW)] = acc;
    }
}

void conv_forward_fused_bias_relu(
        const Tensor* d_input,
        const Tensor* d_weights,
        const float* d_bias,
        Tensor* d_output_relu,
        int pad,
        int stride,
        int blocksize,
        cudaStream_t s
){
    int K = d_weights->H;

    dim3 block(TILE_W, TILE_H, 1);

    int outH = d_output_relu->H;
    int outW = d_output_relu->W;

    int tilesW = (outW + TILE_W - 1) / TILE_W;
    int tilesH = (outH + TILE_H - 1) / TILE_H;
    int tilesPerImg = tilesW * tilesH;

    dim3 grid(d_weights->N, d_input->N * tilesPerImg, 1);

    int shH = TILE_H * stride + (K - 1);
    int shW = TILE_W * stride + (K - 1);
    size_t shBytes = (size_t)shH * shW * sizeof(float);

    conv_forward_tiled_fused_kernel<<<grid, block, shBytes ,s>>>(
            *d_input, *d_weights, d_bias, *d_output_relu, K, pad, stride
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("conv_forward_tiled_fused_kernel Launch Error: %s\n", cudaGetErrorString(err));
    }
}
