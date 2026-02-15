#include <cuda_runtime.h>
#include "tensor.h"
#include <stdio.h>
#include "conv_layer.h"

__device__ __forceinline__ int idx4_nchw(int n,int c,int h,int w,int C,int H,int W){
    return ((n*C + c)*H + h)*W + w; //turn into flat indexing for tensor store
}
__device__ __forceinline__ int idx4_w(int co,int ci,int ky,int kx,int Cin,int K){
    return ((co*Cin + ci)*K + ky)*K + kx; // [Cout,Cin,K,K]
}

__global__ void conv_db_kernel(const float* __restrict__ dY, //one thread per output channel
                               float* __restrict__ db,
                               int B, int Cout, int Hout, int Wout)
{
    int co = blockIdx.x * blockDim.x + threadIdx.x;
    if (co >= Cout) return;

    float acc = 0.0f;
    for (int b = 0; b < B; ++b) {
        for (int ho = 0; ho < Hout; ++ho) {
            for (int wo = 0; wo < Wout; ++wo) {
                int dy_idx = idx4_nchw(b, co, ho, wo, Cout, Hout, Wout);
                acc += dY[dy_idx];
            }
        }
    }
    db[co] = acc;
}

__global__ void conv_dW_kernel(const float* __restrict__ X, //one thread per weight element
                               const float* __restrict__ dY,
                               float* __restrict__ dW,
                               int B, int Cin, int Hin, int Win,
                               int Cout, int Hout, int Wout,
                               int K, int stride, int pad)
{
    // Flatten (co,ci,ky,kx) into one index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Cout * Cin * K * K;
    if (tid >= total) return;

    int kx = tid % K; tid /= K;
    int ky = tid % K; tid /= K;
    int ci = tid % Cin; tid /= Cin;
    int co = tid;

    float acc = 0.0f;

    for (int b = 0; b < B; ++b) {
        for (int ho = 0; ho < Hout; ++ho) {
            int iy = ho * stride - pad + ky;
            if ((unsigned)iy >= (unsigned)Hin) continue;

            for (int wo = 0; wo < Wout; ++wo) {
                int ix = wo * stride - pad + kx;
                if ((unsigned)ix >= (unsigned)Win) continue;

                int x_idx  = idx4_nchw(b, ci, iy, ix, Cin, Hin, Win);
                int dy_idx = idx4_nchw(b, co, ho, wo, Cout, Hout, Wout);
                acc += dY[dy_idx] * X[x_idx];
            }
        }
    }

    int dw_idx = idx4_w(co, ci, ky, kx, Cin, K);
    dW[dw_idx] = acc;
}

__global__ void conv_dX_kernel(const float* __restrict__ dY, //one thread per input element
                               const float* __restrict__ W,
                               float* __restrict__ dX,
                               int B, int Cin, int Hin, int Win,
                               int Cout, int Hout, int Wout,
                               int K, int stride, int pad)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * Cin * Hin * Win;
    if (tid >= total) return;

    int ix = tid % Win; tid /= Win;
    int iy = tid % Hin; tid /= Hin;
    int ci = tid % Cin; tid /= Cin;
    int b  = tid;

    float acc = 0.0f;

    for (int co = 0; co < Cout; ++co) {
        for (int ky = 0; ky < K; ++ky) {
            int ho_num = iy + pad - ky;
            if (ho_num % stride != 0) continue;
            int ho = ho_num / stride;
            if ((unsigned)ho >= (unsigned)Hout) continue;

            for (int kx = 0; kx < K; ++kx) {
                int wo_num = ix + pad - kx;
                if (wo_num % stride != 0) continue;
                int wo = wo_num / stride;
                if ((unsigned)wo >= (unsigned)Wout) continue;

                int dy_idx = idx4_nchw(b, co, ho, wo, Cout, Hout, Wout);
                int w_idx  = idx4_w(co, ci, ky, kx, Cin, K);
                acc += dY[dy_idx] * W[w_idx];
            }
        }
    }

    int dx_idx = idx4_nchw(b, ci, iy, ix, Cin, Hin, Win);
    dX[dx_idx] = acc;
}

void conv_backward(const Tensor& X,
                   const Tensor& W,
                   const Tensor& dY,
                   Tensor& dX,
                   Tensor& dW,
                   Tensor& db,
                   int K, int stride, int pad,
                   int blocksize,
                   cudaStream_t s)
{
    int B   = X.N;
    int Cin = X.C;
    int Hin = X.H;
    int Win = X.W;

    int Cout = dY.C;
    int Hout = dY.H;
    int Wout = dY.W;

    int threads = blocksize;
    if (threads <= 0 || threads > 1024) threads = 256;

    // db
    {
        int blocks = (Cout + threads - 1) / threads;
        conv_db_kernel<<<blocks, threads , 0, s>>>(dY.data, db.data, B, Cout, Hout, Wout);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("conv_db_kernel Launch Error: %s\n", cudaGetErrorString(err));
    }

    // dW
    {
        int total = Cout * Cin * K * K;
        int blocks = (total + threads - 1) / threads;
        conv_dW_kernel<<<blocks, threads , 0, s>>>(X.data, dY.data, dW.data,
                                            B, Cin, Hin, Win,
                                            Cout, Hout, Wout,
                                            K, stride, pad);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("conv_dW_kernel Launch Error: %s\n", cudaGetErrorString(err));
    }

    // dX
    {
        int total = B * Cin * Hin * Win;
        int blocks = (total + threads - 1) / threads;
        conv_dX_kernel<<<blocks, threads , 0, s>>>(dY.data, W.data, dX.data,
                                            B, Cin, Hin, Win,
                                            Cout, Hout, Wout,
                                            K, stride, pad);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("conv_dX_kernel Launch Error: %s\n", cudaGetErrorString(err));
    }
}
