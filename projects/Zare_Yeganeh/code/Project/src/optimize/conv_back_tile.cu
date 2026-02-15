#include <cuda_runtime.h>
#include <cstdio>
#include "tensor.h"
#include "conv_backward.h"

__device__ __forceinline__ int idx4_nchw(int n,int c,int h,int w,int C,int H,int W){
    return ((n*C + c)*H + h)*W + w;
}
__device__ __forceinline__ int idx4_w(int co,int ci,int ky,int kx,int Cin,int K){
    return ((co*Cin + ci)*K + ky)*K + kx;
}

// V1 BASELINE kernels

__global__ void conv_db_kernel_v1(
        const float* __restrict__ dY,
        float* __restrict__ db,
        int B, int Cout, int Hout, int Wout
){
    int co = blockIdx.x * blockDim.x + threadIdx.x;
    if (co >= Cout) return;

    float acc = 0.0f;
    for (int b = 0; b < B; ++b)
        for (int ho = 0; ho < Hout; ++ho)
            for (int wo = 0; wo < Wout; ++wo)
                acc += dY[idx4_nchw(b, co, ho, wo, Cout, Hout, Wout)];

    db[co] = acc;
}

__global__ void conv_dW_kernel_v1(
        const float* __restrict__ X,
        const float* __restrict__ dY,
        float* __restrict__ dW,
        int B, int Cin, int Hin, int Win,
        int Cout, int Hout, int Wout,
        int K, int stride, int pad
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Cout * Cin * K * K;
    if (tid >= total) return;

    int t = tid;
    int kx = t % K; t /= K;
    int ky = t % K; t /= K;
    int ci = t % Cin; t /= Cin;
    int co = t;

    float acc = 0.0f;
    for (int b = 0; b < B; ++b) {
        for (int ho = 0; ho < Hout; ++ho) {
            int iy = ho * stride - pad + ky;
            if ((unsigned)iy >= (unsigned)Hin) continue;

            for (int wo = 0; wo < Wout; ++wo) {
                int ix = wo * stride - pad + kx;
                if ((unsigned)ix >= (unsigned)Win) continue;

                float x  = X[idx4_nchw(b, ci, iy, ix, Cin, Hin, Win)];
                float dy = dY[idx4_nchw(b, co, ho, wo, Cout, Hout, Wout)];
                acc += dy * x;
            }
        }
    }

    dW[idx4_w(co, ci, ky, kx, Cin, K)] = acc;
}

__global__ void conv_dX_kernel_v1(
        const float* __restrict__ dY,
        const float* __restrict__ W,
        float* __restrict__ dX,
        int B, int Cin, int Hin, int Win,
        int Cout, int Hout, int Wout,
        int K, int stride, int pad
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * Cin * Hin * Win;
    if (tid >= total) return;

    int t = tid;
    int ix = t % Win; t /= Win;
    int iy = t % Hin; t /= Hin;
    int ci = t % Cin; t /= Cin;
    int b  = t;

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

                float dy = dY[idx4_nchw(b, co, ho, wo, Cout, Hout, Wout)];
                float w  = W[idx4_w(co, ci, ky, kx, Cin, K)];
                acc += dy * w;
            }
        }
    }

    dX[idx4_nchw(b, ci, iy, ix, Cin, Hin, Win)] = acc;
}

// V2 OPTIMIZED: db reduction + dW partial reduction
// dX kept as v1 baseline

__global__ void conv_db_reduce_kernel_v2(
        const float* __restrict__ dY,
        float* __restrict__ db,
        int B, int Cout, int Hout, int Wout
){
    int co = blockIdx.x;              // one block per output channel
    int tid = threadIdx.x;

    float sum = 0.0f;
    int total = B * Hout * Wout;

    for (int idx = tid; idx < total; idx += blockDim.x) {
        int wo = idx % Wout; int t = idx / Wout;
        int ho = t % Hout;   int b = t / Hout;
        sum += dY[idx4_nchw(b, co, ho, wo, Cout, Hout, Wout)];
    }

    extern __shared__ float smem[];
    smem[tid] = sum;
    __syncthreads();

    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) smem[tid] += smem[tid + off];
        __syncthreads();
    }

    if (tid == 0) db[co] = smem[0];
}

// Specialized for K=3
__global__ void conv_dW_partial_kernel_v2_K3(
        const float* __restrict__ X,
        const float* __restrict__ dY,
        float* __restrict__ dW,
        int B, int Cin, int Hin, int Win,
        int Cout, int Hout, int Wout,
        int stride, int pad
){
    int co = blockIdx.x;
    int ci = blockIdx.y;
    int tid = threadIdx.x;

    // register partial sums for 3x3
    float acc0=0,acc1=0,acc2=0,acc3=0,acc4=0,acc5=0,acc6=0,acc7=0,acc8=0;

    int total = B * Hout * Wout;
    for (int idx = tid; idx < total; idx += blockDim.x) {
        int wo = idx % Wout; int t = idx / Wout;
        int ho = t % Hout;   int b = t / Hout;

        float dy = dY[idx4_nchw(b, co, ho, wo, Cout, Hout, Wout)];

        int iy0 = ho * stride - pad;
        int ix0 = wo * stride - pad;

        // unrolled
        int iy, ix;
        iy = iy0 + 0; ix = ix0 + 0; if ((unsigned)iy<(unsigned)Hin && (unsigned)ix<(unsigned)Win) acc0 += dy * X[idx4_nchw(b,ci,iy,ix,Cin,Hin,Win)];
        iy = iy0 + 0; ix = ix0 + 1; if ((unsigned)iy<(unsigned)Hin && (unsigned)ix<(unsigned)Win) acc1 += dy * X[idx4_nchw(b,ci,iy,ix,Cin,Hin,Win)];
        iy = iy0 + 0; ix = ix0 + 2; if ((unsigned)iy<(unsigned)Hin && (unsigned)ix<(unsigned)Win) acc2 += dy * X[idx4_nchw(b,ci,iy,ix,Cin,Hin,Win)];
        iy = iy0 + 1; ix = ix0 + 0; if ((unsigned)iy<(unsigned)Hin && (unsigned)ix<(unsigned)Win) acc3 += dy * X[idx4_nchw(b,ci,iy,ix,Cin,Hin,Win)];
        iy = iy0 + 1; ix = ix0 + 1; if ((unsigned)iy<(unsigned)Hin && (unsigned)ix<(unsigned)Win) acc4 += dy * X[idx4_nchw(b,ci,iy,ix,Cin,Hin,Win)];
        iy = iy0 + 1; ix = ix0 + 2; if ((unsigned)iy<(unsigned)Hin && (unsigned)ix<(unsigned)Win) acc5 += dy * X[idx4_nchw(b,ci,iy,ix,Cin,Hin,Win)];
        iy = iy0 + 2; ix = ix0 + 0; if ((unsigned)iy<(unsigned)Hin && (unsigned)ix<(unsigned)Win) acc6 += dy * X[idx4_nchw(b,ci,iy,ix,Cin,Hin,Win)];
        iy = iy0 + 2; ix = ix0 + 1; if ((unsigned)iy<(unsigned)Hin && (unsigned)ix<(unsigned)Win) acc7 += dy * X[idx4_nchw(b,ci,iy,ix,Cin,Hin,Win)];
        iy = iy0 + 2; ix = ix0 + 2; if ((unsigned)iy<(unsigned)Hin && (unsigned)ix<(unsigned)Win) acc8 += dy * X[idx4_nchw(b,ci,iy,ix,Cin,Hin,Win)];
    }

    // reduce across threads in shared memory
    extern __shared__ float smem[];
    float* s0 = smem + 0*blockDim.x;
    float* s1 = smem + 1*blockDim.x;
    float* s2 = smem + 2*blockDim.x;
    float* s3 = smem + 3*blockDim.x;
    float* s4 = smem + 4*blockDim.x;
    float* s5 = smem + 5*blockDim.x;
    float* s6 = smem + 6*blockDim.x;
    float* s7 = smem + 7*blockDim.x;
    float* s8 = smem + 8*blockDim.x;

    s0[tid]=acc0; s1[tid]=acc1; s2[tid]=acc2; s3[tid]=acc3; s4[tid]=acc4;
    s5[tid]=acc5; s6[tid]=acc6; s7[tid]=acc7; s8[tid]=acc8;
    __syncthreads();

    for (int off = blockDim.x/2; off>0; off>>=1) {
        if (tid < off) {
            s0[tid]+=s0[tid+off]; s1[tid]+=s1[tid+off]; s2[tid]+=s2[tid+off];
            s3[tid]+=s3[tid+off]; s4[tid]+=s4[tid+off]; s5[tid]+=s5[tid+off];
            s6[tid]+=s6[tid+off]; s7[tid]+=s7[tid+off]; s8[tid]+=s8[tid+off];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&dW[idx4_w(co,ci,0,0,Cin,3)], s0[0]);
        atomicAdd(&dW[idx4_w(co,ci,0,1,Cin,3)], s1[0]);
        atomicAdd(&dW[idx4_w(co,ci,0,2,Cin,3)], s2[0]);
        atomicAdd(&dW[idx4_w(co,ci,1,0,Cin,3)], s3[0]);
        atomicAdd(&dW[idx4_w(co,ci,1,1,Cin,3)], s4[0]);
        atomicAdd(&dW[idx4_w(co,ci,1,2,Cin,3)], s5[0]);
        atomicAdd(&dW[idx4_w(co,ci,2,0,Cin,3)], s6[0]);
        atomicAdd(&dW[idx4_w(co,ci,2,1,Cin,3)], s7[0]);
        atomicAdd(&dW[idx4_w(co,ci,2,2,Cin,3)], s8[0]);
    }
}

static inline int clamp_threads(int bs) {
    if (bs <= 0 || bs > 1024) return 256;
    return bs;
}

void conv_backward(
        const Tensor& X,
        const Tensor& W,
        const Tensor& dY,
        Tensor& dX,
        Tensor& dW,
        Tensor& db,
        int K, int stride, int pad,
        int blocksize,
        cudaStream_t stream,
        ConvBwdVariant variant
){
    int B   = X.N;
    int Cin = X.C;
    int Hin = X.H;
    int Win = X.W;

    int Cout = dY.C;
    int Hout = dY.H;
    int Wout = dY.W;

    int threads = clamp_threads(blocksize);

    if (variant == CONV_BWD_V2_DW_DB_OPT) {
        {
            int t = threads;
            size_t sh = (size_t)t * sizeof(float);
            conv_db_reduce_kernel_v2<<<Cout, t, sh, stream>>>(dY.data, db.data, B, Cout, Hout, Wout);
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("conv_db_reduce_kernel_v2 Launch Error: %s\n", cudaGetErrorString(err));
        }

        if (K == 3) {
            dim3 grid(Cout, Cin, 1);
            int t = threads;
            size_t sh = (size_t)t * 9 * sizeof(float);
            conv_dW_partial_kernel_v2_K3<<<grid, t, sh, stream>>>(
                    X.data, dY.data, dW.data,
                    B, Cin, Hin, Win,
                    Cout, Hout, Wout,
                    stride, pad
            );
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("conv_dW_partial_kernel_v2_K3 Launch Error: %s\n", cudaGetErrorString(err));
            }
        } else {
            // fallback to v1 dW if K != 3
            int total = Cout * Cin * K * K;
            int blocks = (total + threads - 1) / threads;
            conv_dW_kernel_v1<<<blocks, threads, 0, stream>>>(
                    X.data, dY.data, dW.data,
                    B, Cin, Hin, Win,
                    Cout, Hout, Wout,
                    K, stride, pad
            );
        }

        // dX baseline
        {
            int total = B * Cin * Hin * Win;
            int blocks = (total + threads - 1) / threads;
            conv_dX_kernel_v1<<<blocks, threads, 0, stream>>>(
                    dY.data, W.data, dX.data,
                    B, Cin, Hin, Win,
                    Cout, Hout, Wout,
                    K, stride, pad
            );
        }
        cudaError_t err2 = cudaGetLastError();
        if (err2 != cudaSuccess) {
            printf("conv_dX_kernel_v1 Launch Error: %s\n", cudaGetErrorString(err2));
        }

    } else {
        // V1 baseline
        {
            int blocks = (Cout + threads - 1) / threads;
            conv_db_kernel_v1<<<blocks, threads, 0, stream>>>(dY.data, db.data, B, Cout, Hout, Wout);
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("conv_db_kernel_v1 Launch Error: %s\n", cudaGetErrorString(err));
        }

        {
            int total = Cout * Cin * K * K;
            int blocks = (total + threads - 1) / threads;
            conv_dW_kernel_v1<<<blocks, threads, 0, stream>>>(
                    X.data, dY.data, dW.data,
                    B, Cin, Hin, Win,
                    Cout, Hout, Wout,
                    K, stride, pad
            );
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("conv_dW_kernel_v1 Launch Error: %s\n", cudaGetErrorString(err));
        }

        {
            int total = B * Cin * Hin * Win;
            int blocks = (total + threads - 1) / threads;
            conv_dX_kernel_v1<<<blocks, threads, 0, stream>>>(
                    dY.data, W.data, dX.data,
                    B, Cin, Hin, Win,
                    Cout, Hout, Wout,
                    K, stride, pad
            );
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("conv_dX_kernel_v1 Launch Error: %s\n", cudaGetErrorString(err));
        }
    }
}
