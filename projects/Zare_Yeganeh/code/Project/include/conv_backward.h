#pragma once
#include <cuda_runtime.h>
#include "tensor.h"

// Backward variants
enum ConvBwdVariant {
    CONV_BWD_V1_BASELINE = 0,
    CONV_BWD_V2_DW_DB_OPT = 1
};

// Main API (streamed)
void conv_backward(
        const Tensor& X,   // [B,Cin,Hin,Win]
        const Tensor& W,   // [Cout,Cin,K,K] as Tensor N=Cout,C=Cin,H=K,W=K
        const Tensor& dY,  // [B,Cout,Hout,Wout]
        Tensor& dX,        // [B,Cin,Hin,Win]
        Tensor& dW,        // [Cout,Cin,K,K]
        Tensor& db,        // [Cout] stored as contiguous float array (Tensor data)
        int K, int stride, int pad,
        int blocksize,
        cudaStream_t stream,
        ConvBwdVariant variant = CONV_BWD_V2_DW_DB_OPT
);
