// src/loss.cu
#include <cuda_runtime.h>
#include <cstdio>
#include "loss.h"

// logits: (B, K)
// labels: (B)
// dlogits: (B, K)  output gradient
// loss: device scalar (sum of losses)

__global__ void softmax_ce_loss_grad_kernel(
        const float* __restrict__ logits,
        const int* __restrict__ labels,
        float* __restrict__ dlogits,
        float* __restrict__ loss,
        int B, int K
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    int y = labels[b];

    // Defensive check: prevent illegal access if label is bad
    if (y < 0 || y >= K) {
        for (int j = 0; j < K; ++j) dlogits[b*K + j] = 0.0f;
        return;
    }


    // Find max logit for stability
    float m = logits[b * K + 0];
    for (int j = 1; j < K; ++j) {
        float v = logits[b * K + j];
        if (v > m) m = v;
    }

    // Compute sum exp
    float sum = 0.0f;
    for (int j = 0; j < K; ++j) {
        sum += expf(logits[b * K + j] - m);
    }

    // Compute softmax + gradient
    float inv = 1.0f / sum;
    float p_y = 0.0f;

    for (int j = 0; j < K; ++j) {
        float p = expf(logits[b * K + j] - m) * inv;
        dlogits[b * K + j] = p;     // store p first
        if (j == y) p_y = p;
    }

    // gradient: p - onehot
    dlogits[b * K + y] -= 1.0f;

    // loss: -log(p_y)
    // avoid log(0)
    float eps = 1e-12f;
    float l = -logf(fmaxf(p_y, eps));
    atomicAdd(loss, l);
}

void compute_softmax_loss_gradient(
        const float* d_logits,
        const int* d_labels,
        float* d_dlogits,
        float* d_loss,
        int B, int K,
        int blocksize,
        cudaStream_t s
) {
    int threads = blocksize;
    if (threads <= 0 || threads > 1024) threads = 256;
    int blocks = (B + threads - 1) / threads;

    softmax_ce_loss_grad_kernel<<<blocks, threads , 0, s>>>(
            d_logits, d_labels, d_dlogits, d_loss, B, K
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("softmax_ce_loss_grad_kernel Launch Error: %s\n",
               cudaGetErrorString(err));
    }
}
