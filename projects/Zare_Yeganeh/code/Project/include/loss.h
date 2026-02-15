#ifndef LOSS_H
#define LOSS_H

// include/loss.h
#pragma once

void compute_softmax_loss_gradient(
        const float* d_logits,
        const int* d_labels,
        float* d_dlogits,
        float* d_loss,
        int B, int K,
        int blocksize,
        cudaStream_t stream = 0
);


#endif