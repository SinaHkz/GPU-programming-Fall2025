#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#pragma once
#include "tensor.h"

// ReLU forward: Y = max(0, X)
void relu_forward( Tensor& X, Tensor& Y, int blocksize,
                   cudaStream_t stream = 0);

// ReLU backward: dX = dY * (X > 0)
void relu_backward( Tensor& X,
 Tensor& dY,
Tensor& dX,
int blocksize,
cudaStream_t stream = 0
);

#endif