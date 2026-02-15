#pragma once
#include "tensor.h"

// param -= lr * grad
// if zero_grad = true, grad is set to 0 after update
void sgd_update_tensor(Tensor& param,
Tensor& grad,
float lr,
bool zero_grad,
int blocksize,
cudaStream_t stream = 0);
