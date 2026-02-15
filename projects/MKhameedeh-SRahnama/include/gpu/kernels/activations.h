#pragma once

#include "gpu/core/tensor.h"

namespace gpu {

void relu_forward(const Tensor& x, Tensor& y);
void relu_backward(const Tensor& x, const Tensor& grad_y, Tensor& grad_x);

}  // namespace gpu

