#pragma once

#include "gpu/core/tensor.h"

namespace gpu {

void linear_forward(const Tensor& x, const Tensor& w, const Tensor& b, Tensor& y);  // y = x*w^T + b
void linear_backward(const Tensor& x, const Tensor& w, const Tensor& grad_y, Tensor& grad_x, Tensor& grad_w,
                     Tensor& grad_b);

}  // namespace gpu

