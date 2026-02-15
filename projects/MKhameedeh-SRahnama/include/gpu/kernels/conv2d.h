#pragma once

#include "gpu/core/tensor.h"

namespace gpu {

struct Conv2DDesc {
  int in_c{0};
  int out_c{0};
  int k_h{0};
  int k_w{0};
  int stride{1};
  int pad{0};
};

void conv2d_forward_nchw(const Tensor& x, const Tensor& w, const Tensor& b, const Conv2DDesc& d, Tensor& y);
void conv2d_backward_nchw(const Tensor& x, const Tensor& w, const Conv2DDesc& d, const Tensor& grad_y,
                          Tensor& grad_x, Tensor& grad_w, Tensor& grad_b);

}  // namespace gpu

