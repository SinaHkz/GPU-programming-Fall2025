#pragma once

#include "gpu/core/tensor.h"

namespace gpu {

void sgd_step(Tensor& w, const Tensor& grad_w, float lr, float weight_decay);

}  // namespace gpu

