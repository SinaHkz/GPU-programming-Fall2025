#pragma once

#include <cstddef>

#include "gpu/core/tensor.h"

namespace gpu {

// Returns sum(x[i]^2) on host (double).
double tensor_sumsq(const Tensor& x);

// Returns sqrt(sum(x[i]^2)) on host (double).
double tensor_l2_norm(const Tensor& x);

}  // namespace gpu

