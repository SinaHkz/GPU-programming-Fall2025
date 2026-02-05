#pragma once

#include <vector>

#include "gpu/core/tensor.h"

namespace gpu {

// logits: [N, C], labels: host int[N]
// outputs:
// - loss_out: single float on host (average)
// - grad_logits: [N, C]
float softmax_cross_entropy_backward(const Tensor& logits, const std::vector<int>& labels, Tensor& grad_logits);
int argmax_accuracy(const Tensor& logits, const std::vector<int>& labels);

}  // namespace gpu

