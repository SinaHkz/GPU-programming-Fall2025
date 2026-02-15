#pragma once

#include <string>
#include <vector>

#include "gpu/core/tensor.h"
#include "gpu/model/layer.h"

namespace gpu {

class Model {
 public:
  virtual ~Model() = default;
  virtual std::string name() const = 0;

  // x is typically NCHW for image models; logits is [N, num_classes].
  virtual Tensor forward(const Tensor& x) = 0;
  virtual Tensor backward(const Tensor& grad_logits) = 0;
  virtual std::vector<Param> params() const = 0;
};

}  // namespace gpu
