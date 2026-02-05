#pragma once

#include <string>
#include <vector>

#include "gpu/core/tensor.h"

namespace gpu {

struct Param {
  Tensor* w{nullptr};
  Tensor* grad{nullptr};
  std::string name;
};

class Layer {
 public:
  virtual ~Layer() = default;
  virtual std::string name() const = 0;

  virtual Tensor forward(const Tensor& x) = 0;
  virtual Tensor backward(const Tensor& grad_out) = 0;

  virtual std::vector<Param> params() const { return {}; }
};

}  // namespace gpu
