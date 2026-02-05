#pragma once

#include <string>

#include "gpu/kernels/activations.h"
#include "gpu/model/layer.h"

namespace gpu {

class ReLU final : public Layer {
 public:
  explicit ReLU(const std::string& name = "relu") : name_(name) {}
  std::string name() const override { return name_; }
  Tensor forward(const Tensor& x) override;
  Tensor backward(const Tensor& grad_out) override;

 private:
  std::string name_;
  Tensor x_cache_;
};

}  // namespace gpu

