#pragma once

#include <string>

#include "gpu/model/layer.h"

namespace gpu {

class Flatten final : public Layer {
 public:
  explicit Flatten(const std::string& name = "flatten") : name_(name) {}
  std::string name() const override { return name_; }

  Tensor forward(const Tensor& x) override;
  Tensor backward(const Tensor& grad_out) override;

 private:
  std::string name_;
  std::vector<int> in_shape_;
};

}  // namespace gpu

