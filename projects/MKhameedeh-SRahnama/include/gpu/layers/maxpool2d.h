#pragma once

#include <string>

#include "gpu/kernels/maxpool2d.h"
#include "gpu/model/layer.h"

namespace gpu {

class MaxPool2D final : public Layer {
 public:
  MaxPool2D(int k = 2, int stride = 2, const std::string& name = "maxpool2d");

  std::string name() const override { return name_; }
  Tensor forward(const Tensor& x) override;
  Tensor backward(const Tensor& grad_out) override;

 private:
  std::string name_;
  MaxPool2DDesc d_;
  Tensor x_cache_;
  IntTensor mask_;
};

}  // namespace gpu

