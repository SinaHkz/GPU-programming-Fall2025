#pragma once

#include <random>
#include <string>

#include "gpu/kernels/conv2d.h"
#include "gpu/model/layer.h"

namespace gpu {

class Conv2D final : public Layer {
 public:
  Conv2D(int in_c, int out_c, int k, int stride, int pad, int seed, const std::string& name = "conv2d");

  std::string name() const override { return name_; }
  Tensor forward(const Tensor& x) override;
  Tensor backward(const Tensor& grad_out) override;
  std::vector<Param> params() const override;

 private:
  std::string name_;
  Conv2DDesc d_;
  Tensor w_;
  Tensor b_;
  Tensor grad_w_;
  Tensor grad_b_;
  Tensor x_cache_;
};

}  // namespace gpu
