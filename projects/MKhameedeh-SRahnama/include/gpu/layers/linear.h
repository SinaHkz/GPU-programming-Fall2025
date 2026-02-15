#pragma once

#include <random>
#include <string>

#include "gpu/kernels/linear.h"
#include "gpu/model/layer.h"

namespace gpu {

class Linear final : public Layer {
 public:
  Linear(int in_features, int out_features, int seed, const std::string& name = "linear");

  std::string name() const override { return name_; }
  Tensor forward(const Tensor& x) override;
  Tensor backward(const Tensor& grad_out) override;
  std::vector<Param> params() const override;

 private:
  std::string name_;
  int in_{0};
  int out_{0};
  Tensor w_;       // [out, in]
  Tensor b_;       // [out]
  Tensor grad_w_;  // [out, in]
  Tensor grad_b_;  // [out]
  Tensor x_cache_;
};

}  // namespace gpu
