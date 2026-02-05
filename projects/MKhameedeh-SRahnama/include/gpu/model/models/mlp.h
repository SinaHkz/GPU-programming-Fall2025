#pragma once

#include "gpu/model/model.h"
#include "gpu/model/model_base.h"

namespace gpu {

class MlpModel final : public Model {
 public:
  MlpModel(int in_c, int in_h, int in_w, int num_classes, int seed);
  std::string name() const override { return "mlp"; }

  Tensor forward(const Tensor& x) override { return seq_.forward(x); }
  Tensor backward(const Tensor& grad_logits) override { return seq_.backward(grad_logits); }
  std::vector<Param> params() const override { return seq_.params(); }

 private:
  Sequential seq_;
};

}  // namespace gpu
