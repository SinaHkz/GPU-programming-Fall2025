#pragma once

#include <memory>
#include <string>
#include <vector>

#include "gpu/model/layer.h"

namespace gpu {

class Sequential {
 public:
  void add(std::unique_ptr<Layer> layer);

  Tensor forward(const Tensor& x);
  Tensor backward(const Tensor& grad_out);

  std::vector<Param> params() const;

 private:
  std::vector<std::unique_ptr<Layer>> layers_;
};

}  // namespace gpu
