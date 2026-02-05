#include "gpu/model/model.h"

#include <stdexcept>

#include "gpu/utils/cuda_check.h"

namespace gpu {

void Sequential::add(std::unique_ptr<Layer> layer) { layers_.push_back(std::move(layer)); }

Tensor Sequential::forward(const Tensor& x) {
  if (layers_.empty()) {
    throw std::runtime_error("Sequential::forward called with no layers");
  }
  Tensor cur = layers_[0]->forward(x);
  for (size_t i = 1; i < layers_.size(); ++i) {
    cur = layers_[i]->forward(cur);
  }
  return cur;
}

Tensor Sequential::backward(const Tensor& grad_out) {
  if (layers_.empty()) {
    throw std::runtime_error("Sequential::backward called with no layers");
  }
  Tensor grad = layers_.back()->backward(grad_out);
  for (size_t i = layers_.size() - 1; i-- > 0;) {
    grad = layers_[i]->backward(grad);
  }
  return grad;
}

std::vector<Param> Sequential::params() const {
  std::vector<Param> ps;
  for (auto& layer : layers_) {
    const auto lp = layer->params();
    ps.insert(ps.end(), lp.begin(), lp.end());
  }
  return ps;
}

}  // namespace gpu
