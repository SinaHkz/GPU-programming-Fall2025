#include "gpu/layers/conv2d.h"
#include "gpu/layers/flatten.h"
#include "gpu/layers/linear.h"
#include "gpu/layers/maxpool2d.h"
#include "gpu/layers/relu.h"

#include <cmath>
#include <random>
#include <stdexcept>

#include "gpu/utils/cuda_check.h"

namespace gpu {

static void init_xavier_uniform(Tensor& w, int fan_in, int fan_out, int seed) {
  const float limit = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
  std::mt19937 rng(static_cast<uint32_t>(seed));
  std::uniform_real_distribution<float> u(-limit, limit);

  std::vector<float> host(w.numel());
  for (auto& v : host) v = u(rng);
  w.copy_from_host(host.data(), host.size());
}

// ---- Conv2D ----
Conv2D::Conv2D(int in_c, int out_c, int k, int stride, int pad, int seed, const std::string& name)
    : name_(name),
      d_{in_c, out_c, k, k, stride, pad},
      w_({out_c, in_c, k, k}, name + ".w"),
      b_({out_c}, name + ".b"),
      grad_w_({out_c, in_c, k, k}, name + ".grad_w"),
      grad_b_({out_c}, name + ".grad_b") {
  init_xavier_uniform(w_, in_c * k * k, out_c * k * k, seed);
  b_.zero_();
}

Tensor Conv2D::forward(const Tensor& x) {
  x_cache_.resize(x.shape());
  GPU_CUDA_CHECK(cudaMemcpy(x_cache_.data(), x.data(), x.bytes(), cudaMemcpyDeviceToDevice));

  const int n = x.dim(0);
  const int in_h = x.dim(2);
  const int in_w = x.dim(3);
  const int out_h = (in_h + 2 * d_.pad - d_.k_h) / d_.stride + 1;
  const int out_w = (in_w + 2 * d_.pad - d_.k_w) / d_.stride + 1;
  Tensor y({n, d_.out_c, out_h, out_w}, name_ + ".y");
  conv2d_forward_nchw(x, w_, b_, d_, y);
  return y;
}

Tensor Conv2D::backward(const Tensor& grad_out) {
  grad_w_.zero_();
  grad_b_.zero_();

  Tensor grad_x(x_cache_.shape(), name_ + ".grad_x");
  grad_x.zero_();
  conv2d_backward_nchw(x_cache_, w_, d_, grad_out, grad_x, grad_w_, grad_b_);
  return grad_x;
}

std::vector<Param> Conv2D::params() const {
  std::vector<Param> ps;
  ps.push_back(Param{const_cast<Tensor*>(&w_), const_cast<Tensor*>(&grad_w_), name_ + ".w"});
  ps.push_back(Param{const_cast<Tensor*>(&b_), const_cast<Tensor*>(&grad_b_), name_ + ".b"});
  return ps;
}

// ---- ReLU ----
Tensor ReLU::forward(const Tensor& x) {
  x_cache_.resize(x.shape());
  GPU_CUDA_CHECK(cudaMemcpy(x_cache_.data(), x.data(), x.bytes(), cudaMemcpyDeviceToDevice));
  Tensor y(x.shape(), name_ + ".y");
  relu_forward(x, y);
  return y;
}

Tensor ReLU::backward(const Tensor& grad_out) {
  Tensor grad_x(x_cache_.shape(), name_ + ".grad_x");
  relu_backward(x_cache_, grad_out, grad_x);
  return grad_x;
}

// ---- MaxPool2D ----
MaxPool2D::MaxPool2D(int k, int stride, const std::string& name) : name_(name), d_{k, stride} {}

Tensor MaxPool2D::forward(const Tensor& x) {
  x_cache_.resize(x.shape());
  GPU_CUDA_CHECK(cudaMemcpy(x_cache_.data(), x.data(), x.bytes(), cudaMemcpyDeviceToDevice));

  const int n = x.dim(0);
  const int c = x.dim(1);
  const int h = x.dim(2);
  const int w = x.dim(3);
  const int out_h = (h - d_.k) / d_.stride + 1;
  const int out_w = (w - d_.k) / d_.stride + 1;
  Tensor y({n, c, out_h, out_w}, name_ + ".y");
  mask_.resize(y.numel());
  maxpool2d_forward_nchw(x, d_, y, mask_);
  return y;
}

Tensor MaxPool2D::backward(const Tensor& grad_out) {
  Tensor grad_x(x_cache_.shape(), name_ + ".grad_x");
  grad_x.zero_();
  maxpool2d_backward_nchw(x_cache_, d_, grad_out, mask_, grad_x);
  return grad_x;
}

// ---- Flatten ----
Tensor Flatten::forward(const Tensor& x) {
  in_shape_ = x.shape();
  if (in_shape_.size() != 4) throw std::runtime_error("Flatten expects 4D NCHW input");
  const int n = in_shape_[0];
  const int c = in_shape_[1];
  const int h = in_shape_[2];
  const int w = in_shape_[3];
  Tensor y({n, c * h * w}, name_ + ".y");
  GPU_CUDA_CHECK(cudaMemcpy(y.data(), x.data(), x.bytes(), cudaMemcpyDeviceToDevice));
  return y;
}

Tensor Flatten::backward(const Tensor& grad_out) {
  Tensor grad_x(in_shape_, name_ + ".grad_x");
  GPU_CUDA_CHECK(cudaMemcpy(grad_x.data(), grad_out.data(), grad_out.bytes(), cudaMemcpyDeviceToDevice));
  return grad_x;
}

// ---- Linear ----
Linear::Linear(int in_features, int out_features, int seed, const std::string& name)
    : name_(name),
      in_(in_features),
      out_(out_features),
      w_({out_features, in_features}, name + ".w"),
      b_({out_features}, name + ".b"),
      grad_w_({out_features, in_features}, name + ".grad_w"),
      grad_b_({out_features}, name + ".grad_b") {
  init_xavier_uniform(w_, in_features, out_features, seed);
  b_.zero_();
}

Tensor Linear::forward(const Tensor& x) {
  x_cache_.resize(x.shape());
  GPU_CUDA_CHECK(cudaMemcpy(x_cache_.data(), x.data(), x.bytes(), cudaMemcpyDeviceToDevice));
  Tensor y({x.dim(0), out_}, name_ + ".y");
  linear_forward(x, w_, b_, y);
  return y;
}

Tensor Linear::backward(const Tensor& grad_out) {
  grad_w_.zero_();
  grad_b_.zero_();
  Tensor grad_x({grad_out.dim(0), in_}, name_ + ".grad_x");
  grad_x.zero_();
  linear_backward(x_cache_, w_, grad_out, grad_x, grad_w_, grad_b_);
  return grad_x;
}

std::vector<Param> Linear::params() const {
  std::vector<Param> ps;
  ps.push_back(Param{const_cast<Tensor*>(&w_), const_cast<Tensor*>(&grad_w_), name_ + ".w"});
  ps.push_back(Param{const_cast<Tensor*>(&b_), const_cast<Tensor*>(&grad_b_), name_ + ".b"});
  return ps;
}

}  // namespace gpu
