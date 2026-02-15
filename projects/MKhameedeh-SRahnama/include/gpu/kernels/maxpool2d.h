#pragma once

#include "gpu/core/tensor.h"

namespace gpu {

struct MaxPool2DDesc {
  int k{2};
  int stride{2};
};

// mask stores argmax index per output element (int32).
class IntTensor {
 public:
  IntTensor() = default;
  explicit IntTensor(size_t n);
  ~IntTensor();
  IntTensor(const IntTensor&) = delete;
  IntTensor& operator=(const IntTensor&) = delete;
  IntTensor(IntTensor&&) noexcept;
  IntTensor& operator=(IntTensor&&) noexcept;

  int* data() { return data_; }
  const int* data() const { return data_; }
  size_t size() const { return n_; }

  void resize(size_t n);

 private:
  void free_();
  void alloc_();

  size_t n_{0};
  int* data_{nullptr};
};

void maxpool2d_forward_nchw(const Tensor& x, const MaxPool2DDesc& d, Tensor& y, IntTensor& mask);
void maxpool2d_backward_nchw(const Tensor& x, const MaxPool2DDesc& d, const Tensor& grad_y, const IntTensor& mask,
                             Tensor& grad_x);

}  // namespace gpu

