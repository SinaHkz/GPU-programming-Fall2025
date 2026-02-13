#include "gpu/kernels/maxpool2d.h"

#include <cuda_runtime.h>

#include "gpu/core/profiler.h"
#include "gpu/utils/cuda_check.h"

namespace gpu {

IntTensor::IntTensor(size_t n) : n_(n) { alloc_(); }
IntTensor::~IntTensor() { free_(); }

IntTensor::IntTensor(IntTensor&& other) noexcept {
  n_ = other.n_;
  data_ = other.data_;
  other.n_ = 0;
  other.data_ = nullptr;
}

IntTensor& IntTensor::operator=(IntTensor&& other) noexcept {
  if (this == &other) return *this;
  free_();
  n_ = other.n_;
  data_ = other.data_;
  other.n_ = 0;
  other.data_ = nullptr;
  return *this;
}

void IntTensor::alloc_() {
  if (n_ == 0) return;
  GPU_CUDA_CHECK(cudaMalloc(&data_, n_ * sizeof(int)));
}

void IntTensor::free_() {
  if (!data_) return;
  GPU_CUDA_CHECK(cudaFree(data_));
  data_ = nullptr;
  n_ = 0;
}

void IntTensor::resize(size_t n) {
  if (n == n_) return;
  free_();
  n_ = n;
  alloc_();
}

__global__ void maxpool_fwd_kernel(const float* x, float* y, int* mask, int n, int c, int h, int w, int out_h,
                                  int out_w, int k, int stride) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n * c * out_h * out_w;
  if (idx >= total) return;

  const int ow = idx % out_w;
  const int oh = (idx / out_w) % out_h;
  const int cc = (idx / (out_w * out_h)) % c;
  const int nn = idx / (out_w * out_h * c);

  const int ih0 = oh * stride;
  const int iw0 = ow * stride;

  float best = -1e20f;
  int best_i = 0;
  for (int kh = 0; kh < k; ++kh) {
    for (int kw = 0; kw < k; ++kw) {
      const int ih = ih0 + kh;
      const int iw = iw0 + kw;
      const int in_index = ((nn * c + cc) * h + ih) * w + iw;
      const float v = x[in_index];
      if (v > best) {
        best = v;
        best_i = in_index;
      }
    }
  }
  y[idx] = best;
  mask[idx] = best_i;
}

__global__ void maxpool_bwd_kernel(const float* grad_y, const int* mask, float* grad_x, int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int in_index = mask[idx];
  atomicAdd(&grad_x[in_index], grad_y[idx]);
}

void maxpool2d_forward_nchw(const Tensor& x, const MaxPool2DDesc& d, Tensor& y, IntTensor& mask) {
  Profiler::ScopedGpuTimer t("maxpool2d_forward");
  const int n = x.dim(0);
  const int c = x.dim(1);
  const int h = x.dim(2);
  const int w = x.dim(3);
  const int out_h = y.dim(2);
  const int out_w = y.dim(3);
  const int total = n * c * out_h * out_w;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  Profiler::instance().record_kernel_launch("maxpool_fwd_kernel", (const void*)maxpool_fwd_kernel, dim3(blocks),
                                            dim3(threads));
  maxpool_fwd_kernel<<<blocks, threads>>>(x.data(), y.data(), mask.data(), n, c, h, w, out_h, out_w, d.k, d.stride);
  GPU_CUDA_CHECK(cudaGetLastError());
}

void maxpool2d_backward_nchw(const Tensor& x, const MaxPool2DDesc& d, const Tensor& grad_y, const IntTensor& mask,
                             Tensor& grad_x) {
  (void)x;
  (void)d;
  Profiler::ScopedGpuTimer t("maxpool2d_backward");
  const int total = static_cast<int>(grad_y.numel());
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  Profiler::instance().record_kernel_launch("maxpool_bwd_kernel", (const void*)maxpool_bwd_kernel, dim3(blocks),
                                            dim3(threads));
  maxpool_bwd_kernel<<<blocks, threads>>>(grad_y.data(), mask.data(), grad_x.data(), total);
  GPU_CUDA_CHECK(cudaGetLastError());
}

}  // namespace gpu
