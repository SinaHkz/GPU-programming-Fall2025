#include "gpu/kernels/activations.h"

#include "gpu/core/profiler.h"
#include "gpu/utils/cuda_check.h"

namespace gpu {

__global__ void relu_fwd_kernel(const float* x, float* y, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float v = x[i];
  y[i] = v > 0.0f ? v : 0.0f;
}

__global__ void relu_bwd_kernel(const float* x, const float* grad_y, float* grad_x, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  grad_x[i] = x[i] > 0.0f ? grad_y[i] : 0.0f;
}

void relu_forward(const Tensor& x, Tensor& y) {
  Profiler::ScopedGpuTimer t("relu_forward");
  const size_t n = x.numel();
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  Profiler::instance().record_kernel_launch("relu_fwd_kernel", (const void*)relu_fwd_kernel, dim3(blocks),
                                            dim3(threads));
  relu_fwd_kernel<<<blocks, threads>>>(x.data(), y.data(), n);
  GPU_CUDA_CHECK(cudaGetLastError());
}

void relu_backward(const Tensor& x, const Tensor& grad_y, Tensor& grad_x) {
  Profiler::ScopedGpuTimer t("relu_backward");
  const size_t n = x.numel();
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  Profiler::instance().record_kernel_launch("relu_bwd_kernel", (const void*)relu_bwd_kernel, dim3(blocks),
                                            dim3(threads));
  relu_bwd_kernel<<<blocks, threads>>>(x.data(), grad_y.data(), grad_x.data(), n);
  GPU_CUDA_CHECK(cudaGetLastError());
}

}  // namespace gpu
