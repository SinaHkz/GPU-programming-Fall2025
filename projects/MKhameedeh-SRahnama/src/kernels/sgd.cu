#include "gpu/kernels/sgd.h"

namespace gpu {

__global__ void sgd_kernel(float* w, const float* g, size_t n, float lr, float wd) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float gi = g[i] + wd * w[i];
  w[i] -= lr * gi;
}

void sgd_step(Tensor& w, const Tensor& grad_w, float lr, float weight_decay) {
  const size_t n = w.numel();
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  Profiler::instance().record_kernel_launch("sgd_kernel", (const void*)sgd_kernel, dim3(blocks), dim3(threads));
  sgd_kernel<<<blocks, threads>>>(w.data(), grad_w.data(), n, lr, weight_decay);
}

}  // namespace gpu
