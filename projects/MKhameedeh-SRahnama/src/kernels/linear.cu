#include "gpu/kernels/linear.h"

#include "gpu/core/profiler.h"
#include "gpu/utils/cuda_check.h"

namespace gpu {

__global__ void linear_fwd_kernel(const float* x, const float* w, const float* b, float* y, int n, int in,
                                 int out) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;  // n
  const int col = blockIdx.x * blockDim.x + threadIdx.x;  // out
  if (row >= n || col >= out) return;

  float acc = b ? b[col] : 0.0f;
  const float* xrow = x + row * in;
  const float* wrow = w + col * in;
  for (int k = 0; k < in; ++k) acc += xrow[k] * wrow[k];
  y[row * out + col] = acc;
}

__global__ void linear_bwd_gradx_kernel(const float* grad_y, const float* w, float* grad_x, int n, int in, int out) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;  // n
  const int col = blockIdx.x * blockDim.x + threadIdx.x;  // in
  if (row >= n || col >= in) return;

  float acc = 0.0f;
  for (int o = 0; o < out; ++o) acc += grad_y[row * out + o] * w[o * in + col];
  grad_x[row * in + col] = acc;
}

__global__ void linear_bwd_gradw_kernel(const float* x, const float* grad_y, float* grad_w, int n, int in, int out) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;  // out
  const int col = blockIdx.x * blockDim.x + threadIdx.x;  // in
  if (row >= out || col >= in) return;

  float acc = 0.0f;
  for (int i = 0; i < n; ++i) acc += grad_y[i * out + row] * x[i * in + col];
  grad_w[row * in + col] = acc;
}

__global__ void linear_bwd_gradb_kernel(const float* grad_y, float* grad_b, int n, int out) {
  const int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= out) return;
  float acc = 0.0f;
  for (int i = 0; i < n; ++i) acc += grad_y[i * out + o];
  grad_b[o] = acc;
}

void linear_forward(const Tensor& x, const Tensor& w, const Tensor& b, Tensor& y) {
  Profiler::ScopedGpuTimer t("linear_forward");
  const int n = x.dim(0);
  const int in = x.dim(1);
  const int out = w.dim(0);
  dim3 threads(16, 16);
  dim3 blocks((out + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);
  linear_fwd_kernel<<<blocks, threads>>>(x.data(), w.data(), b.data(), y.data(), n, in, out);
  GPU_CUDA_CHECK(cudaGetLastError());
}

void linear_backward(const Tensor& x, const Tensor& w, const Tensor& grad_y, Tensor& grad_x, Tensor& grad_w,
                     Tensor& grad_b) {
  {
    Profiler::ScopedGpuTimer t("linear_backward_gradx");
    const int n = x.dim(0);
    const int in = x.dim(1);
    const int out = w.dim(0);
    dim3 threads(16, 16);
    dim3 blocks((in + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);
    linear_bwd_gradx_kernel<<<blocks, threads>>>(grad_y.data(), w.data(), grad_x.data(), n, in, out);
    GPU_CUDA_CHECK(cudaGetLastError());
  }
  {
    Profiler::ScopedGpuTimer t("linear_backward_gradw");
    const int n = x.dim(0);
    const int in = x.dim(1);
    const int out = w.dim(0);
    dim3 threads(16, 16);
    dim3 blocks((in + threads.x - 1) / threads.x, (out + threads.y - 1) / threads.y);
    linear_bwd_gradw_kernel<<<blocks, threads>>>(x.data(), grad_y.data(), grad_w.data(), n, in, out);
    GPU_CUDA_CHECK(cudaGetLastError());
    linear_bwd_gradb_kernel<<<(out + 255) / 256, 256>>>(grad_y.data(), grad_b.data(), n, out);
    GPU_CUDA_CHECK(cudaGetLastError());
  }
}

}  // namespace gpu
