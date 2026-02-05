#include "gpu/kernels/conv2d.h"

#include <cuda_runtime.h>

#include "gpu/core/profiler.h"
#include "gpu/utils/cuda_check.h"

namespace gpu {

__global__ void conv2d_fwd_kernel(const float* x, const float* w, const float* b, float* y, int n, int in_c, int in_h,
                                 int in_w, int out_c, int out_h, int out_w, int k_h, int k_w, int stride, int pad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n * out_c * out_h * out_w;
  if (idx >= total) return;

  const int ow = idx % out_w;
  const int oh = (idx / out_w) % out_h;
  const int oc = (idx / (out_w * out_h)) % out_c;
  const int nn = idx / (out_w * out_h * out_c);

  float acc = b ? b[oc] : 0.0f;
  for (int ic = 0; ic < in_c; ++ic) {
    for (int kh = 0; kh < k_h; ++kh) {
      const int ih = oh * stride + kh - pad;
      if (ih < 0 || ih >= in_h) continue;
      for (int kw = 0; kw < k_w; ++kw) {
        const int iw = ow * stride + kw - pad;
        if (iw < 0 || iw >= in_w) continue;
        const int x_idx = ((nn * in_c + ic) * in_h + ih) * in_w + iw;
        const int w_idx = (((oc * in_c + ic) * k_h + kh) * k_w + kw);
        acc += x[x_idx] * w[w_idx];
      }
    }
  }
  y[idx] = acc;
}

__global__ void conv2d_gradb_kernel(const float* grad_y, float* grad_b, int n, int out_c, int out_h, int out_w) {
  const int oc = blockIdx.x * blockDim.x + threadIdx.x;
  if (oc >= out_c) return;
  float acc = 0.0f;
  const int plane = out_h * out_w;
  for (int i = 0; i < n; ++i) {
    const float* gy = grad_y + (i * out_c + oc) * plane;
    for (int p = 0; p < plane; ++p) acc += gy[p];
  }
  grad_b[oc] = acc;
}

__global__ void conv2d_gradw_kernel(const float* x, const float* grad_y, float* grad_w, int n, int in_c, int in_h,
                                   int in_w, int out_c, int out_h, int out_w, int k_h, int k_w, int stride, int pad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = out_c * in_c * k_h * k_w;
  if (idx >= total) return;

  const int kw = idx % k_w;
  const int kh = (idx / k_w) % k_h;
  const int ic = (idx / (k_w * k_h)) % in_c;
  const int oc = idx / (k_w * k_h * in_c);

  float acc = 0.0f;
  for (int nn = 0; nn < n; ++nn) {
    for (int oh = 0; oh < out_h; ++oh) {
      const int ih = oh * stride + kh - pad;
      if (ih < 0 || ih >= in_h) continue;
      for (int ow = 0; ow < out_w; ++ow) {
        const int iw = ow * stride + kw - pad;
        if (iw < 0 || iw >= in_w) continue;
        const int x_idx = ((nn * in_c + ic) * in_h + ih) * in_w + iw;
        const int gy_idx = ((nn * out_c + oc) * out_h + oh) * out_w + ow;
        acc += x[x_idx] * grad_y[gy_idx];
      }
    }
  }
  grad_w[idx] = acc;
}

__global__ void conv2d_gradx_kernel(const float* grad_y, const float* w, float* grad_x, int n, int in_c, int in_h,
                                   int in_w, int out_c, int out_h, int out_w, int k_h, int k_w, int stride, int pad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n * in_c * in_h * in_w;
  if (idx >= total) return;

  const int iw = idx % in_w;
  const int ih = (idx / in_w) % in_h;
  const int ic = (idx / (in_w * in_h)) % in_c;
  const int nn = idx / (in_w * in_h * in_c);

  float acc = 0.0f;
  for (int oc = 0; oc < out_c; ++oc) {
    for (int kh = 0; kh < k_h; ++kh) {
      const int oh_num = ih + pad - kh;
      if (oh_num < 0) continue;
      if (oh_num % stride != 0) continue;
      const int oh = oh_num / stride;
      if (oh < 0 || oh >= out_h) continue;
      for (int kw = 0; kw < k_w; ++kw) {
        const int ow_num = iw + pad - kw;
        if (ow_num < 0) continue;
        if (ow_num % stride != 0) continue;
        const int ow = ow_num / stride;
        if (ow < 0 || ow >= out_w) continue;

        const int gy_idx = ((nn * out_c + oc) * out_h + oh) * out_w + ow;
        const int w_idx = (((oc * in_c + ic) * k_h + kh) * k_w + kw);
        acc += grad_y[gy_idx] * w[w_idx];
      }
    }
  }
  grad_x[idx] = acc;
}

void conv2d_forward_nchw(const Tensor& x, const Tensor& w, const Tensor& b, const Conv2DDesc& d, Tensor& y) {
  Profiler::ScopedGpuTimer t("conv2d_forward");
  const int n = x.dim(0);
  const int in_c = x.dim(1);
  const int in_h = x.dim(2);
  const int in_w = x.dim(3);
  const int out_c = d.out_c;
  const int out_h = y.dim(2);
  const int out_w = y.dim(3);
  const int total = n * out_c * out_h * out_w;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  conv2d_fwd_kernel<<<blocks, threads>>>(x.data(), w.data(), b.data(), y.data(), n, in_c, in_h, in_w, out_c, out_h,
                                        out_w, d.k_h, d.k_w, d.stride, d.pad);
  GPU_CUDA_CHECK(cudaGetLastError());
}

void conv2d_backward_nchw(const Tensor& x, const Tensor& w, const Conv2DDesc& d, const Tensor& grad_y, Tensor& grad_x,
                          Tensor& grad_w, Tensor& grad_b) {
  const int n = x.dim(0);
  const int in_c = x.dim(1);
  const int in_h = x.dim(2);
  const int in_w = x.dim(3);
  const int out_c = d.out_c;
  const int out_h = grad_y.dim(2);
  const int out_w = grad_y.dim(3);

  {
    Profiler::ScopedGpuTimer t("conv2d_backward_gradb");
    conv2d_gradb_kernel<<<(out_c + 255) / 256, 256>>>(grad_y.data(), grad_b.data(), n, out_c, out_h, out_w);
    GPU_CUDA_CHECK(cudaGetLastError());
  }
  {
    Profiler::ScopedGpuTimer t("conv2d_backward_gradw");
    const int total = out_c * in_c * d.k_h * d.k_w;
    conv2d_gradw_kernel<<<(total + 255) / 256, 256>>>(x.data(), grad_y.data(), grad_w.data(), n, in_c, in_h, in_w,
                                                      out_c, out_h, out_w, d.k_h, d.k_w, d.stride, d.pad);
    GPU_CUDA_CHECK(cudaGetLastError());
  }
  {
    Profiler::ScopedGpuTimer t("conv2d_backward_gradx");
    const int total = n * in_c * in_h * in_w;
    conv2d_gradx_kernel<<<(total + 255) / 256, 256>>>(grad_y.data(), w.data(), grad_x.data(), n, in_c, in_h, in_w,
                                                      out_c, out_h, out_w, d.k_h, d.k_w, d.stride, d.pad);
    GPU_CUDA_CHECK(cudaGetLastError());
  }
}

}  // namespace gpu
