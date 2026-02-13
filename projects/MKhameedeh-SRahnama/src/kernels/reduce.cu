#include "gpu/kernels/reduce.h"

#include <cmath>
#include <vector>

#include "gpu/core/profiler.h"
#include "gpu/utils/cuda_check.h"
#include "gpu/utils/cuda_profiled.h"

namespace gpu {

namespace {

struct ReduceTmp {
  float* partial{nullptr};
  size_t partial_cap{0};

  ~ReduceTmp() {
    if (partial) cudaFree(partial);
  }

  float* ensure_partial(size_t n) {
    if (n <= partial_cap) return partial;
    if (partial) cudaFree(partial);
    GPU_CUDA_CHECK(cudaMalloc(&partial, n * sizeof(float)));
    partial_cap = n;
    return partial;
  }
};

ReduceTmp& tmp() {
  static ReduceTmp t;
  return t;
}

__global__ void sumsq_partial_kernel(const float* x, float* partial, size_t n) {
  const int tid = threadIdx.x;
  const int block = blockIdx.x;
  const int threads = blockDim.x;
  const size_t base = static_cast<size_t>(block) * static_cast<size_t>(threads) * 4;

  float acc = 0.0f;
  #pragma unroll
  for (int k = 0; k < 4; ++k) {
    const size_t i = base + static_cast<size_t>(tid) + static_cast<size_t>(k) * static_cast<size_t>(threads);
    if (i < n) {
      const float v = x[i];
      acc += v * v;
    }
  }

  __shared__ float smem[256];
  smem[tid] = acc;
  __syncthreads();

  for (int offset = threads / 2; offset > 0; offset /= 2) {
    if (tid < offset) smem[tid] += smem[tid + offset];
    __syncthreads();
  }
  if (tid == 0) partial[block] = smem[0];
}

}  // namespace

double tensor_sumsq(const Tensor& x) {
  Profiler::ScopedGpuTimer t("reduce_sumsq");
  const size_t n = x.numel();
  if (n == 0) return 0.0;
  const int threads = 256;
  const int blocks = static_cast<int>((n + static_cast<size_t>(threads) * 4 - 1) / (static_cast<size_t>(threads) * 4));
  float* d_partial = tmp().ensure_partial(static_cast<size_t>(blocks));
  Profiler::instance().record_kernel_launch("sumsq_partial_kernel", (const void*)sumsq_partial_kernel, dim3(blocks),
                                            dim3(threads));
  sumsq_partial_kernel<<<blocks, threads>>>(x.data(), d_partial, n);
  GPU_CUDA_CHECK(cudaGetLastError());

  std::vector<float> host(static_cast<size_t>(blocks));
  GPU_CUDA_CHECK(
      cudaMemcpyProfiled(host.data(), d_partial, host.size() * sizeof(float), cudaMemcpyDeviceToHost));
  double sum = 0.0;
  for (float v : host) sum += static_cast<double>(v);
  return sum;
}

double tensor_l2_norm(const Tensor& x) { return std::sqrt(tensor_sumsq(x)); }

}  // namespace gpu
