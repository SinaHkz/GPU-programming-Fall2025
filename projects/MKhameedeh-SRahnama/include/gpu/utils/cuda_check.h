#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace gpu {

inline void cuda_check(cudaError_t status, const char* expr, const char* file, int line) {
  if (status == cudaSuccess) return;
  std::fprintf(stderr, "CUDA error: %s (%d)\n  expr: %s\n  at: %s:%d\n",
               cudaGetErrorString(status), static_cast<int>(status), expr, file, line);
  std::fflush(stderr);
  std::abort();
}

}  // namespace gpu

#define GPU_CUDA_CHECK(expr) ::gpu::cuda_check((expr), #expr, __FILE__, __LINE__)

