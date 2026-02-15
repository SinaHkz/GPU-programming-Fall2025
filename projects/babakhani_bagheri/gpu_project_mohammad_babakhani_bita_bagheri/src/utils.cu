#include "utils.h"

#include <cstdio>
#include <cstdlib>

void gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(code), file, line);
    std::exit(static_cast<int>(code));
  }
}

void checkCudaKernel(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA kernel error (%s): %s\n", msg, cudaGetErrorString(err));
    std::exit(static_cast<int>(err));
  }
}

int divUp(int a, int b) {
  return (a + b - 1) / b;
}
