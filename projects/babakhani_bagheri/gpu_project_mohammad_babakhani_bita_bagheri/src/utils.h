#pragma once

#include <cuda_runtime.h>

void gpuAssert(cudaError_t code, const char *file, int line);
void checkCudaKernel(const char *msg);
int divUp(int a, int b);

#define CUDA_CHECK(x) gpuAssert((x), __FILE__, __LINE__)
#define CUDA_CHECK_LAST(msg) checkCudaKernel((msg))
