#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "gpu/core/profiler.h"
#include "gpu/utils/time.h"

namespace gpu {

inline const char* memcpy_kind_label(cudaMemcpyKind kind) {
  switch (kind) {
    case cudaMemcpyHostToDevice:
      return "memcpy_h2d";
    case cudaMemcpyDeviceToHost:
      return "memcpy_d2h";
    case cudaMemcpyDeviceToDevice:
      return "memcpy_d2d";
    case cudaMemcpyHostToHost:
      return "memcpy_h2h";
    default:
      return "memcpy_unknown";
  }
}

inline cudaError_t cudaMemcpyProfiled(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
  const double t0 = steady_seconds();
  cudaError_t err = cudaMemcpy(dst, src, bytes, kind);
  const double ms = (steady_seconds() - t0) * 1000.0;
  if (err == cudaSuccess) {
    Profiler::instance().record_ms_bytes(memcpy_kind_label(kind), ms, "memcpy", static_cast<uint64_t>(bytes));
  }
  return err;
}

}  // namespace gpu
