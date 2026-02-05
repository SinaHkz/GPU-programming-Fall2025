#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "gpu/utils/cuda_check.h"

namespace gpu {

// Simple float32 tensor on device (NCHW by convention for 4D).
class Tensor {
 public:
  Tensor() = default;
  Tensor(const std::vector<int>& shape, const std::string& name = "");
  ~Tensor();

  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  Tensor(Tensor&& other) noexcept;
  Tensor& operator=(Tensor&& other) noexcept;

  void resize(const std::vector<int>& shape);
  void zero_();

  float* data() { return data_; }
  const float* data() const { return data_; }

  const std::vector<int>& shape() const { return shape_; }
  int dim(int i) const { return shape_.at(static_cast<size_t>(i)); }
  size_t numel() const { return numel_; }
  size_t bytes() const { return numel_ * sizeof(float); }

  const std::string& name() const { return name_; }

  void copy_from_host(const float* host, size_t n);
  void copy_to_host(float* host, size_t n) const;

 private:
  void free_();
  void alloc_();

  std::vector<int> shape_;
  size_t numel_{0};
  float* data_{nullptr};
  std::string name_;
};

// Tracks device allocations performed by this project (not total CUDA allocations).
struct AllocatorStats {
  uint64_t current_bytes{0};
  uint64_t peak_bytes{0};
};

AllocatorStats allocator_stats();
void allocator_note_alloc(size_t bytes);
void allocator_note_free(size_t bytes);

}  // namespace gpu

