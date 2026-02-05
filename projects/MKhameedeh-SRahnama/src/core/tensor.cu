#include "gpu/core/tensor.h"

#include <atomic>
#include <numeric>

namespace gpu {

static std::atomic<uint64_t> g_current_bytes{0};
static std::atomic<uint64_t> g_peak_bytes{0};

AllocatorStats allocator_stats() {
  AllocatorStats s;
  s.current_bytes = g_current_bytes.load();
  s.peak_bytes = g_peak_bytes.load();
  return s;
}

void allocator_note_alloc(size_t bytes) {
  const uint64_t after = g_current_bytes.fetch_add(static_cast<uint64_t>(bytes)) + static_cast<uint64_t>(bytes);
  uint64_t prev_peak = g_peak_bytes.load();
  while (after > prev_peak && !g_peak_bytes.compare_exchange_weak(prev_peak, after)) {
  }
}

void allocator_note_free(size_t bytes) { g_current_bytes.fetch_sub(static_cast<uint64_t>(bytes)); }

static size_t numel_from_shape(const std::vector<int>& shape) {
  if (shape.empty()) return 0;
  size_t n = 1;
  for (int d : shape) n *= static_cast<size_t>(d);
  return n;
}

Tensor::Tensor(const std::vector<int>& shape, const std::string& name) : shape_(shape), name_(name) {
  numel_ = numel_from_shape(shape_);
  alloc_();
}

Tensor::~Tensor() { free_(); }

Tensor::Tensor(Tensor&& other) noexcept {
  shape_ = std::move(other.shape_);
  numel_ = other.numel_;
  data_ = other.data_;
  name_ = std::move(other.name_);
  other.numel_ = 0;
  other.data_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this == &other) return *this;
  free_();
  shape_ = std::move(other.shape_);
  numel_ = other.numel_;
  data_ = other.data_;
  name_ = std::move(other.name_);
  other.numel_ = 0;
  other.data_ = nullptr;
  return *this;
}

void Tensor::resize(const std::vector<int>& shape) {
  const size_t new_numel = numel_from_shape(shape);
  if (new_numel == numel_ && shape == shape_) return;
  free_();
  shape_ = shape;
  numel_ = new_numel;
  alloc_();
}

void Tensor::alloc_() {
  if (numel_ == 0) return;
  GPU_CUDA_CHECK(cudaMalloc(&data_, bytes()));
  allocator_note_alloc(bytes());
}

void Tensor::free_() {
  if (!data_) return;
  GPU_CUDA_CHECK(cudaFree(data_));
  allocator_note_free(bytes());
  data_ = nullptr;
  numel_ = 0;
  shape_.clear();
}

void Tensor::zero_() {
  if (!data_) return;
  GPU_CUDA_CHECK(cudaMemset(data_, 0, bytes()));
}

void Tensor::copy_from_host(const float* host, size_t n) {
  if (n > numel_) n = numel_;
  GPU_CUDA_CHECK(cudaMemcpy(data_, host, n * sizeof(float), cudaMemcpyHostToDevice));
}

void Tensor::copy_to_host(float* host, size_t n) const {
  if (n > numel_) n = numel_;
  GPU_CUDA_CHECK(cudaMemcpy(host, data_, n * sizeof(float), cudaMemcpyDeviceToHost));
}

}  // namespace gpu

