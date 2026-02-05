#include "gpu/datasets/synthetic.h"

#include <algorithm>

namespace gpu {

SyntheticDataset::SyntheticDataset(int n_train, int n_test, int c, int h, int w, int classes, int seed)
    : n_train_(n_train),
      n_test_(n_test),
      c_(c),
      h_(h),
      w_(w),
      classes_(classes),
      rng_(static_cast<uint32_t>(seed)) {}

void SyntheticDataset::fill_batch(int batch_size, int& idx, Batch& out) {
  const int remaining = std::max(0, n_train_ - idx);
  const int n = std::min(batch_size, remaining);
  out.n = n;
  out.c = c_;
  out.h = h_;
  out.w = w_;
  out.num_classes = classes_;
  out.x.resize(static_cast<size_t>(n) * c_ * h_ * w_);
  out.y.resize(static_cast<size_t>(n));

  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  std::uniform_int_distribution<int> yi(0, classes_ - 1);
  for (size_t i = 0; i < out.x.size(); ++i) out.x[i] = u(rng_);
  for (int i = 0; i < n; ++i) out.y[static_cast<size_t>(i)] = yi(rng_);

  idx += n;
}

bool SyntheticDataset::next_train_batch(int batch_size, Batch& out) {
  if (train_idx_ >= n_train_) return false;
  fill_batch(batch_size, train_idx_, out);
  return out.n > 0;
}

bool SyntheticDataset::next_test_batch(int batch_size, Batch& out) {
  if (test_idx_ >= n_test_) return false;
  // For simplicity, reuse the same generator (not deterministic across train/test).
  const int remaining = std::max(0, n_test_ - test_idx_);
  const int n = std::min(batch_size, remaining);
  out.n = n;
  out.c = c_;
  out.h = h_;
  out.w = w_;
  out.num_classes = classes_;
  out.x.resize(static_cast<size_t>(n) * c_ * h_ * w_);
  out.y.resize(static_cast<size_t>(n));

  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  std::uniform_int_distribution<int> yi(0, classes_ - 1);
  for (size_t i = 0; i < out.x.size(); ++i) out.x[i] = u(rng_);
  for (int i = 0; i < n; ++i) out.y[static_cast<size_t>(i)] = yi(rng_);

  test_idx_ += n;
  return out.n > 0;
}

}  // namespace gpu

