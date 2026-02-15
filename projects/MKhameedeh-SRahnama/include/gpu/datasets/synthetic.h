#pragma once

#include <random>
#include <string>

#include "gpu/datasets/dataset.h"

namespace gpu {

class SyntheticDataset final : public Dataset {
 public:
  SyntheticDataset(int n_train, int n_test, int c, int h, int w, int classes, int seed);

  std::string name() const override { return "synthetic"; }
  int num_classes() const override { return classes_; }
  int channels() const override { return c_; }
  int height() const override { return h_; }
  int width() const override { return w_; }

  size_t size_train() const override { return static_cast<size_t>(n_train_); }
  size_t size_test() const override { return static_cast<size_t>(n_test_); }

  bool next_train_batch(int batch_size, Batch& out) override;
  bool next_test_batch(int batch_size, Batch& out) override;

  void reset_train() override { train_idx_ = 0; }
  void reset_test() override { test_idx_ = 0; }

 private:
  void fill_batch(int batch_size, int& idx, Batch& out);

  int n_train_{0};
  int n_test_{0};
  int c_{0};
  int h_{0};
  int w_{0};
  int classes_{0};
  int train_idx_{0};
  int test_idx_{0};

  std::mt19937 rng_;
};

}  // namespace gpu

