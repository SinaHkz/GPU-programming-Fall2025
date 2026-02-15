#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "gpu/datasets/dataset.h"

namespace gpu {

class Cifar10Dataset final : public Dataset {
 public:
  Cifar10Dataset(const std::filesystem::path& root, int seed);

  std::string name() const override { return "cifar10"; }
  int num_classes() const override { return 10; }
  int channels() const override { return 3; }
  int height() const override { return 32; }
  int width() const override { return 32; }

  size_t size_train() const override { return train_labels_.size(); }
  size_t size_test() const override { return test_labels_.size(); }

  bool next_train_batch(int batch_size, Batch& out) override;
  bool next_test_batch(int batch_size, Batch& out) override;

  void reset_train() override;
  void reset_test() override;
  void set_shuffle_train(bool enabled) override { shuffle_train_ = enabled; }

 private:
  void load_bin(const std::filesystem::path& p, std::vector<uint8_t>& images, std::vector<uint8_t>& labels);

  std::vector<uint8_t> train_images_;
  std::vector<uint8_t> train_labels_;
  std::vector<uint8_t> test_images_;
  std::vector<uint8_t> test_labels_;

  std::vector<size_t> train_perm_;
  std::vector<size_t> test_perm_;
  int seed_{0};
  bool shuffle_train_{true};
  uint64_t train_reset_count_{0};

  size_t train_idx_{0};
  size_t test_idx_{0};
};

}  // namespace gpu
