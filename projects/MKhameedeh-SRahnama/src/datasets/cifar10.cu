#include "gpu/datasets/cifar10.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>

#include "gpu/core/logger.h"

namespace gpu {

void Cifar10Dataset::load_bin(const std::filesystem::path& p, std::vector<uint8_t>& images,
                             std::vector<uint8_t>& labels) {
  std::ifstream in(p, std::ios::binary);
  if (!in) throw std::runtime_error("CIFAR-10: cannot open " + p.string());

  // Record format: 1 label byte + 3072 image bytes (1024 per channel).
  constexpr int kRecord = 1 + 32 * 32 * 3;
  std::vector<uint8_t> buf(kRecord);
  while (in.read(reinterpret_cast<char*>(buf.data()), kRecord)) {
    labels.push_back(buf[0]);
    images.insert(images.end(), buf.begin() + 1, buf.end());
  }
}

Cifar10Dataset::Cifar10Dataset(const std::filesystem::path& root, int seed) : seed_(seed) {
  Logger::instance().info("Loading CIFAR-10 (binary) from " + root.string());
  // Expected layout: data/cifar10/cifar-10-batches-bin/
  const auto base = root / "cifar-10-batches-bin";
  for (int i = 1; i <= 5; ++i) load_bin(base / ("data_batch_" + std::to_string(i) + ".bin"), train_images_, train_labels_);
  load_bin(base / "test_batch.bin", test_images_, test_labels_);

  train_perm_.resize(size_train());
  std::iota(train_perm_.begin(), train_perm_.end(), 0);
  test_perm_.resize(size_test());
  std::iota(test_perm_.begin(), test_perm_.end(), 0);
}

static void make_cifar_batch(const std::vector<uint8_t>& images, const std::vector<uint8_t>& labels, size_t& idx,
                             const std::vector<size_t>& perm, int batch_size, Batch& out) {
  const size_t n_total = perm.size();
  if (idx >= n_total) return;
  const int n = static_cast<int>(std::min<size_t>(static_cast<size_t>(batch_size), n_total - idx));

  out.n = n;
  out.c = 3;
  out.h = 32;
  out.w = 32;
  out.num_classes = 10;
  out.x.resize(static_cast<size_t>(n) * 3 * 32 * 32);
  out.y.resize(static_cast<size_t>(n));

  constexpr size_t kImgBytes = 3 * 32 * 32;
  for (int i = 0; i < n; ++i) {
    const size_t sample = perm[idx + static_cast<size_t>(i)];
    out.y[static_cast<size_t>(i)] = static_cast<int>(labels[sample]);
    const size_t src_off = sample * kImgBytes;
    const size_t dst_off = static_cast<size_t>(i) * kImgBytes;

    // CIFAR-10 binary is channel-major: 1024 R, 1024 G, 1024 B.
    for (int ch = 0; ch < 3; ++ch) {
      for (int p = 0; p < 32 * 32; ++p) {
        const uint8_t v = images[src_off + static_cast<size_t>(ch * 1024 + p)];
        out.x[dst_off + static_cast<size_t>(ch * 32 * 32 + p)] = static_cast<float>(v) / 255.0f;
      }
    }
  }

  idx += static_cast<size_t>(n);
}

bool Cifar10Dataset::next_train_batch(int batch_size, Batch& out) {
  if (train_idx_ >= size_train()) return false;
  make_cifar_batch(train_images_, train_labels_, train_idx_, train_perm_, batch_size, out);
  return out.n > 0;
}

bool Cifar10Dataset::next_test_batch(int batch_size, Batch& out) {
  if (test_idx_ >= size_test()) return false;
  make_cifar_batch(test_images_, test_labels_, test_idx_, test_perm_, batch_size, out);
  return out.n > 0;
}

void Cifar10Dataset::reset_train() {
  train_idx_ = 0;
  if (shuffle_train_) {
    std::mt19937 rng(static_cast<uint32_t>(seed_) + static_cast<uint32_t>(train_reset_count_));
    std::shuffle(train_perm_.begin(), train_perm_.end(), rng);
  } else {
    std::iota(train_perm_.begin(), train_perm_.end(), 0);
  }
  train_reset_count_ += 1;
}

void Cifar10Dataset::reset_test() { test_idx_ = 0; }

}  // namespace gpu
