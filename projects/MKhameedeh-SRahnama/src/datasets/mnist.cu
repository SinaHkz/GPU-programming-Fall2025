#include "gpu/datasets/mnist.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>

#include "gpu/core/logger.h"

namespace gpu {

static uint32_t read_be_u32(std::ifstream& in) {
  uint8_t b[4] = {0, 0, 0, 0};
  in.read(reinterpret_cast<char*>(b), 4);
  return (static_cast<uint32_t>(b[0]) << 24) | (static_cast<uint32_t>(b[1]) << 16) |
         (static_cast<uint32_t>(b[2]) << 8) | static_cast<uint32_t>(b[3]);
}

std::vector<uint8_t> MnistDataset::load_idx_u8(const std::filesystem::path& p, int expected_dims) {
  std::ifstream in(p, std::ios::binary);
  if (!in) throw std::runtime_error("MNIST: cannot open " + p.string());

  const uint32_t magic = read_be_u32(in);
  const uint32_t dims = magic & 0xFF;
  if (static_cast<int>(dims) != expected_dims) {
    throw std::runtime_error("MNIST: unexpected IDX dims in " + p.string());
  }

  std::vector<uint32_t> shape;
  shape.reserve(dims);
  size_t n = 1;
  for (uint32_t i = 0; i < dims; ++i) {
    const uint32_t d = read_be_u32(in);
    shape.push_back(d);
    n *= static_cast<size_t>(d);
  }

  std::vector<uint8_t> data(n);
  in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(n));
  if (!in) throw std::runtime_error("MNIST: failed reading data " + p.string());
  return data;
}

MnistDataset::MnistDataset(const std::filesystem::path& root, int seed) : seed_(seed) {
  const auto img_tr = root / "train-images-idx3-ubyte";
  const auto lab_tr = root / "train-labels-idx1-ubyte";
  const auto img_te = root / "t10k-images-idx3-ubyte";
  const auto lab_te = root / "t10k-labels-idx1-ubyte";

  Logger::instance().info("Loading MNIST from " + root.string());
  train_images_ = load_idx_u8(img_tr, 3);
  train_labels_ = load_idx_u8(lab_tr, 1);
  test_images_ = load_idx_u8(img_te, 3);
  test_labels_ = load_idx_u8(lab_te, 1);

  if (train_labels_.size() != size_train() || test_labels_.size() != size_test()) {
    throw std::runtime_error("MNIST: label/image size mismatch");
  }

  train_perm_.resize(size_train());
  std::iota(train_perm_.begin(), train_perm_.end(), 0);
  test_perm_.resize(size_test());
  std::iota(test_perm_.begin(), test_perm_.end(), 0);
}

static void make_mnist_batch(const std::vector<uint8_t>& images, const std::vector<uint8_t>& labels,
                            const std::vector<size_t>& perm, size_t& idx, int batch_size, Batch& out) {
  const size_t n_total = perm.size();
  if (idx >= n_total) return;
  const int n = static_cast<int>(std::min<size_t>(static_cast<size_t>(batch_size), n_total - idx));

  out.n = n;
  out.c = 1;
  out.h = 28;
  out.w = 28;
  out.num_classes = 10;
  out.x.resize(static_cast<size_t>(n) * 28 * 28);
  out.y.resize(static_cast<size_t>(n));

  for (int i = 0; i < n; ++i) {
    const size_t sample = perm[idx + static_cast<size_t>(i)];
    out.y[static_cast<size_t>(i)] = static_cast<int>(labels[sample]);
    const size_t src_off = sample * 28 * 28;
    const size_t dst_off = static_cast<size_t>(i) * 28 * 28;
    for (int p = 0; p < 28 * 28; ++p) {
      out.x[dst_off + static_cast<size_t>(p)] =
          static_cast<float>(images[src_off + static_cast<size_t>(p)]) / 255.0f;
    }
  }

  idx += static_cast<size_t>(n);
}

bool MnistDataset::next_train_batch(int batch_size, Batch& out) {
  if (train_idx_ >= size_train()) return false;
  make_mnist_batch(train_images_, train_labels_, train_perm_, train_idx_, batch_size, out);
  return out.n > 0;
}

bool MnistDataset::next_test_batch(int batch_size, Batch& out) {
  if (test_idx_ >= size_test()) return false;
  make_mnist_batch(test_images_, test_labels_, test_perm_, test_idx_, batch_size, out);
  return out.n > 0;
}

void MnistDataset::reset_train() {
  train_idx_ = 0;
  if (shuffle_train_) {
    std::mt19937 rng(static_cast<uint32_t>(seed_) + static_cast<uint32_t>(train_reset_count_));
    std::shuffle(train_perm_.begin(), train_perm_.end(), rng);
  } else {
    std::iota(train_perm_.begin(), train_perm_.end(), 0);
  }
  train_reset_count_ += 1;
}

void MnistDataset::reset_test() { test_idx_ = 0; }

}  // namespace gpu
