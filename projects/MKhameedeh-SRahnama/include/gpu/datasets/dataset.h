#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace gpu {

struct Batch {
  // Host buffers (float32). Images are NCHW contiguous.
  std::vector<float> x;
  std::vector<int> y;
  int n{0};
  int c{0};
  int h{0};
  int w{0};
  int num_classes{0};
};

class Dataset {
 public:
  virtual ~Dataset() = default;
  virtual std::string name() const = 0;
  virtual int num_classes() const = 0;
  virtual int channels() const = 0;
  virtual int height() const = 0;
  virtual int width() const = 0;

  virtual size_t size_train() const = 0;
  virtual size_t size_test() const = 0;

  virtual bool next_train_batch(int batch_size, Batch& out) = 0;
  virtual bool next_test_batch(int batch_size, Batch& out) = 0;

  virtual void reset_train() = 0;
  virtual void reset_test() = 0;

  // Optional: control shuffling behavior for training.
  virtual void set_shuffle_train(bool enabled) { (void)enabled; }
};

}  // namespace gpu
