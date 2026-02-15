#include "gpu/model/models/lenet.h"
#include "gpu/model/models/mlp.h"
#include "gpu/model/models/simple_cnn.h"

#include <stdexcept>

#include "gpu/layers/conv2d.h"
#include "gpu/layers/flatten.h"
#include "gpu/layers/linear.h"
#include "gpu/layers/maxpool2d.h"
#include "gpu/layers/relu.h"
#include "gpu/utils/cuda_check.h"

namespace gpu {

LeNetModel::LeNetModel(int in_c, int in_h, int in_w, int num_classes, int seed) {
  // LeNet-ish for MNIST: (N,1,28,28)
  if (in_c != 1 || in_h != 28 || in_w != 28) throw std::runtime_error("lenet expects input 1x28x28");

  seq_.add(std::make_unique<Conv2D>(1, 6, 5, 1, 0, seed + 100, "conv1"));
  seq_.add(std::make_unique<ReLU>("relu1"));
  seq_.add(std::make_unique<MaxPool2D>(2, 2, "pool1"));
  seq_.add(std::make_unique<Conv2D>(6, 16, 5, 1, 0, seed + 101, "conv2"));
  seq_.add(std::make_unique<ReLU>("relu2"));
  seq_.add(std::make_unique<MaxPool2D>(2, 2, "pool2"));
  seq_.add(std::make_unique<Flatten>("flat"));
  seq_.add(std::make_unique<Linear>(16 * 4 * 4, 120, seed + 102, "fc1"));
  seq_.add(std::make_unique<ReLU>("relu3"));
  seq_.add(std::make_unique<Linear>(120, 84, seed + 103, "fc2"));
  seq_.add(std::make_unique<ReLU>("relu4"));
  seq_.add(std::make_unique<Linear>(84, num_classes, seed + 104, "fc3"));
}

SimpleCnnModel::SimpleCnnModel(int in_c, int in_h, int in_w, int num_classes, int seed) {
  // Small CNN for CIFAR-10: conv->relu->pool->conv->relu->pool->fc
  if (in_c != 3 || in_h != 32 || in_w != 32) throw std::runtime_error("simple_cnn expects input 3x32x32");
  seq_.add(std::make_unique<Conv2D>(3, 32, 3, 1, 1, seed + 200, "conv1"));
  seq_.add(std::make_unique<ReLU>("relu1"));
  seq_.add(std::make_unique<MaxPool2D>(2, 2, "pool1"));  // 16x16
  seq_.add(std::make_unique<Conv2D>(32, 64, 3, 1, 1, seed + 201, "conv2"));
  seq_.add(std::make_unique<ReLU>("relu2"));
  seq_.add(std::make_unique<MaxPool2D>(2, 2, "pool2"));  // 8x8
  seq_.add(std::make_unique<Flatten>("flat"));
  seq_.add(std::make_unique<Linear>(64 * 8 * 8, 256, seed + 202, "fc1"));
  seq_.add(std::make_unique<ReLU>("relu3"));
  seq_.add(std::make_unique<Linear>(256, num_classes, seed + 203, "fc2"));
}

MlpModel::MlpModel(int in_c, int in_h, int in_w, int num_classes, int seed) {
  seq_.add(std::make_unique<Flatten>("flat"));
  seq_.add(std::make_unique<Linear>(in_c * in_h * in_w, 256, seed + 300, "fc1"));
  seq_.add(std::make_unique<ReLU>("relu1"));
  seq_.add(std::make_unique<Linear>(256, num_classes, seed + 301, "fc2"));
}

}  // namespace gpu

