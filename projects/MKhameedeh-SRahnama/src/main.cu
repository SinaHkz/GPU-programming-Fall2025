#include "gpu/app/args.h"

#include <cuda_runtime.h>

#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "gpu/core/logger.h"
#include "gpu/datasets/cifar10.h"
#include "gpu/datasets/mnist.h"
#include "gpu/datasets/synthetic.h"
#include "gpu/model/factory.h"
#include "gpu/train/trainer.h"
#include "gpu/utils/cuda_check.h"

int main(int argc, char** argv) {
  try {
    auto cfg = gpu::parse_args(argc, argv);

    int dev = 0;
    GPU_CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    GPU_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    gpu::Logger::instance().info(std::string("CUDA device: ") + prop.name);

    std::unique_ptr<gpu::Dataset> dataset;
    if (cfg.dataset == "mnist") {
      dataset = std::make_unique<gpu::MnistDataset>(std::filesystem::path(cfg.data_dir) / "mnist", cfg.seed);
    } else if (cfg.dataset == "cifar10") {
      dataset = std::make_unique<gpu::Cifar10Dataset>(std::filesystem::path(cfg.data_dir) / "cifar10", cfg.seed);
    } else if (cfg.dataset == "synthetic") {
      const int c = (cfg.arch == "lenet") ? 1 : 3;
      const int h = (cfg.arch == "lenet") ? 28 : 32;
      const int w = (cfg.arch == "lenet") ? 28 : 32;
      dataset = std::make_unique<gpu::SyntheticDataset>(6000, 1000, c, h, w, 10, cfg.seed);
    } else {
      throw std::runtime_error("Unknown dataset: " + cfg.dataset);
    }

    auto model = gpu::make_model(cfg.arch, dataset->channels(), dataset->height(), dataset->width(),
                                 dataset->num_classes(), cfg.seed);
    gpu::Trainer t(cfg, std::move(dataset), std::move(model));
    t.run();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Fatal: " << e.what() << "\n";
    return 1;
  }
}

