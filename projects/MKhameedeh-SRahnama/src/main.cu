#include "gpu/app/args.h"

#include <cuda_runtime.h>

#include <filesystem>
#include <iomanip>
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

namespace {

std::unique_ptr<gpu::Dataset> make_dataset(const gpu::TrainConfig& cfg) {
  if (cfg.dataset == "mnist") {
    return std::make_unique<gpu::MnistDataset>(std::filesystem::path(cfg.data_dir) / "mnist", cfg.seed);
  }
  if (cfg.dataset == "cifar10") {
    return std::make_unique<gpu::Cifar10Dataset>(std::filesystem::path(cfg.data_dir) / "cifar10", cfg.seed);
  }
  if (cfg.dataset == "synthetic") {
    const int c = (cfg.arch == "lenet") ? 1 : 3;
    const int h = (cfg.arch == "lenet") ? 28 : 32;
    const int w = (cfg.arch == "lenet") ? 28 : 32;
    return std::make_unique<gpu::SyntheticDataset>(6000, 1000, c, h, w, 10, cfg.seed);
  }
  throw std::runtime_error("Unknown dataset: " + cfg.dataset);
}

std::unique_ptr<gpu::Model> make_model_for_cfg(const gpu::TrainConfig& cfg, const gpu::Dataset& ds) {
  return gpu::make_model(cfg.arch, ds.channels(), ds.height(), ds.width(), ds.num_classes(), cfg.seed);
}

gpu::TrainSummary run_once(const gpu::TrainConfig& cfg) {
  auto dataset = make_dataset(cfg);
  auto model = make_model_for_cfg(cfg, *dataset);
  gpu::Trainer t(cfg, std::move(dataset), std::move(model));
  return t.run();
}

std::string with_suffix(const std::string& base, const std::string& suffix) {
  if (base.empty()) return suffix;
  return base + "_" + suffix;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    auto cfg = gpu::parse_args(argc, argv);

    int dev = 0;
    GPU_CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    GPU_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    gpu::Logger::instance().info(std::string("CUDA device: ") + prop.name);

    if (cfg.benchmark_compare) {
      gpu::TrainConfig before = cfg;
      gpu::TrainConfig after = cfg;

      const int bench_steps = (cfg.max_steps > 0) ? cfg.max_steps : ((cfg.benchmark_steps > 0) ? cfg.benchmark_steps : 200);
      before.max_steps = bench_steps;
      after.max_steps = bench_steps;
      // Benchmark should measure core training-step performance, not eval/checkpoint side work.
      before.eval_every = 0;
      after.eval_every = 0;
      before.save_every = 0;
      after.save_every = 0;

      before.enable_h2d_pipeline = false;
      before.enable_log_sync_optimizations = false;
      before.enable_async_checkpoint = false;
      before.enable_cuda_graph_sgd = false;
      before.enable_async_eval = false;
      before.norm_log_multiplier = 1;

      after.enable_h2d_pipeline = true;
      after.enable_log_sync_optimizations = true;
      after.enable_async_checkpoint = true;
      // Keep benchmark robust across environments: force graph capture off.
      after.enable_cuda_graph_sgd = false;
      // Async eval competes for the same GPU and can make step-time benchmark look slower.
      after.enable_async_eval = false;
      if (after.norm_log_multiplier < 1) after.norm_log_multiplier = 1;

      before.run_name = with_suffix(cfg.run_name, "before");
      after.run_name = with_suffix(cfg.run_name, "after");

      const auto before_sum = run_once(before);
      const auto after_sum = run_once(after);

      std::cout << std::fixed << std::setprecision(6);
      std::cout << "BENCHMARK_BEFORE steps=" << before_sum.steps << " avg_step_s=" << before_sum.avg_step_s << "\n";
      std::cout << "BENCHMARK_AFTER  steps=" << after_sum.steps << " avg_step_s=" << after_sum.avg_step_s << "\n";
      if (before_sum.avg_step_s > 0.0 && after_sum.avg_step_s > 0.0) {
        const double speedup = before_sum.avg_step_s / after_sum.avg_step_s;
        const double pct = (1.0 - (after_sum.avg_step_s / before_sum.avg_step_s)) * 100.0;
        std::cout << "BENCHMARK_SPEEDUP x" << speedup << " (" << pct << "% faster)\n";
      } else {
        std::cout << "BENCHMARK_SPEEDUP unavailable (insufficient steps)\n";
      }
      std::cout.flush();
    } else {
      (void)run_once(cfg);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Fatal: " << e.what() << "\n";
    return 1;
  }
}
