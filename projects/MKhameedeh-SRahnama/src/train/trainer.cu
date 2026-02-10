#include "gpu/train/trainer.h"

#include <cuda_runtime.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "gpu/core/logger.h"
#include "gpu/core/metrics.h"
#include "gpu/core/profiler.h"
#include "gpu/core/system_monitor.h"
#include "gpu/core/tensor.h"
#include "gpu/kernels/sgd.h"
#include "gpu/kernels/softmax_ce.h"
#include "gpu/kernels/reduce.h"
#include "gpu/train/checkpoint.h"
#include "gpu/utils/cuda_check.h"
#include "gpu/utils/fs.h"
#include "gpu/utils/time.h"

namespace gpu {

static std::filesystem::path make_run_dir(const TrainConfig& cfg) {
  const auto t = unix_millis();
  std::ostringstream name;
  if (!cfg.run_name.empty()) name << cfg.run_name << "_";
  name << t << "_" << cfg.arch << "_" << cfg.dataset;
  std::filesystem::path dir = std::filesystem::path(cfg.out_dir) / name.str();
  ensure_dir(dir);
  // Marker for "latest".
  ensure_dir(cfg.out_dir);
  std::FILE* f = std::fopen((std::filesystem::path(cfg.out_dir) / "latest.txt").string().c_str(), "wb");
  if (f) {
    const auto s = dir.string();
    std::fwrite(s.data(), 1, s.size(), f);
    std::fclose(f);
  }
  return dir;
}

static void write_config_json(const std::filesystem::path& dir, const TrainConfig& cfg) {
  std::ofstream out(dir / "config.json", std::ios::trunc);
  out << "{";
  out << "\"arch\":\"" << cfg.arch << "\",";
  out << "\"dataset\":\"" << cfg.dataset << "\",";
  out << "\"data_dir\":\"" << cfg.data_dir << "\",";
  out << "\"out_dir\":\"" << cfg.out_dir << "\",";
  out << "\"run_name\":\"" << cfg.run_name << "\",";
  out << "\"resume_from\":\"" << cfg.resume_from << "\",";
  out << "\"epochs\":" << cfg.epochs << ",";
  out << "\"batch_size\":" << cfg.batch_size << ",";
  out << "\"lr\":" << cfg.lr << ",";
  out << "\"weight_decay\":" << cfg.weight_decay << ",";
  out << "\"seed\":" << cfg.seed << ",";
  out << "\"log_every\":" << cfg.log_every << ",";
  out << "\"eval_every\":" << cfg.eval_every << ",";
  out << "\"save_every\":" << cfg.save_every << ",";
  out << "\"profile_interval_ms\":" << cfg.profile_interval_ms << ",";
  out << "\"shuffle_train\":" << (cfg.shuffle_train ? "true" : "false");
  out << "}\n";
}

static void write_device_json(const std::filesystem::path& dir) {
  int dev = 0;
  GPU_CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  GPU_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  int clock_khz = 0;
  cudaError_t clock_err = cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, dev);
  std::ofstream out(dir / "device.json", std::ios::trunc);
  out << "{";
  out << "\"device\":" << dev << ",";
  out << "\"name\":\"" << prop.name << "\",";
  out << "\"sm_count\":" << prop.multiProcessorCount << ",";
  if (clock_err == cudaSuccess) {
    out << "\"clock_khz\":" << clock_khz << ",";
  } else {
    out << "\"clock_khz\":0,";
  }
  out << "\"mem_total_bytes\":" << prop.totalGlobalMem << ",";
  out << "\"major\":" << prop.major << ",";
  out << "\"minor\":" << prop.minor;
  out << "}\n";
}

Trainer::Trainer(TrainConfig cfg, std::unique_ptr<Dataset> dataset, std::unique_ptr<Model> model)
    : cfg_(std::move(cfg)), dataset_(std::move(dataset)), model_(std::move(model)) {}

void Trainer::run() {
  const auto run_dir = make_run_dir(cfg_);
  write_config_json(run_dir, cfg_);
  write_device_json(run_dir);

  Logger::instance().set_run_dir(run_dir);
  MetricsSink metrics;
  metrics.set_run_dir(run_dir);
  Profiler::instance().set_run_dir(run_dir);

  SystemMonitor monitor;
  int dev = 0;
  GPU_CUDA_CHECK(cudaGetDevice(&dev));
  monitor.start(run_dir, cfg_.profile_interval_ms, dev);

  Logger::instance().info("Run dir: " + run_dir.string());

  // Main loop.
  uint64_t global_step = 0;
  if (!cfg_.resume_from.empty()) {
    const auto meta = load_checkpoint(cfg_.resume_from, *model_);
    global_step = meta.step;
    Logger::instance().info("Resumed from checkpoint step " + std::to_string(global_step));
  }

  const auto ckpt_dir = run_dir / "checkpoints";

  auto run_eval = [&](int epoch_tag, uint64_t step_tag) {
    dataset_->reset_test();
    Batch tb;
    int eval_seen = 0;
    int eval_correct = 0;
    float eval_loss_sum = 0.0f;
    while (dataset_->next_test_batch(cfg_.batch_size, tb)) {
      Tensor x_d({tb.n, tb.c, tb.h, tb.w}, "x_eval");
      x_d.copy_from_host(tb.x.data(), tb.x.size());
      Tensor logits = model_->forward(x_d);
      Tensor grad_logits({tb.n, tb.num_classes}, "grad_logits_eval");
      const float loss = softmax_cross_entropy_backward(logits, tb.y, grad_logits);
      eval_loss_sum += loss * static_cast<float>(tb.n);
      eval_correct += argmax_accuracy(logits, tb.y);
      eval_seen += tb.n;
      if (eval_seen >= 1000) break;  // keep eval cheap
    }
    const float eval_loss = eval_seen ? (eval_loss_sum / static_cast<float>(eval_seen)) : 0.0f;
    const float eval_acc = eval_seen ? (static_cast<float>(eval_correct) / static_cast<float>(eval_seen)) : 0.0f;
    Logger::instance().info("Eval: loss=" + std::to_string(eval_loss) + " acc=" + std::to_string(eval_acc) +
                            " n=" + std::to_string(eval_seen));

    MetricPoint p;
    p.t_ms = unix_millis();
    p.scalars["epoch"] = static_cast<double>(epoch_tag);
    p.scalars["step"] = static_cast<double>(step_tag);
    p.scalars["eval_loss"] = eval_loss;
    p.scalars["eval_acc"] = eval_acc;
    metrics.write_point(p);
  };

  for (int epoch = 1; epoch <= cfg_.epochs; ++epoch) {
    dataset_->set_shuffle_train(cfg_.shuffle_train);
    dataset_->reset_train();
    Batch batch;
    double epoch_start = steady_seconds();
    double samples = 0.0;

    while (dataset_->next_train_batch(cfg_.batch_size, batch)) {
      const double step_start = steady_seconds();

      Tensor x_d({batch.n, batch.c, batch.h, batch.w}, "x");
      {
        Profiler::ScopedCpuTimer t("step_h2d_copy");
        x_d.copy_from_host(batch.x.data(), batch.x.size());
      }

      Tensor logits;
      {
        Profiler::ScopedCpuTimer t("step_forward");
        logits = model_->forward(x_d);
      }

      Tensor grad_logits({batch.n, batch.num_classes}, "grad_logits");
      float loss = 0.0f;
      {
        Profiler::ScopedCpuTimer t("step_loss_backward");
        loss = softmax_cross_entropy_backward(logits, batch.y, grad_logits);
        (void)model_->backward(grad_logits);
      }

      // SGD update.
      {
        Profiler::ScopedCpuTimer t("step_sgd");
        for (auto p : model_->params()) sgd_step(*p.w, *p.grad, cfg_.lr, cfg_.weight_decay);
      }

      int correct = 0;
      float acc = 0.0f;
      {
        Profiler::ScopedCpuTimer t("step_accuracy");
        correct = argmax_accuracy(logits, batch.y);
        acc = static_cast<float>(correct) / static_cast<float>(batch.n);
      }

      size_t free_b = 0;
      size_t total_b = 0;
      GPU_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
      const auto a = allocator_stats();

      const double step_s = steady_seconds() - step_start;
      samples += static_cast<double>(batch.n);

      const uint64_t log_every = (cfg_.log_every <= 0) ? 1ULL : static_cast<uint64_t>(cfg_.log_every);
      if ((global_step % log_every) == 0) {
        std::ostringstream msg;
        msg << "epoch " << epoch << " step " << global_step << " loss=" << loss << " acc=" << acc
            << " step_s=" << step_s;
        Logger::instance().info(msg.str());

        MetricPoint p;
        p.t_ms = unix_millis();
        p.scalars["step"] = static_cast<double>(global_step);
        p.scalars["epoch"] = static_cast<double>(epoch);
        p.scalars["lr"] = static_cast<double>(cfg_.lr);
        p.scalars["loss"] = loss;
        p.scalars["acc"] = acc;
        p.scalars["step_s"] = step_s;
        p.scalars["samples_per_s"] = static_cast<double>(batch.n) / step_s;
        p.scalars["cuda_mem_free_mb"] = static_cast<double>(free_b) / (1024.0 * 1024.0);
        p.scalars["cuda_mem_total_mb"] = static_cast<double>(total_b) / (1024.0 * 1024.0);
        p.scalars["cuda_mem_used_mb"] =
            static_cast<double>(total_b - free_b) / (1024.0 * 1024.0);
        p.scalars["alloc_current_mb"] = static_cast<double>(a.current_bytes) / (1024.0 * 1024.0);
        p.scalars["alloc_peak_mb"] = static_cast<double>(a.peak_bytes) / (1024.0 * 1024.0);

        double w_sumsq = 0.0;
        double g_sumsq = 0.0;
        for (const auto& prm : model_->params()) {
          w_sumsq += tensor_sumsq(*prm.w);
          g_sumsq += tensor_sumsq(*prm.grad);
        }
        p.scalars["param_l2"] = std::sqrt(w_sumsq);
        p.scalars["grad_l2"] = std::sqrt(g_sumsq);

        metrics.write_point(p);
      }

      const uint64_t eval_every = (cfg_.eval_every <= 0) ? 0ULL : static_cast<uint64_t>(cfg_.eval_every);
      if (eval_every > 0 && global_step > 0 && (global_step % eval_every) == 0) {
        Profiler::ScopedCpuTimer te("eval");
        run_eval(epoch, global_step);
      }

      if (cfg_.save_every > 0 && global_step > 0 &&
          (global_step % static_cast<uint64_t>(cfg_.save_every) == 0)) {
        Profiler::ScopedCpuTimer ts("checkpoint_save");
        save_checkpoint(ckpt_dir, *model_, global_step);
      }

      global_step += 1;
    }

    const double epoch_s = steady_seconds() - epoch_start;
    Logger::instance().info("Epoch " + std::to_string(epoch) + " done: " + std::to_string(samples / epoch_s) +
                            " samples/s");

    {
      Profiler::ScopedCpuTimer te("eval_epoch_end");
      run_eval(epoch, global_step);
    }
  }

  Profiler::instance().flush();
}

}  // namespace gpu
