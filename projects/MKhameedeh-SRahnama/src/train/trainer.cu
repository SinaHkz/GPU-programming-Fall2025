#include "gpu/train/trainer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "gpu/core/logger.h"
#include "gpu/core/metrics.h"
#include "gpu/core/profiler.h"
#include "gpu/core/tensor.h"
#include "gpu/datasets/cifar10.h"
#include "gpu/datasets/mnist.h"
#include "gpu/datasets/synthetic.h"
#include "gpu/kernels/sgd.h"
#include "gpu/kernels/softmax_ce.h"
#include "gpu/kernels/reduce.h"
#include "gpu/model/factory.h"
#include "gpu/train/checkpoint.h"
#include "gpu/utils/cuda_check.h"
#include "gpu/utils/fs.h"
#include "gpu/utils/time.h"

namespace gpu {

namespace {

class H2DPipeline {
 public:
  explicit H2DPipeline(size_t max_input_elems) : max_input_elems_(max_input_elems) {
    GPU_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    for (int i = 0; i < 2; ++i) {
      GPU_CUDA_CHECK(cudaEventCreateWithFlags(&ready_[i], cudaEventDisableTiming));
      if (max_input_elems_ > 0) {
        GPU_CUDA_CHECK(cudaHostAlloc(&host_[i], max_input_elems_ * sizeof(float), cudaHostAllocDefault));
      }
    }
  }

  ~H2DPipeline() {
    for (int i = 0; i < 2; ++i) {
      if (ready_[i]) cudaEventDestroy(ready_[i]);
      if (host_[i]) cudaFreeHost(host_[i]);
    }
    if (stream_) cudaStreamDestroy(stream_);
  }

  H2DPipeline(const H2DPipeline&) = delete;
  H2DPipeline& operator=(const H2DPipeline&) = delete;

  void stage_batch_async(int slot, const Batch& b) {
    if (slot < 0 || slot > 1) throw std::runtime_error("H2DPipeline: invalid slot");
    if (b.x.size() > max_input_elems_) {
      throw std::runtime_error("H2DPipeline: batch exceeds pinned staging capacity");
    }
    if (b.x.empty()) return;

    std::memcpy(host_[slot], b.x.data(), b.x.size() * sizeof(float));
    x_d_[slot].resize({b.n, b.c, b.h, b.w});
    GPU_CUDA_CHECK(cudaMemcpyAsync(x_d_[slot].data(), host_[slot], b.x.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream_));
    GPU_CUDA_CHECK(cudaEventRecord(ready_[slot], stream_));
  }

  void wait_until_ready(int slot) {
    if (slot < 0 || slot > 1) throw std::runtime_error("H2DPipeline: invalid slot");
    GPU_CUDA_CHECK(cudaStreamWaitEvent(0, ready_[slot], 0));
  }

  Tensor& tensor(int slot) { return x_d_[slot]; }

 private:
  size_t max_input_elems_{0};
  cudaStream_t stream_{};
  std::array<float*, 2> host_{{nullptr, nullptr}};
  std::array<cudaEvent_t, 2> ready_{{nullptr, nullptr}};
  std::array<Tensor, 2> x_d_{};
};

std::unique_ptr<Dataset> make_dataset_for_cfg(const TrainConfig& cfg) {
  if (cfg.dataset == "mnist") {
    return std::make_unique<MnistDataset>(std::filesystem::path(cfg.data_dir) / "mnist", cfg.seed);
  }
  if (cfg.dataset == "cifar10") {
    return std::make_unique<Cifar10Dataset>(std::filesystem::path(cfg.data_dir) / "cifar10", cfg.seed);
  }
  if (cfg.dataset == "synthetic") {
    const int c = (cfg.arch == "lenet") ? 1 : 3;
    const int h = (cfg.arch == "lenet") ? 28 : 32;
    const int w = (cfg.arch == "lenet") ? 28 : 32;
    return std::make_unique<SyntheticDataset>(6000, 1000, c, h, w, 10, cfg.seed);
  }
  throw std::runtime_error("Unknown dataset: " + cfg.dataset);
}

class SgdGraphRunner {
 public:
  explicit SgdGraphRunner(bool enabled) : enabled_(enabled) {}
  ~SgdGraphRunner() {
    if (exec_) cudaGraphExecDestroy(exec_);
    if (graph_) cudaGraphDestroy(graph_);
  }

  void run(const std::vector<Param>& params, float lr, float weight_decay) {
    if (!enabled_ || failed_) {
      for (auto p : params) sgd_step(*p.w, *p.grad, lr, weight_decay);
      return;
    }

    if (!captured_) {
      if (capture_and_instantiate(params, lr, weight_decay)) {
        captured_ = true;
      } else {
        failed_ = true;
        Logger::instance().warn("CUDA graph capture for SGD disabled; using normal kernel launches.");
        for (auto p : params) sgd_step(*p.w, *p.grad, lr, weight_decay);
        return;
      }
    }

    GPU_CUDA_CHECK(cudaGraphLaunch(exec_, 0));
  }

 private:
  static void cleanup_capture_if_needed() {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(0, &status) != cudaSuccess || status == cudaStreamCaptureStatusNone) return;
    cudaGraph_t dropped{};
    (void)cudaStreamEndCapture(0, &dropped);
    if (dropped) cudaGraphDestroy(dropped);
  }

  bool capture_and_instantiate(const std::vector<Param>& params, float lr, float weight_decay) {
    if (params.empty()) return false;

    (void)cudaGetLastError();  // clear stale error state before attempting capture.
    cudaError_t st = cudaStreamBeginCapture(0, cudaStreamCaptureModeGlobal);
    if (st != cudaSuccess) return false;

    for (auto p : params) sgd_step(*p.w, *p.grad, lr, weight_decay);

    cudaGraph_t captured_graph{};
    st = cudaStreamEndCapture(0, &captured_graph);
    if (st != cudaSuccess || captured_graph == nullptr) {
      cleanup_capture_if_needed();
      return false;
    }
    graph_ = captured_graph;

    st = cudaGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0);
    if (st != cudaSuccess || exec_ == nullptr) {
      if (graph_) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
      }
      exec_ = nullptr;
      return false;
    }
    return true;
  }

  bool enabled_{false};
  bool captured_{false};
  bool failed_{false};
  cudaGraph_t graph_{};
  cudaGraphExec_t exec_{};
};

class AsyncEvaluator {
 public:
  AsyncEvaluator(const TrainConfig& cfg, const std::filesystem::path& ckpt_dir, MetricsSink* metrics)
      : cfg_(cfg), ckpt_dir_(ckpt_dir), metrics_(metrics) {
    dataset_ = make_dataset_for_cfg(cfg_);
    model_ = make_model(cfg_.arch, dataset_->channels(), dataset_->height(), dataset_->width(),
                        dataset_->num_classes(), cfg_.seed);
    worker_ = std::thread([this]() { worker_main_(); });
  }

  ~AsyncEvaluator() {
    flush();
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
  }

  void request(int epoch, uint64_t step) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      pending_epoch_ = epoch;
      pending_step_ = step;
      has_pending_ = true;
    }
    cv_.notify_one();
  }

  void flush() {
    std::unique_lock<std::mutex> lk(mu_);
    drained_cv_.wait(lk, [&]() { return !has_pending_ && !running_; });
  }

 private:
  void worker_main_() {
    while (true) {
      int epoch = 0;
      uint64_t step = 0;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&]() { return stop_ || has_pending_; });
        if (stop_ && !has_pending_) return;
        epoch = pending_epoch_;
        step = pending_step_;
        has_pending_ = false;
        running_ = true;
      }

      evaluate_(epoch, step);

      {
        std::lock_guard<std::mutex> lk(mu_);
        running_ = false;
        if (!has_pending_) drained_cv_.notify_all();
      }
    }
  }

  void evaluate_(int epoch, uint64_t step) {
    const auto json_path = ckpt_dir_ / ("checkpoint_step_" + std::to_string(step) + ".json");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!std::filesystem::exists(json_path)) {
      if (std::chrono::steady_clock::now() > deadline) {
        Logger::instance().warn("Async eval skipped: checkpoint not ready for step " + std::to_string(step));
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    try {
      (void)load_checkpoint(json_path, *model_);
      dataset_->reset_test();
      Batch tb;
      int eval_seen = 0;
      int eval_correct = 0;
      float eval_loss_sum = 0.0f;
      Tensor grad_logits_eval;

      while (dataset_->next_test_batch(cfg_.batch_size, tb)) {
        Tensor x_d({tb.n, tb.c, tb.h, tb.w}, "x_eval_async");
        x_d.copy_from_host(tb.x.data(), tb.x.size());
        Tensor logits = model_->forward(x_d);
        grad_logits_eval.resize({tb.n, tb.num_classes});
        const float loss = softmax_cross_entropy_backward(logits, tb.y, grad_logits_eval);
        eval_loss_sum += loss * static_cast<float>(tb.n);
        eval_correct += argmax_accuracy(logits, tb.y);
        eval_seen += tb.n;
        if (eval_seen >= 1000) break;
      }

      if (eval_seen <= 0) return;
      const float eval_loss = eval_loss_sum / static_cast<float>(eval_seen);
      const float eval_acc = static_cast<float>(eval_correct) / static_cast<float>(eval_seen);
      Logger::instance().info("Async Eval: step=" + std::to_string(step) + " loss=" + std::to_string(eval_loss) +
                              " acc=" + std::to_string(eval_acc) + " n=" + std::to_string(eval_seen));

      if (metrics_) {
        MetricPoint p;
        p.t_ms = unix_millis();
        p.scalars["epoch"] = static_cast<double>(epoch);
        p.scalars["step"] = static_cast<double>(step);
        p.scalars["eval_loss"] = eval_loss;
        p.scalars["eval_acc"] = eval_acc;
        metrics_->write_point(p);
      }
    } catch (const std::exception& e) {
      Logger::instance().warn(std::string("Async eval failed: ") + e.what());
    }
  }

  TrainConfig cfg_;
  std::filesystem::path ckpt_dir_;
  MetricsSink* metrics_{nullptr};
  std::unique_ptr<Dataset> dataset_;
  std::unique_ptr<Model> model_;

  std::thread worker_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::condition_variable drained_cv_;
  bool stop_{false};
  bool running_{false};
  bool has_pending_{false};
  int pending_epoch_{0};
  uint64_t pending_step_{0};
};

}  // namespace

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
  out << "\"max_steps\":" << cfg.max_steps << ",";
  out << "\"enable_h2d_pipeline\":" << (cfg.enable_h2d_pipeline ? "true" : "false") << ",";
  out << "\"enable_log_sync_optimizations\":" << (cfg.enable_log_sync_optimizations ? "true" : "false") << ",";
  out << "\"enable_async_checkpoint\":" << (cfg.enable_async_checkpoint ? "true" : "false") << ",";
  out << "\"enable_cuda_graph_sgd\":" << (cfg.enable_cuda_graph_sgd ? "true" : "false") << ",";
  out << "\"enable_async_eval\":" << (cfg.enable_async_eval ? "true" : "false") << ",";
  out << "\"norm_log_multiplier\":" << cfg.norm_log_multiplier << ",";
  out << "\"benchmark_compare\":" << (cfg.benchmark_compare ? "true" : "false") << ",";
  out << "\"benchmark_steps\":" << cfg.benchmark_steps << ",";
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

TrainSummary Trainer::run() {
  TrainSummary summary;
  double step_s_sum = 0.0;

  const auto run_dir = make_run_dir(cfg_);
  write_config_json(run_dir, cfg_);
  write_device_json(run_dir);

  Logger::instance().set_run_dir(run_dir);
  MetricsSink metrics;
  metrics.set_run_dir(run_dir);
  Profiler::instance().set_run_dir(run_dir);

  Logger::instance().info("Run dir: " + run_dir.string());

  // Main loop.
  uint64_t global_step = 0;
  if (!cfg_.resume_from.empty()) {
    const auto meta = load_checkpoint(cfg_.resume_from, *model_);
    global_step = meta.step;
    Logger::instance().info("Resumed from checkpoint step " + std::to_string(global_step));
  }

  const auto ckpt_dir = run_dir / "checkpoints";
  std::unique_ptr<AsyncCheckpointWriter> ckpt_writer;
  if ((cfg_.save_every > 0 || cfg_.enable_async_eval) && cfg_.enable_async_checkpoint) {
    ckpt_writer = std::make_unique<AsyncCheckpointWriter>(ckpt_dir, *model_);
  }
  std::unique_ptr<AsyncEvaluator> async_evaluator;
  if (cfg_.enable_async_eval) {
    async_evaluator = std::make_unique<AsyncEvaluator>(cfg_, ckpt_dir, &metrics);
  }
  uint64_t last_checkpoint_request_step = std::numeric_limits<uint64_t>::max();
  auto request_checkpoint = [&](uint64_t step) {
    if (step == last_checkpoint_request_step) return;
    if (ckpt_writer) ckpt_writer->enqueue(step);
    else save_checkpoint(ckpt_dir, *model_, step);
    last_checkpoint_request_step = step;
  };
  SgdGraphRunner sgd_graph(cfg_.enable_cuda_graph_sgd);

  auto run_eval = [&](int epoch_tag, uint64_t step_tag) {
    dataset_->reset_test();
    Batch tb;
    Tensor grad_logits_eval;
    int eval_seen = 0;
    int eval_correct = 0;
    float eval_loss_sum = 0.0f;

    if (cfg_.enable_h2d_pipeline) {
      const size_t max_input_elems = static_cast<size_t>(cfg_.batch_size) *
                                     static_cast<size_t>(dataset_->channels()) *
                                     static_cast<size_t>(dataset_->height()) *
                                     static_cast<size_t>(dataset_->width());
      H2DPipeline eval_h2d(max_input_elems);

      if (!dataset_->next_test_batch(cfg_.batch_size, tb)) {
        Logger::instance().warn("No evaluation samples available.");
        return;
      }

      int cur_slot = 0;
      eval_h2d.stage_batch_async(cur_slot, tb);

      while (true) {
        Batch next_tb;
        const bool has_next = dataset_->next_test_batch(cfg_.batch_size, next_tb);
        const int next_slot = cur_slot ^ 1;
        if (has_next) {
          eval_h2d.stage_batch_async(next_slot, next_tb);
        }

        eval_h2d.wait_until_ready(cur_slot);
        Tensor& x_d = eval_h2d.tensor(cur_slot);
        Tensor logits = model_->forward(x_d);
        grad_logits_eval.resize({tb.n, tb.num_classes});
        const float loss = softmax_cross_entropy_backward(logits, tb.y, grad_logits_eval);
        eval_loss_sum += loss * static_cast<float>(tb.n);
        eval_correct += argmax_accuracy(logits, tb.y);
        eval_seen += tb.n;

        if (eval_seen >= 1000 || !has_next) break;  // keep eval cheap
        tb = std::move(next_tb);
        cur_slot = next_slot;
      }
    } else {
      while (dataset_->next_test_batch(cfg_.batch_size, tb)) {
        Tensor x_d({tb.n, tb.c, tb.h, tb.w}, "x_eval");
        x_d.copy_from_host(tb.x.data(), tb.x.size());
        Tensor logits = model_->forward(x_d);
        grad_logits_eval.resize({tb.n, tb.num_classes});
        const float loss = softmax_cross_entropy_backward(logits, tb.y, grad_logits_eval);
        eval_loss_sum += loss * static_cast<float>(tb.n);
        eval_correct += argmax_accuracy(logits, tb.y);
        eval_seen += tb.n;
        if (eval_seen >= 1000) break;  // keep eval cheap
      }
      if (eval_seen == 0) {
        Logger::instance().warn("No evaluation samples available.");
        return;
      }
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

  bool stop_training = false;
  for (int epoch = 1; epoch <= cfg_.epochs; ++epoch) {
    dataset_->set_shuffle_train(cfg_.shuffle_train);
    dataset_->reset_train();
    Batch batch;
    double epoch_start = steady_seconds();
    double samples = 0.0;
    const uint64_t log_every = (cfg_.log_every <= 0) ? 1ULL : static_cast<uint64_t>(cfg_.log_every);
    const uint64_t norm_mult = static_cast<uint64_t>(std::max(1, cfg_.norm_log_multiplier));
    const uint64_t norm_every =
        (log_every > (std::numeric_limits<uint64_t>::max() / norm_mult))
            ? std::numeric_limits<uint64_t>::max()
            : (log_every * norm_mult);
    bool have_norm = false;
    double last_param_l2 = 0.0;
    double last_grad_l2 = 0.0;
    Tensor grad_logits;

    auto train_one_step = [&](Tensor& x_d) -> bool {
      const double step_start = steady_seconds();
      Tensor logits = model_->forward(x_d);
      grad_logits.resize({batch.n, batch.num_classes});
      const bool do_log = ((global_step % log_every) == 0);
      const bool compute_step_scalars = do_log || !cfg_.enable_log_sync_optimizations;

      const float loss = softmax_cross_entropy_backward(logits, batch.y, grad_logits, compute_step_scalars);
      (void)model_->backward(grad_logits);

      // SGD update (optionally via CUDA graph replay).
      sgd_graph.run(model_->params(), cfg_.lr, cfg_.weight_decay);

      const double step_s = steady_seconds() - step_start;
      samples += static_cast<double>(batch.n);
      step_s_sum += step_s;
      summary.steps += 1;

      int correct = 0;
      float acc = 0.0f;
      size_t free_b = 0;
      size_t total_b = 0;
      AllocatorStats a{};

      if (compute_step_scalars) {
        correct = argmax_accuracy(logits, batch.y);
        acc = static_cast<float>(correct) / static_cast<float>(batch.n);
        GPU_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        a = allocator_stats();
      }

      if (do_log) {
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

        const bool do_norm = (!have_norm) || ((global_step % norm_every) == 0);
        if (do_norm) {
          double w_sumsq = 0.0;
          double g_sumsq = 0.0;
          for (const auto& prm : model_->params()) {
            w_sumsq += tensor_sumsq(*prm.w);
            g_sumsq += tensor_sumsq(*prm.grad);
          }
          last_param_l2 = std::sqrt(w_sumsq);
          last_grad_l2 = std::sqrt(g_sumsq);
          have_norm = true;
        }
        if (have_norm) {
          p.scalars["param_l2"] = last_param_l2;
          p.scalars["grad_l2"] = last_grad_l2;
        }

        metrics.write_point(p);
      }

      const uint64_t eval_every = (cfg_.eval_every <= 0) ? 0ULL : static_cast<uint64_t>(cfg_.eval_every);
      if (eval_every > 0 && global_step > 0 && (global_step % eval_every) == 0) {
        if (cfg_.enable_async_eval && async_evaluator) {
          request_checkpoint(global_step);
          async_evaluator->request(epoch, global_step);
        } else {
          run_eval(epoch, global_step);
        }
      }

      if (cfg_.save_every > 0 && global_step > 0 &&
          (global_step % static_cast<uint64_t>(cfg_.save_every) == 0)) {
        request_checkpoint(global_step);
      }

      global_step += 1;
      if (cfg_.max_steps > 0 && global_step >= static_cast<uint64_t>(cfg_.max_steps)) {
        return false;
      }
      return true;
    };

    if (cfg_.enable_h2d_pipeline) {
      const size_t max_input_elems = static_cast<size_t>(cfg_.batch_size) *
                                     static_cast<size_t>(dataset_->channels()) *
                                     static_cast<size_t>(dataset_->height()) *
                                     static_cast<size_t>(dataset_->width());
      H2DPipeline h2d(max_input_elems);

      if (!dataset_->next_train_batch(cfg_.batch_size, batch)) {
        Logger::instance().warn("No training samples for epoch " + std::to_string(epoch));
        continue;
      }

      int cur_slot = 0;
      h2d.stage_batch_async(cur_slot, batch);

      while (true) {
        Batch next_batch;
        const bool has_next = dataset_->next_train_batch(cfg_.batch_size, next_batch);
        const int next_slot = cur_slot ^ 1;
        if (has_next) {
          h2d.stage_batch_async(next_slot, next_batch);
        }

        h2d.wait_until_ready(cur_slot);
        Tensor& x_d = h2d.tensor(cur_slot);
        if (!train_one_step(x_d)) {
          stop_training = true;
          break;
        }

        if (!has_next) break;
        batch = std::move(next_batch);
        cur_slot = next_slot;
      }
    } else {
      while (dataset_->next_train_batch(cfg_.batch_size, batch)) {
        Tensor x_d({batch.n, batch.c, batch.h, batch.w}, "x");
        x_d.copy_from_host(batch.x.data(), batch.x.size());
        if (!train_one_step(x_d)) {
          stop_training = true;
          break;
        }
      }
    }

    const double epoch_s = steady_seconds() - epoch_start;
    Logger::instance().info("Epoch " + std::to_string(epoch) + " done: " + std::to_string(samples / epoch_s) +
                            " samples/s");

    if (!stop_training) {
      if (cfg_.enable_async_eval && async_evaluator) {
        request_checkpoint(global_step);
        async_evaluator->request(epoch, global_step);
      } else {
        run_eval(epoch, global_step);
      }
    }
    if (stop_training) break;
  }

  if (ckpt_writer) ckpt_writer->flush();
  if (async_evaluator) async_evaluator->flush();
  Profiler::instance().flush();
  summary.avg_step_s = summary.steps ? (step_s_sum / static_cast<double>(summary.steps)) : 0.0;
  Logger::instance().info("Train summary: steps=" + std::to_string(summary.steps) +
                          " avg_step_s=" + std::to_string(summary.avg_step_s));
  return summary;
}

}  // namespace gpu
