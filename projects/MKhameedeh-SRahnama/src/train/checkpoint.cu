#include "gpu/train/checkpoint.h"

#include <cuda_runtime.h>

#include <condition_variable>
#include <deque>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "gpu/core/tensor.h"
#include "gpu/core/logger.h"
#include "gpu/utils/cuda_check.h"
#include "gpu/utils/fs.h"

namespace gpu {

namespace {

struct CheckpointEntryMeta {
  std::string name;
  std::vector<int> shape;
  uint64_t offset_bytes{0};
  uint64_t numel{0};
};

static std::string json_escape(const std::string& s) {
  std::ostringstream o;
  for (char c : s) {
    switch (c) {
      case '\\':
      case '"':
        o << '\\' << c;
        break;
      case '\n':
        o << "\\n";
        break;
      case '\r':
        o << "\\r";
        break;
      case '\t':
        o << "\\t";
        break;
      default:
        o << c;
    }
  }
  return o.str();
}

static void write_shape(std::ostream& out, const std::vector<int>& shape) {
  out << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) out << ",";
    out << shape[i];
  }
  out << "]";
}

static std::filesystem::path write_checkpoint_files(const std::filesystem::path& dir, const std::string& model_name,
                                                    uint64_t step, const std::vector<CheckpointEntryMeta>& entries,
                                                    const std::vector<const float*>& host_ptrs) {
  ensure_dir(dir);
  if (entries.size() != host_ptrs.size()) {
    throw std::runtime_error("Checkpoint write: entries/data size mismatch");
  }

  const std::filesystem::path bin_path = dir / ("checkpoint_step_" + std::to_string(step) + ".bin");
  const std::filesystem::path json_path = dir / ("checkpoint_step_" + std::to_string(step) + ".json");

  std::ofstream bin(bin_path, std::ios::binary | std::ios::trunc);
  if (!bin) throw std::runtime_error("Failed opening checkpoint bin for write: " + bin_path.string());

  for (size_t i = 0; i < entries.size(); ++i) {
    const auto& e = entries[i];
    const auto* ptr = host_ptrs[i];
    if (!ptr && e.numel > 0) {
      throw std::runtime_error("Checkpoint write: null param buffer for " + e.name);
    }
    bin.write(reinterpret_cast<const char*>(ptr), static_cast<std::streamsize>(e.numel * sizeof(float)));
  }
  bin.flush();

  std::ofstream json(json_path, std::ios::trunc);
  if (!json) throw std::runtime_error("Failed opening checkpoint json for write: " + json_path.string());

  json << "{";
  json << "\"model\":\"" << json_escape(model_name) << "\",";
  json << "\"step\":" << step << ",";
  json << "\"bin_file\":\"" << json_escape(bin_path.filename().string()) << "\",";
  json << "\"params\":[";
  for (size_t i = 0; i < entries.size(); ++i) {
    if (i) json << ",";
    const auto& e = entries[i];
    json << "{";
    json << "\"name\":\"" << json_escape(e.name) << "\",";
    json << "\"shape\":";
    write_shape(json, e.shape);
    json << ",";
    json << "\"offset_bytes\":" << e.offset_bytes << ",";
    json << "\"numel\":" << e.numel;
    json << "}";
  }
  json << "]";
  json << "}\n";
  return json_path;
}

struct AsyncSnapshot {
  explicit AsyncSnapshot(size_t n_params) : host_ptrs(n_params, nullptr) {}
  ~AsyncSnapshot() {
    if (ready) cudaEventDestroy(ready);
    for (auto* p : host_ptrs) {
      if (p) cudaFreeHost(p);
    }
  }

  uint64_t step{0};
  std::vector<float*> host_ptrs;
  cudaEvent_t ready{};
};

struct AsyncParamState {
  Tensor* weight{nullptr};
  Tensor staging;
  CheckpointEntryMeta meta;
};

}  // namespace

std::filesystem::path save_checkpoint(const std::filesystem::path& dir, const Model& model, uint64_t step) {
  std::vector<CheckpointEntryMeta> entries;
  std::vector<std::vector<float>> host_data;
  std::vector<const float*> host_ptrs;
  entries.reserve(model.params().size());
  host_data.reserve(model.params().size());
  host_ptrs.reserve(model.params().size());

  uint64_t offset = 0;
  for (const auto& p : model.params()) {
    CheckpointEntryMeta e;
    e.name = p.name;
    e.shape = p.w->shape();
    e.numel = static_cast<uint64_t>(p.w->numel());
    e.offset_bytes = offset;

    host_data.emplace_back(p.w->numel());
    auto& host = host_data.back();
    p.w->copy_to_host(host.data(), host.size());
    host_ptrs.push_back(host.data());

    offset += static_cast<uint64_t>(host.size() * sizeof(float));
    entries.push_back(std::move(e));
  }
  const auto json_path = write_checkpoint_files(dir, model.name(), step, entries, host_ptrs);

  Logger::instance().info("Saved checkpoint: " + json_path.string());
  return json_path;
}

// Minimal JSON parser for our own checkpoint format.
static std::string find_string_field(const std::string& s, const std::string& key) {
  const std::string needle = "\"" + key + "\":\"";
  const size_t pos = s.find(needle);
  if (pos == std::string::npos) return {};
  size_t i = pos + needle.size();
  std::string out;
  while (i < s.size()) {
    char c = s[i++];
    if (c == '"') break;
    if (c == '\\' && i < s.size()) {
      char n = s[i++];
      if (n == 'n') out.push_back('\n');
      else if (n == 'r') out.push_back('\r');
      else if (n == 't') out.push_back('\t');
      else out.push_back(n);
    } else {
      out.push_back(c);
    }
  }
  return out;
}

static uint64_t find_u64_field(const std::string& s, const std::string& key) {
  const std::string needle = "\"" + key + "\":";
  const size_t pos = s.find(needle);
  if (pos == std::string::npos) return 0;
  size_t i = pos + needle.size();
  while (i < s.size() && (s[i] == ' ')) ++i;
  size_t j = i;
  while (j < s.size() && (s[j] >= '0' && s[j] <= '9')) ++j;
  return static_cast<uint64_t>(std::stoull(s.substr(i, j - i)));
}

CheckpointMeta load_checkpoint(const std::filesystem::path& checkpoint_json, Model& model) {
  const std::string txt = read_text_file(checkpoint_json);
  if (txt.empty()) throw std::runtime_error("Failed reading checkpoint json: " + checkpoint_json.string());

  const std::string bin_file = find_string_field(txt, "bin_file");
  const uint64_t step = find_u64_field(txt, "step");
  if (bin_file.empty()) throw std::runtime_error("Checkpoint missing bin_file: " + checkpoint_json.string());

  const auto bin_path = checkpoint_json.parent_path() / bin_file;
  std::ifstream bin(bin_path, std::ios::binary);
  if (!bin) throw std::runtime_error("Failed opening checkpoint bin: " + bin_path.string());

  // Our save format writes params in the same order as model.params().
  uint64_t offset = 0;
  for (const auto& p : model.params()) {
    std::vector<float> host(p.w->numel());
    bin.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    bin.read(reinterpret_cast<char*>(host.data()), static_cast<std::streamsize>(host.size() * sizeof(float)));
    if (!bin) throw std::runtime_error("Checkpoint read failed at param: " + p.name);
    p.w->copy_from_host(host.data(), host.size());
    offset += static_cast<uint64_t>(host.size() * sizeof(float));
  }

  Logger::instance().info("Loaded checkpoint: " + checkpoint_json.string());
  CheckpointMeta m;
  m.step = step;
  return m;
}

struct AsyncCheckpointWriter::Impl {
  Impl(const std::filesystem::path& dir, const Model& model, size_t max_pending)
      : dir_(dir), model_name_(model.name()), max_pending_(max_pending == 0 ? 1 : max_pending) {
    ensure_dir(dir_);
    GPU_CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));

    uint64_t offset = 0;
    for (const auto& p : model.params()) {
      AsyncParamState st;
      st.weight = p.w;
      st.meta.name = p.name;
      st.meta.shape = p.w->shape();
      st.meta.numel = static_cast<uint64_t>(p.w->numel());
      st.meta.offset_bytes = offset;
      st.staging = Tensor(st.meta.shape, p.name + ".ckpt_staging");
      offset += st.meta.numel * sizeof(float);
      params_.push_back(std::move(st));
    }

    worker_ = std::thread([this]() { worker_main_(); });
  }

  ~Impl() {
    flush();
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_work_.notify_all();
    if (worker_.joinable()) worker_.join();
    if (copy_stream_) cudaStreamDestroy(copy_stream_);
  }

  void enqueue(uint64_t step) {
    if (params_.empty()) return;

    auto snap = std::make_unique<AsyncSnapshot>(params_.size());
    snap->step = step;
    GPU_CUDA_CHECK(cudaEventCreateWithFlags(&snap->ready, cudaEventDisableTiming));

    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_space_.wait(lk, [&]() { return stop_ || (pending_.size() + active_) < max_pending_; });
      if (stop_) throw std::runtime_error("AsyncCheckpointWriter is stopping");
    }

    for (size_t i = 0; i < params_.size(); ++i) {
      const auto bytes = params_[i].meta.numel * sizeof(float);
      if (bytes == 0) continue;
      GPU_CUDA_CHECK(cudaHostAlloc(&snap->host_ptrs[i], bytes, cudaHostAllocDefault));
      GPU_CUDA_CHECK(cudaMemcpyAsync(params_[i].staging.data(), params_[i].weight->data(), bytes,
                                     cudaMemcpyDeviceToDevice, 0));
    }

    cudaEvent_t stage_done{};
    GPU_CUDA_CHECK(cudaEventCreateWithFlags(&stage_done, cudaEventDisableTiming));
    GPU_CUDA_CHECK(cudaEventRecord(stage_done, 0));
    GPU_CUDA_CHECK(cudaStreamWaitEvent(copy_stream_, stage_done, 0));
    GPU_CUDA_CHECK(cudaEventDestroy(stage_done));

    for (size_t i = 0; i < params_.size(); ++i) {
      const auto bytes = params_[i].meta.numel * sizeof(float);
      if (bytes == 0) continue;
      GPU_CUDA_CHECK(cudaMemcpyAsync(snap->host_ptrs[i], params_[i].staging.data(), bytes, cudaMemcpyDeviceToHost,
                                     copy_stream_));
    }
    GPU_CUDA_CHECK(cudaEventRecord(snap->ready, copy_stream_));

    {
      std::lock_guard<std::mutex> lk(mu_);
      pending_.push_back(std::move(snap));
    }
    cv_work_.notify_one();
  }

  void flush() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_drained_.wait(lk, [&]() { return pending_.empty() && active_ == 0; });
  }

  void worker_main_() {
    while (true) {
      std::unique_ptr<AsyncSnapshot> snap;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_work_.wait(lk, [&]() { return stop_ || !pending_.empty(); });
        if (stop_ && pending_.empty()) return;

        snap = std::move(pending_.front());
        pending_.pop_front();
        active_ += 1;
        cv_space_.notify_all();
      }

      try {
        GPU_CUDA_CHECK(cudaEventSynchronize(snap->ready));
        std::vector<CheckpointEntryMeta> entries;
        std::vector<const float*> host_ptrs;
        entries.reserve(params_.size());
        host_ptrs.reserve(params_.size());
        for (size_t i = 0; i < params_.size(); ++i) {
          entries.push_back(params_[i].meta);
          host_ptrs.push_back(snap->host_ptrs[i]);
        }
        const auto json_path = write_checkpoint_files(dir_, model_name_, snap->step, entries, host_ptrs);
        Logger::instance().info("Saved checkpoint (async): " + json_path.string());
      } catch (const std::exception& e) {
        Logger::instance().error(std::string("Async checkpoint failed: ") + e.what());
      }

      {
        std::lock_guard<std::mutex> lk(mu_);
        if (active_ > 0) active_ -= 1;
        if (pending_.empty() && active_ == 0) cv_drained_.notify_all();
        cv_space_.notify_all();
      }
    }
  }

  std::filesystem::path dir_;
  std::string model_name_;
  std::vector<AsyncParamState> params_;
  size_t max_pending_{2};

  cudaStream_t copy_stream_{};
  std::thread worker_;
  std::mutex mu_;
  std::condition_variable cv_work_;
  std::condition_variable cv_space_;
  std::condition_variable cv_drained_;
  std::deque<std::unique_ptr<AsyncSnapshot>> pending_;
  size_t active_{0};
  bool stop_{false};
};

AsyncCheckpointWriter::AsyncCheckpointWriter(const std::filesystem::path& dir, const Model& model, size_t max_pending)
    : impl_(std::make_unique<Impl>(dir, model, max_pending)) {}

AsyncCheckpointWriter::~AsyncCheckpointWriter() = default;

void AsyncCheckpointWriter::enqueue(uint64_t step) { impl_->enqueue(step); }

void AsyncCheckpointWriter::flush() { impl_->flush(); }

}  // namespace gpu
