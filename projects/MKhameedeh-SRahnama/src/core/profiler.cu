#include "gpu/core/profiler.h"

#include <algorithm>
#include <fstream>

#include "gpu/utils/cuda_check.h"
#include "gpu/utils/time.h"

#if defined(ENABLE_NVTX) && ENABLE_NVTX
#include <nvToolsExt.h>
#endif

namespace gpu {

Profiler& Profiler::instance() {
  static Profiler p;
  return p;
}

void Profiler::set_run_dir(const std::filesystem::path& run_dir) {
  std::scoped_lock lk(mu_);
  kernel_path_ = run_dir / "kernels.jsonl";
  std::ofstream out(kernel_path_, std::ios::app);
}

void Profiler::record_ms(const std::string& name, double ms) {
  std::scoped_lock lk(mu_);
  auto& st = stats_[name];
  st.calls += 1;
  st.total_ms += ms;
  st.max_ms = std::max(st.max_ms, ms);

  if (!kernel_path_.empty()) {
    std::ofstream out(kernel_path_, std::ios::app);
    out << "{\"t_ms\":" << unix_millis() << ",\"name\":\"" << name << "\",\"ms\":" << ms << "}\n";
  }
}

void Profiler::flush() {
  std::scoped_lock lk(mu_);
  if (kernel_path_.empty()) return;
  std::ofstream out(kernel_path_.parent_path() / "kernel_summary.json", std::ios::trunc);
  out << "{";
  bool first = true;
  for (const auto& kv : stats_) {
    if (!first) out << ",";
    first = false;
    out << "\"" << kv.first << "\":{\"calls\":" << kv.second.calls << ",\"total_ms\":" << kv.second.total_ms
        << ",\"max_ms\":" << kv.second.max_ms << "}";
  }
  out << "}\n";
}

Profiler::ScopedGpuTimer::ScopedGpuTimer(const char* name) : name_(name) {
#if defined(ENABLE_NVTX) && ENABLE_NVTX
  nvtxRangePushA(name_);
#endif
  GPU_CUDA_CHECK(cudaEventCreate(&start_));
  GPU_CUDA_CHECK(cudaEventCreate(&stop_));
  GPU_CUDA_CHECK(cudaEventRecord(start_));
}

Profiler::ScopedGpuTimer::~ScopedGpuTimer() {
  GPU_CUDA_CHECK(cudaEventRecord(stop_));
  GPU_CUDA_CHECK(cudaEventSynchronize(stop_));
  float ms = 0.0f;
  GPU_CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
  GPU_CUDA_CHECK(cudaEventDestroy(start_));
  GPU_CUDA_CHECK(cudaEventDestroy(stop_));
#if defined(ENABLE_NVTX) && ENABLE_NVTX
  nvtxRangePop();
#endif
#if defined(ENABLE_PROFILING) && ENABLE_PROFILING
  Profiler::instance().record_ms(name_, static_cast<double>(ms));
#endif
}

}  // namespace gpu

