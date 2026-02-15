#pragma once

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

namespace gpu {

struct KernelStat {
  uint64_t calls{0};
  double total_ms{0.0};
  double max_ms{0.0};
  uint64_t total_bytes{0};
};

class Profiler {
 public:
  static Profiler& instance();

  void set_run_dir(const std::filesystem::path& run_dir);
  void record_ms(const std::string& name, double ms);
  void record_ms(const std::string& name, double ms, const char* kind);
  void record_ms_bytes(const std::string& name, double ms, const char* kind, uint64_t bytes);
  void record_kernel_launch(const std::string& name, const void* kernel_func, dim3 grid, dim3 block,
                            size_t dynamic_shared_bytes = 0, cudaStream_t stream = nullptr);
  void flush();

  class ScopedGpuTimer {
   public:
    explicit ScopedGpuTimer(const char* name);
    ~ScopedGpuTimer();

   private:
    const char* name_{nullptr};
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
  };

  class ScopedCpuTimer {
   public:
    explicit ScopedCpuTimer(const char* name);
    ~ScopedCpuTimer();

   private:
    const char* name_{nullptr};
    double t0_s_{0.0};
  };

 private:
  Profiler() = default;

  std::mutex mu_;
  std::unordered_map<std::string, KernelStat> stats_;
  std::filesystem::path kernel_path_;
  std::filesystem::path events_csv_path_;
  std::filesystem::path summary_csv_path_;
  std::filesystem::path kernel_launches_csv_path_;
};

}  // namespace gpu
