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
};

class Profiler {
 public:
  static Profiler& instance();

  void set_run_dir(const std::filesystem::path& run_dir);
  void record_ms(const std::string& name, double ms);
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

 private:
  Profiler() = default;

  std::mutex mu_;
  std::unordered_map<std::string, KernelStat> stats_;
  std::filesystem::path kernel_path_;
};

}  // namespace gpu

