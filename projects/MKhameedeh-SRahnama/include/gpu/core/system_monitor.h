#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <thread>

namespace gpu {

// Periodically samples process + system + GPU metrics and appends them to CSV files
// in `<run_dir>/profiling/`.
//
// Designed to be best-effort:
// - GPU utilization/temperature/power require NVML; if unavailable, only CUDA memory
//   info is recorded.
// - System-level CPU utilization is not recorded; process CPU time + derived %
//   are recorded instead.
class SystemMonitor {
 public:
  SystemMonitor() = default;
  ~SystemMonitor();

  SystemMonitor(const SystemMonitor&) = delete;
  SystemMonitor& operator=(const SystemMonitor&) = delete;

  void start(const std::filesystem::path& run_dir, int interval_ms, int device_index = 0);
  void stop();

 private:
  void thread_main_();

  std::atomic<bool> running_{false};
  std::thread thread_;

  std::filesystem::path profiling_dir_;
  std::filesystem::path system_csv_;
  std::filesystem::path gpu_csv_;
  int interval_ms_{0};
  int device_index_{0};
};

}  // namespace gpu

