#pragma once

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>

namespace gpu {

struct MetricPoint {
  uint64_t t_ms{0};
  std::unordered_map<std::string, double> scalars;
};

class MetricsSink {
 public:
  void set_run_dir(const std::filesystem::path& run_dir);
  void write_point(const MetricPoint& p);

 private:
  std::mutex mu_;
  std::filesystem::path metrics_path_;
  std::filesystem::path metrics_csv_path_;
};

}  // namespace gpu
