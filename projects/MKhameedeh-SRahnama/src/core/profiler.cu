#include "gpu/core/profiler.h"

#include <algorithm>
#include <fstream>
#include <sstream>

#include "gpu/utils/cuda_check.h"
#include "gpu/utils/fs.h"
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

  const auto profiling_dir = run_dir / "profiling";
  ensure_dir(profiling_dir);
  events_csv_path_ = profiling_dir / "functions_events.csv";
  summary_csv_path_ = profiling_dir / "functions_summary.csv";
  kernel_launches_csv_path_ = profiling_dir / "kernel_launches.csv";

  {
    std::ofstream csv(events_csv_path_, std::ios::trunc);
    csv << "t_ms,kind,name,ms,bytes,gbps\n";
  }
  {
    std::ofstream csv(summary_csv_path_, std::ios::trunc);
    csv << "kind,name,calls,total_ms,max_ms,avg_ms,total_bytes,avg_gbps\n";
  }
  {
    std::ofstream csv(kernel_launches_csv_path_, std::ios::trunc);
    csv << "t_ms,name,device_index,sm_count,warp_size,max_threads_per_sm,"
           "grid_x,grid_y,grid_z,block_x,block_y,block_z,threads_per_block,warps_per_block,"
           "dynamic_shared_bytes,static_shared_bytes,regs_per_thread,local_bytes_per_thread,max_threads_per_block,"
           "active_blocks_per_sm,active_warps_per_sm,max_warps_per_sm,occupancy_pct\n";
  }
}

void Profiler::record_ms(const std::string& name, double ms) { record_ms(name, ms, "gpu"); }

static void write_csv_escaped(std::ostream& os, const std::string& s) {
  bool needs_quotes = false;
  for (char c : s) {
    if (c == '"' || c == ',' || c == '\n' || c == '\r') {
      needs_quotes = true;
      break;
    }
  }
  if (!needs_quotes) {
    os << s;
    return;
  }
  os << '"';
  for (char c : s) {
    if (c == '"') os << "\"\"";
    else os << c;
  }
  os << '"';
}

static double bytes_to_gbps(uint64_t bytes, double ms) {
  if (bytes == 0 || ms <= 0.0) return 0.0;
  const double gb = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
  const double s = ms / 1000.0;
  return s > 0.0 ? (gb / s) : 0.0;
}

void Profiler::record_ms(const std::string& name, double ms, const char* kind) {
  record_ms_bytes(name, ms, kind, 0);
}

void Profiler::record_ms_bytes(const std::string& name, double ms, const char* kind, uint64_t bytes) {
  std::scoped_lock lk(mu_);
  const std::string kind_s = kind ? kind : "unknown";
  const std::string k = kind_s + ":" + name;
  auto& st = stats_[k];
  st.calls += 1;
  st.total_ms += ms;
  st.max_ms = std::max(st.max_ms, ms);
  st.total_bytes += bytes;

  if (!kernel_path_.empty() && kind_s == "gpu") {
    std::ofstream out(kernel_path_, std::ios::app);
    out << "{\"t_ms\":" << unix_millis() << ",\"name\":\"" << name << "\",\"ms\":" << ms << "}\n";
  }

  if (!events_csv_path_.empty()) {
    std::ofstream csv(events_csv_path_, std::ios::app);
    csv << unix_millis() << ",";
    write_csv_escaped(csv, kind ? kind : "unknown");
    csv << ",";
    write_csv_escaped(csv, name);
    csv << "," << ms << "," << bytes << "," << bytes_to_gbps(bytes, ms) << "\n";
  }
}

void Profiler::record_kernel_launch(const std::string& name, const void* kernel_func, dim3 grid, dim3 block,
                                    size_t dynamic_shared_bytes, cudaStream_t /*stream*/) {
  std::scoped_lock lk(mu_);
  if (kernel_launches_csv_path_.empty() || kernel_func == nullptr) return;

  struct AttrCache {
    int static_shared_bytes{-1};
    int regs_per_thread{-1};
    int local_bytes_per_thread{-1};
    int max_threads_per_block{-1};
  };
  static std::unordered_map<const void*, AttrCache> g_attr_cache;

  int dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, dev);

  AttrCache ac;
  auto it = g_attr_cache.find(kernel_func);
  if (it != g_attr_cache.end()) {
    ac = it->second;
  } else {
    cudaFuncAttributes attr{};
    cudaError_t attr_err = cudaFuncGetAttributes(&attr, kernel_func);
    if (attr_err == cudaSuccess) {
      ac.static_shared_bytes = attr.sharedSizeBytes;
      ac.regs_per_thread = attr.numRegs;
      ac.local_bytes_per_thread = attr.localSizeBytes;
      ac.max_threads_per_block = attr.maxThreadsPerBlock;
    }
    g_attr_cache[kernel_func] = ac;
  }

  int active_blocks_per_sm = 0;
  cudaError_t occ_err =
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks_per_sm, kernel_func,
                                                    static_cast<int>(block.x * block.y * block.z),
                                                    dynamic_shared_bytes);

  const int threads_per_block = static_cast<int>(block.x * block.y * block.z);
  const int warp_size = prop.warpSize ? prop.warpSize : 32;
  const int warps_per_block = (threads_per_block + warp_size - 1) / warp_size;
  const int max_warps_per_sm = (prop.maxThreadsPerMultiProcessor > 0) ? (prop.maxThreadsPerMultiProcessor / warp_size)
                                                                     : 0;

  const int grid_blocks = static_cast<int>(grid.x * grid.y * grid.z);
  // Total blocks that can physically reside on the GPU simultaneously for this kernel
  const int max_resident_blocks_gpu = prop.multiProcessorCount * active_blocks_per_sm;
  // Actual blocks that will reside (limited by grid size and resource availability)
  const int actual_resident_blocks = std::min(grid_blocks, max_resident_blocks_gpu);
  const int actual_resident_warps = actual_resident_blocks * warps_per_block;
  const int max_capacity_warps_gpu = prop.multiProcessorCount * max_warps_per_sm;

  double occ_pct = 0.0;
  if (max_capacity_warps_gpu > 0) {
    occ_pct = 100.0 * static_cast<double>(actual_resident_warps) / static_cast<double>(max_capacity_warps_gpu);
  }

  // For CSV logging, we'll output the "achieved" metrics per SM
  const double achieved_blocks_per_sm = static_cast<double>(actual_resident_blocks) / prop.multiProcessorCount;
  const double active_warps_per_sm = static_cast<double>(actual_resident_warps) / prop.multiProcessorCount;

  std::ofstream csv(kernel_launches_csv_path_, std::ios::app);
  csv << unix_millis() << ",";
  write_csv_escaped(csv, name);
  csv << "," << dev << "," << prop.multiProcessorCount << "," << warp_size << "," << prop.maxThreadsPerMultiProcessor
      << "," << grid.x << "," << grid.y << "," << grid.z << "," << block.x << "," << block.y << "," << block.z << ","
      << threads_per_block << "," << warps_per_block << "," << dynamic_shared_bytes << "," << ac.static_shared_bytes
      << "," << ac.regs_per_thread << "," << ac.local_bytes_per_thread << "," << ac.max_threads_per_block << ","
      << achieved_blocks_per_sm << "," << active_warps_per_sm << "," << max_warps_per_sm << "," << occ_pct << "\n";
}

void Profiler::flush() {
  std::scoped_lock lk(mu_);
  if (kernel_path_.empty()) return;
  std::ofstream out(kernel_path_.parent_path() / "kernel_summary.json", std::ios::trunc);
  out << "{";
  bool first = true;
  for (const auto& kv : stats_) {
    const auto pos = kv.first.find(':');
    const std::string kind = (pos == std::string::npos) ? "unknown" : kv.first.substr(0, pos);
    if (kind != "gpu") continue;
    const std::string name = (pos == std::string::npos) ? kv.first : kv.first.substr(pos + 1);

    if (!first) out << ",";
    first = false;
    out << "\"" << name << "\":{\"calls\":" << kv.second.calls << ",\"total_ms\":" << kv.second.total_ms
        << ",\"max_ms\":" << kv.second.max_ms << "}";
  }
  out << "}\n";

  if (!summary_csv_path_.empty()) {
    std::ofstream csv(summary_csv_path_, std::ios::trunc);
    csv << "kind,name,calls,total_ms,max_ms,avg_ms,total_bytes,avg_gbps\n";
    for (const auto& kv : stats_) {
      const auto& key = kv.first;
      const auto pos = key.find(':');
      const std::string kind = (pos == std::string::npos) ? "unknown" : key.substr(0, pos);
      const std::string name = (pos == std::string::npos) ? key : key.substr(pos + 1);
      const double avg = kv.second.calls ? (kv.second.total_ms / static_cast<double>(kv.second.calls)) : 0.0;
      const double avg_gbps = bytes_to_gbps(kv.second.total_bytes, kv.second.total_ms);
      write_csv_escaped(csv, kind);
      csv << ",";
      write_csv_escaped(csv, name);
      csv << "," << kv.second.calls << "," << kv.second.total_ms << "," << kv.second.max_ms << "," << avg << ","
          << kv.second.total_bytes << "," << avg_gbps << "\n";
    }
  }
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
  Profiler::instance().record_ms(name_, static_cast<double>(ms), "gpu");
#endif
}

Profiler::ScopedCpuTimer::ScopedCpuTimer(const char* name) : name_(name) {
#if defined(ENABLE_NVTX) && ENABLE_NVTX
  nvtxRangePushA(name_);
#endif
  t0_s_ = steady_seconds();
}

Profiler::ScopedCpuTimer::~ScopedCpuTimer() {
  const double ms = (steady_seconds() - t0_s_) * 1000.0;
  Profiler::instance().record_ms(name_, ms, "cpu");
#if defined(ENABLE_NVTX) && ENABLE_NVTX
  nvtxRangePop();
#endif
}

}  // namespace gpu
