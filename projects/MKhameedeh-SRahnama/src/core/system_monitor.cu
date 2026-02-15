#include "gpu/core/system_monitor.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include <cuda_runtime.h>

#include "gpu/utils/cuda_check.h"
#include "gpu/utils/fs.h"
#include "gpu/utils/time.h"

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#else
#include <sys/sysinfo.h>
#include <unistd.h>
#include <dlfcn.h>
#endif

namespace gpu {

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

static bool get_process_times(double& user_s, double& kernel_s) {
#if defined(_WIN32)
  FILETIME create_t{}, exit_t{}, kernel_t{}, user_t{};
  if (!GetProcessTimes(GetCurrentProcess(), &create_t, &exit_t, &kernel_t, &user_t)) return false;
  ULARGE_INTEGER ku{};
  ku.LowPart = kernel_t.dwLowDateTime;
  ku.HighPart = kernel_t.dwHighDateTime;
  ULARGE_INTEGER uu{};
  uu.LowPart = user_t.dwLowDateTime;
  uu.HighPart = user_t.dwHighDateTime;
  kernel_s = static_cast<double>(ku.QuadPart) / 1e7;
  user_s = static_cast<double>(uu.QuadPart) / 1e7;
  return true;
#else
  // /proc/self/stat: fields 14 (utime), 15 (stime) in clock ticks.
  std::FILE* f = std::fopen("/proc/self/stat", "r");
  if (!f) return false;
  char buf[4096];
  if (!std::fgets(buf, sizeof(buf), f)) {
    std::fclose(f);
    return false;
  }
  std::fclose(f);

  // Skip pid (1) and comm (2, may contain spaces but is inside parentheses).
  // Then parse until utime/stime.
  std::string s(buf);
  const auto rparen = s.rfind(')');
  if (rparen == std::string::npos) return false;
  std::string rest = s.substr(rparen + 2);

  // Now rest starts at field 3 (state). We need fields 14 and 15 overall => 12 and 13 in rest.
  unsigned long long uticks = 0;
  unsigned long long sticks = 0;
  int idx = 0;
  std::string tok;
  std::istringstream iss(rest);
  while (iss >> tok) {
    idx++;
    if (idx == 12) uticks = std::strtoull(tok.c_str(), nullptr, 10);
    if (idx == 13) {
      sticks = std::strtoull(tok.c_str(), nullptr, 10);
      break;
    }
  }
  const long hz = sysconf(_SC_CLK_TCK);
  if (hz <= 0) return false;
  user_s = static_cast<double>(uticks) / static_cast<double>(hz);
  kernel_s = static_cast<double>(sticks) / static_cast<double>(hz);
  return true;
#endif
}

static bool get_process_rss_mb(double& rss_mb) {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS pmc{};
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return false;
  rss_mb = static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
  return true;
#else
  std::FILE* f = std::fopen("/proc/self/status", "r");
  if (!f) return false;
  char line[512];
  while (std::fgets(line, sizeof(line), f)) {
    if (std::strncmp(line, "VmRSS:", 6) == 0) {
      unsigned long kb = 0;
      if (std::sscanf(line + 6, "%lu", &kb) == 1) {
        rss_mb = static_cast<double>(kb) / 1024.0;
        std::fclose(f);
        return true;
      }
    }
  }
  std::fclose(f);
  return false;
#endif
}

static bool get_system_memory_mb(double& total_mb, double& avail_mb) {
#if defined(_WIN32)
  MEMORYSTATUSEX st{};
  st.dwLength = sizeof(st);
  if (!GlobalMemoryStatusEx(&st)) return false;
  total_mb = static_cast<double>(st.ullTotalPhys) / (1024.0 * 1024.0);
  avail_mb = static_cast<double>(st.ullAvailPhys) / (1024.0 * 1024.0);
  return true;
#else
  struct sysinfo info {};
  if (sysinfo(&info) != 0) return false;
  const double unit = static_cast<double>(info.mem_unit);
  total_mb = (static_cast<double>(info.totalram) * unit) / (1024.0 * 1024.0);
  avail_mb = (static_cast<double>(info.freeram) * unit) / (1024.0 * 1024.0);
  return true;
#endif
}

// Minimal NVML dynamic loader (optional).
struct Nvml {
  bool ok{false};

#if defined(_WIN32)
  HMODULE lib{nullptr};
#else
  void* lib{nullptr};
#endif

  // Types
  using nvmlReturn_t = int;
  using nvmlDevice_t = struct nvmlDevice_st*;
  struct nvmlUtilization_t {
    unsigned int gpu;
    unsigned int memory;
  };
  struct nvmlMemory_t {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
  };

  // Constants
  static constexpr int NVML_SUCCESS = 0;
  static constexpr int NVML_TEMPERATURE_GPU = 0;
  static constexpr int NVML_CLOCK_SM = 0;
  static constexpr int NVML_CLOCK_MEM = 2;

  // Function pointers (subset)
  nvmlReturn_t (*nvmlInit_v2)() = nullptr;
  nvmlReturn_t (*nvmlShutdown)() = nullptr;
  nvmlReturn_t (*nvmlDeviceGetHandleByIndex_v2)(unsigned int, nvmlDevice_t*) = nullptr;
  nvmlReturn_t (*nvmlDeviceGetUtilizationRates)(nvmlDevice_t, nvmlUtilization_t*) = nullptr;
  nvmlReturn_t (*nvmlDeviceGetMemoryInfo)(nvmlDevice_t, nvmlMemory_t*) = nullptr;
  nvmlReturn_t (*nvmlDeviceGetPowerUsage)(nvmlDevice_t, unsigned int*) = nullptr;  // mW
  nvmlReturn_t (*nvmlDeviceGetTemperature)(nvmlDevice_t, unsigned int, unsigned int*) = nullptr;
  nvmlReturn_t (*nvmlDeviceGetClockInfo)(nvmlDevice_t, unsigned int, unsigned int*) = nullptr;  // MHz

  nvmlDevice_t dev{};

  void* sym_(const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(lib, name));
#else
    return dlsym(lib, name);
#endif
  }

  bool load(int device_index) {
#if defined(_WIN32)
    lib = LoadLibraryA("nvml.dll");
#else
    lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
#endif
    if (!lib) return false;

    nvmlInit_v2 = reinterpret_cast<nvmlReturn_t (*)()>(sym_("nvmlInit_v2"));
    nvmlShutdown = reinterpret_cast<nvmlReturn_t (*)()>(sym_("nvmlShutdown"));
    nvmlDeviceGetHandleByIndex_v2 =
        reinterpret_cast<nvmlReturn_t (*)(unsigned int, nvmlDevice_t*)>(sym_("nvmlDeviceGetHandleByIndex_v2"));
    nvmlDeviceGetUtilizationRates =
        reinterpret_cast<nvmlReturn_t (*)(nvmlDevice_t, nvmlUtilization_t*)>(sym_("nvmlDeviceGetUtilizationRates"));
    nvmlDeviceGetMemoryInfo =
        reinterpret_cast<nvmlReturn_t (*)(nvmlDevice_t, nvmlMemory_t*)>(sym_("nvmlDeviceGetMemoryInfo"));
    nvmlDeviceGetPowerUsage =
        reinterpret_cast<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*)>(sym_("nvmlDeviceGetPowerUsage"));
    nvmlDeviceGetTemperature =
        reinterpret_cast<nvmlReturn_t (*)(nvmlDevice_t, unsigned int, unsigned int*)>(sym_("nvmlDeviceGetTemperature"));
    nvmlDeviceGetClockInfo =
        reinterpret_cast<nvmlReturn_t (*)(nvmlDevice_t, unsigned int, unsigned int*)>(sym_("nvmlDeviceGetClockInfo"));

    if (!nvmlInit_v2 || !nvmlShutdown || !nvmlDeviceGetHandleByIndex_v2 || !nvmlDeviceGetUtilizationRates ||
        !nvmlDeviceGetMemoryInfo || !nvmlDeviceGetPowerUsage || !nvmlDeviceGetTemperature || !nvmlDeviceGetClockInfo) {
      return false;
    }

    if (nvmlInit_v2() != NVML_SUCCESS) return false;
    if (nvmlDeviceGetHandleByIndex_v2(static_cast<unsigned int>(device_index), &dev) != NVML_SUCCESS) return false;
    ok = true;
    return true;
  }

  void unload() {
    if (ok && nvmlShutdown) nvmlShutdown();
    ok = false;
#if defined(_WIN32)
    if (lib) FreeLibrary(lib);
#else
    if (lib) dlclose(lib);
#endif
    lib = nullptr;
  }

  ~Nvml() { unload(); }
};

SystemMonitor::~SystemMonitor() { stop(); }

void SystemMonitor::start(const std::filesystem::path& run_dir, int interval_ms, int device_index) {
  if (running_.load()) return;
  if (interval_ms <= 0) return;

  profiling_dir_ = run_dir / "profiling";
  ensure_dir(profiling_dir_);
  system_csv_ = profiling_dir_ / "system_metrics.csv";
  gpu_csv_ = profiling_dir_ / "gpu_metrics.csv";
  interval_ms_ = interval_ms;
  device_index_ = device_index;

  {
    std::ofstream out(system_csv_, std::ios::trunc);
    out << "t_ms,process_cpu_user_s,process_cpu_kernel_s,process_cpu_pct,process_rss_mb,system_mem_total_mb,system_mem_avail_mb\n";
  }
  {
    std::ofstream out(gpu_csv_, std::ios::trunc);
    out << "t_ms,device_index,cuda_mem_free_mb,cuda_mem_total_mb,cuda_mem_used_mb,"
           "nvml_util_gpu_pct,nvml_util_mem_pct,nvml_mem_used_mb,nvml_mem_total_mb,"
           "nvml_power_w,nvml_temp_c,nvml_sm_clock_mhz,nvml_mem_clock_mhz\n";
  }

  running_.store(true);
  thread_ = std::thread([this]() { thread_main_(); });
}

void SystemMonitor::stop() {
  if (!running_.exchange(false)) return;
  if (thread_.joinable()) thread_.join();
}

void SystemMonitor::thread_main_() {
  Nvml nvml;
  (void)nvml.load(device_index_);

  double last_user_s = 0.0;
  double last_kernel_s = 0.0;
  double last_wall_s = steady_seconds();
  (void)get_process_times(last_user_s, last_kernel_s);

  while (running_.load()) {
    const uint64_t t_ms = unix_millis();

    // System/process metrics
    double user_s = -1.0;
    double kernel_s = -1.0;
    double rss_mb = -1.0;
    double mem_total_mb = -1.0;
    double mem_avail_mb = -1.0;
    (void)get_process_times(user_s, kernel_s);
    (void)get_process_rss_mb(rss_mb);
    (void)get_system_memory_mb(mem_total_mb, mem_avail_mb);

    const double wall_s = steady_seconds();
    const double dt_wall = wall_s - last_wall_s;
    const double dt_cpu = (user_s - last_user_s) + (kernel_s - last_kernel_s);
    const double cpu_pct = (dt_wall > 0.0 && dt_cpu >= 0.0) ? (dt_cpu / dt_wall * 100.0) : -1.0;
    last_user_s = user_s;
    last_kernel_s = kernel_s;
    last_wall_s = wall_s;

    {
      std::ofstream out(system_csv_, std::ios::app);
      out << t_ms << "," << user_s << "," << kernel_s << "," << cpu_pct << "," << rss_mb << "," << mem_total_mb << ","
          << mem_avail_mb << "\n";
    }

    // GPU metrics
    size_t free_b = 0;
    size_t total_b = 0;
    cudaError_t mem_err = cudaMemGetInfo(&free_b, &total_b);
    const double cuda_free_mb = (mem_err == cudaSuccess) ? static_cast<double>(free_b) / (1024.0 * 1024.0) : -1.0;
    const double cuda_total_mb = (mem_err == cudaSuccess) ? static_cast<double>(total_b) / (1024.0 * 1024.0) : -1.0;
    const double cuda_used_mb =
        (mem_err == cudaSuccess) ? static_cast<double>(total_b - free_b) / (1024.0 * 1024.0) : -1.0;

    int util_gpu = -1;
    int util_mem = -1;
    double nvml_mem_used_mb = -1.0;
    double nvml_mem_total_mb = -1.0;
    double power_w = -1.0;
    int temp_c = -1;
    int sm_clock_mhz = -1;
    int mem_clock_mhz = -1;

    if (nvml.ok) {
      Nvml::nvmlUtilization_t u{};
      if (nvml.nvmlDeviceGetUtilizationRates(nvml.dev, &u) == Nvml::NVML_SUCCESS) {
        util_gpu = static_cast<int>(u.gpu);
        util_mem = static_cast<int>(u.memory);
      }

      Nvml::nvmlMemory_t m{};
      if (nvml.nvmlDeviceGetMemoryInfo(nvml.dev, &m) == Nvml::NVML_SUCCESS) {
        nvml_mem_used_mb = static_cast<double>(m.used) / (1024.0 * 1024.0);
        nvml_mem_total_mb = static_cast<double>(m.total) / (1024.0 * 1024.0);
      }

      unsigned int mw = 0;
      if (nvml.nvmlDeviceGetPowerUsage(nvml.dev, &mw) == Nvml::NVML_SUCCESS) {
        power_w = static_cast<double>(mw) / 1000.0;
      }

      unsigned int tc = 0;
      if (nvml.nvmlDeviceGetTemperature(nvml.dev, Nvml::NVML_TEMPERATURE_GPU, &tc) == Nvml::NVML_SUCCESS) {
        temp_c = static_cast<int>(tc);
      }

      unsigned int clk = 0;
      if (nvml.nvmlDeviceGetClockInfo(nvml.dev, Nvml::NVML_CLOCK_SM, &clk) == Nvml::NVML_SUCCESS) {
        sm_clock_mhz = static_cast<int>(clk);
      }
      if (nvml.nvmlDeviceGetClockInfo(nvml.dev, Nvml::NVML_CLOCK_MEM, &clk) == Nvml::NVML_SUCCESS) {
        mem_clock_mhz = static_cast<int>(clk);
      }
    }

    {
      std::ofstream out(gpu_csv_, std::ios::app);
      out << t_ms << "," << device_index_ << "," << cuda_free_mb << "," << cuda_total_mb << "," << cuda_used_mb << ","
          << util_gpu << "," << util_mem << "," << nvml_mem_used_mb << "," << nvml_mem_total_mb << "," << power_w
          << "," << temp_c << "," << sm_clock_mhz << "," << mem_clock_mhz << "\n";
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));
  }
}

}  // namespace gpu
