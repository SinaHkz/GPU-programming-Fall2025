#pragma once

#include <filesystem>
#include <mutex>
#include <string>

namespace gpu {

enum class LogLevel { kDebug = 0, kInfo = 1, kWarn = 2, kError = 3 };

class Logger {
 public:
  static Logger& instance();

  void set_level(LogLevel level);
  void set_run_dir(const std::filesystem::path& run_dir);

  void debug(const std::string& msg);
  void info(const std::string& msg);
  void warn(const std::string& msg);
  void error(const std::string& msg);

 private:
  Logger() = default;
  void log(LogLevel level, const std::string& msg);

  std::mutex mu_;
  LogLevel level_{LogLevel::kInfo};
  std::filesystem::path log_path_;
};

}  // namespace gpu

