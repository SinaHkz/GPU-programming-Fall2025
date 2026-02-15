#include "gpu/core/logger.h"

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "gpu/utils/time.h"

namespace gpu {

static const char* level_name(LogLevel level) {
  switch (level) {
    case LogLevel::kDebug:
      return "DEBUG";
    case LogLevel::kInfo:
      return "INFO";
    case LogLevel::kWarn:
      return "WARN";
    case LogLevel::kError:
      return "ERROR";
  }
  return "INFO";
}

Logger& Logger::instance() {
  static Logger g;
  return g;
}

void Logger::set_level(LogLevel level) { level_ = level; }

void Logger::set_run_dir(const std::filesystem::path& run_dir) {
  std::scoped_lock lk(mu_);
  log_path_ = run_dir / "train.log";
}

void Logger::debug(const std::string& msg) { log(LogLevel::kDebug, msg); }
void Logger::info(const std::string& msg) { log(LogLevel::kInfo, msg); }
void Logger::warn(const std::string& msg) { log(LogLevel::kWarn, msg); }
void Logger::error(const std::string& msg) { log(LogLevel::kError, msg); }

void Logger::log(LogLevel level, const std::string& msg) {
  if (static_cast<int>(level) < static_cast<int>(level_)) return;

  const uint64_t ms = unix_millis();
  std::ostringstream line;
  line << ms << " [" << level_name(level) << "] " << msg;

  std::scoped_lock lk(mu_);
  std::cout << line.str() << "\n";
  std::cout.flush();

  if (!log_path_.empty()) {
    std::ofstream out(log_path_, std::ios::app);
    out << line.str() << "\n";
  }
}

}  // namespace gpu

