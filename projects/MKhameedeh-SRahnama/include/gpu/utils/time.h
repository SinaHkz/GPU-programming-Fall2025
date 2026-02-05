#pragma once

#include <chrono>
#include <cstdint>

namespace gpu {

inline uint64_t unix_millis() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

inline double steady_seconds() {
  using namespace std::chrono;
  return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

}  // namespace gpu

