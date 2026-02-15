#pragma once

#include <filesystem>
#include <cstdio>
#include <string>

namespace gpu {

inline void ensure_dir(const std::filesystem::path& p) { std::filesystem::create_directories(p); }

inline std::string read_text_file(const std::filesystem::path& p) {
  std::FILE* f = std::fopen(p.string().c_str(), "rb");
  if (!f) return {};
  std::fseek(f, 0, SEEK_END);
  long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::string s;
  s.resize(static_cast<size_t>(n));
  if (n > 0) std::fread(s.data(), 1, static_cast<size_t>(n), f);
  std::fclose(f);
  return s;
}

}  // namespace gpu
