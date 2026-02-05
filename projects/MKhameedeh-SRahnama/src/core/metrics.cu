#include "gpu/core/metrics.h"

#include <fstream>

namespace gpu {

void MetricsSink::set_run_dir(const std::filesystem::path& run_dir) {
  std::scoped_lock lk(mu_);
  metrics_path_ = run_dir / "metrics.jsonl";
  std::ofstream out(metrics_path_, std::ios::app);
}

static void write_json_escaped(std::ostream& os, const std::string& s) {
  os << '"';
  for (char c : s) {
    switch (c) {
      case '\\':
      case '"':
        os << '\\' << c;
        break;
      case '\n':
        os << "\\n";
        break;
      case '\r':
        os << "\\r";
        break;
      case '\t':
        os << "\\t";
        break;
      default:
        os << c;
    }
  }
  os << '"';
}

void MetricsSink::write_point(const MetricPoint& p) {
  std::scoped_lock lk(mu_);
  if (metrics_path_.empty()) return;
  std::ofstream out(metrics_path_, std::ios::app);
  out << "{";
  out << "\"t_ms\":" << p.t_ms;
  for (const auto& kv : p.scalars) {
    out << ",";
    write_json_escaped(out, kv.first);
    out << ":" << kv.second;
  }
  out << "}\n";
}

}  // namespace gpu

