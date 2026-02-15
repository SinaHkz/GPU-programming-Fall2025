#include "gpu/core/metrics.h"

#include <fstream>

#include "gpu/utils/fs.h"

namespace gpu {

void MetricsSink::set_run_dir(const std::filesystem::path& run_dir) {
  std::scoped_lock lk(mu_);
  metrics_path_ = run_dir / "metrics.jsonl";
  std::ofstream out(metrics_path_, std::ios::app);

  const auto profiling_dir = run_dir / "profiling";
  ensure_dir(profiling_dir);
  metrics_csv_path_ = profiling_dir / "train_metrics.csv";
  std::ofstream csv(metrics_csv_path_, std::ios::trunc);
  csv << "t_ms,metric,value\n";
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

  if (!metrics_csv_path_.empty()) {
    std::ofstream csv(metrics_csv_path_, std::ios::app);
    for (const auto& kv : p.scalars) {
      csv << p.t_ms << ",";
      write_csv_escaped(csv, kv.first);
      csv << "," << kv.second << "\n";
    }
  }
}

}  // namespace gpu
