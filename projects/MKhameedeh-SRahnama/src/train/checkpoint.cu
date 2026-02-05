#include "gpu/train/checkpoint.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "gpu/core/logger.h"
#include "gpu/utils/fs.h"

namespace gpu {

static std::string json_escape(const std::string& s) {
  std::ostringstream o;
  for (char c : s) {
    switch (c) {
      case '\\':
      case '"':
        o << '\\' << c;
        break;
      case '\n':
        o << "\\n";
        break;
      case '\r':
        o << "\\r";
        break;
      case '\t':
        o << "\\t";
        break;
      default:
        o << c;
    }
  }
  return o.str();
}

static void write_shape(std::ostream& out, const std::vector<int>& shape) {
  out << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) out << ",";
    out << shape[i];
  }
  out << "]";
}

std::filesystem::path save_checkpoint(const std::filesystem::path& dir, const Model& model, uint64_t step) {
  ensure_dir(dir);
  const std::filesystem::path bin_path = dir / ("checkpoint_step_" + std::to_string(step) + ".bin");
  const std::filesystem::path json_path = dir / ("checkpoint_step_" + std::to_string(step) + ".json");

  std::ofstream bin(bin_path, std::ios::binary | std::ios::trunc);
  if (!bin) throw std::runtime_error("Failed opening checkpoint bin for write: " + bin_path.string());

  struct Entry {
    std::string name;
    std::vector<int> shape;
    uint64_t offset_bytes{0};
    uint64_t numel{0};
  };

  std::vector<Entry> entries;
  uint64_t offset = 0;
  for (const auto& p : model.params()) {
    Entry e;
    e.name = p.name;
    e.shape = p.w->shape();
    e.numel = static_cast<uint64_t>(p.w->numel());
    e.offset_bytes = offset;

    std::vector<float> host(p.w->numel());
    p.w->copy_to_host(host.data(), host.size());
    bin.write(reinterpret_cast<const char*>(host.data()), static_cast<std::streamsize>(host.size() * sizeof(float)));

    offset += static_cast<uint64_t>(host.size() * sizeof(float));
    entries.push_back(std::move(e));
  }
  bin.flush();

  std::ofstream json(json_path, std::ios::trunc);
  if (!json) throw std::runtime_error("Failed opening checkpoint json for write: " + json_path.string());

  json << "{";
  json << "\"model\":\"" << json_escape(model.name()) << "\",";
  json << "\"step\":" << step << ",";
  json << "\"bin_file\":\"" << json_escape(bin_path.filename().string()) << "\",";
  json << "\"params\":[";
  for (size_t i = 0; i < entries.size(); ++i) {
    if (i) json << ",";
    const auto& e = entries[i];
    json << "{";
    json << "\"name\":\"" << json_escape(e.name) << "\",";
    json << "\"shape\":";
    write_shape(json, e.shape);
    json << ",";
    json << "\"offset_bytes\":" << e.offset_bytes << ",";
    json << "\"numel\":" << e.numel;
    json << "}";
  }
  json << "]";
  json << "}\n";

  Logger::instance().info("Saved checkpoint: " + json_path.string());
  return json_path;
}

// Minimal JSON parser for our own checkpoint format.
static std::string find_string_field(const std::string& s, const std::string& key) {
  const std::string needle = "\"" + key + "\":\"";
  const size_t pos = s.find(needle);
  if (pos == std::string::npos) return {};
  size_t i = pos + needle.size();
  std::string out;
  while (i < s.size()) {
    char c = s[i++];
    if (c == '"') break;
    if (c == '\\' && i < s.size()) {
      char n = s[i++];
      if (n == 'n') out.push_back('\n');
      else if (n == 'r') out.push_back('\r');
      else if (n == 't') out.push_back('\t');
      else out.push_back(n);
    } else {
      out.push_back(c);
    }
  }
  return out;
}

static uint64_t find_u64_field(const std::string& s, const std::string& key) {
  const std::string needle = "\"" + key + "\":";
  const size_t pos = s.find(needle);
  if (pos == std::string::npos) return 0;
  size_t i = pos + needle.size();
  while (i < s.size() && (s[i] == ' ')) ++i;
  size_t j = i;
  while (j < s.size() && (s[j] >= '0' && s[j] <= '9')) ++j;
  return static_cast<uint64_t>(std::stoull(s.substr(i, j - i)));
}

CheckpointMeta load_checkpoint(const std::filesystem::path& checkpoint_json, Model& model) {
  const std::string txt = read_text_file(checkpoint_json);
  if (txt.empty()) throw std::runtime_error("Failed reading checkpoint json: " + checkpoint_json.string());

  const std::string bin_file = find_string_field(txt, "bin_file");
  const uint64_t step = find_u64_field(txt, "step");
  if (bin_file.empty()) throw std::runtime_error("Checkpoint missing bin_file: " + checkpoint_json.string());

  const auto bin_path = checkpoint_json.parent_path() / bin_file;
  std::ifstream bin(bin_path, std::ios::binary);
  if (!bin) throw std::runtime_error("Failed opening checkpoint bin: " + bin_path.string());

  // Our save format writes params in the same order as model.params().
  uint64_t offset = 0;
  for (const auto& p : model.params()) {
    std::vector<float> host(p.w->numel());
    bin.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    bin.read(reinterpret_cast<char*>(host.data()), static_cast<std::streamsize>(host.size() * sizeof(float)));
    if (!bin) throw std::runtime_error("Checkpoint read failed at param: " + p.name);
    p.w->copy_from_host(host.data(), host.size());
    offset += static_cast<uint64_t>(host.size() * sizeof(float));
  }

  Logger::instance().info("Loaded checkpoint: " + checkpoint_json.string());
  CheckpointMeta m;
  m.step = step;
  return m;
}

}  // namespace gpu

