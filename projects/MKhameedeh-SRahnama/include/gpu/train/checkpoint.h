#pragma once

#include <filesystem>
#include <memory>

#include "gpu/model/model_base.h"

namespace gpu {

struct CheckpointMeta {
  uint64_t step{0};
};

// Saves model parameters to:
//   <dir>/checkpoint_step_<step>.json
//   <dir>/checkpoint_step_<step>.bin
// Returns the JSON path.
std::filesystem::path save_checkpoint(const std::filesystem::path& dir, const Model& model, uint64_t step);

// Loads a checkpoint from its JSON path (the JSON references a sibling .bin).
CheckpointMeta load_checkpoint(const std::filesystem::path& checkpoint_json, Model& model);

class AsyncCheckpointWriter {
 public:
  AsyncCheckpointWriter(const std::filesystem::path& dir, const Model& model, size_t max_pending = 2);
  ~AsyncCheckpointWriter();

  AsyncCheckpointWriter(const AsyncCheckpointWriter&) = delete;
  AsyncCheckpointWriter& operator=(const AsyncCheckpointWriter&) = delete;

  // Queues checkpoint creation for this step.
  void enqueue(uint64_t step);

  // Blocks until all queued checkpoints are fully written to disk.
  void flush();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gpu
