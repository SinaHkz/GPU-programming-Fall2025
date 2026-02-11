#pragma once

#include <cstdint>
#include <string>

namespace gpu {

struct TrainConfig {
  std::string arch{"lenet"};
  std::string dataset{"mnist"};
  std::string data_dir{"data"};
  std::string out_dir{"runs"};
  std::string run_name{""};     // optional prefix
  std::string resume_from{""};  // checkpoint json path (or empty)

  int epochs{2};
  int batch_size{64};
  float lr{0.01f};
  float weight_decay{0.0f};

  int seed{1337};
  int log_every{50};
  int eval_every{200};
  int save_every{0};  // steps (0 disables)
  int max_steps{0};   // 0 disables step cap

  bool shuffle_train{true};
  bool use_fp16{false};  // reserved

  // Performance toggles.
  bool enable_h2d_pipeline{true};
  bool enable_log_sync_optimizations{true};
  bool enable_async_checkpoint{true};
  bool enable_cuda_graph_sgd{false};
  bool enable_async_eval{false};
  int norm_log_multiplier{5};  // compute norms every (log_every * multiplier)

  // Runs baseline+optimized back-to-back and prints speedup.
  bool benchmark_compare{false};
  int benchmark_steps{200};
};

}  // namespace gpu
