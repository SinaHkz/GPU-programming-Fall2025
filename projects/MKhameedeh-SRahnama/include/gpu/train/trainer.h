#pragma once

#include <memory>

#include "gpu/app/config.h"
#include "gpu/datasets/dataset.h"
#include "gpu/model/model_base.h"

namespace gpu {

struct TrainSummary {
  uint64_t steps{0};
  double avg_step_s{0.0};
};

class Trainer {
 public:
  Trainer(TrainConfig cfg, std::unique_ptr<Dataset> dataset, std::unique_ptr<Model> model);
  TrainSummary run();

 private:
  TrainConfig cfg_;
  std::unique_ptr<Dataset> dataset_;
  std::unique_ptr<Model> model_;
};

}  // namespace gpu
