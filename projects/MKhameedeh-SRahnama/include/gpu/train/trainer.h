#pragma once

#include <memory>

#include "gpu/app/config.h"
#include "gpu/datasets/dataset.h"
#include "gpu/model/model_base.h"

namespace gpu {

class Trainer {
 public:
  Trainer(TrainConfig cfg, std::unique_ptr<Dataset> dataset, std::unique_ptr<Model> model);
  void run();

 private:
  TrainConfig cfg_;
  std::unique_ptr<Dataset> dataset_;
  std::unique_ptr<Model> model_;
};

}  // namespace gpu
