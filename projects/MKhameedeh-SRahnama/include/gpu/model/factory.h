#pragma once

#include <memory>
#include <string>

#include "gpu/model/model_base.h"

namespace gpu {

std::unique_ptr<Model> make_model(const std::string& arch, int in_c, int in_h, int in_w, int num_classes, int seed);

}  // namespace gpu

