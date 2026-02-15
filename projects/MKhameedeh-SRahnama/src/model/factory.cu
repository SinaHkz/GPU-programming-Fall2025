#include "gpu/model/factory.h"

#include <stdexcept>

#include "gpu/model/models/lenet.h"
#include "gpu/model/models/mlp.h"
#include "gpu/model/models/simple_cnn.h"

namespace gpu {

std::unique_ptr<Model> make_model(const std::string& arch, int in_c, int in_h, int in_w, int num_classes, int seed) {
  if (arch == "lenet") return std::make_unique<LeNetModel>(in_c, in_h, in_w, num_classes, seed);
  if (arch == "simple_cnn") return std::make_unique<SimpleCnnModel>(in_c, in_h, in_w, num_classes, seed);
  if (arch == "mlp") return std::make_unique<MlpModel>(in_c, in_h, in_w, num_classes, seed);
  throw std::runtime_error("Unknown arch: " + arch);
}

}  // namespace gpu

