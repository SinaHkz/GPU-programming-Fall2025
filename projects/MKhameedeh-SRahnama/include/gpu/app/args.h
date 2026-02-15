#pragma once

#include <string>
#include <vector>

#include "gpu/app/config.h"

namespace gpu {

TrainConfig parse_args(int argc, char** argv);
void print_help(const char* prog);

}  // namespace gpu

