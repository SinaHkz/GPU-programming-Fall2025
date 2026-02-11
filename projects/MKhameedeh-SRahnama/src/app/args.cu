#include "gpu/app/args.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace gpu {

void print_help(const char* prog) {
  std::cout << "Usage: " << prog
            << " --arch <lenet|simple_cnn|mlp> --dataset <mnist|cifar10|synthetic> [options]\n"
               "Options:\n"
               "  --data-dir <path>\n"
               "  --out-dir <path>\n"
               "  --run-name <str>\n"
               "  --resume <checkpoint.json>\n"
               "  --epochs <int>\n"
               "  --batch <int>\n"
               "  --lr <float>\n"
               "  --save-every <steps>\n"
               "  --seed <int>\n"
               "  --log-every <int>\n"
               "  --eval-every <int>\n"
               "  --weight-decay <float>\n"
               "  --max-steps <int>\n"
               "  --no-h2d-pipeline\n"
               "  --no-log-sync-opt\n"
               "  --no-async-checkpoint\n"
               "  --cuda-graph-sgd\n"
               "  --async-eval\n"
               "  --norm-log-mult <int>\n"
               "  --benchmark-compare\n"
               "  --benchmark-steps <int>\n"
               "  --no-shuffle\n";
}

static bool has_arg(const std::string& a, const char* k) { return a == k; }

TrainConfig parse_args(int argc, char** argv) {
  TrainConfig cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&](const char* key) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value for ") + key);
      return argv[++i];
    };

    if (has_arg(a, "--help") || has_arg(a, "-h")) {
      print_help(argv[0]);
      std::exit(0);
    } else if (has_arg(a, "--arch")) {
      cfg.arch = need("--arch");
    } else if (has_arg(a, "--dataset")) {
      cfg.dataset = need("--dataset");
    } else if (has_arg(a, "--data-dir")) {
      cfg.data_dir = need("--data-dir");
    } else if (has_arg(a, "--out-dir")) {
      cfg.out_dir = need("--out-dir");
    } else if (has_arg(a, "--run-name")) {
      cfg.run_name = need("--run-name");
    } else if (has_arg(a, "--resume")) {
      cfg.resume_from = need("--resume");
    } else if (has_arg(a, "--epochs")) {
      cfg.epochs = std::stoi(need("--epochs"));
    } else if (has_arg(a, "--batch")) {
      cfg.batch_size = std::stoi(need("--batch"));
    } else if (has_arg(a, "--lr")) {
      cfg.lr = std::stof(need("--lr"));
    } else if (has_arg(a, "--save-every")) {
      cfg.save_every = std::stoi(need("--save-every"));
    } else if (has_arg(a, "--seed")) {
      cfg.seed = std::stoi(need("--seed"));
    } else if (has_arg(a, "--log-every")) {
      cfg.log_every = std::stoi(need("--log-every"));
    } else if (has_arg(a, "--eval-every")) {
      cfg.eval_every = std::stoi(need("--eval-every"));
    } else if (has_arg(a, "--weight-decay")) {
      cfg.weight_decay = std::stof(need("--weight-decay"));
    } else if (has_arg(a, "--max-steps")) {
      cfg.max_steps = std::stoi(need("--max-steps"));
    } else if (has_arg(a, "--no-h2d-pipeline")) {
      cfg.enable_h2d_pipeline = false;
    } else if (has_arg(a, "--no-log-sync-opt")) {
      cfg.enable_log_sync_optimizations = false;
    } else if (has_arg(a, "--no-async-checkpoint")) {
      cfg.enable_async_checkpoint = false;
    } else if (has_arg(a, "--cuda-graph-sgd")) {
      cfg.enable_cuda_graph_sgd = true;
    } else if (has_arg(a, "--async-eval")) {
      cfg.enable_async_eval = true;
    } else if (has_arg(a, "--norm-log-mult")) {
      cfg.norm_log_multiplier = std::stoi(need("--norm-log-mult"));
    } else if (has_arg(a, "--benchmark-compare")) {
      cfg.benchmark_compare = true;
    } else if (has_arg(a, "--benchmark-steps")) {
      cfg.benchmark_steps = std::stoi(need("--benchmark-steps"));
    } else if (has_arg(a, "--no-shuffle")) {
      cfg.shuffle_train = false;
    } else {
      throw std::runtime_error("Unknown arg: " + a);
    }
  }
  return cfg;
}

}  // namespace gpu
