# GPU CNN Trainer (CUDA, from scratch)

This repository is a CUDA C++ training platform for CNN/MLP models implemented from scratch (no PyTorch/TensorFlow).  
It now includes a full streaming/pipelining optimization path, async checkpoint/eval, and a built-in before/after benchmark mode.

## Current capabilities

- Model architectures:
  - `lenet`
  - `simple_cnn`
  - `mlp`
- Datasets:
  - `mnist`
  - `cifar10`
  - `synthetic`
- Training system:
  - Layer-by-layer `forward`/`backward`
  - SGD update kernels
  - Checkpoint save/load (`--save-every`, `--resume`)
  - Run artifacts (`config.json`, `device.json`, `metrics.jsonl`, logs, checkpoints)
- Performance features:
  - Host-to-device (H2D) batch streaming pipeline with double-buffered pinned memory
  - Reduced log-path synchronizations (metrics/loss-gather only when needed)
  - Async checkpoint pipeline (device snapshot -> pinned host -> background file write)
  - Norm logging throttling (`param_l2`/`grad_l2` computed every `log_every * norm_log_multiplier`)
  - Memory allocation reuse (`Tensor::resize` reuses storage when shape numel is unchanged)
  - Async eval decoupled from training (`--async-eval`)
  - Benchmark compare mode (`--benchmark-compare`) that runs baseline and optimized configs back-to-back
  - Optional CUDA graph capture for SGD (`--cuda-graph-sgd`, currently experimental)
- Profiling/observability:
  - Optional kernel timing export (`PROFILE=1`)
  - Optional NVTX ranges (`NVTX=1`)
  - Dashboard viewer (`tools/serve_dashboard.py`)

## Quick start

Prerequisites:
- NVIDIA GPU + CUDA Toolkit (`nvcc` available)
- GNU Make
- Python 3

Download MNIST:

```bash
make data-mnist
```

Build:

```bash
make build
```

Run training:

```bash
build/gpu_trainer \
  --arch lenet --dataset mnist \
  --data-dir data --out-dir runs \
  --epochs 2 --batch 64 --lr 0.01 \
  --log-every 50 --save-every 200
```

Launch dashboard:

```bash
python3 tools/serve_dashboard.py --run-dir runs/latest
```

## Benchmark mode (before vs after)

Use this to quantify optimization impact on your machine:

```bash
build/gpu_trainer \
  --arch lenet --dataset mnist \
  --data-dir data --out-dir runs \
  --epochs 2 --batch 64 --lr 0.01 \
  --log-every 50 --save-every 200 \
  --benchmark-compare --benchmark-steps 400
```

This prints:
- `BENCHMARK_BEFORE ...`
- `BENCHMARK_AFTER  ...`
- `BENCHMARK_SPEEDUP ...`

Benchmark details:
- `before` disables the optimization stack.
- `after` enables streaming/pipelining + async features.
- `--benchmark-steps` controls the step count for each run.

## Performance and behavior flags

- `--max-steps <int>`: hard step cap for quick tests.
- `--no-h2d-pipeline`: disable H2D streaming pipeline.
- `--no-log-sync-opt`: disable log-path sync reductions.
- `--no-async-checkpoint`: use synchronous checkpoint writes.
- `--norm-log-mult <int>`: norm compute period multiplier.
- `--async-eval`: evaluate from checkpoints in a background thread.
- `--cuda-graph-sgd`: enable CUDA graph capture/replay for SGD update (experimental).
- `--benchmark-compare`: run built-in before/after benchmark.
- `--benchmark-steps <int>`: number of steps for benchmark runs.

## Makefile commands

- `make build`: build `build/gpu_trainer`.
- `make train ...`: training with Makefile variables (basic flow).
- `make data-mnist`: download MNIST into `data/mnist`.
- `make data-cifar10`: download CIFAR-10 into `data/cifar10`.
- `make dashboard`: serve dashboard for `runs/latest`.

Examples:

```bash
make train ARCH=lenet DATASET=mnist EPOCHS=2 BATCH=64 LR=0.01 SAVE_EVERY=200
make train RESUME=runs/.../checkpoints/checkpoint_step_200.json
make build PROFILE=1 NVTX=1
```

## Run outputs

Each run directory contains:
- `config.json`: full resolved run config.
- `device.json`: CUDA device metadata.
- `metrics.jsonl`: time-series metrics.
- `kernels.jsonl` and `kernel_summary.json` (when profiling enabled).
- `checkpoints/`: JSON checkpoints.
- log files used by the dashboard.

## Known caveats

- `--cuda-graph-sgd` may fail on some environments/drivers with stream-capture restrictions.
- Benchmark mode currently forces CUDA-graph off in the `after` path for robustness.

## Windows options

- PowerShell scripts:
  - `scripts/build.ps1`
  - `scripts/train.ps1`
- CMake:
  - Configure: `cmake -S . -B build_cmake`
  - Build: `cmake --build build_cmake --config Release`

