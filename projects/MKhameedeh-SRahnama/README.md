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
This repo contains a small **CUDA C++** training platform to build and train **CNNs from scratch** (no PyTorch/TensorFlow), with a performance-first mindset (kernel timing, memory metrics, NVTX ranges).

## What you get

- Multiple model architectures (each as its own class; select via `--arch`)
- Multiple datasets (select via `--dataset`)
- OOP, layer-by-layer design (each layer has `forward/backward`)
- Metrics exporting (loss/acc, throughput, GPU memory, allocator bytes, per-kernel timing)
- Logging to console + run directory
- Optional NVTX ranges for Nsight Systems/Compute
- Checkpoint save/resume (`--save-every`, `--resume`)
- Simple dashboard to view exported metrics

## Quick start

Prereqs:
- NVIDIA GPU + CUDA Toolkit (so `nvcc` works)
- GNU Make (MSYS2/WSL/Linux) or adapt the build commands

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
Build + train LeNet on MNIST:

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

Then launch the dashboard:

```bash
python tools/serve_dashboard.py --run-dir runs/latest
```

Or, launch the dashboard server (training starts from the UI):

```bash
python tools/serve_dashboard.py
```

Windows (PowerShell):

```powershell
.\scripts\dashboard.ps1
```

Optional auto-start from a config:

```bash
python tools/run_dashboard.py --start --config configs/lenet_mnist.json
```

The dashboard can:
- Switch between runs (dropdown)
- Live-plot `profiling/*.csv` (GPU power/util/memory, process CPU/RSS, kernel occupancy, function summaries)
- Launch a new run from the UI (optional; requires the `build/gpu_trainer` binary to exist)
- Auto-download MNIST/CIFAR-10 if missing when you press Start
- Show CPU↔GPU memcpy bandwidth in the functions table (Total MB / Avg GB/s)

Note: the dashboard vendors uPlot (offline-capable charts) in `dashboard/vendor/`.

## Key commands

- `make build` builds the binary into `build/gpu_trainer`
- `make train ...` runs training with Makefile-provided config (passed as CLI args)
- `make data-mnist` downloads MNIST into `data/mnist/`
- `make data-cifar10` downloads CIFAR-10 (binary) into `data/cifar10/`
- `make train RESUME=runs/.../checkpoints/checkpoint_step_200.json` resumes from a checkpoint

## Notes

- This project is intentionally small and readable; kernels are written directly in CUDA.
- For performance work, enable profiling and NVTX:
  - `make train PROFILE=1 NVTX=1 ...`
- Nsight Systems / Compute:
  - `make nsys NVTX=1 PROFILE=1 EPOCHS=1` (timeline + NVTX ranges)
  - `make ncu PROFILE=1 EPOCHS=1 KERNEL=.*conv2d.*` (deep per-kernel metrics)
  - `make ncu-metrics CONFIG=configs/lenet_mnist.json NCU_METRICS=achieved_occupancy`
    (exports `runs/<run>/profiling/kernel_metrics.csv` for achieved occupancy in the dashboard)
- CSV profiling outputs are written to `runs/<run>/profiling/`:
  - `functions_events.csv`, `functions_summary.csv`
  - `train_metrics.csv`
  - `system_metrics.csv`, `gpu_metrics.csv`
- Control background sampling rate with `--profile-interval-ms` (set to `0` to disable).
- Disable training shuffling (debugging):
  - `make train NO_SHUFFLE=1 ...`

## Windows options

- PowerShell scripts:
  - `scripts/build.ps1`
  - `scripts/train.ps1`
- CMake:
- If you don’t have `make`, use:
  - `scripts/build.ps1` and `scripts/train.ps1`
- Nsight Compute metrics export:
  - `scripts/ncu_metrics.ps1 -Config configs\lenet_mnist.json -Metrics achieved_occupancy`
- You can also use CMake/Visual Studio:
  - Configure: `cmake -S . -B build_cmake`
  - Build: `cmake --build build_cmake --config Release`
