# GPU CNN Trainer (CUDA, from scratch)

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

Build + train LeNet on MNIST:

```bash
make train ARCH=lenet DATASET=mnist EPOCHS=2 BATCH=64 LR=0.01 SAVE_EVERY=200
```

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

## Windows build options

- If you don’t have `make`, use:
  - `scripts/build.ps1` and `scripts/train.ps1`
- Nsight Compute metrics export:
  - `scripts/ncu_metrics.ps1 -Config configs\lenet_mnist.json -Metrics achieved_occupancy`
- You can also use CMake/Visual Studio:
  - Configure: `cmake -S . -B build_cmake`
  - Build: `cmake --build build_cmake --config Release`
