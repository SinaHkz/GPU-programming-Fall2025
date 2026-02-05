# Performance workflow

## Built-in profiling

- Build with per-kernel timing export:
  - `make build PROFILE=1`
- Build with NVTX ranges for Nsight Systems:
  - `make build NVTX=1`

Outputs in the run directory:
- `kernels.jsonl`: per-kernel timing events
- `kernel_summary.json`: aggregated timing stats
- `metrics.jsonl`: training metrics + memory

## Nsight Systems example

1. Build with NVTX: `make build NVTX=1 PROFILE=1`
2. Run: `nsys profile -o runs/nsys_report build/gpu_trainer --arch lenet --dataset mnist ...`

## Things to try

- Increase `BATCH` to improve occupancy/throughput.
- Compare `lenet` vs `mlp` to isolate conv behavior.
- Inspect `conv2d_*` kernel time fractions in `kernel_summary.json`.

