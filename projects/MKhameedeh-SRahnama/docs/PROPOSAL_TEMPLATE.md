# Proposal (Topic 1 — CUDA CNN Training From Scratch)

## Dataset

- Dataset: (MNIST / CIFAR-10)
- Input shape: (e.g. 1×28×28 or 3×32×32)
- Preprocessing: (normalization, batching, shuffling)

## Architecture

- Model: (e.g. LeNet / small CNN / custom)
- Layer list (in order):
  - Conv2D(...)
  - ReLU
  - MaxPool2D(...)
  - ...
- Parameter count (estimate):

## CUDA Scope (what is “from scratch” here)

- CUDA kernels implemented:
  - Conv2D forward/backward
  - Activation forward/backward
  - Pooling forward/backward
  - Linear forward/backward
  - Softmax + cross-entropy
  - SGD update
- CPU side:
  - Dataset loading and preprocessing only

## Expected bottlenecks

- Convolution forward/backward
- Gradient reductions (e.g. bias gradients)
- Memory bandwidth vs compute intensity

## Planned optimizations

- Improve memory coalescing (NCHW access patterns)
- Shared memory tiling for conv/linear
- Kernel fusion where beneficial (e.g. bias+activation)
- Mixed precision exploration (optional)

## Correctness + performance evaluation

- Correctness:
  - Loss decreases across epochs
  - Accuracy improves vs baseline
- Performance:
  - Per-kernel timings and call counts
  - Throughput (samples/sec)
  - GPU memory usage and allocator peak
  - Nsight Systems/Compute traces (with NVTX ranges enabled)

