param(
  [string]$OutDir = "build",
  [string]$OutName = "gpu_trainer.exe",
  [string]$CudaArch = "sm_75",
  [switch]$Profile,
  [switch]$Nvtx
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$defines = @()
if ($Profile) { $defines += "-DENABLE_PROFILING=1" }
if ($Nvtx) { $defines += "-DENABLE_NVTX=1" }

$src = @(
  "src/app/args.cu",
  "src/main.cu",
  "src/core/tensor.cu",
  "src/core/logger.cu",
  "src/core/metrics.cu",
  "src/core/profiler.cu",
  "src/datasets/mnist.cu",
  "src/datasets/cifar10.cu",
  "src/datasets/synthetic.cu",
  "src/model/model.cu",
  "src/model/layers.cu",
  "src/model/models.cu",
  "src/model/factory.cu",
  "src/train/trainer.cu",
  "src/train/checkpoint.cu",
  "src/kernels/activations.cu",
  "src/kernels/conv2d.cu",
  "src/kernels/linear.cu",
  "src/kernels/maxpool2d.cu",
  "src/kernels/softmax_ce.cu",
  "src/kernels/reduce.cu",
  "src/kernels/sgd.cu"
)

$libs = @("-lcudart")
if ($Nvtx) { $libs += "-lnvToolsExt" }

Write-Host "Building with nvcc (requires MSVC Build Tools in PATH)..."
& nvcc "-arch=$CudaArch" -O3 --use_fast_math -lineinfo -std=c++17 -Iinclude @defines -o (Join-Path $OutDir $OutName) @src @libs

Write-Host "Built: $(Join-Path $OutDir $OutName)"
