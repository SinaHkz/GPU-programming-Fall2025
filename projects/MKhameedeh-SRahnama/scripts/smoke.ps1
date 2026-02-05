param(
  [string]$DataDir = "data"
)

$ErrorActionPreference = "Continue"

Write-Host "Checking prerequisites..."
if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
  Write-Host "Missing: nvcc (install CUDA Toolkit)"
  exit 1
}

if (-not (Get-Command make -ErrorAction SilentlyContinue)) {
  Write-Host "Note: make not found (install MSYS2/MinGW make or use scripts/build.ps1)"
}

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
  Write-Host "Note: cl.exe not found in PATH (install 'Build Tools for Visual Studio' and run from a Developer PowerShell)"
}

Write-Host "Checking datasets..."
$mnist = Join-Path $DataDir "mnist/train-images-idx3-ubyte"
if (-not (Test-Path $mnist)) {
  Write-Host "MNIST not found. Run: python tools/download_mnist.py --out $DataDir"
} else {
  Write-Host "MNIST OK"
}

$cifar = Join-Path $DataDir "cifar10/cifar-10-batches-bin/test_batch.bin"
if (-not (Test-Path $cifar)) {
  Write-Host "CIFAR-10 not found. Run: python tools/download_cifar10.py --out $DataDir"
} else {
  Write-Host "CIFAR-10 OK"
}

Write-Host "Smoke checks done."

