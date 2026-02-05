param(
  [string]$Exe = "build/gpu_trainer.exe",
  [string]$Arch = "lenet",
  [string]$Dataset = "mnist",
  [int]$Epochs = 2,
  [int]$Batch = 64,
  [double]$Lr = 0.01,
  [int]$Seed = 1337,
  [string]$DataDir = "data",
  [string]$OutDir = "runs",
  [string]$RunName = "",
  [string]$Resume = "",
  [int]$SaveEvery = 0,
  [switch]$NoShuffle
)

$ErrorActionPreference = "Stop"

$args = @("--arch", $Arch, "--dataset", $Dataset, "--epochs", $Epochs, "--batch", $Batch, "--lr", $Lr, "--seed", $Seed, "--data-dir", $DataDir, "--out-dir", $OutDir, "--run-name", $RunName, "--save-every", $SaveEvery)
if ($Resume -ne "") { $args += @("--resume", $Resume) }
if ($NoShuffle) { $args += "--no-shuffle" }
& $Exe @args
