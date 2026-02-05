SHELL := bash

NVCC ?= nvcc
CXX ?= g++

# Build for the local GPU by default (override with `make CUDA_ARCH=-arch=sm_XX` if needed).
CUDA_ARCH ?= -arch=native
BUILD_DIR := build
BIN := $(BUILD_DIR)/gpu_trainer

SRC := \
  src/app/args.cu \
  src/main.cu \
  src/core/tensor.cu \
  src/core/logger.cu \
  src/core/metrics.cu \
  src/core/profiler.cu \
  src/datasets/mnist.cu \
  src/datasets/cifar10.cu \
  src/datasets/synthetic.cu \
  src/model/model.cu \
  src/model/layers.cu \
  src/model/models.cu \
  src/model/factory.cu \
  src/train/trainer.cu \
  src/train/checkpoint.cu \
  src/kernels/activations.cu \
  src/kernels/conv2d.cu \
  src/kernels/linear.cu \
  src/kernels/maxpool2d.cu \
  src/kernels/softmax_ce.cu \
  src/kernels/reduce.cu \
  src/kernels/sgd.cu

INC := -Iinclude

CUDA_FLAGS := $(CUDA_ARCH) -O3 --use_fast_math -lineinfo
HOST_FLAGS := -O3 -std=c++17

LDFLAGS :=
LDLIBS := -lcudart

ifeq ($(NVTX),1)
  CUDA_FLAGS += -DENABLE_NVTX=1
  LDLIBS += -lnvToolsExt
endif

ifeq ($(PROFILE),1)
  CUDA_FLAGS += -DENABLE_PROFILING=1
endif

.PHONY: all build clean train data-mnist data-cifar10 run-synth dashboard help

help:
	@echo "Targets:"
	@echo "  build             Build CUDA trainer"
	@echo "  train             Train with Makefile-provided config"
	@echo "  data-mnist        Download MNIST into data/mnist/"
	@echo "  data-cifar10      Download CIFAR-10 into data/cifar10/"
	@echo "  dashboard         Serve dashboard for runs/latest"

all: build

$(BUILD_DIR)/.dir:
	mkdir -p $(BUILD_DIR)
	touch $(BUILD_DIR)/.dir

$(BIN): $(SRC) | $(BUILD_DIR)/.dir
	$(NVCC) $(CUDA_FLAGS) $(HOST_FLAGS) $(INC) -o $(BIN) $(SRC) $(LDFLAGS) $(LDLIBS)

build: $(BIN)

clean:
	rm -rf $(BUILD_DIR)

# ---- Data ----
data-mnist:
	python tools/download_mnist.py --out data

data-cifar10:
	python tools/download_cifar10.py --out data/cifar10

# ---- Runs ----
ARCH ?= lenet
DATASET ?= mnist
EPOCHS ?= 2
BATCH ?= 64
LR ?= 0.01
SEED ?= 1337
DATA_DIR ?= data
OUT_DIR ?= runs
RUN_NAME ?=
SAVE_EVERY ?= 0
RESUME ?=
NO_SHUFFLE ?= 0

train: build
	$(BIN) --arch $(ARCH) --dataset $(DATASET) --epochs $(EPOCHS) --batch $(BATCH) --lr $(LR) --seed $(SEED) --data-dir $(DATA_DIR) --out-dir $(OUT_DIR) --run-name "$(RUN_NAME)" --save-every $(SAVE_EVERY) $(if $(RESUME),--resume $(RESUME),) $(if $(filter 1,$(NO_SHUFFLE)),--no-shuffle,)

run-synth: build
	$(BIN) --arch $(ARCH) --dataset synthetic --epochs 1 --batch $(BATCH) --lr $(LR) --seed $(SEED) --out-dir $(OUT_DIR) --run-name "$(RUN_NAME)" --save-every $(SAVE_EVERY)

dashboard:
	python tools/serve_dashboard.py --run-dir $(OUT_DIR)/latest --port 8080
