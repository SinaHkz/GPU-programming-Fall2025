# -------- Project settings --------
TARGET := cnn

NVCC   := nvcc
CXX    := g++

INCLUDE_DIR := include
SRC_DIR := src
#LAYER_DIR := layer
# Change this if needed (e.g. sm_75, sm_86, sm_89, sm_90)
CUDA_ARCH ?= sm_61

# -------- Flags --------
# Baseline flags
NVCCFLAGS := -O3 -std=c++14 -I$(INCLUDE_DIR) -arch=$(CUDA_ARCH)
CXXFLAGS  := -O3 -std=c++14 -I$(INCLUDE_DIR)

# Link flags
LDFLAGS := -arch=$(CUDA_ARCH)

# -------- Source files --------
#CU_SRCS := \
#  $(SRC_DIR)/main.cu \
#  $(SRC_DIR)/train.cu \
#  $(SRC_DIR)/layers/conv_forward.cu \
#  $(SRC_DIR)/layers/conv_backward.cu \
#  $(SRC_DIR)/layers/pool_forward.cu \
#  $(SRC_DIR)/layers/pool_backward.cu \
#  $(SRC_DIR)/layers/fc_forward.cu \
#  $(SRC_DIR)/layers/fc_backward.cu \
#  $(SRC_DIR)/ops/softmax_loss.cu \
#  $(SRC_DIR)/ops/sgd_update.cu \
#  $(SRC_DIR)/ops/relu_activate.cu\
#  $(SRC_DIR)/utils/cuda_utils.cu \
#  $(SRC_DIR)/utils/timer.cu

#CU_SRCS := \
#  $(SRC_DIR)/main.cu \
#  $(SRC_DIR)/train_fues.cu \
#  $(SRC_DIR)/optimize/conv_fuse.cu \
#  $(SRC_DIR)/layers/pool_forward.cu \
#  $(SRC_DIR)/layers/pool_backward.cu \
#  $(SRC_DIR)/layers/fc_forward.cu \
#  $(SRC_DIR)/layers/fc_backward.cu \
#  $(SRC_DIR)/ops/softmax_loss.cu \
#  $(SRC_DIR)/ops/sgd_update.cu \
#  $(SRC_DIR)/layers/conv_backward.cu\
#  $(SRC_DIR)/ops/relu_activate.cu\
#  $(SRC_DIR)/utils/cuda_utils.cu \
#  $(SRC_DIR)/utils/timer.cu

CU_SRCS := \
  $(SRC_DIR)/main.cu \
  $(SRC_DIR)/train_op.cu \
  $(SRC_DIR)/optimize/conv_fuse.cu \
  $(SRC_DIR)/layers/pool_forward.cu \
  $(SRC_DIR)/layers/pool_backward.cu \
  $(SRC_DIR)/layers/fc_forward.cu \
  $(SRC_DIR)/layers/fc_backward.cu \
  $(SRC_DIR)/ops/softmax_loss.cu \
  $(SRC_DIR)/ops/sgd_update.cu \
  $(SRC_DIR)/optimize/conv_back_tile.cu\
  $(SRC_DIR)/ops/relu_activate.cu\
  $(SRC_DIR)/utils/cuda_utils.cu \
  $(SRC_DIR)/utils/timer.cu


CPP_SRCS := \
  $(SRC_DIR)/mnist_loader.cpp

# Objects (mirror filenames into build/)
BUILD_DIR := build
CU_OBJS  := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SRCS))
CPP_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SRCS))
OBJS := $(CU_OBJS) $(CPP_OBJS)

# -------- Rules --------
all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(LDFLAGS) -o $@ $^

# Compile CUDA .cu files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile C++ .cpp files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

print:
	@echo "CU_SRCS = $(CU_SRCS)"
	@echo "CPP_SRCS = $(CPP_SRCS)"

.PHONY: all clean print
