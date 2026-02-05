cmake_minimum_required(VERSION 3.24)

project(gpu_trainer LANGUAGES CXX CUDA)

option(ENABLE_PROFILING "Enable kernel timing export via cudaEvent" OFF)
option(ENABLE_NVTX "Enable NVTX ranges (Nsight Systems)" OFF)

find_package(CUDAToolkit REQUIRED)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  # Avoid building a binary that doesn't contain device code for the local GPU.
  # `native` is supported by nvcc (and CMake >= 3.24) and picks the installed GPU's SM.
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(gpu_trainer
  src/app/args.cu
  src/main.cu
  src/core/tensor.cu
  src/core/logger.cu
  src/core/metrics.cu
  src/core/profiler.cu
  src/datasets/mnist.cu
  src/datasets/cifar10.cu
  src/datasets/synthetic.cu
  src/model/model.cu
  src/model/layers.cu
  src/model/models.cu
  src/model/factory.cu
  src/train/trainer.cu
  src/train/checkpoint.cu
  src/kernels/activations.cu
  src/kernels/conv2d.cu
  src/kernels/linear.cu
  src/kernels/maxpool2d.cu
  src/kernels/softmax_ce.cu
  src/kernels/reduce.cu
  src/kernels/sgd.cu
)

target_include_directories(gpu_trainer PRIVATE include)
target_link_libraries(gpu_trainer PRIVATE CUDA::cudart)

target_compile_options(gpu_trainer PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

if (MSVC)
  target_compile_options(gpu_trainer PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/O2>)
else()
  target_compile_options(gpu_trainer PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
endif()

if (ENABLE_PROFILING)
  target_compile_definitions(gpu_trainer PRIVATE ENABLE_PROFILING=1)
endif()

if (ENABLE_NVTX)
  target_compile_definitions(gpu_trainer PRIVATE ENABLE_NVTX=1)
  if (TARGET CUDA::nvToolsExt)
    target_link_libraries(gpu_trainer PRIVATE CUDA::nvToolsExt)
  endif()
endif()

set_target_properties(gpu_trainer PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
