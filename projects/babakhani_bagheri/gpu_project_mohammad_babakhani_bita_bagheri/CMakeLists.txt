cmake_minimum_required(VERSION 3.18)
project(nbody_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)

add_executable(nbody
  app/bench.cpp
  app/cli.cpp
  app/config.cpp
  app/main.cpp
  app/render_utils.cpp
  app/sim.cpp
  src/init.cpp
  src/integrator.cu
  src/kernels.cu
  src/utils.cu
  vis/renderer.cpp
)

target_include_directories(nbody PRIVATE src vis app)

target_link_libraries(nbody PRIVATE CUDA::cudart OpenGL::GL glfw GLEW::GLEW)

set_target_properties(nbody PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

add_custom_command(TARGET nbody POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_SOURCE_DIR}/shaders
          $<TARGET_FILE_DIR:nbody>/shaders
)
