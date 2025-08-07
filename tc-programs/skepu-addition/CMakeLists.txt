cmake_minimum_required(VERSION 3.20)

project(Test)

enable_language(CUDA)

set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_86,code=sm_86")


find_package(SkePU REQUIRED)

include_directories(CMAKE_CURRENT_SOURCE_DIR/../cutlass/include)
include_directories(CMAKE_CURRENT_SOURCE_DIR/../cutlass/tools/util/include)

skepu_add_executable(main CUDA SKEPUSRC my_thing.cpp)