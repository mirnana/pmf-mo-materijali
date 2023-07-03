cmake_minimum_required (VERSION 3.8)
project(cuda-1)

enable_language(C CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

add_executable(axpy  axpy.cu)
set_target_properties(axpy PROPERTIES CUDA_ARCHITECTURES "75")
#set_target_properties(axpy PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

