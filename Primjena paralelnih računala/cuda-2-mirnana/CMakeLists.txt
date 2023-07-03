cmake_minimum_required(VERSION 3.8)
project(cuda-2)

enable_language(C CXX CUDA)

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CUDA_STANDARD 11)

#set (ARCH_FLAGS -arch=sm_30)
#set (CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS}; "-O3 ${ARCH_FLAGS}")

add_executable(main main.cu)
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "75")
