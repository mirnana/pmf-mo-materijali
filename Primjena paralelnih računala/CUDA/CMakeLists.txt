cmake_minimum_required(VERSION 3.9)
project(cuda-1)
enable_language(CUDA)

add_executable(add-vec add-vec.cu)
set_target_properties(add-vec PROPERTIES CUDA_ARCHITECTURES "75")

add_executable(add-mat add-mat.cu)
set_target_properties(add-mat PROPERTIES CUDA_ARCHITECTURES "75")

add_executable(capability capability.cu)
set_target_properties(capability PROPERTIES CUDA_ARCHITECTURES "75")

add_executable(mul-mat mul-mat.cu)
set_target_properties(mul-mat PROPERTIES CUDA_ARCHITECTURES "75")

add_executable(reduction reduction.cu)
set_target_properties(reduction PROPERTIES CUDA_ARCHITECTURES "75")