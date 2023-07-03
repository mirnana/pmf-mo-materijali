cmake_minimum_required(VERSION 3.1)

project(nbody CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)
add_compile_options(${OpenMP_CXX_FLAGS} "-Wall")
add_link_options(${OpenMP_CXX_FLAGS})

add_executable(main main.cpp nbody.cpp nbody.h)
