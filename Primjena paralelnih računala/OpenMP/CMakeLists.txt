cmake_minimum_required(VERSION 3.13)
project(openmp)

find_package(OpenMP REQUIRED)

add_compile_options(${OpenMP_CXX_FLAGS} "-Wall" "--std=c++17")
add_link_options(${OpenMP_CXX_FLAGS})
# ${OpenMP_CXX_FLAGS} == -fopenmp

add_executable(hello_world hello_world.cpp)
add_executable(race race.cpp)
add_executable(sum sum.cpp)
add_executable(dot dot.cpp)
add_executable(histogram histogram.cpp)
add_executable(pi pi.cpp)
add_executable(mc mc.cpp)
add_executable(mat_vec mat_vec.cpp)
add_executable(histogram_critical histogram_critical.cpp)
add_executable(race_atomic race_atomic.cpp)
add_executable(tokenizer tokenizer.cpp)
add_executable(task_intro task_intro.cpp)
add_executable(list list.cpp)
add_executable(quicksort quicksort.cpp)