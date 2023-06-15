cmake_minimum_required(VERSION 3.14)
project (ts_map)
set(CMAKE_CXX_STANDARD 20)

find_package(Threads REQUIRED)

add_executable(main ts_map_main.cpp ts_map.h)
target_link_libraries(main Threads::Threads)
