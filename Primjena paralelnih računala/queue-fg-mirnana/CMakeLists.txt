cmake_minimum_required(VERSION 3.5)
project(queue)
find_package(Threads REQUIRED)

add_executable(queue main.cpp tests.cpp tsqueue-fine-graded.h)
target_link_libraries(queue Threads::Threads)
