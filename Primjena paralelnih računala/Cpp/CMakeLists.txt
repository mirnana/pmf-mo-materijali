cmake_minimum_required(VERSION 3.13)
project(Cpp_threads CXX)

find_package(Threads REQUIRED)

add_executable(hello hello.cpp)
target_link_libraries(hello Threads::Threads)

add_executable(test test.cpp)
target_link_libraries(test Threads::Threads)

add_executable(accum accum.cpp)
target_link_libraries(accum Threads::Threads)

add_executable(account account_main.cpp account.cpp account.h)
target_link_libraries(account Threads::Threads)

add_executable(stack stack.cpp stack.h)
target_link_libraries(stack Threads::Threads)

add_executable(sinhro sinhro.cpp)
target_link_libraries(sinhro Threads::Threads)

add_executable(tsqueue tsqueue.cpp)
target_link_libraries(tsqueue Threads::Threads)

add_executable(future future.cpp)
target_link_libraries(future Threads::Threads)

add_executable(future1 future1.cpp)
target_link_libraries(future1 Threads::Threads)

add_executable(packaged_task packaged_task.cpp)
target_link_libraries(packaged_task Threads::Threads)

add_executable(async async.cpp)
target_link_libraries(async Threads::Threads)

add_executable(future_ali_async future_ali_async.cpp)
target_link_libraries(future_ali_async Threads::Threads)

add_executable(compare_exchange compare_exchange.cpp)
target_link_libraries(compare_exchange Threads::Threads)