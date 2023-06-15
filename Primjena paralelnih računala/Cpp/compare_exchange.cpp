#include <atomic>
#include <thread>
#include <iostream>
#include <mutex>
#include <chrono>

std::atomic<long> current{2};
std::mutex cout_mut;

void f(int i){
    long expected = current.load();
    long desired;
    do {
        desired = expected*expected;
        std::this_thread::sleep_for(std::chrono::microseconds(12));
    }
    while (!current.compare_exchange_weak(expected, desired));

    std::lock_guard<std::mutex> lg(cout_mut);
    std::cout << "f(" << i <<"): " << desired << std::endl;
}

void g(int i){
    long expected = current.load();
    long desired;
    do {
        desired = expected+expected;
        std::this_thread::sleep_for(std::chrono::microseconds(12));
    }
    while (!current.compare_exchange_weak(expected, desired));

    std::lock_guard<std::mutex> lg(cout_mut);
    std::cout << "g(" << i <<"): " << desired << std::endl;
}

int main(){
    int i = 0;
    std::thread t1(f, ++i);
    std::thread t2(g, ++i);
    std::thread t3(f, ++i);
    std::thread t4(g, ++i);
    std::thread t5(f, ++i);

    t1.join(); t2.join(); t3.join(); t4.join(); t5.join();
    std::cout << "Konačno : " << current.load() << std::endl;
}