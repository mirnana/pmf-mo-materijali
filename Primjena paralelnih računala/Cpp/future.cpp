#include <future>
#include <thread>
#include <chrono>
#include <iostream>

using namespace std::literals::chrono_literals;

void calculate(std::promise<double>&& prom){ 
    std::this_thread::sleep_for(2s);  // izračunavanje vrijednosti ...
    prom.set_value(61.1);  // spremi vrijednost za std::future<> objekt 
}

int main(){
    std::promise<double> promise; 
    std::future<double> future = promise.get_future(); 
    std::thread t(calculate, std::move(promise));   // promise objekt se ne može kopirati, stoga calculate uzima desnu referencu i koristimo std::move za prijenos vrijednosti
    std::cout << " future value = " << future.get() << std::endl; 
    t.join(); 
    return 0;
}