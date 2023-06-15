#include <iostream>
#include <future>  // za async i future
#include <thread> // za sleep_for

// Kod koji želimo izvršiti paralelno s glavnim programom
// zatvorimo u funkciju.
void f(char c){
    for(int i=0; i<10; ++i){
        std::cout.put(c).flush();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

int main(){
    std::future<void> f_future = std::async(f, '+');  // asinhrono izvršavanje 
    f('.');    // paralelno izvršavanje u glavnoj programskoj niti  
    f_future.wait(); // čekaj na async. 
    return 0;
}