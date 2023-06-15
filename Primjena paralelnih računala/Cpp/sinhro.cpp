#include <iostream>
#include <mutex>
#include <thread>
#include <chrono>
#include <condition_variable>

int data;
bool flag = false;
std::condition_variable cv;
std::mutex data_mutex;

void set_data() {
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    std::lock_guard<std::mutex> lg(data_mutex);
    data = 13;
    flag = true;
    cv.notify_all();
}

void read_data() {
    std::unique_lock<std::mutex> ul(data_mutex);
    /*while(!flag) {
        ul.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ul.lock();
    }*/

    // ako je predikat (tj lambda izraz) vratio da je flag=false, onda cv unlocka ul \
    // i stavlja nit na spavanje. nit spava sve dok cv ne pošalje notify_all
    cv.wait(ul, []{return flag == true;});  

    // sad je flag = true i mutex zakljucan pa smijemo citati data
    std::cout << data << "\n";
}

int main() {
    std::thread t1(read_data);
    std::thread t2(set_data);
    std::thread t3(read_data);

    t1.join();
    t2.join();
    t3.join();

    return 0;
}