#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>

#include <array>
#include <random>
#include <ctime>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <atomic>

#include <iostream>

template<typename T>
class Queue {
    public:
        Queue() = default;
        Queue(Queue const &) = delete;
        Queue & operator=(Queue const &) = delete; 

        void push(T const & t) {
            std::lock_guard<std::mutex> lg(mut);
            data.push(t);
            cv.notify_all();
        }
        bool try_pop(T & t) {  // vraća vrijednost gornjeg elementa kao referencu
            std::lock_guard<std::mutex> lg(mut);
            if(data.empty()) {
                return false;
            }
            else {
                t = data.front();
                data.pop();
                return true;
            }
        }
        void wait_pop(T & t) {
            std::unique_lock<std::mutex> ul(mut);

            cv.wait(ul, [this]{return !data.empty();}); // čekamo dok data ne postane neprazan
            // lambda u nekom objektu mora u uglatim zagradama dobiti this tog objekta!
            t = data.front();
            data.pop();
        }

    private:
        std::queue<T> data;
        std::mutex mut;      
        std::condition_variable cv;
};

Queue<int> tsqueue;
// za ispis dobivenih vrijednosti
std::array<std::vector<int>,3>  v;
std::mutex stop_mut;  // štiti samo stop zastavicu
bool stop = false;

void generator(){
    int n = 1;
    while(n <= 100) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        tsqueue.push(n);
        n++;
    }
    std::lock_guard<std::mutex> lg(stop_mut);
    stop = true;
}

void worker(std::vector<int> & vec){
    int n = 0;
    while(true) {
        if(tsqueue.try_pop(n)){
            vec.push_back(n);
        }
        else {
            std::lock_guard<std::mutex> ul(stop_mut);
            if(stop) return;
        }
    }
}

void print(int k){
    std::cout << "thread " << k+2 << " :\n";
    for(auto x : v[k])
        std::cout << x << ",";
    std::cout << std::endl;
}

int main() {
    std::thread t1(generator);
    std::thread t2(worker, std::ref(v[0]));
    std::thread t3(worker, std::ref(v[1]));
    std::thread t4(worker, std::ref(v[2]));

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // Ispis vrijednosti
    print(0);
    print(1);
    print(2);
    std::cout << "Received size = " << v[0].size() + v[1].size() + v[2].size() << std::endl;

    
    return 0;
}