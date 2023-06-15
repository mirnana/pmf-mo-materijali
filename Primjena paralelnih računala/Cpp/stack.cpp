#include "stack.h"
#include <iostream>
#include <string>
#include <thread>

Stack<std::string> stog;
using namespace std::chrono_literals;

void generate(){
    for(int i=0; i<50; ++i){
        std::string str{"some text "};
        str += std::to_string(i);
        str += "\n";
        stog.push(str);
    }
}

void consume(char c){
    while(true){
        try {
            //std::this_thread::sleep_for(2ms);
            auto p = stog.pop();
            std::cout << c << " : " << *p;
        } catch(std::runtime_error & e) {
            break;
        }
    }
}

int main() {
    generate();
    std::thread t1(consume, 'A');
    std::thread t2(consume, 'B');

    t1.join();
    t2.join();

    return 0;
}