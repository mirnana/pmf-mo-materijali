#include "tsqueue-fine-graded.h"
#include <thread>
#include <stdexcept>

Queue<int> kju;

void kju_fill() {
    for(int i = 0; i < 200; i++) {
        if(i % 20 == 0) {
            kju.print();
        }
        kju.push(i);
    }
} 

void kju_rearrange() {
    for(int i = 2; i <= 200; i *= 2) {
        int popped;
        if(kju.try_pop(popped)) {
            kju.push(popped);
        }
    }
}

void kju_wait() {
    int popped = 0;
    while(popped != 100) {
        kju.wait_pop(popped);
    }
}

void kju_empty() {
    for(int i = 1; i < 200; i *= 2) {
        if(kju.empty()) {
            kju.push(i - 1);
        }
    }
}

void test0();
void test1();
void test2();
void test3();

int main(){
	test0();
    test1();    
    test2();
    test3();

	// Vaši testovi dolaze ovdje. 
    std::thread t1(kju_wait);
    std::thread t2(kju_rearrange);
    std::thread t3(kju_fill);
    std::thread t4(kju_empty);

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    kju.print();

    int popped;
    kju.try_pop(popped);
    if(popped != 101) {
        throw std::runtime_error("Izvođenje je završilo na neočekivani način."); 
    }

    return 0;
}
