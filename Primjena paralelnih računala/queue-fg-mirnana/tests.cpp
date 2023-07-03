#include "tsqueue-fine-graded.h"

#include <thread>
#include <chrono>
#include <string>
#include <vector>
#include <cassert>
#include <algorithm>


// Testovi za tsqueue-fine-graded.cpp

void generator(Queue<int> & queue, int N){
  using namespace std::literals;
  for(int i=0; i<N; ++i){
    queue.push(i);
    std::this_thread::sleep_for(10ms);
  }
}

void sink(Queue<int> & queue, int N){
  using namespace std::literals;
  int i=0, cnt = 0;
  while(i<N){
    cnt++;
    int n;
    if(!queue.try_pop(n))
      std::this_thread::sleep_for(50ms);
    else
      i++;
  }
  //std::cout << "Broj poziva try_pop() metode = " << cnt << "\n";
}

template <typename T>
void worker(Queue<T> & queue, std::vector<T> & vec, int max){
    T n;
	int k = 0;
    while(k<max){
       queue.wait_pop(n);
       vec.push_back(n);
	   ++k;
    }
}

// ====  testovi, main =====

void test0(){
    Queue<int> queue;
    queue.push(3);
    queue.push(4);
    queue.push(5);
    
    int n = 0;
    assert(queue.try_pop(n));
    assert(n == 3);
    assert(queue.try_pop(n));
    assert(n == 4);
    assert(queue.try_pop(n));
    assert(n == 5);
    assert(!queue.try_pop(n));
}

void test1(){
    Queue<int> queue;
   
    std::thread t1(generator, std::ref(queue), 100);
    std::thread t2(sink, std::ref(queue), 100);

    t1.join();
    t2.join();

	assert(queue.empty());
    //queue.print();
}


void test2(){
    Queue<std::string> squeue;
    squeue.push("dodo");
    squeue.push("vlado");
    squeue.push("medo");
    
    std::string name;
    assert(squeue.try_pop(name));
    assert(name == "dodo");
    squeue.wait_pop(name);
    assert(name == "vlado");
    squeue.wait_pop(name);
    assert(name == "medo");
    assert(!squeue.try_pop(name));
	assert(squeue.empty());
}

void test3(){
    Queue<int> queue;
	std::vector<int> v1, v2, v3;
    std::thread t1(generator,   std::ref(queue), 100);
    std::thread t2(worker<int>, std::ref(queue), std::ref(v1), 20);
    std::thread t3(worker<int>, std::ref(queue), std::ref(v2), 20);
    std::thread t4(worker<int>, std::ref(queue), std::ref(v3), 20);

	t1.join();
	t2.join();
	t3.join();
	t4.join();

	// Testiranje. Vrijednosti moraju biti sortirane
    assert( std::is_sorted(v1.begin(), v1.end()) ) ;
    assert( std::is_sorted(v2.begin(), v2.end()) );
    assert( std::is_sorted(v3.begin(), v3.end()) );

    // Ne smije biti duplikata
    for(auto x : v1){
      assert(!std::binary_search(v2.begin(), v2.end(), x));
      assert(!std::binary_search(v3.begin(), v3.end(), x));
    }

    for(auto x : v2){
      assert(!std::binary_search(v3.begin(), v3.end(), x));
    }

    assert(v1.size() + v2.size() + v3.size() == 60);

}
