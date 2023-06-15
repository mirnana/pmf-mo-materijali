#include <iostream>
#include <thread>
#include <string>
#include <chrono>

void err(std::string const & polje){  
    for(int i=0; i<polje.size(); ++i)
        std::cout << polje[i] << " ";
    std::cout << std::endl;
}

void test() {
    char niz[]= "abcdef";
    //std::thread t1(err, niz);  
    std::thread t1(err, std::string(niz));  
    t1.detach();               
    //t1.join();
}

int main()
{
    test();
    std::this_thread::sleep_for(std::chrono::seconds(1)); 
    return 0;
}