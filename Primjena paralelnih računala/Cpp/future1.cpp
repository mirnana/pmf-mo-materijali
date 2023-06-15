#include <future>
#include <string>
#include <iostream>
#include <stdexcept>
#include <thread>

void input(std::promise<std::string> && promise){
    try{
        std::string result;
        char c;
        while(true){
            std::cin.get(c);
            if(c == '?')
                break;  
            else if(c == 'x')
                throw std::runtime_error("x not allowed!"); 
            else
                result.push_back(c);
        }
        promise.set_value(std::move(result)); 
    }
    catch(...){
        promise.set_exception(std::current_exception()); 
    }
}

int main(){
    try{
        std::promise<std::string> promise;
        auto future = promise.get_future();
        std::thread t(input, std::move(promise));  
        t.detach();
        std::cout << "Input is : " << future.get() << std::endl; 
    }
    catch(std::exception const & e){
        std::cout << "Exception: " << e.what() << std::endl; 
    }
    catch(...){
        std::cout << "Exception! " <<  std::endl;
    }
    return 0;
}