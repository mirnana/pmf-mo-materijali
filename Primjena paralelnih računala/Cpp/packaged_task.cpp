// future1 pomocu packaged taska

#include <future>
#include <string>
#include <iostream>
#include <stdexcept>
#include <thread>

std::string input(){
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
    
    return result;
}

int main(){
    try{
        //std::promise<std::string> promise;
        std::packaged_task<std::string(void)> pack(input);
        auto future = pack.get_future();
        std::thread t(std::move(pack));  
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