#include <iostream>
#include <numeric>
#include <vector>
#include <thread>

template <typename T, typename Iterator>
void block_accumulate(Iterator first, Iterator last, T & result){
    T init = 0;
    result = std::accumulate(first, last, init);
}

template <typename T, typename Iterator>
T parallel_accumulate(Iterator first, Iterator last, T init){
    auto length = std::distance(first, last);
    if(!length)
        return init; 

    int phard = std::thread::hardware_concurrency();
    if(!phard)
        phard = 2;  
    
    int n = 1000;  
    int pmax = (length > n) ? length/n : 1;  
    int p = std::min(pmax, phard);  

    std::vector<T> results(p);
    std::vector<std::thread> threads(p-1);

    auto block_size = length/p;
    auto block_first = first;
    auto block_last = block_first;
    for(int i=0; i < p-1; ++i){
        std::advance(block_last, block_size);
        threads[i] = std::thread(block_accumulate<T,Iterator>,
                                 block_first, block_last, std::ref(results[i]));
        block_first = block_last;
    }
    
    results[p-1] = std::accumulate(block_first, last, T());  

    for(int i=0; i < p-1; ++i)
        threads[i].join();  
    
    return std::accumulate(results.begin(), results.end(), init); 
}

int main() {

    int N = 1000;
    std::vector<int> vec(N);
    for(int i=0; i < N; i++) {
        vec[i] = i+1;
    }

    std::cout << parallel_accumulate( vec.begin()
                                    , vec.end()
                                    , 0) 
            << "\n";


    return 0;
}