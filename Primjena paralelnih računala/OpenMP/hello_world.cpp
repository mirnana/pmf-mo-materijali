#include <iostream>
#include <omp.h>

int main() 
{
#pragma omp parallel //num_threads(7)
{
    auto my_thread  = 0;
    auto no_threads = 1;
    #ifdef _OPENMP
        my_thread = omp_get_thread_num();
        no_threads = omp_get_num_threads();
    #endif
    //if(my_thread == 0) std::cout << "num_of_threads =" << no_threads << " \n";
    #pragma omp single 
    {
        std::cout << "num_of_threads = " << no_threads << " \n";
        std::cout << "my_thread = " << my_thread << "\n";
    }

    std::cout << "hello from " << my_thread << "\n";
}
    return 0;
}

