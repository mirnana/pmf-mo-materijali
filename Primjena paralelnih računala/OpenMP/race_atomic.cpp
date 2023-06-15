#include <iostream>

int main() {
    int x = 0;
    #pragma omp parallel 
    {
        #pragma omp atomic
        ++x;
    }

    std::cout << "broj threadova = " << x << "\n";

    return 0;
}