#include <iostream>

int main() {
    int x = 5;
    #pragma omp parallel firstprivate(x)
    {
        ++x;
        std::cout << "privatni x = " << x << "\n";
    }

    std::cout << "x = " << x << "\n";

    return 0;
}