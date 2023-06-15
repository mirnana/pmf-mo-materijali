#include <iostream>

int main() {
    double a[] {1, 2, 3, 4, 5};
    double b[] {1, 1, 1, 1, 1};
    
    double dot = 0.0;
    
    #pragma omp parallel for reduction(+: dot)
    for(int i=0; i<5; ++i)
        dot += a[i]*b[i];

    std::cout << dot << "\n";
    return 0;
}