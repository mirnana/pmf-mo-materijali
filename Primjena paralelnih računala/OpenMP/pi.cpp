#include <iostream>

int main() {
    int N = 10000000;
    const int no_threads = 4;
    //double factor = 1.0;
    double sum = 0.0;
    #pragma omp parallel for num_threads(no_threads) \
                             reduction(+:sum) 
    for (int k = 0; k < N; ++k) {
        double factor = (k%2 == 0) ? 1 : -1;
        sum += factor/(2*k+1);
    }
    double pi_approx = 4.0*sum;

    std::cout << pi_approx << "\n";
    //std::cout << 


    return 0;
}