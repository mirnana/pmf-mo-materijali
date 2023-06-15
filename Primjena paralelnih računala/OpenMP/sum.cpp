#include <iostream>
#include <omp.h>

void sum( double* __restrict a
        , double* __restrict b
        , double* __restrict c
        , int N) {

    #pragma omp parallel
    {
    int no_threads = omp_get_num_threads();
    int this_thread = omp_get_thread_num();
    
    int size =  N / no_threads;
    int excess = N % no_threads;

    int start = 0;
    int end = 0;

    if (this_thread < excess) {
        start = (size+1) * this_thread;
        end = start + size + 1;
    }
    else {
        start = this_thread * size + excess;
        end = start + size;
    }
    
    for (int i = start; i < end; i++) {
        c[i] = a[i] + b[i];
    }
    }
}

void sum1(double* __restrict a
        , double* __restrict b
        , double* __restrict c
        , int N) {

    #pragma omp parallel for
    for(int i=0; i<N; i++) 
        c[i] = a[i]+b[i];
}

int main() {
    
    int N = 10'000'000;
    double* a = new double [N];
    double* b = new double [N];
    double* c = new double [N];

    for(int i = 0; i < N; i++) {
        a[i] = 3.0;
        b[i] = 4.0;
    }

    double begin = omp_get_wtime();
    sum1(a, b, c, N);
    double end = omp_get_wtime();
    std::cout << "vrijeme je " << end-begin << "\n";

    //std::cout << c[10] << "\n";

    delete[] c;
    delete[] b;
    delete[] a;
 
    return 0;

}
