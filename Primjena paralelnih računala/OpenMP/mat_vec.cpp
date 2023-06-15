#include <iostream>
#include <omp.h>

void dmvm(const double * __restrict mat
        , const double * __restrict py
        , double * __restrict px
        , int rows
        , int cols) {
    #pragma omp parallel for
    for(int i=0; i<rows; ++i){
        px[i] = 0.0;
        for(int j=0; j<cols; ++j)
            px[i] +=  mat[i*cols+j]*py[j];  // r_i = A(i,j)*v_j
    }
}

void dmtrvm(  const double * __restrict mat
            , const double * __restrict px
            , double * __restrict py
            , int rows
            , int cols) {

    # pragma omp parallel
    {
        #pragma omp for
        for(int i=0; i<cols; ++i)
            py[i] = 0.0;

        for(int j=0; j<rows; ++j) {
            #pragma omp for schedule(static) nowait
            for(int i=0; i<cols; ++i){
                py[i] +=  mat[i*cols+j]*px[j]; 
            }   // implicitna barijera! nowait ju uklanja
        }

    }
}

int main() {

    int rows = 1000;
    int cols = 600;

    // x = Ay
    double *pmat = new double[rows*cols];
    double *py   = new double[cols];
    double *px   = new double[rows];

    for (int i = 0; i < rows; i++) {
        px[i] = 0.0;
        for(int j = 0; j < cols; j++) {
            pmat[i*cols+j] = i + j + 1.0; // pmat[i,j] =i+j+1
            py[j] = 1.0;
        }
    }

    double time = 0.0;
    for(int k = 0; k < 50; k++) {
        double begin = omp_get_wtime();
        dmvm(pmat, py, px, rows, cols);
        double end = omp_get_wtime();

        time += end-begin;
    }
    

    for(int i= 0; i < rows; i++) {
        std::cout << px[i] << ", ";
    }
    std::cout << "\n vrijeme je " << time/50 << "\n";

    time = 0.0;
    for(int k = 0; k < 50; k++) {
        double begin = omp_get_wtime();
        dmtrvm(pmat, px, py, rows, cols);
        double end = omp_get_wtime();

        time += end-begin;
    }
    

    for(int i= 0; i < rows; i++) {
        std::cout << py[i] << ", ";
    }
    std::cout << "\n vrijeme je " << time/50 << "\n";
    return 0;
}