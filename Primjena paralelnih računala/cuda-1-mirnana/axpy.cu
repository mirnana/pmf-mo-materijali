#include <stdio.h>
#include <time.h>
#include <math.h>

__global__
void axpyGlobal(float *a, float *x, float *y, float *z, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N)
        z[i] = (*a) * x[i] + y[i];
}

__host__
void axpyHost(float *a, float *x, float *y, float *z, int N) {
    for(int i = 0; i < N; i++) {
        z[i] = (*a) * x[i] + y[i];
    }
}

__host__ 
void check(float *a, float *x, float *y, float *z, int N) {
    for(int i = 0; i < N; i++) {
        float tmp = (*a) * x[i] + y[i];
        if(z[i] != tmp) 
            printf("Error, z[%d]=%f", i, z[i]);
    }
}

// Broj ponavljanja za mjerenje vremena.
#define NTIMES 16  

int main(int argc, char *argv[])
{
    int N = 80000000; // dimenzije polja
    
    double GPU_time = 0.0, CPU_time = 0.0;

    for(int n = 0; n < NTIMES; n++) {

        // alokacija i inicijalizacija x,y na CPU
        float *h_a = (float*) malloc(1 * sizeof(float));
        float *h_x = (float*) malloc(N * sizeof(float));
        float *h_y = (float*) malloc(N * sizeof(float));
        float *h_z = (float*) malloc(N * sizeof(float));
        *h_a = 3.0f;
        for(int i = 0; i < N; i++) {
            h_x[i] = 8.0f;
            h_y[i] = 5.0f;
        }

        // prebacivanje a,x,y,z na GPU
        float *d_a, *d_x, *d_y, *d_z;
        cudaMalloc((void**) &d_a, 1 * sizeof(float));
        cudaMalloc((void**) &d_x, N * sizeof(float));
        cudaMalloc((void**) &d_y, N * sizeof(float));
        cudaMalloc((void**) &d_z, N * sizeof(float));

        cudaMemcpy(d_a, h_a, 1 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);

        // izvršavanje axpy jezgre na GPU uz mjerenje vremena i kopiranje polja z na CPU
        cudaDeviceSynchronize();
        clock_t start = clock();

        axpyGlobal<<<ceil(N/1024.0), 1024>>>(d_a, d_x, d_y, d_z, N); 
        cudaMemcpy(h_z, d_z, N * sizeof(float), cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();
        clock_t stop = clock();
        GPU_time += 1000.0*(stop-start)/CLOCKS_PER_SEC;

        // dealokacija na GPU
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_z);
        
        // provjera da GPU i CPU daju isti rezultat
        check(h_a, h_x, h_y, h_z, N);

        // mjerenje vremena izvršavanja na CPU
        start = clock();
        axpyHost(h_a, h_x, h_y, h_z, N);
        stop = clock();
        CPU_time += 1000.0*(stop-start)/CLOCKS_PER_SEC;

        // dealokacija na CPU
        free(h_x);
        free(h_y);
        free(h_z);
    }

    // printanje vremena izvršavanja
    printf("Vrijeme izvršavanja na GPU: %lf ms\n", GPU_time/NTIMES);
    printf("Vrijeme izvršavanja na CPU: %lf ms\n", CPU_time/NTIMES);

    return 0;
}
