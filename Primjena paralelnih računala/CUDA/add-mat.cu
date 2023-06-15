#include <iostream>

__global__ 
void addMat(float * a, float * b, float * c, int Nrow, int Ncol) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int i = row * Ncol + col;
    if (row < Nrow && col < Ncol)
        c[i] = a[i] + b[i];
}

__host__
void check(float * a, float * b, float * c, int N) {
    for(int i = 0; i < N; i++) {
        float tmp = a[i] + b[i];
        if(c[i] != tmp)
            printf("Error, c[%d]=%f\n", i, c[i]);
    }
} 


int main(void) {
    const int Nrow = 100, Ncol = 100;
    const int N = Nrow * Ncol;

    float * h_A = (float*) malloc(N * sizeof(float));
    float * h_B = (float*) malloc(N * sizeof(float));
    float * h_C = (float*) malloc(N * sizeof(float));

    for(int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = N-i;
    }

    float *d_A, *d_B, *d_C;

    cudaMalloc((void**) &d_A, N * sizeof(float));
    cudaMalloc((void**) &d_B, N * sizeof(float));
    cudaMalloc((void**) &d_C, N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // konfiguracijski parametri ne moraju biti cijeli brojevi (kad jesu, onda se ustvari sprema samo .x vrijednost)
    // parametri su utstvari strukture tipa dim3
    dim3 blockDim(10, 10, 1);
    dim3 gridDim(ceil(Nrow/10.0), ceil(Ncol/10.0), 1);

    cudaDeviceSynchronize();    // ova sinkronizacija se mora obaviti kad se mjeri vrijeme, i kod starta i kod stopa!!
    clock_t start = clock();

    addMat<<<gridDim, blockDim>>>(d_A, d_B, d_C, Nrow, Ncol);  

    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    clock_t stop = clock();

    printf("Time = %lf ms\n", 1000.0*(stop-start)/CLOCKS_PER_SEC);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    check(h_A, h_B, h_C, N);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}