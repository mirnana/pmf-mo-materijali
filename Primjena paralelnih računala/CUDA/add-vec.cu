#include <iostream>

// __global__ fje se izvršavaju na GPU, a pohranjene su u (tj pozivaju se iz) CPU
__global__ 
void addVec(float * a, float * b, float * c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // threadIdx je indeks threada
    // blockIdx.x je indeks bloka
    // blockDim.x je drugi konfiguracijski parametar (vidi poziv fje addVec) - dimenzija bloka!

    // može se dogoditi i>=N pa treba pripaziti
    if (i < N)
        c[i] = a[i] + b[i];
}

// __host__ je default, ali svejedno se ova oznaka može dodati
// izvršava se na CPU
__host__
void check(float * a, float * b, float * c, int N) {
    for(int i = 0; i < N; i++) {
        float tmp = a[i] + b[i];
        if(c[i] != tmp)
            printf("Error, c[%d]=%f\n", i, c[i]);
    }
} 


int main(void) {
    const int N = 1000000;

    // host = CPU, device = GPU

    // alokacija na hostu
    float * h_a = (float*) malloc(N * sizeof(float));
    float * h_b = (float*) malloc(N * sizeof(float));
    float * h_c = (float*) malloc(N * sizeof(float));

    for(int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_c;

    // alokacija na deviceu
    cudaMalloc((void**) &d_a, N * sizeof(float));
    cudaMalloc((void**) &d_b, N * sizeof(float));
    cudaMalloc((void**) &d_c, N * sizeof(float));

    // kopiranje ulaznih podataka s hosta na device
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // računanje zbroja
    // threadovi se dijele u grupe, moramo definirati koliko

    addVec<<<ceil(N/256.0), 256>>>(d_a, d_b, d_c, N);  //<<< , >>> je za KONFIGURACIJSKE PARAMETRE
    // drugi konfiguracijski parametar je broj grupa threadova (256)
    // prvi je broj elemenata arraya koji taj thread hendla, tj N/256, ali ne zelimo cjelobrojno dijeljenje pa dijelimo s 256.0

    // kopiranje izlaznih podataka s devicea na host
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    check(h_a, h_b, h_c, N);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}