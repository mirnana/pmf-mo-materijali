#include <stdlib.h> // za rand
#include <stdio.h>

__global__
void mul_mat_kernel_NAIVNA_VERZIJA(float *A, float *B, float *C, int n) {
    // na kojem podatku radi jezgra - mjesto(row, col)
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    
    // n je broj redaka i broj stupaca --- matrice su kvadratne
    float tmp = 0.0f;
    if (row < n && col < n) {
        for(int k = 0; k < n; k++) {
            // A[row][k] * B[k][col]
            tmp += A[row * n + k] * B[k * n + col];
        }

        C[row * n + col] = tmp; //C[row][col]
    }
}

# define BLOCK_SIZE 32
///////////////////////////////////////////////// OVO NE RADI!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
__global__
void mul_mat_kernel(float *A, float *B, float *C, int n) {

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int col  = bx * BLOCK_SIZE + tx;  
    int row  = by * BLOCK_SIZE + ty;

    // uvodimo blokove dijeljene memorije!
    __shared__ float Aloc[BLOCK_SIZE][BLOCK_SIZE];// za smještanje komada matrice A
    __shared__ float Bloc[BLOCK_SIZE][BLOCK_SIZE];// za smještanje komada matrice B

    int k = 0;
    float tmp = 0.0f;
    while(k * BLOCK_SIZE < n) {
        Aloc[ty][tx] = (row < n && tx + BLOCK_SIZE*k < n) 
                        ? 
                        A[row * n + (tx + BLOCK_SIZE*k)] 
                        : 
                        0.0f;
        Bloc[ty][tx] = (col < n && (ty + BLOCK_SIZE*k) < n)
                        ? 
                        B[(ty + BLOCK_SIZE*k) * n  + col]
                        :
                        0.0f;

        __syncthreads();
        ++k;

        for(int i = 0; i < BLOCK_SIZE; i++) {
            tmp += Aloc[ty][i] + Bloc[i][tx];
        }
        __syncthreads();    // jer svaka programska nit ima svoju varijablu tmp pa se moraju syncati
    }
    
    if(row < n && col < n)
        C[row * n + col] = tmp;
}

__host__
void check(float *A, float *B, float *C, int n) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            float tmp = 0.0f;
            for(int k = 0; k < n; k++) {
                tmp += A[i*n+k] * B[k*n+j];
            }

            float diff = fabs(tmp - C[i*n+j]);
            float mod = fabs(tmp);
            float EPS = 1E-6;
            if(diff > EPS * mod)
                printf("Error, C[%d][%d]=%f, tmp = %f\n", i, j, C[i*n+j], tmp);
        }
    }
}

int main(void) {
    int nrows =  1020;
    int N = nrows * nrows;

    //C=A*B
    int numBytes = nrows*nrows*sizeof(float); // broj bajtova za alokaciju matrica
    float *h_A = (float*) malloc(numBytes);
    float *h_B = (float*) malloc(numBytes);
    float *h_C = (float*) malloc(numBytes);
    
    srand(777);
    for(int i = 0; i < N; i++) {
        h_A[i] = 2.0f*rand()/RAND_MAX;
        h_B[i] = 2.0f*rand()/RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, numBytes);
    cudaMalloc(&d_B, numBytes);
    cudaMalloc(&d_C, numBytes);

    cudaMemcpy(d_A, h_A, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, numBytes, cudaMemcpyHostToDevice);

    dim3 gridDim(ceil(nrows/(float)BLOCK_SIZE), ceil(nrows/(float)BLOCK_SIZE), 1);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    mul_mat_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, nrows);

    cudaMemcpy(h_C, d_C, numBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    check(h_A, h_B, h_C, nrows);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}