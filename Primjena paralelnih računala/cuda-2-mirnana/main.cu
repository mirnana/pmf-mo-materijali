#include <stdio.h>
#define NTIMES 1

#define R 2
// OVO MI NIJE JASNO. BLOK_SIZE TREBA BITI ULAZNI BLOK (ONAJ VEĆI), A TILE_SIZE 
// IZLAZNI BLOK (MANJI). 
//#define BLOCK_SIZE 4 * R    // input tile
//#define TILE_SIZE 2 * R     // output tile
// PRIRODNO JE UZETI
#define BLOCK_SIZE 32            // input tile
#define TILE_SIZE  (32 - 2*(R))  // output tile


__constant__ float d_K[2*R+1][2*R+1] =
    {
        {0.1f, 0.2f, 0.3f, 0.2f, 0.1f},
        {0.2f, 0.3f, 0.4f, 0.3f, 0.2f},
        {0.3f, 0.4f, 0.5f, 0.4f, 0.3f},
        {0.2f, 0.3f, 0.4f, 0.3f, 0.2f},
        {0.1f, 0.2f, 0.3f, 0.2f, 0.1f}
    };

// TREBALA SU BITI 2 KERNELA - JEDNOSTAVAN I OPTIMIZIRAN. 
__global__
void convolutionKernel(float *A, float *B, int noRows, int noCols) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col  = blockIdx.x * TILE_SIZE + tx; // OVO BI TREBAKI BITI INDEKSI U IZLAZNO POLJE!
    int row  = blockIdx.y * TILE_SIZE + ty;

    int tile_border_L = blockIdx.x * TILE_SIZE;                 // left
    int tile_border_R = ((blockIdx.x + 1) * TILE_SIZE < noCols) 
                        ? (blockIdx.x + 1) * TILE_SIZE 
                        : noCols;                               // right
    int tile_border_T = blockIdx.y * TILE_SIZE;                 // top
    int tile_border_B = ((blockIdx.y + 1) * TILE_SIZE < noRows) 
                        ? (blockIdx.y + 1) * TILE_SIZE 
                        : noRows;                               // bottom

    __shared__ float localA[BLOCK_SIZE][BLOCK_SIZE];
    if(row >= tile_border_T && row < tile_border_B
        && col >= tile_border_L && col < tile_border_R)
        localA[ty][tx] = A[row * noCols + col];
    else
        localA[ty][tx] = 0.0f;

    __syncthreads();

    float result = 0.0f;
    for(int m=0; m<2*R+1; ++m){
        for(int n=0; n<2*R+1; ++n){
            int convRow = row + m - R;
            int convCol = col + n - R;
            if(convRow >= tile_border_T && convRow < tile_border_B
                && convCol >= tile_border_L && convCol < tile_border_R)
                result += d_K[m][n] * localA[convRow][convCol];
        }
    }
    __syncthreads();

    if(col < noCols && row < noRows)
        B[row * noCols + col] = result;
}


// B JE STALNO NULA, TREBALO JE PROVJERAVATI KOD GREŠKE DA SE VIDI O ČEMU SE RADI!
__host__ 
void check(float *A, float *B, float K[2*R+1][2*R+1], int M, int N) {
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {

            float tmp = 0.0;
            for(int m = 0; m < 2*R+1; m++) {
                for(int n = 0; n < 2*R+1; n++) {
                    int row = i + m - R;
                    int col = j + n - R;
                    if(row >= 0 && row < M && col >= 0 && col < N)
                        tmp += K[m][n] * A[row * N + col];
                }
            }

            if(tmp != B[i * N + j])
                printf("Error, B[%d][%d]=%f, tmp=%f\n", i, j, B[i * N + j], tmp);
        }
    }
}

int main(void) {

    int M = 1000;  // broj redaka
    int N = 1000; // broj stupaca
    int bytes = M * N * sizeof(float);
    double time = 0.0;
    
    for(int l = 0; l < NTIMES; l++) {
        
        float *h_A = (float*) malloc(bytes);
        float *h_B = (float*) malloc(bytes);
        for(int i = 0; i < M; i++) {
            for(int j = 0; j < N; j++) {
                h_A[i*N+j]=  ((float)rand()/(float)(RAND_MAX)) * 2.0f;  // 1.0f;  MALO SLOŽENIJE!
            }
        }

        float h_K[2*R+1][2*R+1] =
            {
                {0.1, 0.2, 0.3, 0.2, 0.1},
                {0.2, 0.3, 0.4, 0.3, 0.2},
                {0.3, 0.4, 0.5, 0.4, 0.3},
                {0.2, 0.3, 0.4, 0.3, 0.2},
                {0.1, 0.2, 0.3, 0.2, 0.1}
            };

        float *d_A, *d_B;
        cudaMalloc(&d_A, bytes);
        cudaMalloc(&d_B, bytes);
        
        cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);

        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
        dim3 gridDim(ceil(M/(float)TILE_SIZE), ceil(N/(float)TILE_SIZE), 1);

        cudaDeviceSynchronize();
        clock_t start = clock();

        convolutionKernel<<<gridDim, blockDim>>>(d_A, d_B, M, N);  

        cudaDeviceSynchronize();
        clock_t stop = clock();
        
        cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost);

        time += 1000.0*(stop-start)/CLOCKS_PER_SEC;

        cudaFree(d_A);
        cudaFree(d_B);

        check(h_A, h_B, h_K, M, N);

        free(h_A);
        free(h_B);
    }

    printf("Prosječno vrijeme u %d izvršavanja: %f\n", NTIMES, time/NTIMES);
    return 0;
}
