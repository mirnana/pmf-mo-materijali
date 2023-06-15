#define BLOCK_SIZE 32

__global__
void reductionKernel(float * A, float * sum, int N)
{
    int t = threadIdx.x;

    __shared__ float input_A[BLOCK_SIZE];
    input_A[t] = A[t] + A[t+BLOCK_SIZE];

    for(int stride = blockDim.x/2; stride >= 1; stride /= 2){
        __syncthreads();
        if(t < stride)
            input_A[t]  += input_A[t + stride];
    }
    if(t == 0)
        *sum = input_A[0];
}

int main(void) {

    int N = 512;
    float *h_arr = (float*) malloc(N * sizeof(float));
    float *h_sum = (float*) malloc(1 * sizeof(float));

    for(int i = 0; i < N; i++) {
        h_arr[i] = 3.0f;
    }

    float *d_arr, *d_sum;
    cudaMalloc(&d_arr, N * sizeof(float));
    cudaMalloc(&d_sum, 1 * sizeof(float));

    cudaMemcpy(d_arr, h_arr, N * sizeof(float), cudaMemcpyHostToDevice);

    reductionKernel<<<32, 32>>>(d_arr, d_sum, N);

    cudaMemcpy(h_arr, d_arr, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
    cudaFree(d_sum);

    printf("%f\n", *h_sum);

    free(h_arr);
    free(h_sum);

    return 0;
}