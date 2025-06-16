#include <stdio.h>
#include <stdlib.h>

// code=sum && nvcc -arch=sm_35 -o $code.o $code.cu && nvprof ./$code.o

#define PRINTIT true
#define THREADS_PER_BLOCK 256

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ void warpReduce(volatile int* arr, int t) {
	arr[t] += arr[t + 32];
	arr[t] += arr[t + 16];
	arr[t] += arr[t + 8];
	arr[t] += arr[t + 4];
	arr[t] += arr[t + 2];
	arr[t] += arr[t + 1];
}

__global__ void d_sum(int* d_a, int* d_b, int size) {
	extern __shared__ int s_sums[];

	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x; //
	s_sums[threadIdx.x] = d_a[i] + d_a[i + blockDim.x];
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
		if (threadIdx.x < stride)
			s_sums[threadIdx.x] += s_sums[threadIdx.x + stride];
		if (threadIdx.x == 0 && (stride / 2) * 2 != stride) // on vérifie si le stride est bien impaire
			s_sums[0] += s_sums[stride - 1];
		__syncthreads();
	}

	if (threadIdx.x < 32) {
		warpReduce(s_sums, threadIdx.x);
	}

	if (threadIdx.x == 0) {
		atomicAdd(d_b, s_sums[0]);
	}
}

void h_sum(int* a, int* b, int size) {
    *b = 0;
    for (int i = 0; i < size; i++) {
        *b += a[i];
    }
}

void sum(int* a, int* b, int size) {
    int* deviceCount = (int*) malloc(sizeof(int));
    cudaGetDeviceCount(deviceCount);
	int sizePerBlock = 2 * THREADS_PER_BLOCK;
    if (*deviceCount == 0 || size < sizePerBlock) {
        h_sum(a, b, size);
    } else {
        int threadsPerBlock = THREADS_PER_BLOCK;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
		int effectiveBlocksPerGrid = blocksPerGrid / 2;
		int segments = size / sizePerBlock;
		int d_size = segments * sizePerBlock;
		int h_size = size - d_size;
        printf(
            "threadsPerBlock=%d, blocksPerGrid=%d, effectiveBlocksPerGrid=%d, sizePerBlock=%d, segments=%d, d_size=%d, h_size=%d\n",
            threadsPerBlock, blocksPerGrid, effectiveBlocksPerGrid, sizePerBlock, segments, d_size, h_size
        );

        int* d_a;
        int* d_b;

        gpuErrchk(cudaMalloc((void**) &d_a, d_size * sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_b, d_size * sizeof(int)));

        gpuErrchk(cudaMemcpy(d_a, a, d_size * sizeof(int), cudaMemcpyHostToDevice));

        d_sum<<<effectiveBlocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_a, d_b, d_size);
        gpuErrchk(cudaPeekAtLastError());

        gpuErrchk(cudaMemcpy(b, d_b, d_size * sizeof(int), cudaMemcpyDeviceToHost));

        gpuErrchk(cudaFree(d_a));
        gpuErrchk(cudaFree(d_b));

		gpuErrchk(cudaDeviceSynchronize());

		int* h_b = (int*) malloc(sizeof(int));
		h_sum(&a[d_size], h_b, h_size);
		*b += *h_b;
		free(h_b);
    }
    printf("deviceCount = %d\n", *deviceCount);
    free(deviceCount);
}

void initialize(int* v, int size) {
    for (int i = 0; i < size; i++) {
        v[i] = 1;
    }
}

void print(int* v, int size) {
    printf("[");
    for (int i = 0; i < size - 1; i++) {
        printf("%d, ", v[i]);
    }
    if (size != 0) printf("%d", v[size - 1]);
    printf("]\n");
}

int main(int argc, char** argv) {
    int SIZE = 1<<21;
    if (argc > 1) SIZE = (int) atoi(argv[1]);
    printf("SIZE=%d\n", SIZE);

    int* a = (int*) malloc(SIZE * sizeof(int));
    int* b = (int*) malloc(SIZE * sizeof(int));

    initialize(a, SIZE);
    if (!PRINTIT) {
        printf("a = ");
        print(a, SIZE);
    }

    sum(a, b, SIZE);

    if (PRINTIT) {
        printf("b = ");
        print(b, 1);
    }

    free(a);
    free(b);

    return 0;
}