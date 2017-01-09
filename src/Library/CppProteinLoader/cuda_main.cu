#include <iostream>
#include <THC/THC.h>

__global__ void add(float *a, float *b, float *c, int n){
	//c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
	//c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if (idx<n)
		c[idx] = a[idx] + b[idx];
}
#define N (16*16)
#define M 128
int main(void){
	float *a, *b, *c;
	float *d_a, *d_b, *d_c;
	int size = N*sizeof(float);

	a = (float*)malloc(size);
	b = (float*)malloc(size);
	c = (float*)malloc(size);
	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);
	
	for(int i=0;i<N;i++){
		a[i]=1.2;
		b[i]=2.6;
	}

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	add<<<(N + M -1)/M,M>>>(d_a, d_b, d_c, N);

	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	free(a);
	free(b);
	free(c);

	for(int i=0;i<N;i++){
		std::cout<<c[i]<<", ";
	}
	return 0;
}