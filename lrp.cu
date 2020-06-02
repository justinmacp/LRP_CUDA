#include <iostream>
#include <math.h>

// function to convolve the elements of two arrays
__global__ void conv(float *out, float *in, float *kernel, int width, int height, int infeat, int outfeat) {
	int x = threadIdx.x;
	int y = threadIdx.y;
	int inf = threadIdx.z;
	int outf = blockIdx.x;
	for (int u = 0; u < 3; u++){
		for (int v = 0; v < 3; v++) {
			out[x + y * width + outf * width * height] += in[(x - 1 + u) + (y - 1 + v) * width + inf * width * height] * kernel[u + v * 3 + inf * 9 + outf * 9 * infeat];
		}
	}					
}

int main(void) {
	int N = 5;
	int M = 5;
	int infeat = 2;
	int outfeat = 2;
	int kernelh = 3;
	int kernelw = 3;

	float *x;
	float *y;
	float *kernel;

	// initialize x and y arrays on the host
	// Allocate Unified Memory – accessible from CPU or GPU
	cudaMallocManaged(&x, N*M*infeat*sizeof(float));
	cudaMallocManaged(&y, N*M*infeat*sizeof(float));
	cudaMallocManaged(&kernel, 9*sizeof(float));

	for (int i = 0; i < infeat * N * M; i++) {
		x[i] = 1.0f;
	}

	for (int i = 0; i < outfeat * N * M; i++) {
		x[i] = 1.0f;
	}

	for (int i = 0; i < infeat * outfeat * kernelw * kernelh; i++) {
		kernel[i] = 1.0f;
	}

	// Run kernel on 1M elements on the CPU
	conv<<<outfeat, dim3(N, M, infeat)>>>(y, x, kernel, N, M, infeat, outfeat);

	cudaDeviceSynchronize();

	// Check for errors (all values should be 3.0f)
	float maxError = 0.0f;
	for (int i = 1; i < N - 1; i++) {
		for (int j = 1; j < M - 1; j++) {
			for (int o = 0; o < outfeat; o++) {
				maxError = fmax(maxError, fabs(y[i + j * N + o * N * M]-18.0f));
			}
		}
	}
	std::cout << "Max error: " << maxError << std::endl;

	for (int i = 0; i < N * M * outfeat; i++) {
		std::cout << y[i] << std::endl;
	}


	// Free memory
	cudaFree(x);
	cudaFree(y);
	cudaFree(kernel);

	return 0;
}
