#include <iostream>
#include <math.h>

// function to convolve the elements of two arrays
__global__ void conv(float *out, float *in, float *kernel, int width, int height)
{
	for (int m = 1; m < width - 1; m++) {
		for (int n = 1; n < height - 1; n++){
			for (int u = 0; u < 3; u++){
				for (int v = 0; v < 3; v++) {
					out[(m) + (n) * width] += in[(m - 1 + u) + (n - 1 + v) * width] * kernel[u + v * 3];
				}
			}
		}
	}					
}

int main(void) {
	int N = 5;
	int M = 5;

	float *x;
	float *y;
	float *kernel;

	// initialize x and y arrays on the host
	// Allocate Unified Memory – accessible from CPU or GPU
	cudaMallocManaged(&x, N*M*sizeof(float));
	cudaMallocManaged(&y, N*M*sizeof(float));
	cudaMallocManaged(&kernel, 9*sizeof(float));

	for (int i = 0; i < N * M; i++) {
		x[i] = 1.0f;
		y[i] = 0.0f;
	}

	for (int i = 0; i < 9; i++) {
		kernel[i] = 1.0f;
	}

	// Run kernel on 1M elements on the CPU
	conv<<<1, 1>>>(y, x, kernel, N, M);

	cudaDeviceSynchronize();

	// Check for errors (all values should be 3.0f)
	float maxError = 0.0f;
	for (int i = 1; i < N - 1; i++) {
		for (int j = 1; j < M - 1; j++) {
			maxError = fmax(maxError, fabs(y[i + j * N]-9.0f));
		}
	}
	std::cout << "Max error: " << maxError << std::endl;
	for (int i = 0; i < N * M; i++) {
		std::cout << y[i] << std::endl;
	}

	// Free memory
	cudaFree(x);
	cudaFree(y);
	cudaFree(kernel);

	return 0;
}
