#include <iostream>
#include <math.h>

using namespace std;


// function to convolve the elements of two arrays
__global__ void conv(float *out, float *in, float *kernel, int width, int height, int infeat, int outfeat) {
	int outf = blockIdx.x;
	int x = threadIdx.x;
	int y = threadIdx.y;
	int outIdx = x + 1 + (y + 1) * width + outf * width * height;
	for (int inf = 0; inf < infeat; inf++) {
		for (int u = 0; u < 3; u++){
			for (int v = 0; v < 3; v++) {
				int inIdx = (x + u) + (y + v) * width + inf * width * height;
				out[outIdx] += in[inIdx] * kernel[u + v * 3 + inf * 9 + outf * 9 * infeat];
			}
		}
	}		
}


int main(void) {
	int N = 4;
	int M = 4;
	int infeat = 2;
	int outfeat = 2;
	int kernelh = 3;
	int kernelw = 3;

	float *x, *y, *kernel;
	
	// initialize x and y arrays on the host
	// Allocate Unified Memory – accessible from CPU or GPU
	cudaMallocManaged(&x, N * M * infeat * sizeof(float));
	cudaMallocManaged(&y, N * M * outfeat * sizeof(float));
	cudaMallocManaged(&kernel, kernelh * kernelw * sizeof(float));

	for (int i = 0; i < infeat * N * M; i++) {
		x[i] = 1.0f;
	}

	for (int i = 0; i < outfeat * N * M; i++) {
		y[i] = 0.0f;
	}

	for (int i = 0; i < infeat * outfeat * kernelw * kernelh; i++) {
		kernel[i] = 1.0f;
	}


	// Run kernel on 1M elements on the CPU
	conv <<<outfeat, dim3(N - 2, M - 2)>>> (y, x, kernel, N, M, infeat, outfeat);

	cudaDeviceSynchronize();

	// Check for errors (all values should be 3.0f)
	float maxError = 0.0f;
	for (int i = 1; i < N - 1; i++) {
		for (int j = 1; j < M - 1; j++) {
			for (int o = 0; o < outfeat; o++) {
				maxError = fmax(maxError, fabs(y[i + j * N + o * N * M] - 18.0f));
			}
		}
	}
	cout << "Max error: " << maxError << endl;

	for (int k = 0; k < outfeat; k ++) {
		for (int j = 0; j < M; j++) {
			for (int i = 0; i < N; i++) {
				cout << y[i + j * N + k * N * M] << " ";
			}
			cout << endl;
		}
		cout << endl;
	}

	// Free memory
	cudaFree(x);
	cudaFree(y);
	cudaFree(kernel);

	return 0;
}
