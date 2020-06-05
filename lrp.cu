#include <iostream>
#include <math.h>

using namespace std;


// function to convolve the elements of two arrays
__global__ void conv_relu(float *out, float *in, float *kernel, int width, int height, int infeat, int outfeat) {
	int outf = blockIdx.x;
	int x = threadIdx.x;
	int y = threadIdx.y;
	int outIdx = x + 1 + (y + 1) * width + outf * width * height;
	for (int inf = 0; inf < infeat; inf++){	
		for (int u = 0; u < 3; u++){
			for (int v = 0; v < 3; v++) {
				out[outIdx] += in[(x + u) + (y + v) * width + inf * width * height] * kernel[u + v * 3 + inf * 9 + outf * 9 * infeat];
			}
		}
	}
	if (out[outIdx] < 0) {out[outIdx] = 0.0f; }	
}


__global__ void lrp(float *relevance, float *out, float *kernel, int width, int height, int infeat, int outfeat) {
	int inf = blockIdx.x;
	int x = threadIdx.x;
	int y = threadIdx.y;
	float activations[25 * infeat * outfeat];
	for (int inf = 0; inf < infeat; inf++) {
		for (int u = 0; u < 5; u++) {
			for (int v = 0; v < 5; v++) {
				activations[u + v * 5 + inf * 25 + outf * infeat * 25] = in[(x + u) + (y + v) * width + inf * width * height] * kernel[u + v * 3 + inf * 9 + outf * 9 * infeat];
			}	
		}		
	}
}


int main(void) {
	
	//Define arrays
	float *input, *output, *kernel, *activations;
	cudaError_t s;

	//Define sizes
	int imgHeight = 30;
	int imgWidth = 30;
	int kernelHeight = 3;
	int kernelWidth = 3;
	int inFeat = 2;
	int outFeat = 2;
	int nInput = imgHeight * imgWidth * inFeat;
	int nOutput = imgHeight * imgWidth * outFeat;
	int nWeights = kernelHeight * kernelWidth * inFeat * outFeat;
	int nActivations = (kernelHeight * 2 - 1) * (kernelWidth * 2 - 1) * inFeat * outFeat;
	float output_[nOutput];
	float input_[nInput];
	float kernel_[nWeights];
	float activations_[nActivations]
	
	//Initialize inputs
	for (int i = 0; i < nInput; i++) {
		input_[i] = 1.0f;
	}
	for (int i = 0; i < nWeights; i++) {
		kernel_[i] = 1.0f;
	}
	
	//Perform memory operations
	cudaMalloc((void **) &input, nInput * sizeof(float));
	cudaMalloc((void **) &output, nOutput * sizeof(float));
	cudaMalloc((void **) &kernel, nWeights * sizeof(float));
	cudaMalloc((void **) &activations,l nActivations * sizeof(float));
	cudaMemset((void **) output, 0, nOutput * sizeof(float));
	cudaMemset((void **) activations, 0, nActivations * sizeof(float));
	cudaMemcpy(input, input_, nInput * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(kernel, kernel_, nWeights * sizeof(float), cudaMemcpyHostToDevice);

	// Run kernel on 1M elements on the CPU
	conv_relu <<<outFeat, dim3(imgWidth - 2, imgHeight - 2)>>> (output, input, kernel, imgWidth, imgHeight, inFeat, outFeat);

	cudaDeviceSynchronize();

	s = cudaMemcpy(output_, output, nOutput * sizeof(float), cudaMemcpyDeviceToHost);
	cout << cudaGetErrorName(s) << endl;

	// Check for errors (all values should be 3.0f)
	float maxError = 0.0f;
	for (int i = 1; i < imgWidth - 1; i++) {
		for (int j = 1; j < imgHeight - 1; j++) {
			for (int o = 0; o < outFeat; o++) {
				maxError = fmax(maxError, fabs(output_[i + j * imgWidth + o * imgHeight * imgWidth] - 18.0f));
			}
		}
	}
	cout << "Max error: " << maxError << endl;

	for (int k = 0; k < outFeat; k ++) {
		for (int i = 0; i < imgWidth; i++) {
			for (int j = 0; j < imgHeight; j++) {
				cout << output_[i + j * imgWidth + k * imgWidth * imgHeight] << " ";
			}
			cout << endl;
		}
		cout << endl;
	}

	// Free memory
	cudaFree(input);
	cudaFree(output);
	cudaFree(kernel);
	return 0;
}
