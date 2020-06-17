#include <iostream>
#include <math.h>

using namespace std;

#define CLASSES 2


// function to convolve the elements of two arrays
__global__ void dense_relu(float *out, float *in, float *weights, int infeats) {
	int batch = blockIdx.x;
	int outfeat = threadIdx.y;
	int infeat = threadIdx.x;
	int outIndivId = outfeat;
	int outId = batch * CLASSES + outfeat;
	int wId = batch * CLASSES * infeats + outfeat * infeats + infeat;
	int inId = batch * infeats + infeat;
	__shared__ float outTemp[CLASSES];
	__syncthreads();
	outTemp[outIndivId] += in[inId] * weights[wId];
	__syncthreads();
	out[outId] = outTemp[outIndivId];
	__syncthreads();
	printf("%d %d %d %f %f\n", batch, outfeat, infeat, in[inId] * weights[wId], outTemp[outId]);
	if (infeat == 0 && out[outId] < 0.0f) {
		out[outId] = 0.0f;
	}
}

int main(void) {
	
	//Define arrays
	float *input, *output, *weights;
	cudaError_t s;

	//Define sizes
	int inFeatures = 10;
	int batchSize = 2;
	int nInput = inFeatures * batchSize;
	int nOutput = CLASSES * batchSize;
	int nWeights = inFeatures * CLASSES * batchSize;
	float output_[nOutput];
	float input_[nInput];
	float weights_[nWeights];
	
	//Initialize inputs
	for (int i = 0; i < nInput; i++) {
		input_[i] = 1.0f;
	}
	for (int i = 0; i < nWeights; i++) {
		weights_[i] = 1.0f;
	}
	
	//Perform memory operations
	cudaMalloc((void **) &input, nInput * sizeof(float));
	cudaMalloc((void **) &output, nOutput * sizeof(float));
	cudaMalloc((void **) &weights, nWeights * sizeof(float));
	cudaMemset((void **) output, 0, nOutput * sizeof(float));
	s = cudaMemcpy(input, input_, nInput * sizeof(float), cudaMemcpyHostToDevice);
	cout << cudaGetErrorName(s) << endl;
	s = cudaMemcpy(weights, weights_, nWeights * sizeof(float), cudaMemcpyHostToDevice);
	cout << cudaGetErrorName(s) << endl;

	// Run kernel on 1M elements on the CPU
	dense_relu <<<batchSize, dim3(inFeatures, CLASSES), (nInput + nOutput + nWeights) * sizeof(float)>>> (output, input, weights, inFeatures);

	s = cudaDeviceSynchronize();
	cout << cudaGetErrorName(s) << endl;

	s = cudaMemcpy(output_, output, nOutput * sizeof(float), cudaMemcpyDeviceToHost);
	cout << cudaGetErrorName(s) << endl;

	// Check for errors (all values should be 3.0f)
	float maxError = 0.0f;
	float expectedAnswer = inFeatures;
	for (int i = 0; i < nOutput; i++) {
		maxError = fmax(maxError, fabs(output_[i] - expectedAnswer));
	}
	cout << "Max error: " << maxError << endl;

	for (int i = 0; i < CLASSES; i++) {
		for (int j = 0; j < batchSize; j++) {
			cout << output_[j * CLASSES + i] << "\t";
		}
		cout << endl;
	}

	// Free memory
	cudaFree(input);
	cudaFree(output);
	cudaFree(weights);
	return 0;
}
