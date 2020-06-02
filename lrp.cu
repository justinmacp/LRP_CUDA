#include <stdlib.h>

#include "cudnn.h"

__global__ void forward_pass(float *pInputs, float *pOutputs) {
	int Inx = blockIdx.x, Iny = blockIdx.y, 
	}

int lrp() {
	kernel = get_parameter(weight_NCHW_Name128, 9*128*128);
	bnBias = get_parameter(bnBiasName128, 128);
	bnScale = get_parameter(bnScaleName128, 128);
	float* eMean = get_parameter(eMeanName128, 128);
	float* eVar = get_parameter(eVarName128, 128);
	float *l_eMean, *l_eVar;
	nInput = 16*16*128, nOutput = 14*14*128, nWeights = 3*3*128*128, nBias = 128;

	cudaMalloc((void **) &output, nOutput<<2);
	cudaMalloc((void **) &l_weights, nWeights<<2);
	cudaMalloc((void **) &l_bias, nBias<<2);
	cudaMemcpy(l_weights, kernel, nWeights<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(l_bias, bias, nBias<<2, cudaMemcpyHostToDevice);

	cudaMalloc((void **) &l_eMean, nBias<<2);
	cudaMalloc((void **) &l_eVar, nBias<<2);
	cudaMemcpy(l_bnBias, bnBias, nBias<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(l_bnScale, bnScale, nBias<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(l_eMean, eMean, nBias<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(l_eVar, eVar, nBias<<2, cudaMemcpyHostToDevice);

	cudaMemset((void *) output, 0, nOutput<<2);

	float tmp_cudnn[nOutput];


	/*  2. cuDNN preparation  */
	cudnnStatus_t status;
	float one = 1.0, zero = 0.0;
	int size;

	cudnnHandle_t handle;
	status = cudnnCreate(&handle);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed1\n");

	cudnnTensorDescriptor_t xdesc, ydesc, bdesc;
	cudnnFilterDescriptor_t wdesc; // CUDNN_TENSOR_NHWC, CUDNN_TENSOR_NCHW
	status = cudnnCreateTensorDescriptor(&xdesc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed2\n");
	status = cudnnSetTensor4dDescriptor(xdesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 128, 16, 16);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed3\n");
	status = cudnnCreateTensorDescriptor(&ydesc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed4\n");
	status = cudnnSetTensor4dDescriptor(ydesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 128, 14, 14);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed5\n");
	status = cudnnCreateFilterDescriptor(&wdesc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed6\n");
	status = cudnnSetFilter4dDescriptor(wdesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 128, 128, 3, 3);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed7\n");
	status = cudnnCreateTensorDescriptor(&bdesc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed8\n");
	status = cudnnSetTensor4dDescriptor(bdesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 128, 1, 1);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed9\n");
	cudnnConvolutionDescriptor_t conv_desc;
	status = cudnnCreateConvolutionDescriptor(&conv_desc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed10\n");
	status = cudnnSetConvolution2dDescriptor(conv_desc, 0,0, 1,1,1,1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT); //CUDNN_CONVOLUTION
	if (status != CUDNN_STATUS_SUCCESS) printf("failed11\n");

	cudnnActivationDescriptor_t act_desc;
	status = cudnnCreateActivationDescriptor(&act_desc);  
	if (status != CUDNN_STATUS_SUCCESS) printf("failed12\n");
	status = cudnnSetActivationDescriptor(act_desc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed13\n");

	cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc;
	status = cudnnCreateTensorDescriptor(&bnScaleBiasMeanVarDesc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed14\n");
	status = cudnnSetTensor4dDescriptor(bnScaleBiasMeanVarDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 128, 1, 1);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed15\n");

	cudnnConvolutionFwdAlgo_t algo = (cudnnConvolutionFwdAlgo_t)6;

	status = cudnnGetConvolutionForwardWorkspaceSize(handle,
	   xdesc,
	   wdesc,
	   conv_desc,
	   ydesc,
	   algo,
	   (size_t *)&(size));

	float *extra;
	cudaMalloc((void **) &extra, size);


	/*  3. Computing  */
	nT1_cudnn = getTimeMicroseconds64();

	status = cudnnConvolutionForward(handle, &one,
		xdesc, input, wdesc, l_weights, 
		conv_desc, algo, 
		extra, size, &zero,
		ydesc, output);
	if (status != CUDNN_STATUS_SUCCESS) printf("No Success1\n");

	status = cudnnActivationForward(handle, act_desc, &one,
		ydesc, output, &zero,
		ydesc, output);
	if (status != CUDNN_STATUS_SUCCESS) printf("No Success3\n");

	cudaDeviceSynchronize();
	nT2_cudnn = getTimeMicroseconds64();
	printf("cuDNN TotalTime = %d us\n", nT2_cudnn-nT1_cudnn);


	/*  4. Copy back and free  */
	s = cudaMemcpy(tmp_cudnn, output, nOutput<<2, cudaMemcpyDeviceToHost);
	printf("%s\n", cudaGetErrorName(s));

	cudaFree(extra);
	cudaFree(input);
	cudaFree(output);
	cudaFree(l_weights);
	cudaFree(l_bias);

	free(bias);
	free(kernel);
	free(input_);

	return (nT2_cudnn-nT1_cudnn);
}
