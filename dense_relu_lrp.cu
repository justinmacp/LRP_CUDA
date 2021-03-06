#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>


#define IN_FEATS 200
#define OUT_CLASSES 10
#define W_SIZE IN_FEATS * OUT_CLASSES


uint64_t getTimeMicroseconds64()
{
  uint64_t nTime;
  struct timespec tSpec;
  clock_gettime(CLOCK_REALTIME, &tSpec);
  nTime = (uint64_t)tSpec.tv_nsec / 1000;
  return nTime;
}


__global__ void fwd_perc(float *in, float *out, float *weights, float *activations, float *activation_sum)
{
  int b = blockIdx.x;
  int t = threadIdx.x;
  __shared__ float z[IN_FEATS], sum_z;
  sum_z = 0;
  __syncthreads();
  z[t] = in[t] * weights[b * IN_FEATS + t];
  atomicAdd(&sum_z, z[t]);
  __syncthreads();
  activation_sum[b] = sum_z;
  activations[b * IN_FEATS + t ] = z[t];
  if (sum_z < 0) { out[b] = 0; } else { out[b] = sum_z; }
}


__global__ void lrp_perc(float *out, float *relevance, float *activations, float *activation_sum)
{
  int b = blockIdx.x;
  int t = threadIdx.x;
  __shared__ float z[OUT_CLASSES], rel, sum_z[OUT_CLASSES], r_m[OUT_CLASSES];
  z[t] = activations[t * IN_FEATS + b];
  rel = 0;
  sum_z[t] = activation_sum[t];
  r_m[t] = out[t];
  __syncthreads();
  atomicAdd(&rel, z[t] * r_m[t] / sum_z[t]);
  __syncthreads();
  relevance[b] = rel;
}


void lrp_perc_gm(float *in, float *out, float *relevance, float *weights, float *activations, float *activation_sum, int n, int m)
{
  for (int j = 0; j < m; j++) {
    for (int i_prime = 0; i_prime < n; i_prime++) {
      activations[j * n + i_prime] = in[i_prime] * weights[j * n + i_prime];
      activation_sum[j] += activations[j * n + i_prime];
    }
    if (activation_sum[j] < 0) { out[j] = 0; } else { out[j] = activation_sum[j]; }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      relevance[i] += (activations[j * n + i] * out[j]) / activation_sum[j];
    }
  }
}


int main(void)
{
  uint64_t dT1 = 0, dT2 = 0, hT1 = 0, hT2 = 0;
  float input[IN_FEATS], golden_out[OUT_CLASSES], cuda_out[OUT_CLASSES], weights[W_SIZE], golden_relevance[IN_FEATS], cuda_relevance[IN_FEATS], golden_activations[W_SIZE], cuda_activations[W_SIZE], golden_asum[OUT_CLASSES], cuda_asum[OUT_CLASSES];
  cudaError_t s;
  
  // initialize variables on host
  for (int i = 0; i < IN_FEATS; i++) {
    input[i] = rand() % 10;
    golden_relevance[i] = 0;
    cuda_relevance[i] = 0;
    for (int j = 0; j < OUT_CLASSES; j++) {
      weights[j * IN_FEATS + i] = rand() % 10;
      golden_activations[j * IN_FEATS + i] = 0;
      cuda_activations[j * IN_FEATS + i] = 0;
    }
  }
  for (int i = 0; i < OUT_CLASSES; i++) {
    golden_out[i] = 0;
    cuda_out[i] = 0;
    golden_asum[i] = 0;
    cuda_asum[i] = 0;
  }

  // allocating memory for variables for device
  float *input_, *weights_, *output_, *relevance_, *activations_, *asum_;
  cudaMalloc(&input_, IN_FEATS * sizeof(float)); 
  cudaMalloc(&weights_, W_SIZE * sizeof(float)); 
  cudaMalloc(&output_, OUT_CLASSES * sizeof(float));
  cudaMalloc(&relevance_, IN_FEATS * sizeof(float));
  cudaMalloc(&activations_, W_SIZE * sizeof(float)); 
  cudaMalloc(&asum_, OUT_CLASSES * sizeof(float));
  
  // run version with static shared memory
  cudaMemcpy(input_, input, IN_FEATS * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(weights_, weights, W_SIZE *sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(output_, 0, OUT_CLASSES * sizeof(float));
  cudaMemset(relevance_, 0, IN_FEATS * sizeof(float));
  cudaMemset(activations_, 0, W_SIZE * sizeof(float));
  cudaMemset(asum_, 0, OUT_CLASSES * sizeof(float));

  // run cuda kernel and host function and compare the results
  hT1 = getTimeMicroseconds64();
  lrp_perc_gm(input, golden_out, golden_relevance, weights, golden_activations, golden_asum, IN_FEATS, OUT_CLASSES);
  hT2 = getTimeMicroseconds64();
  dT1 = getTimeMicroseconds64();
  fwd_perc<<<OUT_CLASSES, IN_FEATS>>>(input_, output_, weights_, activations_, asum_);
  lrp_perc<<<IN_FEATS, OUT_CLASSES>>>(output_, relevance_, activations_, asum_);
  s = cudaDeviceSynchronize();
  dT2 = getTimeMicroseconds64();
  printf("%s\n", cudaGetErrorName(s));

  // relvance
  printf("### RELEVANCE ###\n");
  s = cudaMemcpyAsync(cuda_relevance, relevance_, IN_FEATS * sizeof(float), cudaMemcpyDeviceToHost);
  printf("%s\n", cudaGetErrorName(s));
  for (int i = 0; i < IN_FEATS; i++) {
    if (golden_relevance[i] != cuda_relevance[i]) {
      printf("Error: golden_relevance[%d]!=cuda_relevance[%d] (%f, %f)\n", i, i, golden_relevance[i], cuda_relevance[i]);
    }
  }

  // out
  printf("### OUT ###\n");
  s = cudaMemcpy(cuda_out, output_, OUT_CLASSES * sizeof(float), cudaMemcpyDeviceToHost);
  printf("%s\n", cudaGetErrorName(s));
  for (int i = 0; i < OUT_CLASSES; i++) {
    if (golden_out[i] != cuda_out[i]) {
      printf("Error: golden_out[%d]!=cuda_out[%d] (%f, %f)\n", i, i, golden_out[i], cuda_out[i]);
    }
  }

  // activations
  printf("### ACTIVATIONS ###\n");
  s = cudaMemcpy(cuda_activations, activations_, W_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
  printf("%s\n", cudaGetErrorName(s));
  for (int i = 0; i < W_SIZE; i++) {
    if (golden_activations[i] != cuda_activations[i]) {
      printf("Error: golden_activations[%d]!=cuda_activations[%d] (%f, %f)\n", i, i, golden_activations[i], cuda_activations[i]);
    }
  }

  // asum
  printf("### ASUM ###\n");
  s = cudaMemcpy(cuda_asum, asum_, OUT_CLASSES * sizeof(float), cudaMemcpyDeviceToHost);
  printf("%s\n", cudaGetErrorName(s));
  for (int i = 0; i < OUT_CLASSES; i++) {
    if (golden_asum[i] != cuda_asum[i]) {
      printf("Error: golden_asum[%d]!=cuda_asum[%d] (%f, %f)\n", i, i, golden_asum[i], cuda_asum[i]);
    }
  }
 
  printf("GPU time: %lu, \tCPU time: %lu\n", (dT2 - dT1) << 16, (hT2 - hT1) << 16);

  cudaFree(input_); 
  cudaFree(weights_); 
  cudaFree(output_);
  cudaFree(relevance_);
  cudaFree(activations_); 
  cudaFree(asum_);
}
