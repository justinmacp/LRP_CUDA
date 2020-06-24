#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>


uint getTimeMicroseconds64() 
{
  uint nTime;
  struct timespec tSpec;
  clock_gettime(CLOCK_REALTIME, &tSpec);
  nTime = (uint)tSpec.tv_sec * 1000000 + (uint)tSpec.tv_nsec / 1000;
  return nTime;
}


__global__ void lrp_perc(int *in, int *out, int *relevance, int *weights, int *activations, int *activation_sum, int n, int m)
{
  int out_idx = threadIdx.y;
  int in_idx = threadIdx.x;
  activations[out_idx * n + in_idx] = in[in_idx] * weights[out_idx * n + in_idx];
  atomicAdd(&activation_sum[out_idx], activations[out_idx * n + in_idx]);
  __syncthreads();
  if (activation_sum[out_idx] < 0) { out[out_idx] = 0; } else { out[out_idx] = activation_sum[out_idx]; }
  atomicAdd(&relevance[in_idx], (activations[out_idx * n + in_idx] * out[out_idx]) / activation_sum[out_idx]);
}


void lrp_perc_gm(int *in, int *out, int *relevance, int *weights, int *activations, int *activation_sum, int n, int m)
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


__global__ void v_m_mul(int *in, int *out, int *weights, int n, int m)
{
  int b = blockIdx.x;
  int t = threadIdx.x;
  __syncthreads();
  int mul = in[t] * weights[b * n + t];
  atomicAdd(&out[b], mul);
}


void v_m_mul_gm(int *in, int *out, int *weights, int n, int m)
{
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      out[j] += in[i] * weights[j * n + i];
    }
  }
}


__global__ void add(int *in, int n)
{
  __shared__ int s;
  int t = threadIdx.x;
  s = 0;
  __syncthreads();
  atomicAdd(&s, in[t]);
  __syncthreads();
  in[t] = s;
}


void add_gm(int *in, int n)
{
  int tmp;
  tmp = 0;
  for (int i = 0; i < n; i++) {
   tmp += in[i];
  }
  for (int i = 0; i < n; i++) {
   in[i] = tmp;
  }
}


int main(void)
{
  const int n = 128, m = 8;
  uint dT1 = 0, dT2 = 0, hT1 = 0, hT2 = 0;
  int input[n], golden_out[m], cuda_out[m], weights[m * n], golden_relevance[n], cuda_relevance[n], golden_activations[m * n], cuda_activations[m * n], golden_asum[m], cuda_asum[m];
  cudaError_t s;
  
  // initialize variables on host
  for (int i = 0; i < n; i++) {
    input[i] = rand() % 10;
    golden_relevance[i] = 0;
    cuda_relevance[i] = 0;
    for (int j = 0; j < m; j++) {
      weights[j * n + i] = 1;
      golden_activations[j * n + i] = 0;
      cuda_activations[j * n + i] = 0;
    }
  }
  for (int i = 0; i < m; i++) {
    golden_out[i] = 0;
    cuda_out[i] = 0;
    golden_asum[i] = 0;
    cuda_asum[i] = 0;
  }

  // allocating memory for variables for device
  int *input_, *weights_, *output_, *relevance_, *activations_, *asum_;
  cudaMalloc(&input_, n * sizeof(int)); 
  cudaMalloc(&weights_, m * n * sizeof(int)); 
  cudaMalloc(&output_, m * sizeof(int));
  cudaMalloc(&relevance_, n * sizeof(int));
  cudaMalloc(&activations_, m * n * sizeof(int)); 
  cudaMalloc(&asum_, m * sizeof(int));
  
  // run version with static shared memory
  cudaMemcpy(input_, input, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(weights_, weights, n * m *sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(output_, 0, m * sizeof(int));
  cudaMemset(relevance_, 0, n * sizeof(int));
  cudaMemset(activations_, 0, m * n * sizeof(int));
  cudaMemset(asum_, 0, m * sizeof(int));

  // run cuda kernel and host function and compare the results
  hT1 = getTimeMicroseconds64();
  lrp_perc_gm(input, golden_out, golden_relevance, weights, golden_activations, golden_asum, n, m);
  hT2 = getTimeMicroseconds64();
  dT1 = getTimeMicroseconds64();
  lrp_perc<<<1,dim3(n,m)>>>(input_, output_, relevance_, weights_, activations_, asum_,  n, m);
  s = cudaDeviceSynchronize();
  dT2 = getTimeMicroseconds64();
  printf("%s\n", cudaGetErrorName(s));

  // relvance
  printf("### RELEVANCE ###\n");
  s = cudaMemcpy(cuda_relevance, relevance_, n * sizeof(int), cudaMemcpyDeviceToHost);
  printf("%s\n", cudaGetErrorName(s));
  for (int i = 0; i < n; i++) {
    if (golden_relevance[i] != cuda_relevance[i]) {
      printf("Error: golden_relevance[%d]!=cuda_relevance[%d] (%d, %d)\n", i, i, golden_relevance[i], cuda_relevance[i]);
    }
  }

  // out
  printf("### OUT ###\n");
  s = cudaMemcpy(cuda_out, output_, m * sizeof(int), cudaMemcpyDeviceToHost);
  printf("%s\n", cudaGetErrorName(s));
  for (int i = 0; i < m; i++) {
    if (golden_out[i] != cuda_out[i]) {
      printf("Error: golden_out[%d]!=cuda_out[%d] (%d, %d)\n", i, i, golden_out[i], cuda_out[i]);
    }
  }

  // activations
  printf("### ACTIVATIONS ###\n");
  s = cudaMemcpy(cuda_activations, activations_, m * n * sizeof(int), cudaMemcpyDeviceToHost);
  printf("%s\n", cudaGetErrorName(s));
  for (int i = 0; i < m * n; i++) {
    if (golden_activations[i] != cuda_activations[i]) {
      printf("Error: golden_activations[%d]!=cuda_activations[%d] (%d, %d)\n", i, i, golden_activations[i], cuda_activations[i]);
    }
  }

  // asum
  printf("### ASUM ###\n");
  s = cudaMemcpy(cuda_asum, asum_, m * sizeof(int), cudaMemcpyDeviceToHost);
  printf("%s\n", cudaGetErrorName(s));
  for (int i = 0; i < m; i++) {
    if (golden_asum[i] != cuda_asum[i]) {
      printf("Error: golden_asum[%d]!=cuda_asum[%d] (%d, %d)\n", i, i, golden_asum[i], cuda_asum[i]);
    }
  }
 
  printf("GPU time: %d, \tCPU time: %d\n", (dT2 - dT1) << 16, (hT2 - hT1) << 16);
}
