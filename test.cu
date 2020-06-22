#include <stdio.h>


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
  const int n = 64, m = 4;
  int tmp;
  int input[n], golden_answer[m], cuda_answer[m], weights[m*n];
  cudaError_t s;
  
  for (int i = 0; i < n; i++) {
    tmp = 1;
    input[i] = tmp;
    for (int j = 0; j < m; j++) {
      weights[j * n + i] = tmp;
    }
  }
  for (int i = 0; i < m; i++) {
    golden_answer[i] = 0;
    cuda_answer[i] = 0;
  }

  int *input_, *weights_, *output_;
  cudaMalloc(&input_, n * sizeof(int)); 
  cudaMalloc(&weights_, m * n * sizeof(int)); 
  cudaMalloc(&output_, m * sizeof(int)); 
  
  // run version with static shared memory
  cudaMemcpy(input_, input, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(weights_, weights, n * m *sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(output_, 0, m * sizeof(int));
  v_m_mul<<<m,n>>>(input_, output_, weights_, n, m);
  v_m_mul_gm(input, golden_answer, weights, n, m);
  s = cudaDeviceSynchronize();
  printf("%s\n", cudaGetErrorName(s));
  s = cudaMemcpy(cuda_answer, output_, m * sizeof(int), cudaMemcpyDeviceToHost);
  printf("%s\n", cudaGetErrorName(s));
  for (int i = 0; i < m; i++) {
    if (golden_answer[i] != cuda_answer[i]) {
      printf("Error: golden_answer[%d]!=cuda_answer[%d] (%d, %d)\n", i, i, golden_answer[i], cuda_answer[i]);
    }
    printf("%d\n", cuda_answer[i]);
  }
}
