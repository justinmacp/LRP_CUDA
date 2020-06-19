#include <stdio.h>


__global__ void v_m_mul(int *in, int *out, int *weights, int n, int m)
{
  __shared__ int s;
  int b = blockIdx.x;
  int t = threadIdx.x;
  int mul;
  __syncthreads();
  mul = in[t] * weights[b * n +];
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

  int *d_d;
  cudaMalloc(&d_d, n * sizeof(int)); 
  
  // run version with static shared memory
  cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
  add<<<1,n>>>(d_d, n);
  add_gm(r, n);
  s = cudaMemcpy(d, d_d, n*sizeof(int), cudaMemcpyDeviceToHost);
  printf("%s", cudaGetErrorName(s));
  for (int i = 0; i < n; i++) {
    if (d[i] != r[i]) {
      printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);
    }
    printf("%d\n", d[i]);
  }
}
