#include <stdio.h>


__global__ void add(int *d, int n)
{
  __shared__ int s;
  int t = threadIdx.x;
  s = 0;
  __syncthreads();
  atomicAdd(&s, d[t]);
  __syncthreads();
  d[t] = s;
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
  const int n = 64;
  int tmp;
  int a[n], r[n], d[n];
  cudaError_t s;
  
  for (int i = 0; i < n; i++) {
    tmp = 1;
    a[i] = tmp;
    r[i] = tmp;
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
