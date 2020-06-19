#include <stdio.h>


__global__ void add(int *d, int n)
{
  __shared__ int s[64];
  int t = threadIdx.x;
  int tr = n-t-1;
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];
}


void add_gm(int *in, int n)
{
  int tmp[64];
  for (int i = 0; i < n; i++) {
   tmp[i] = in[i];
  }
  for (int i = 0; i < n; i++) {
   int tr = n - i - 1;
   in[i] = tmp[tr];
  }
}


int main(void)
{
  const int n = 64;
  int tmp;
  int a[n], r[n], d[n];
  cudaError_t s;
  
  for (int i = 0; i < n; i++) {
    tmp = i;
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
