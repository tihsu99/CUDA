// Includes
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <gsl/gsl_rng.h>
#include <math.h>

gsl_rng *rng=NULL;

// Variables
// Functions
void RandomUniform(float*, int);
void RandomMetro(float*, float*, int);

__global__ void sum_uniform(const float* data, float* m, float* s, int N_sample)
{
  extern __shared__ float cache[];

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int cacheIndex = threadIdx.x;

  float denominator = 1.;
  float temp_mean   = 0.;
  float temp_sigma  = 0.;
  while (i < N_sample){
    denominator = 1.;
    for(int j = 0; j < 10; j++){
      denominator += pow(data[i*10+j],2);
    }
    temp_mean  += 1./denominator;
    temp_sigma += pow(1./denominator,2);
    i += blockDim.x*gridDim.x;
  }
  cache[cacheIndex] = temp_mean;
  cache[cacheIndex + blockDim.x] = temp_sigma;

  __syncthreads();

  int ib = blockDim.x/2;
  while (ib != 0){
    if(cacheIndex < ib){
      cache[cacheIndex] += cache[cacheIndex + ib];
      cache[cacheIndex + blockDim.x] += cache[cacheIndex + ib + blockDim.x];
    }
    __syncthreads();
    ib /= 2;
  }
  if(cacheIndex == 0){
    m[blockIdx.x] = cache[0];
    s[blockIdx.x] = cache[blockDim.x];
  }
}
__global__ void sum_metro(const float* data, const float* w, float*m, float*s, int N_sample)
{
  extern __shared__ float cache[];

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int cacheIndex = threadIdx.x;

  float denominator = 1.;
  float temp_mean   = 0.;
  float temp_sigma  = 0.;
  while( i < N_sample){
    denominator = 1.;
    for(int j = 0; j < 10; j++){
      denominator += pow(data[i*10+j],2);
    }
    temp_mean += 1./denominator/w[i];
    temp_sigma += pow(1./denominator/w[i],2);
    i += blockDim.x * gridDim.x;
  }
  cache[cacheIndex] = temp_mean;
  cache[cacheIndex + blockDim.x] = temp_sigma;

  __syncthreads();

  int ib = blockDim.x/2;
  while(ib != 0){
    if(cacheIndex < ib){
      cache[cacheIndex] += cache[cacheIndex + ib];
      cache[cacheIndex + blockDim.x] += cache[cacheIndex + ib + blockDim.x];
    }
    __syncthreads();
    ib /= 2;
  }
  if(cacheIndex == 0){
    m[blockIdx.x] = cache[0];
    s[blockIdx.x] = cache[blockDim.x];
  }
}

int main(void)
{
  rng = gsl_rng_alloc(gsl_rng_mt19937);
  int threadsPerBlock, blocksPerGrid, sb, sm;
  int cpu_thread_id = 0;
  int Dev[2] = {0,1};
  long N_sample, size, size_sample;
  float mean, sigma, denominator, time;
  float* h_rng, *h_w, *h_m, *h_s;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for(int n = 6; n < 17; n++){
    N_sample = pow(2,n);
    printf("* Sample with N = %d samples.\n",N_sample);
    size = N_sample*10*sizeof(float);
    size_sample = N_sample*sizeof(float);
    h_rng = (float*)malloc(size);
    h_w   = (float*)malloc(size_sample);
    // Simple sampling - CPU
    cudaEventRecord(start,0);

    RandomUniform(h_rng, N_sample*10);
    mean = 0.;
    sigma = 0.;
    for(int i = 0; i < N_sample; i++){
      denominator = 1.;
      for(int j = 0; j < 10; j++){
        denominator += pow(h_rng[i*10+j],2);
      }
      mean += (1./denominator);
      sigma += pow(1./denominator,2);
    }
    mean /= N_sample;
    sigma = sqrt((sigma/N_sample-mean*mean)/N_sample);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime( &time, start, stop);
    printf("  - simple sampling(CPU): \t I = %5.10f +/- %5.10f  time: %f(ms)\n",mean,sigma, time);


    // Simple sampling - NGPU
    for(int ngpu = 1; ngpu < 3; ngpu++){
      omp_set_num_threads(ngpu);
      threadsPerBlock = 64/ ngpu;
      blocksPerGrid = N_sample / threadsPerBlock / ngpu; // 4 grid.
      sb = (blocksPerGrid * ngpu) * sizeof(float);
      sm =  threadsPerBlock*sizeof(float)*2;
      h_m = (float*)malloc(sb);
      h_s = (float*)malloc(sb);

      cudaEventRecord(start,0);
      RandomUniform(h_rng, N_sample*10);
      #pragma omp parallel private(cpu_thread_id)
      {
	float *d_rng, *d_m, *d_s;
        cpu_thread_id = omp_get_thread_num();
	cudaSetDevice(Dev[cpu_thread_id]);
	cudaMalloc((void**)&d_rng, size/ngpu);
	cudaMalloc((void**)&d_m,   sb/ngpu);
	cudaMalloc((void**)&d_s,   sb/ngpu);
 
	cudaMemcpy(d_rng, h_rng + N_sample*10/ngpu*cpu_thread_id, size/ngpu, cudaMemcpyHostToDevice);
        #pragma omp barrier
	dim3 block(blocksPerGrid);
	dim3 thread(threadsPerBlock);
        sum_uniform<<<block, thread, sm>>>(d_rng, d_m, d_s, N_sample/ngpu);

	cudaDeviceSynchronize();

	cudaMemcpy(h_m + sb/sizeof(float)/ngpu*cpu_thread_id, d_m, sb/ngpu, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_s + sb/sizeof(float)/ngpu*cpu_thread_id, d_s, sb/ngpu, cudaMemcpyDeviceToHost);

	cudaFree(d_m);
	cudaFree(d_s);
	cudaFree(d_rng);
//	cudaDeviceReset();
      }
      mean = 0.;
      sigma = 0.;
      for(int k = 0; k < sb/sizeof(float); k++)
      {  
        mean += h_m[k];
	sigma += h_s[k];
      }
      mean /= N_sample;
      sigma  = sqrt((sigma/N_sample - mean*mean)/N_sample);
      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&time, start, stop);
      
      printf("  - simple sampling (%dGPU):\t I = %5.10f +/- %5.10f  time: %f(ms)\n",ngpu, mean,sigma,time);
      free(h_m);
      free(h_s);
      
    }

    // Importance sampling - CPU
    cudaEventRecord(start,0);

    RandomMetro(h_rng, h_w, N_sample);
    mean = 0.;
    sigma = 0.;
    for(int i = 0; i < N_sample; i++){
      denominator = 1.;
      for(int j = 0; j < 10; j++){
	denominator += pow(h_rng[i*10+j],2);
      }
      mean += (1./denominator)/h_w[i];
      sigma += pow(1./denominator/h_w[i],2);
    }
    mean /= N_sample;
    sigma = sqrt((sigma/N_sample-mean*mean)/N_sample);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime( &time, start, stop);
    printf("  - importance sampling(CPU): \t I = %5.10f +/- %5.10f  time: %f(ms)\n",mean,sigma,time);
  // Importance sampling - GPU
    for(int ngpu = 1; ngpu < 3; ngpu++){
      omp_set_num_threads(ngpu);
      threadsPerBlock = 64/ngpu;
      blocksPerGrid   = N_sample / threadsPerBlock / ngpu; // Only one grid
      sb = (blocksPerGrid * ngpu) * sizeof(float);
      sm = threadsPerBlock*sizeof(float)*2;
      h_m = (float*)malloc(sb);
      h_s = (float*)malloc(sb);

      cudaEventRecord(start,0);
      RandomMetro(h_rng, h_w, N_sample);
      #pragma omp parallel private(cpu_thread_id)
      {
        float *d_rng, *d_m, *d_s, *d_w;
        cpu_thread_id = omp_get_thread_num();
        cudaSetDevice(Dev[cpu_thread_id]);
        cudaMalloc((void**)&d_rng, size/ngpu);
        cudaMalloc((void**)&d_m, sb/ngpu);
        cudaMalloc((void**)&d_s, sb/ngpu);
        cudaMalloc((void**)&d_w, size_sample/ngpu);

        cudaMemcpy(d_rng, h_rng + N_sample*10/ngpu*cpu_thread_id, size/ngpu, cudaMemcpyHostToDevice);
        cudaMemcpy(d_w, h_w + N_sample/ngpu*cpu_thread_id, size_sample/ngpu, cudaMemcpyHostToDevice);
        #pragma omp barrier
        dim3 block(blocksPerGrid);
        dim3 thread(threadsPerBlock);
        sum_metro<<<block, thread, sm>>>(d_rng, d_w, d_m,d_s,N_sample/ngpu);

        cudaDeviceSynchronize();

        cudaMemcpy(h_m + sb/sizeof(float)/ngpu*cpu_thread_id, d_m, sb/ngpu, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_s + sb/sizeof(float)/ngpu*cpu_thread_id, d_s, sb/ngpu, cudaMemcpyDeviceToHost);

        cudaFree(d_m);
        cudaFree(d_s);
        cudaFree(d_w);
        cudaFree(d_rng);
      }
      mean = 0.;
      sigma = 0.;
      for(int k = 0; k < sb/sizeof(float); k++){
        mean += h_m[k];
        sigma += h_s[k];
      }
      mean /= N_sample;
      sigma = sqrt((sigma/N_sample - mean*mean)/N_sample);

      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&time, start, stop);
    
      printf("  - importance sampling(%dGPU): \t I = %5.10f +/- %5.10f  time: %f(ms)\n",ngpu,mean,sigma,time);
      free(h_m);
      free(h_s);
    }
  }
}

void RandomUniform(float* data, int n)
{
  for(int i = 0; i < n; i++){
    data[i] = (float) gsl_rng_uniform(rng);
  }
}

void RandomMetro(float* data, float* w, int N_sample)
{
  // Initialize
  RandomUniform(data,N_sample*10);
  float w_old, w_new, r;

  w_old = 1.;
  for(int i=0; i < 10; i++){
    w_old *= exp(-1.*data[i])/(1.-exp(-1.));
  }
  w[0] = w_old;

  for(int i=1; i < N_sample; i++){
    w_new = 1.;
    for(int j=0; j < 10; j++){
      w_new *= exp(-1.*data[i*10+j])/(1.-exp(-1.));
    }
    if(w_old <= w_new){
      w_old = w_new; // Keep new as final result.
    }
    else
    {
      r = gsl_rng_uniform(rng);
      if(r < w_new/w_old){
        w_old = w_new; // Keep new as final result.
      }
      else{
        for(int j=0; j < 10; j++){
	  data[i*10+j] = data[(i-1)*10+j]; // Keep old as final result
	}
      }
    }
    w[i] = w_old;
  }
}



