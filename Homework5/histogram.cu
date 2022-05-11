#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// Variables
float *h_data;
unsigned int *h_hist_global;
unsigned int *h_hist_share;
float *d_data_global;
float *d_data_share;
unsigned int *d_hist_global;
unsigned int *d_hist_share;
unsigned int *c_hist;

void RandomUniform(float*, long);
void RandomExponential(float*, long);

__global__ void hist_globalmem(float *data, const long N,
	                       unsigned int *hist, const float Rmin,
			       const int nbins, const float binsize)
{
  // use global memory 

  long i = threadIdx.x + blockIdx.x * blockDim.x;
  long strip = blockDim.x * gridDim.x;

  while(i < N){
    int index = (int)((data[i]-Rmin)/binsize);
    if(index < nbins) atomicAdd(&hist[index],1);
    i += strip;  // jump to next grid
  }
  __syncthreads();
}

__global__ void hist_sharemem(float *data, const long N,
		              unsigned int *hist, const int nbins,
			      const float Rmin,	const float binsize)
{
  // use share memory
  extern __shared__ unsigned int temp[]; // length = nbins
  temp[threadIdx.x] = 0;
  __syncthreads();

  long i = threadIdx.x + blockIdx.x * blockDim.x;
  long strip = blockDim.x * gridDim.x;

  while(i < N){
    int index = (int)((data[i]-Rmin)/binsize);
    if(index < nbins) atomicAdd(&temp[index],1);
    i += strip;
  }
  __syncthreads();
  if(threadIdx.x < nbins){
    atomicAdd(&(hist[threadIdx.x]), temp[threadIdx.x]);
  }
}

int main(void)
{
  int gpu_id = 0;
  cudaError_t err = cudaSuccess;

  err = cudaSetDevice(gpu_id);
  if (err != cudaSuccess){
    printf("Can not set device %d\n", gpu_id);
    exit(1);
  }
  cudaSetDevice(gpu_id);

  long N = 81920000;
  int nbins, index;
  float Rmin, Rmax, binsize;

  // Hist information
  Rmin  = 0;
  Rmax  = 16;
  nbins = 16;
  binsize = (Rmax-Rmin)/nbins;

  // vector size
  long size  = N * sizeof(float);
  int  bsize = nbins*sizeof(int);

  // Allocate vector
  h_data = (float*)malloc(size);
  h_hist_global = (unsigned int*)malloc(bsize);
  h_hist_share  = (unsigned int*)malloc(bsize);
  c_hist        = (unsigned int*)malloc(bsize);
  memset(c_hist       , 0, bsize);


  srand(time(NULL));

  RandomExponential(h_data, N);

  // create the timer
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float time;

  // iterate blocksize / gridsize

  int blocksize_list[5] = {16,32,64,128,256};
  int gridsize_list[5]  = {16,32,64,128,256};

  int best_blocksize_global = -1;
  int best_blocksize_share  = -1;
  int best_gridsize_global  = -1;
  int best_gridsize_share   = -1;
  float best_performance_global = 9999;
  float best_performance_share  = 9999;  

  for(int j=0; j < 5; j++){
    for(int k=0; k < 5; k++){

      // Initialize
      int gridsize  = gridsize_list[j];
      int blocksize = blocksize_list[k];

      printf("**(gridsize, blocksize):(%d,%d)\n",gridsize,blocksize);
      // Global memory
      float p_time;

      time = 0.;
      for(int t=0; t<3; t++){
	memset(h_hist_global, 0, bsize);
	cudaEventRecord(start,0);

        cudaMalloc((void**)&d_hist_global, bsize);
        cudaMalloc((void**)&d_data_global, size);

        cudaMemcpy(d_data_global, h_data, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_hist_global, h_hist_global, bsize, cudaMemcpyHostToDevice);

        hist_globalmem<<<gridsize, blocksize>>>(d_data_global,N,d_hist_global,
		                              Rmin, nbins, binsize);
        cudaMemcpy(h_hist_global, d_hist_global, bsize, cudaMemcpyDeviceToHost);

        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime( &p_time, start, stop);

        cudaFree(d_data_global);
	cudaFree(d_hist_global);
	time += p_time;
      }
      time /= 3.;

      printf("  - global memory(GPU): %f (ms)\n",time);
      // Save if it has the best performance
      if(time < best_performance_global){
        best_performance_global = time;
	best_blocksize_global   = blocksize;
	best_gridsize_global    = gridsize;
	FILE *out_global;
	out_global = fopen("hist_gpu_globalmem.dat","w");
	for(int i=0; i<nbins; i++){
	  float x = Rmin+(i+0.5)*binsize;
	  fprintf(out_global,"%f %d \n",x,h_hist_global[i]);
	}
	fclose(out_global);
      }

      // Share memory
      time = 0.;
      for(int t=0; t<3; t++){

	memset(h_hist_share, 0, bsize);
        cudaEventRecord(start,0);
      
        cudaMalloc((void**)&d_hist_share, bsize);
        cudaMalloc((void**)&d_data_share, size);

        cudaMemcpy(d_data_share, h_data, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_hist_share, h_hist_global, bsize, cudaMemcpyDeviceToHost);

        int sm = blocksize*sizeof(int);

        hist_sharemem<<< gridsize, blocksize, sm>>>(d_data_share, N, d_hist_share,
 		                                  nbins, Rmin, binsize);
        cudaMemcpy(h_hist_share, d_hist_share, bsize, cudaMemcpyDeviceToHost);
     
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&p_time, start, stop);

	cudaFree(d_data_share);
	cudaFree(d_hist_share);

	time += p_time;
      }

      time /= 3.;
      if(time < best_performance_share){
        best_performance_share = time;
	best_blocksize_share   = blocksize;
	best_gridsize_share    = gridsize;
	FILE *out_share;
	out_share = fopen("hist_gpu_sharemem.dat","w");
	for(int i=0; i<nbins; i++){
	  float x = Rmin+(i+0.5)*binsize;
	  fprintf(out_share, "%f %d \n",x,h_hist_share[i]);
	}
	fclose(out_share);
      }

      printf("  - share memory(GPU): %f (ms)\n",time);
      printf("\n");
    }
  }
  printf(" best setting for GPU global memory: (gridsize, blocksize, performance(ms)): (%d, %d, %f ms)\n", best_gridsize_global,best_blocksize_global, best_performance_global);
  printf(" best setting for GPU share memory: (gridsize, blocksize, performance(ms)): (%d, %d, %f ms)\n", best_gridsize_share, best_blocksize_share,  best_performance_share);

  // CPU solution
  cudaEventRecord(start,0);
  for(int i=0; i<N; i++){
    index = (int)((h_data[i]-Rmin)/binsize);
    if(index < nbins)  c_hist[index]++;
  }
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime( &time, start, stop);
  printf("Process time for CPU: %f (ms) \n",time);
  
  // Save CPU result
  FILE *cpu_out;
  cpu_out = fopen("hist_cpu.dat","w");
  for(int i=0; i<nbins; i++){
    float x = Rmin + (i+0.5)*binsize;
    fprintf(cpu_out,"%f %d \n",x,c_hist[i]);
  }

  // End process
  fclose(cpu_out);
  free(h_data);
  free(h_hist_global);
  free(h_hist_share);
  free(c_hist);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;

}

void RandomUniform(float* data, long n)
{
  for(long i = 0; i < n; i++){
    data[i] = rand()/(float)RAND_MAX;
  }
}

void RandomExponential(float* data, long n)
{
  for(long i=0; i < n; i++){
    double y = (double) rand() / (float)RAND_MAX;
    double x = -log(1.0-y);
    data[i]  = x;
  }
}
