// Includes
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>

// Variables ( Host vectors )
float* h_A;
float* h_B;
float* h_C;

void RandomInit(float*, int);

//Device code ( GPU )
__global__ void vecDot_ngpu(const float* A, const float* B, float* C, int N)
{
  extern __shared__ float cache[];
  
  int i = blockDim.x * blockIdx.x + threadIdx.x; // 1D array
  int cacheIndex = threadIdx.x;

  float temp = 0.0;
  while( i < N ) {
    temp += A[i]*B[i];
    i += blockDim.x*gridDim.x;
  }
  cache[cacheIndex] = temp;
 
  __syncthreads();

  int ib = blockDim.x/2;
  while (ib != 0){
    if(cacheIndex < ib) cache[cacheIndex] += cache[cacheIndex + ib];  //parallel reduction

    __syncthreads();

    ib /= 2;
  }
  if(cacheIndex == 0)  C[blockIdx.x] = cache[0];
  

}

int main(void)
{
  int  N_vec  = 40960000;
  int  N_GPU  = 2;
  int  cpu_thread_id = 0;
  int  Dev[2] = {0,1};
  long size   = N_vec*sizeof(float);

  int List_threadsPerBlock[6] = {4,8,16,32,64,128};
  int List_blocksPerGrid[6]   = {64,128,256,512,1024,2048};

  // Initialize host vector 
  h_A = (float*)malloc(size);
  h_B = (float*)malloc(size);
  RandomInit(h_A, N_vec);
  RandomInit(h_B, N_vec);

  cudaEvent_t start, stop;
  float Intime, GPUtime, Outtime, CPUtime, Totaltime;

  omp_set_num_threads(N_GPU);

  int best_blocksPerGrid = -1;
  int best_threadsPerBlock = -1;
  float performance = 999999.;

  printf("--------- (gridsize, blocksize) ----------\n");
  for(int i = 0; i < 6; i++){
    for(int j = 0; j < 6; j++){
    
      int threadsPerBlock = List_threadsPerBlock[i];
      int blocksPerGrid   = List_blocksPerGrid[j];
      int sb              = (blocksPerGrid * N_GPU) * sizeof(float);
      int sm              = threadsPerBlock*sizeof(float);
      h_C = (float*)malloc(sb);
      printf("----------- Structure: (%d, %d) ---------------- \n", blocksPerGrid, threadsPerBlock);

      #pragma omp parallel private(cpu_thread_id)
      {
        float *d_A, *d_B, *d_C;

	cpu_thread_id = omp_get_thread_num();
	cudaSetDevice(Dev[cpu_thread_id]);

	if(cpu_thread_id == 0){
	  cudaEventCreate(&start);
	  cudaEventCreate(&stop);
	  cudaEventRecord(start,0);
	}

	cudaMalloc((void**)&d_A, size/N_GPU);
	cudaMalloc((void**)&d_B, size/N_GPU);
	cudaMalloc((void**)&d_C, sb/N_GPU);

	// Copy from host memory
	cudaMemcpy(d_A, h_A + N_vec/N_GPU*cpu_thread_id, size/N_GPU, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B + N_vec/N_GPU*cpu_thread_id, size/N_GPU, cudaMemcpyHostToDevice);

        #pragma omp barrier

	if(cpu_thread_id == 0){
	  cudaEventRecord(stop,0);
	  cudaEventSynchronize(stop);
	  cudaEventElapsedTime( &Intime, start, stop);

	  cudaEventRecord(start,0);
	}

	dim3 block(blocksPerGrid);
	dim3 thread(threadsPerBlock);
	vecDot_ngpu<<<block, thread,sm>>>(d_A, d_B, d_C, N_vec/N_GPU);

	cudaDeviceSynchronize();

	if(cpu_thread_id == 0){
	  cudaEventRecord(stop,0);
	  cudaEventSynchronize(stop);
	  cudaEventElapsedTime( &GPUtime, start, stop);

	  cudaEventRecord(start,0);
	}

	cudaMemcpy(h_C + sb/sizeof(float)/N_GPU*cpu_thread_id, d_C, sb/N_GPU, cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	if(cpu_thread_id == 0){
	  cudaEventRecord(stop,0);
	  cudaEventSynchronize(stop);
	  cudaEventElapsedTime( &Outtime, start, stop);
	}
      }

      for( int l = 0; l < N_GPU; l++){
        cudaSetDevice(l);
	cudaDeviceReset();
      }
      cudaEventRecord(start,0);
      float result = 0.;
      for(int k = 0; k < sb/sizeof(float); k++) result += h_C[k];
      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime( &CPUtime, start, stop);

      Totaltime = Intime + GPUtime + Outtime + CPUtime;

      printf("  *input time:        %f (ms) \n", Intime);
      printf("  *process time(GPU): %f (ms) \n", GPUtime);
      printf("  *output time:       %f (ms) \n", Outtime);
      printf("  *process time(CPU): %f (ms) \n", CPUtime);
      printf("  *total time:        %f (ms) \n", Totaltime);
      printf("  *result:            %f \n",result);
      free(h_C);
   
      if(Totaltime < performance){
        performance = Totaltime;
	best_blocksPerGrid   = blocksPerGrid;
	best_threadsPerBlock = threadsPerBlock;
      }

    }
  }
  free(h_A);
  free(h_B);
  printf("---------- best setting (%d, %d) -----------\n",best_blocksPerGrid, best_threadsPerBlock);
  printf("  *performance = %f (ms) \n", performance);
}


void RandomInit(float* data, int n)
{
  for (int i = 0; i < n; i++){
    data[i] = rand()/(float)RAND_MAX;
  }
}
