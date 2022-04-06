// Problem set 2
// compile with the following command
//
//
// (for GTX1060)
// nvcc -arch=compute_61 -code=sm_61,sm_61 -O3 -m64 -o vecAbs vecAbs.cu

// Includes
#include <stdio.h>
#include <stdlib.h>

// Variables
float* h_A;
float* h_B;
float* d_A;
float* d_B;
float  v_CPU, v_GPU;

//Functions
void RandomInit(float*, int);

// Device code
__global__ void vecAbs(const float*A, float*B, int N)
{
  extern __shared__ float cache[];
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int cacheIndex = threadIdx.x;

  float temp = 0.0;
  while( i < N ){
    if(fabs(A[i]) > temp){
       temp = fabs(A[i]);
    }
    i += blockDim.x*gridDim.x;
  }

  cache[cacheIndex] = temp;

  __syncthreads();

  int ib = blockDim.x/2;
  while (ib != 0){
    if(cacheIndex < ib){
      if(cache[cacheIndex + ib] > cache[cacheIndex]) 
	 cache[cacheIndex] = cache[cacheIndex + ib];
    }
    __syncthreads();
    ib /=2;
  }
  if(cacheIndex == 0)
    B[blockIdx.x] = cache[0];
}

int main(void)
{
  int GPUid = 1;
  cudaError_t err = cudaSetDevice(GPUid);
  if (err != cudaSuccess){
    printf(" Fail to select GPU with device ID = %d\n", GPUid);
    exit(1);
  }
  cudaSetDevice(GPUid);

  int N = 81920007;
  int size = N * sizeof(float);
  h_A = (float*)malloc(size);

  RandomInit(h_A, N);

  float time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  v_CPU = 0.;
  for(int i = 0; i < N; i++){
    if(fabs(h_A[i]) > v_CPU) v_CPU = fabs(h_A[i]);
  }

  cudaEventRecord(stop,0);
  cudaEventElapsedTime( &time, start, stop);
  printf("Processing time for CPU: %f (ms) , result: %f\n", time, v_CPU);

  int List_threadsPerBlock[6] = {8,16,32,64,128,256};
  int List_blocksPerGrid[6]   = {256,512,1024,2048,4096,8192};

  int best_blocksPerGrid = -1;
  int best_threadsPerBlock = -1;
  float performance = 999.;
  printf("-------- (gridsize,blocksize) ------------\n");
  for(int i = 0; i < 6; i++){
    for(int j = 0; j < 6; j++){

      int threadsPerBlock = List_threadsPerBlock[i];
      int blocksPerGrid   = List_blocksPerGrid[j];
      int sb = blocksPerGrid * sizeof(float);
      h_B = (float*)malloc(sb);
      printf("--------- Structure: (%d, %d) ----------- \n", blocksPerGrid, threadsPerBlock);
      float intime, processtime1, outtime, processtime2;
      
      cudaEventRecord(start,0);

      cudaMalloc((void**)&d_A, size);
      cudaMalloc((void**)&d_B, sb);

      cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime( &intime, start, stop);

      cudaEventRecord(start,0);
 
      int sm = threadsPerBlock*sizeof(float);
      vecAbs <<< blocksPerGrid, threadsPerBlock, sm>>>(d_A,d_B,N);

      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime( &processtime1, start, stop);

      cudaEventRecord(start,0);

      cudaMemcpy(h_B, d_B, sb, cudaMemcpyDeviceToHost);

      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime( &outtime, start, stop);
  
      cudaFree(d_A);
      cudaFree(d_B);

      cudaEventRecord(start,0);
      v_GPU = 0.0;
      for(int k = 0; k < blocksPerGrid; k++){
        if(h_B[k] > v_GPU) v_GPU = h_B[k];
      }
      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime( &processtime2, start, stop);
      float total_time = intime + processtime1 + outtime + processtime2;
      printf("  *input time:        %f (ms) \n", intime);
      printf("  *process time(GPU): %f (ms) \n", processtime1);
      printf("  *output time:       %f (ms) \n", outtime);
      printf("  *process time(CPU): %f (ms) \n", processtime2);
      printf("  *total time:        %f (ms) \n", total_time);
      printf("  *result:            %f \n",v_GPU);
      if(total_time < performance){
        performance = total_time;
	best_blocksPerGrid = blocksPerGrid;
	best_threadsPerBlock = threadsPerBlock;
      }
    }
  }
  printf("------- best setting (%d, %d) ------------\n",best_blocksPerGrid,best_threadsPerBlock);
  printf("  *performance = %f (ms) \n", performance);

}

void RandomInit(float* data, int n)
{
  for (int i=0; i < n; i++){
    data[i] = 2.0*rand()/(float)RAND_MAX - 1.0;
  }
}
