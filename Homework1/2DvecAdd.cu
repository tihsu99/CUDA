// 2D Vector addition
// compile with the following
// nvcc -arch=compute_61 -code=sm_61,sm61 -O2 -m64 -o 2DvecAdd 2DvecAdd.cu

#include <stdio.h>
#include <stdlib.h>

// Variables
/*float** h_A; // host vectors
float** h_B;
float** h_C;
float** d_A;
float** d_B;
float** d_C;
*/

// Declare Functions
void RandomInit2D(float**, int);
// GPU code
__global__ void VecAdd(const float* A, const float* B, float* C, int n)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  int index = i*n + j;
  if(i < n && j < n){
    C[index] = A[index] + B[index];
  }

  __syncthreads();

}

int main()
{
  int GPUid = 1; // GPU ID
  cudaError_t err = cudaSetDevice(GPUid);
  if (err != cudaSuccess){
    printf(" Fail to select GPU with device ID = %d\n", GPUid);
    exit(1);
  }
  cudaSetDevice(GPUid);
  int N = 6400;
  // Allocate host input vectors
  float* d_A;
  float* d_B;
  float* d_C;
  float** h_A = new float*[N];
  float** h_B = new float*[N];
  float** h_C = new float*[N];
  float** h_D = new float*[N];
  h_A[0] = new float[N*N];
  h_B[0] = new float[N*N];
  h_C[0] = new float[N*N];
  h_D[0] = new float[N*N];
  for(int i = 1; i < N; i++){
    h_A[i] = h_A[i-1] + N;
    h_B[i] = h_B[i-1] + N;
    h_C[i] = h_C[i-1] + N;
    h_D[i] = h_D[i-1] + N;
  }
  RandomInit2D(h_A, N);
  RandomInit2D(h_B, N);

  float time;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  for(int i = 0; i < N; i++){
    for(int j = 0; j < N; j++){
      h_D[i][j] = h_A[i][j] + h_B[i][j];
    }
  }

  cudaEventRecord(stop,0);
  cudaEventElapsedTime( & time, start, stop);
  printf("Processing time for CPU: %f (ms) \n", time);

  int List_threadsPerBlock[6] = {4,8,10,16,20,32};
  for(int n = 0; n < 6; n++){
    // CUDA parameter setting
    int threadsPerBlock = List_threadsPerBlock[n];
    int blocksPerGrid = (N + threadsPerBlock -1)/threadsPerBlock;
    dim3 dimBlock(threadsPerBlock,threadsPerBlock);
    dim3 dimGrid(blocksPerGrid,blocksPerGrid);
    printf("-------- Block size: (%d, %d) --------- \n",threadsPerBlock, threadsPerBlock);

    float intime,processtime,outtime;
    // Start timer
    cudaEventRecord(start,0);

    // Allocate vectors in device memory
    cudaMalloc((void**)&d_A, sizeof(float)*N*N);
    cudaMalloc((void**)&d_B, sizeof(float)*N*N);
    cudaMalloc((void**)&d_C, sizeof(float)*N*N);

    // Copy vectors from host memory to device memory

    cudaMemcpy(d_A, h_A[0], sizeof(float)*N*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B[0], sizeof(float)*N*N, cudaMemcpyHostToDevice);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime( &intime, start, stop);

    cudaEventRecord(start,0);
    
    VecAdd<<<dimGrid,dimBlock>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime( &processtime, start, stop);

    cudaEventRecord(start,0);

    cudaMemcpy(h_C[0], d_C, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime( &outtime, start, stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime( &time, start, stop);
    printf("   *input time:      %f (ms) \n", intime);
    printf("   *processing time: %f (ms) \n", processtime);
    printf("   *output time:     %f (ms) \n", outtime);
    printf("   *total time:      %f (ms) \n", intime + processtime + outtime);

    float diff = 0.;
    for(int i = 0; i < N; i++){
      for(int j = 0; j < N; j++){
        diff+=pow((h_C[i][j]-h_D[i][j]),2);
      }
    }
    printf("    Difference with ans from CPU: %e \n", sqrt(diff));

  } 
    cudaEventDestroy(start);
    cudaEventDestroy(start);

    cudaDeviceReset();
} 

void RandomInit2D(float** data, int n)
{
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      data[i][j] = rand() / (float)RAND_MAX;
    }
  }
}
