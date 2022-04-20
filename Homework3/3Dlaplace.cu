
// Includes
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// field variables

//////////
// Host //
//////////

float* h_old;
float* h_new;
float* h_source;
float* h_C;
float* g_new;

////////////
// Device //
////////////

float* d_old;
float* d_new;
float* d_source;
float* d_C;

int    MAX = 10000000;
double eps = 1.0e-10;

__global__ void laplacian(float* phi_old, float* phi_new, float* source, 
		          float* C, bool flag, cudaTextureObject_t texOld,
			  cudaTextureObject_t texNew, cudaTextureObject_t texSource)
{
    extern __shared__ float cache[];
    float t,l,c,r,b,u,d,s; // top, left, center, right, bottom, up ,down, source
    float diff;
    int   site, xm1, ym1, zm1, xp1, yp1, zp1;

    int Nx = blockDim.x*gridDim.x;
    int Ny = blockDim.y*gridDim.y;
    int Nz = blockDim.z*gridDim.z;
    int x  = blockDim.x*blockIdx.x + threadIdx.x;
    int y  = blockDim.y*blockIdx.y + threadIdx.y;
    int z  = blockDim.z*blockIdx.z + threadIdx.z;
    int cacheIndex = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;

    site = x + y*Nx + z*Nx*Ny;

    if((x == 0) || (x == Nx-1) ||
       (y == 0) || (y == Ny-1) ||
       (z == 0) || (z == Nz-1) ){
       diff = 0.0;
    }
    else {
      xm1 = site - 1;
      xp1 = site + 1;
      ym1 = site - Nx;
      yp1 = site + Nx;
      zm1 = site - Nx*Ny;
      zp1 = site + Nx*Ny;
      if(flag){
        b = tex1Dfetch<float>(texOld, ym1);
	l = tex1Dfetch<float>(texOld, xm1);
	d = tex1Dfetch<float>(texOld, zm1);
	c = tex1Dfetch<float>(texOld, site);
	r = tex1Dfetch<float>(texOld, xp1);
	t = tex1Dfetch<float>(texOld, yp1);
	u = tex1Dfetch<float>(texOld, zp1);
	s = tex1Dfetch<float>(texSource, site);
	phi_new[site] = 1./6.*(b+l+d+r+t+u+s);
	diff = phi_new[site]-c;
      }
      else {
        b = tex1Dfetch<float>(texNew, ym1);
	l = tex1Dfetch<float>(texNew, xm1);
	d = tex1Dfetch<float>(texNew, zm1);
	r = tex1Dfetch<float>(texNew, xp1);
	t = tex1Dfetch<float>(texNew, yp1);
	u = tex1Dfetch<float>(texNew, zp1);
	c = tex1Dfetch<float>(texNew, site);
	s = tex1Dfetch<float>(texSource, site);
	phi_old[site] = 1./6.*(b+l+d+r+t+u+s);
	diff = phi_old[site]-c;
      }
    }
    cache[cacheIndex] = diff*diff;
    __syncthreads();

    // parallel reduction in each block

    int ib = blockDim.x*blockDim.y*blockDim.z/2;
    while(ib != 0) {
      if(cacheIndex < ib)  cache[cacheIndex] += cache[cacheIndex + ib];
      __syncthreads();
      ib /= 2;      
    }
    int blockIndex = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
    if(cacheIndex == 0) C[blockIndex] = cache[0];
}

int main(void)
{
  double error;
  int    gid = 1;

  cudaError_t err = cudaSuccess;
  err = cudaSetDevice(gid);
  if (err != cudaSuccess){
      printf("Oh no! Cannot select GPU with device ID = %d\n", gid);
      exit(1);
  }
  cudaSetDevice(gid);
  printf("Select GPU with device ID = %d\n", gid);

  int Nx,Ny,Nz;
  int tx,ty,tz;
  int bx,by,bz;

  // Fix threads per block in this study;
  tx = 4; ty = 4; tz = 4;
  dim3 threads(tx,ty,tz);
  int sm = tx*ty*tz*sizeof(float);
  
  int L[4] = {8,16,32,64};

  int N,size,sb,middle_index, iter;
  volatile bool flag;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);


  for(int i = 0; i < 4; i++){
    Nx = L[i];  Ny = L[i];  Nz = L[i];
    bx = Nx/tx; by = Ny/ty; bz = Nz/tz;
    dim3 blocks(bx,by,bz);
    N = Nx*Ny*Nz; size = N*sizeof(float); sb = bx*by*bz*sizeof(float);

    printf("-------- %d X %d X %d -----------\n\n",Nx,Ny,Nz);

    h_old    = (float*)malloc(size);
    h_new    = (float*)malloc(size);
    h_source = (float*)malloc(size);
    g_new    = (float*)malloc(size);
    h_C      = (float*)malloc(size);
    memset(h_source, 0, size);
    memset(h_old,    0, size);
    memset(h_new,    0, size);
    
    // Initialize the charge source condition.

    middle_index = (Nx/2) + (Ny/2)*Nx + (Nz/2)*Ny*Nx;
    h_source[middle_index] = 1.;

    // Other parameter

    error = 100*eps; iter = 0; flag = true;

    // Create timer

    cudaEventRecord(start,0);
    
    // Allocate vectors in device memory

    cudaMalloc((void**)&d_new,    size);
    cudaMalloc((void**)&d_old,    size);
    cudaMalloc((void**)&d_source, size);
    cudaMalloc((void**)&d_C,      sb);

    // Setup for texture.

    cudaTextureObject_t texOld, texNew, texSource;
    struct cudaResourceDesc resDesc;
    struct cudaTextureDesc  texDesc;

    memset(&texDesc, 0, sizeof(texDesc));
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr      = d_old;
    resDesc.res.linear.desc        = cudaCreateChannelDesc<float>();
    resDesc.res.linear.sizeInBytes = size;
    cudaCreateTextureObject(&texOld, &resDesc, &texDesc, NULL);

    memset(&texDesc, 0, sizeof(texDesc));
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr      = d_new;
    resDesc.res.linear.desc        = cudaCreateChannelDesc<float>();
    resDesc.res.linear.sizeInBytes = size;
    cudaCreateTextureObject(&texNew, &resDesc, &texDesc, NULL);

    memset(&texDesc, 0, sizeof(texDesc));
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr      = d_source;
    resDesc.res.linear.desc        = cudaCreateChannelDesc<float>();
    resDesc.res.linear.sizeInBytes = size;
    cudaCreateTextureObject(&texSource, &resDesc, &texDesc, NULL);

    // Copy from host to device

    cudaMemcpy(d_new,    h_new,    size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_old,    h_old,    size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_source, h_source, size, cudaMemcpyHostToDevice);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float InputTime;
    cudaEventElapsedTime( &InputTime, start, stop);
    
    cudaEventRecord(start,0);
    while((error > eps) && (iter < MAX)){
      
      laplacian<<<blocks,threads,sm>>>(d_old, d_new, d_source, d_C, flag, texOld, texNew, texSource);
      cudaMemcpy(h_C, d_C, sb, cudaMemcpyDeviceToHost);
      error = 0.0;
      for(int j = 0; j < bx*by*bz; j++){
        error += h_C[j];
      }
      error = sqrt(error);
      iter ++;
      flag = !flag;
    }

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float GPUTime;
    cudaEventElapsedTime(&GPUTime, start, stop);

    cudaEventRecord(start,0);
    
    cudaMemcpy(g_new, d_new, size, cudaMemcpyDeviceToHost);
    
    cudaDestroyTextureObject(texOld);
    cudaDestroyTextureObject(texNew);
    cudaDestroyTextureObject(texSource);
    cudaFree(d_source);
    cudaFree(d_new);
    cudaFree(d_old);
    cudaFree(d_C);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float OuTime;
    cudaEventElapsedTime( &OuTime, start, stop);

    float total_time = InputTime + GPUTime + OuTime;
    printf("Total time: %f (ms)\n",total_time);
    float distance, exact_sol, sim_sol;
    int ref, now;
    for(int k = 1; k < Nx/2; k++){
      distance = sqrt(((float)(k-Nx/2))*((float)(k-Nx/2)));
      exact_sol = (1./(4.*M_PI*distance)) - (1./(4.*M_PI));
      now = k + (Ny/2)*Nx + (Nz/2)*Nx*Ny;
      ref = (Nx/2)-1 + (Ny/2)*Nx + (Nz/2)*Nx*Ny;
      sim_sol = g_new[now] - g_new[ref];
      printf(" r = %2.0f, Phi_sim = % 1.4f, Phi_th = % 1.4f, sim/th = %1.3f\n",distance,sim_sol,exact_sol,sim_sol/exact_sol);
    }
    printf("\n");
  }
  cudaDeviceReset();
  return 1; 
}
