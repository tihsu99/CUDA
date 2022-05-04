// Includes
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>

// field variables
float* h_new;
float* h_old;
float* h_1;
float* h_2;
float* h_C;
float* g_new;
float** d_1;
float** d_2;
float** d_C;

int    iter_max = 1e7;
double eps      = 1.0e-8;

__global__ void heat_diff(float* phi0_old, float* phiL_old, float* phiR_old, float* phiB_old, float* phiT_old, float* phi0_new, float* C, float omega)
{
  extern __shared__ float cache[];
  float t, l, c, r, b;
  float diff;
  int site, skip;

  int Lx = blockDim.x*gridDim.x; // lattice size in each GPU
  int Ly = blockDim.y*gridDim.y;
  int x  = blockDim.x*blockIdx.x + threadIdx.x;
  int y  = blockDim.y*blockIdx.y + threadIdx.y;
  int cacheIndex = threadIdx.x + threadIdx.y*blockDim.x;

  site = x + y*Lx;
  skip = 0;
  diff = 0.0;
  b = 0.; l = 0.; r = 0.; t = 0.; c = phi0_old[site];

  // Left boundary
  if(x == 0){
    if (phiL_old != NULL){
      l = phiL_old[(Lx-1)+y*Lx];
      r = phi0_old[site+1];
    }
    else skip = 1;
  }
  //right boundary
  else if (x == Lx-1){
    if(phiR_old != NULL){
      l = phi0_old[site-1];
      r = phiR_old[y*Lx];
    }
    else skip = 1;
  }
  else {
    l = phi0_old[site-1];
    r = phi0_old[site+1];
  }

  if (y==0) {
    if (phiB_old != NULL){
      b = phiB_old[x+(Ly-1)*Lx];
      t = phi0_old[site+Lx];
    }
    else skip = 1;
  }
  else if (y == Ly-1) {
    if (phiT_old != NULL){
      b = phi0_old[site-Lx];
      t = phiT_old[x];
    }
    else skip = 1;
  }
  else{
    b = phi0_old[site-Lx];
    t = phi0_old[site+Lx];
  }

  if (skip == 0){
    phi0_new[site] = 0.25*omega*(b+l+r+t) + (1.-omega)*c;
    diff = phi0_new[site] - c;
  }
  cache[cacheIndex] = diff*diff;
  __syncthreads();

  // Parallel reduction
  int ib = blockDim.x*blockDim.y/2;
  while (ib != 0){
    if (cacheIndex < ib) cache[cacheIndex] += cache[cacheIndex + ib];
    __syncthreads();
    ib /= 2;
  }

  int blockIndex = blockIdx.x + gridDim.x*blockIdx.y;
  if(cacheIndex == 0) C[blockIndex] = cache[0];
}

int main(void)
{
  volatile bool flag;
  int cpu_thread_id = 0;
  int N_GPU[2]  = {2,1};
  int Dev[2]    = {0,1};
  int N, Nx, Ny, Lx, Ly, NGy, NGx, tx, ty, bx, by, sm, iter;
  float Intime, GPUtime, Outime, Totaltime;
  float error;
  float omega = 1.;
  cudaEvent_t start, stop;

  // Parameter set; 
  int N_set = 3;
  int list_thread[N_set] = {8,16,32};

  // Initialize field
  Nx = 1024;
  Ny = 1024;
  N  = Nx*Ny;
  int size = N*sizeof(float);
  h_1 = (float*)malloc(size);
  h_2 = (float*)malloc(size);
  g_new = (float*)malloc(size);

  printf(" ----------- (NGPU, blocksize_x, blocksize_y) --------\n");

  for(int i = 0; i < 2; i++){
    for(int j = 0; j < N_set; j++){
      // GPU number
      NGx = 1;
      NGy = N_GPU[i];
      int NGPU = NGx * NGy;
      // Lattice per GPU
      Lx = Nx / NGx;
      Ly = Ny / NGy;
      tx = list_thread[j];
      ty = list_thread[j];
      bx = Nx/tx;
      by = Ny/ty;
      dim3 threads(tx,ty);
      dim3 blocks(bx/NGx,by/NGy);
      int  sb = bx*by*sizeof(float);

      printf(" ----- Structure: (%d, %d, %d) ------\n",NGPU, tx,ty);
      
      error = 10*eps;
      flag  = true;

      // Allocate host vector
      h_C   = (float*)malloc(sb);
      // Initialize it
      memset(h_1, 0, size);
      memset(h_2, 0, size);
      for (int x = 0; x < Nx; x++){
        h_1[x+Nx*(Ny-1)] = 400.;
	h_2[x+Nx*(Ny-1)] = 400.;
	h_1[x] = 273.;
	h_2[x] = 273.;
      }
      for (int y = 0; y < Ny; y++){
        h_1[Nx*y] = 273.;
	h_2[Nx*y] = 273.;
	h_1[Nx-1 + Nx*y] = 273.;
	h_2[Nx-1 + Nx*y] = 273.;
      }

      sm = tx*ty*sizeof(float);

      d_1 = (float **)malloc(NGPU*sizeof(float *));
      d_2 = (float **)malloc(NGPU*sizeof(float *));
      d_C = (float **)malloc(NGPU*sizeof(float *));

      omp_set_num_threads(NGPU);
      #pragma omp parallel private(cpu_thread_id)
      {
        int cpuid_x, cpuid_y;
	cpu_thread_id = omp_get_thread_num();
	cpuid_x = cpu_thread_id % NGx;
	cpuid_y = cpu_thread_id / NGx;      
        cudaSetDevice(Dev[cpu_thread_id]);

        int cpuid_r = ((cpuid_x + 1) % NGx) + cpuid_y*NGx;
        cudaDeviceEnablePeerAccess(Dev[cpuid_r],0);
        int cpuid_l = ((cpuid_x+NGx-1)%NGx) + cpuid_y*NGx;
        cudaDeviceEnablePeerAccess(Dev[cpuid_l],0);
        int cpuid_t = cpuid_x + ((cpuid_y+1)%NGy)*NGx;
        cudaDeviceEnablePeerAccess(Dev[cpuid_t],0);
        int cpuid_b = cpuid_x +	((cpuid_y+NGy-1)%NGy)*NGx;
        cudaDeviceEnablePeerAccess(Dev[cpuid_b],0);	
      
	if (cpu_thread_id == 0){
	  cudaEventCreate(&start);
	  cudaEventCreate(&stop);
	  cudaEventRecord(start,0);
	}

	cudaMalloc((void**)&d_1[cpu_thread_id], size/NGPU);
	cudaMalloc((void**)&d_2[cpu_thread_id], size/NGPU);
	cudaMalloc((void**)&d_C[cpu_thread_id], sb/NGPU);

	for(int k = 0; k < Ly; k++){
	  float *h, *d;
	  h = h_1 + cpuid_x*Lx + (cpuid_y*Ly + k)*Nx;
	  d = d_1[cpu_thread_id] + k*Lx;
	  cudaMemcpy(d,h,Lx*sizeof(float), cudaMemcpyHostToDevice);
	}

	for(int k = 0; k < Ly; k++){
	  float *h, *d;
	  h = h_2 + cpuid_x*Lx + (cpuid_y*Ly + k)*Nx;
	  d = d_2[cpu_thread_id] + k*Lx;
	  cudaMemcpy(d, h , Lx*sizeof(float), cudaMemcpyHostToDevice);
	}

        #pragma omp barrier

	//stop the timer
	if (cpu_thread_id == 0){
	  cudaEventRecord(stop, 0);
	  cudaEventSynchronize(stop);
	  cudaEventElapsedTime(&Intime, start, stop);
	}
      }
      cudaEventRecord(start,0);
      iter = 0;
      while((error > eps) && (iter < iter_max)){
        #pragma omp parallel private(cpu_thread_id)
	{
	  int cpuid_x, cpuid_y;
	  cpu_thread_id = omp_get_thread_num();
	  cpuid_x       = cpu_thread_id % NGx;
	  cpuid_y       = cpu_thread_id / NGx;
	  cudaSetDevice(Dev[cpu_thread_id]);
	  float **d_old, **d_new;
	  float *dL_old, *dR_old, *dT_old, *dB_old, *d0_old, *d0_new;
	  d_old   = (flag == true) ? d_1 : d_2;
	  d_new   = (flag == true) ? d_2 : d_1;
	  d0_old = d_old[cpu_thread_id];
          d0_new = d_new[cpu_thread_id];
	  dL_old = (cpuid_x == 0)     ? NULL : d_old[cpuid_x-1+cpuid_y*NGx];
	  dR_old = (cpuid_x == NGx-1) ? NULL : d_old[cpuid_x+1+cpuid_y*NGx];
	  dB_old = (cpuid_y == 0)     ? NULL : d_old[cpuid_x+(cpuid_y-1)*NGx];
	  dT_old = (cpuid_y == NGy-1) ? NULL : d_old[cpuid_x+(cpuid_y+1)*NGx];

	  heat_diff<<<blocks,threads,sm>>>(d0_old, dL_old, dR_old, dB_old, dT_old, d0_new, d_C[cpu_thread_id],omega);

	  cudaDeviceSynchronize();

	  cudaMemcpy(h_C+bx*by/NGPU*cpu_thread_id, d_C[cpu_thread_id], sb/NGPU, cudaMemcpyDeviceToHost);
	} // Open MP

        error = 0.0;
        for(int l=0; l < bx*by; l++){
          error = error + h_C[l];
        }
        error = sqrt(error);

        iter ++;
        flag = !flag;
      }
      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime( &GPUtime, start, stop);
      cudaEventRecord(start,0);

      #pragma omp parallel private(cpu_thread_id)
      {
        int cpuid_x, cpuid_y;
	cpu_thread_id = omp_get_thread_num();
	cpuid_x       = cpu_thread_id % NGx;
	cpuid_y       = cpu_thread_id / NGx;
	cudaSetDevice(Dev[cpu_thread_id]);

	float* d_new = (flag==true) ? d_2[cpu_thread_id] : d_1[cpu_thread_id];
	for (int l = 0; l < Ly; l++){
	  float *g, *d;
	  g = g_new + cpuid_x*Lx + (cpuid_y*Ly+l)*Nx;
	  d = d_new + l*Lx;
	  cudaMemcpy(g,d,Lx*sizeof(float),cudaMemcpyDeviceToHost);
	}
	cudaFree(d_1[cpu_thread_id]);
	cudaFree(d_2[cpu_thread_id]);
	cudaFree(d_C[cpu_thread_id]);
      }
      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime( &Outime, start, stop);
      Totaltime = Intime + GPUtime + Outime;
      printf("  *input time:     %f (ms) \n", Intime);
      printf("  *process time:   %f (ms) \n", GPUtime);
      printf("  *output time:    %f (ms) \n", Outime);
      printf("  *total time:     %f (ms) \n", Totaltime);
      printf("  *Phi[5000]:      %f \n", g_new[5000]);
      free(h_C);
      free(d_1);
      free(d_2);
      free(d_C);
/*      FILE *outg;
      if ((outg = fopen("phi_GPU.dat","w")) == NULL){
        printf("Can not open file.\n");
	exit(1);
      }
      fprintf(outg, "GPU field configuration:\n");
      for(int m = Ny-1; m>-1; m--){
        for(int n = 0 ;n < Ny; n++){
	  fprintf(outg,"%.2e ",g_new[n+m*Nx]);
	}
	fprintf(outg,"\n");
      }
      fclose(outg);*/
    }
  }
  free(h_new);
  free(h_old);
  free(g_new);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
