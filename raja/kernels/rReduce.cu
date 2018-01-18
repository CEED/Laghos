// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <cuda_runtime.h>

#include <device_functions.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void cuReduceMin(const double*, double*, unsigned int);

// *****************************************************************************
// *****************************************************************************
unsigned int nextPow2(unsigned int x){
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}
// *****************************************************************************
void getNumBlocksAndThreads(const int n,
                            const int maxBlocks,
                            const int maxThreads,
                            int &blocks, int &threads){
  if (n == 1){
    threads = 1;
    blocks = 1;
  }else{
    threads = (n < maxThreads*2) ? nextPow2(n / 2) : maxThreads;
    blocks = max(1, n / (threads * 2));
  }
  blocks = min(maxBlocks, blocks);
}
// *****************************************************************************
__attribute__((unused))
static void cuSetup(const int device,
                    const int size,
                    int &numThreads,
                    int &numBlocks){
  cudaDeviceProp prop = { 0 };
  checkCudaErrors(cudaSetDevice(device));
  checkCudaErrors(cudaGetDeviceProperties(&prop, device));
  // Determine the launch configuration (threads, blocks)
  int maxThreads = prop.maxThreadsPerBlock;
  int maxBlocks = prop.multiProcessorCount*(prop.maxThreadsPerMultiProcessor/prop.maxThreadsPerBlock);
  getNumBlocksAndThreads(size, maxBlocks, maxThreads, numBlocks, numThreads);
  int numBlocksPerSm = 0;
  checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm,
                                                                cuReduceMin,
                                                                numThreads,
                                                                numThreads*sizeof(double)));
  int numSms = prop.multiProcessorCount;
  if (numBlocks > numBlocksPerSm * numSms)  {
    numBlocks = numBlocksPerSm * numSms;
  }
  //printf("%d threads, %d blocks\033[m", numThreads, numBlocks);
}



// *****************************************************************************
// * Reduce SUM
// *****************************************************************************
__device__ void cuReduceBlockSum(double *sdata, const cg::thread_block &cta){
  const unsigned int tid = cta.thread_rank();
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
  double beta  = sdata[tid];
  //printf("\033[33m[cuReduceBlockSum] beta=%f\033[m\n",beta);
  double temp;
  for (int i = tile32.size()/2; i > 0; i >>= 1) {
    if (tile32.thread_rank() < i) {
      temp       = sdata[tid+i];
      beta       += temp;
      sdata[tid] = beta;
    }
    cg::sync(tile32);
  }
  cg::sync(cta);
  if (cta.thread_rank() == 0) {
    beta  = 0;
    for (int i = 0; i < blockDim.x; i += tile32.size()) {
      beta  += sdata[i];
    }
    sdata[0] = beta;
  }
  cg::sync(cta);
}
// *****************************************************************************
__global__ void cuReduceSum(const double *g_i1data,
                            const double *g_i2data,
                            double *g_odata,
                            unsigned int n){
  //printf("\033[33m[cuReduceSum]\033[m\n");
  // Handle to thread block group
  cg::thread_block block = cg::this_thread_block();
  cg::grid_group grid = cg::this_grid();
  extern double __shared__ sdata[];
  // Stride over grid and add the values to a shared memory buffer
  const int btr =  block.thread_rank();
  sdata[btr] = 0.0;
  for (int i = grid.thread_rank(); i < n; i += grid.size()) {
    //printf("\t\033[33m[cuReduceSum] g_i1data[%d]=%f, g_i2data[%d]=%f\033[m\n",i,g_i1data[i],i,g_i2data[i]);
    sdata[btr] += g_i1data[i]*g_i2data[i];
    //printf("\t\033[33m[cuReduceSum] sdata[%d]=%f\033[m\n",btr,sdata[btr]);
  }
  //printf("\033[33m[cuReduceSum] Reduce each block (called once per block)\033[m\n");
  cuReduceBlockSum(sdata, block);
  // Write out the result to global memory
  if (btr == 0) {
    //printf("\033[33m[cuReduceSum] Write out the result to global memory\033[m\n");
    g_odata[blockIdx.x] = sdata[0];
  }
  cg::sync(grid);
  if (grid.thread_rank() == 0) {
    for (int block = 1; block < gridDim.x; block++) {
      g_odata[0] += g_odata[block];
    }
  }
}
// *****************************************************************************
void reduceSum(int size,
               const double *d_i1data, const double *d_i2data,
               double *d_odata){
  static int threads=0, blocks=0;
  if (threads==0 && blocks==0){
    //printf("\n\033[33m[reduceSum] Setup[size=%d]: ",size);  
    cuSetup(0,size,threads,blocks);
  }
  assert(blocks==1);
  const dim3 dimBlock(threads, 1, 1);
  const dim3 dimGrid(blocks, 1, 1);
  const int smemSize = threads * sizeof(double);

  //printf("\n\033[31m[reduceSum]\033[m");
  //double t_i1data[size];
  //double t_i2data[size];
  void *kernelArgs[] = {
    (void*)&d_i1data,
    (void*)&d_i2data,
    (void*)&d_odata,
    (void*)&size,
  };
  //printf("\n[reduceSum] [#t=%d,#b=%d] d_i1data:", threads,blocks);
  //for(int i=0;i<size;i+=1) { printf(" [%f]",d_i1data[i]); }
  //printf("\n[reduceSum] d_i2data:");
  //for(int i=0;i<size;i+=1) { printf(" [%f]",d_i2data[i]); }
  cudaLaunchCooperativeKernel((void*)cuReduceSum,
                              dimGrid, dimBlock,
                              kernelArgs, smemSize, NULL);
  cudaDeviceSynchronize();
  //printf("\033[33m[reduceSum] d_odata[0]=%f\n\033[m",d_odata[0]);
}


// *****************************************************************************
// * Reduce MIN
// *****************************************************************************
__device__ void cuReduceBlockMin(double *sdata,
                                 const cg::thread_block &cta){
  const unsigned int tid = cta.thread_rank();
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
  //printf("\033[31m[cuReduceBlock]\033[m\n");
  double beta  = sdata[tid];
  for (int i = tile32.size() / 2; i > 0; i >>= 1) {
    if (tile32.thread_rank() < i) {
      sdata[tid] = sdata[tid]<sdata[tid+i]?sdata[tid]:sdata[tid+i];
    }
    cg::sync(tile32);
  }
  cg::sync(cta);
  if (cta.thread_rank() == 0) {
    beta  = 0;
    for (int i = 0; i < blockDim.x; i += tile32.size()) {
      beta  += sdata[i];
    }
    sdata[0] = beta;
  }
  cg::sync(cta);
}
// *****************************************************************************
__global__ void cuReduceMin(const double *g_idata,
                            double *g_odata,
                            unsigned int n){
  //printf("\033[32m[cuReduceMin]\033[m\n");
  // Handle to thread block group
  cg::thread_block block = cg::this_thread_block();
  cg::grid_group grid = cg::this_grid();
  extern double __shared__ sdata[];
  // Stride over grid and add the values to a shared memory buffer
  sdata[block.thread_rank()] = g_idata[0];
  for (int i = grid.thread_rank(); i < n; i += grid.size()) {
    const double gid = g_idata[i];
    const double sdb = sdata[block.thread_rank()];
    //printf("\t\033[32m[cuReduceMin] for thread\033[m\n");
    sdata[block.thread_rank()] = (sdb<gid)?sdb:gid;
  }
  //printf("\033[32m[cuReduceMin] Reduce each block (called once per block)\033[m\n");
  cuReduceBlockMin(sdata, block);
  //printf("\033[32m[cuReduceMin] Write out the result to global memory\033[m\n");
  if (block.thread_rank() == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }
  cg::sync(grid);
  if (grid.thread_rank() == 0) {
    for (int block = 1; block < gridDim.x; block++) {
      //printf("\033[32m[cuReduceMin] g_odata[block]=%f\033[m\n",g_odata[block]);
      g_odata[0] += g_odata[block];
    }
  }else{
    //printf("\033[32m[cuReduceMin] grid.thread_rank() != 0\033[m\n");
  }
}


// *****************************************************************************
void reduceMinV(int size, const double *d_idata, double *d_odata){
  static int threads=0, blocks=0;
  if (threads==0 && blocks==0){
    //printf("\033[32m\n[reduceMin] Setup: ");
    cuSetup(0,size,threads,blocks);
  }
  assert(blocks==1);
  const dim3 dimBlock(threads, 1, 1);
  const dim3 dimGrid(blocks, 1, 1);
  const int smemSize = threads * sizeof(double);
  
/*  static double *t_idata=NULL;
  if (!t_idata){
    cudaMallocManaged(&t_idata, size*sizeof(double), cudaMemAttachGlobal);
  }
*/
  
  void *kernelArgs[] = {
    (void*)&d_idata,
    (void*)&d_odata,
    (void*)&size,
  };
  
/*  printf("\n\033[32m[reduceMin] [#t=%d,#b=%d] t_idata:\033[m", threads,blocks);
  for(int i=0;i<size;i+=1) {
    //printf(" [%f]",t_idata[i]=(i+18)*0.123456789);
    printf(" [%f]",d_idata[i]);
  }
*/
  cudaLaunchCooperativeKernel((void*)cuReduceMin,
                              dimGrid, dimBlock,
                              kernelArgs, smemSize, NULL);
  cudaDeviceSynchronize();
  //printf("\033[32m[reduceMin] d_odata[0]=%f\n\033[m",d_odata[0]);
//#warning exit(0)
//  exit(0);
}
