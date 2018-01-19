/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

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
static unsigned int nextPow2(unsigned int x){
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}
// *****************************************************************************
static void getNumBlocksAndThreads(const int n,
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
  const int maxThreads = prop.maxThreadsPerBlock;
  const int maxBlocks = prop.multiProcessorCount*(prop.maxThreadsPerMultiProcessor/prop.maxThreadsPerBlock);
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
}



// *****************************************************************************
// * Reduce SUM
// *****************************************************************************
__device__ void cuReduceBlockSum(double *sdata, const cg::thread_block &cta){
  const unsigned int tid = cta.thread_rank();
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
  double beta  = sdata[tid];
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
  // Handle to thread block group
  cg::thread_block block = cg::this_thread_block();
  cg::grid_group grid = cg::this_grid();
  extern double __shared__ sdata[];
  // Stride over grid and add the values to a shared memory buffer
  const int btr =  block.thread_rank();
  sdata[btr] = 0.0;
  for (int i = grid.thread_rank(); i < n; i += grid.size()) {
    sdata[btr] += g_i1data[i]*g_i2data[i];
  }
  cuReduceBlockSum(sdata, block);
  // Write out the result to global memory
  if (btr == 0) {
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
  int threads=1024, blocks=1;
  //static int threads=0, blocks=0;
  if (threads==0 && blocks==0){
    cuSetup(0,size,threads,blocks);
  }
  const dim3 dimBlock(threads, 1, 1);
  const dim3 dimGrid(blocks, 1, 1);
  const int smemSize = threads * sizeof(double);

  void *kernelArgs[] = {
    (void*)&d_i1data,
    (void*)&d_i2data,
    (void*)&d_odata,
    (void*)&size,
  };
  cudaLaunchCooperativeKernel((void*)cuReduceSum,
                              dimGrid, dimBlock,
                              kernelArgs, smemSize, NULL);
}


// *****************************************************************************
// * Reduce MIN
// *****************************************************************************
__device__ void cuReduceBlockMin(double *sdata,
                                 const cg::thread_block &cta){
  const unsigned int tid = cta.thread_rank();
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
  double beta  = sdata[tid];
  double temp;
  for (int i = tile32.size() / 2; i > 0; i >>= 1) {
    if (tile32.thread_rank() < i) {
      temp = sdata[tid+i];
      beta = (beta<sdata[tid+i])?beta:temp;
      sdata[tid] = beta;
    }
    cg::sync(tile32);
  }
  cg::sync(cta);
  if (cta.thread_rank() == 0) {
    for (int i = 0; i < blockDim.x; i += tile32.size()) {
      beta  = (beta<sdata[i])?beta:sdata[i];
    }
    sdata[0] = beta;
  }
  cg::sync(cta);
}
// *****************************************************************************
__global__ void cuReduceMin(const double *g_idata,
                            double *g_odata,
                            unsigned int n){
  // Handle to thread block group
  cg::thread_block block = cg::this_thread_block();
  cg::grid_group grid = cg::this_grid();
  extern double __shared__ sdata[];
  // Stride over grid and add the values to a shared memory buffer
  const int btr =  block.thread_rank();
  sdata[btr] = g_idata[0];
  for (int i = grid.thread_rank(); i < n; i += grid.size()) {
    const double gid = g_idata[i];
    const double sdb = sdata[btr];
    sdata[btr] = (sdb<gid)?sdb:gid;
  }
  cuReduceBlockMin(sdata, block);
  if (block.thread_rank() == 0) {
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
void reduceMin(int size, const double *d_idata, double *d_odata){
  int threads=1024, blocks=1;
  //static int threads=0, blocks=0;
  if (threads==0 && blocks==0)
    cuSetup(0,size,threads,blocks);
  const dim3 dimBlock(threads, 1, 1);
  const dim3 dimGrid(blocks, 1, 1);
  const int smemSize = threads * sizeof(double);
  void *kernelArgs[] = {
    (void*)&d_idata,
    (void*)&d_odata,
    (void*)&size,
  };
  cudaLaunchCooperativeKernel((void*)cuReduceMin,dimGrid, dimBlock,kernelArgs, smemSize, NULL);
}
