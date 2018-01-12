// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <cuda_runtime.h>

#include <device_functions.h>
#include <cooperative_groups.h>
//#include <cuda_device_runtime_api.h>

namespace cg = cooperative_groups;

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n/2 threads
    - only works for power-of-2 arrays

    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    See the CUDA SDK "reduction" sample for more information.
*/
__device__ void reduceBlock(double *sdata, const cg::thread_block &cta){
  const unsigned int tid = cta.thread_rank();
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
  double beta  = sdata[tid];
  double temp;
  for (int i = tile32.size() / 2; i > 0; i >>= 1) {
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

// This reduction kernel reduces an arbitrary size array in a single kernel invocation
// For more details on the reduction algorithm (notably the multi-pass approach), see
// the "reduction" sample in the CUDA SDK.
__global__ void cuReduceSum(const double *g_i1data,
                            const double *g_i2data,
                            double *g_odata,
                            unsigned int n){
  // Handle to thread block group
  cg::thread_block block = cg::this_thread_block();
  cg::grid_group grid = cg::this_grid();
  extern double __shared__ sdata[];
  // Stride over grid and add the values to a shared memory buffer
  sdata[block.thread_rank()] = 0;
  for (int i = grid.thread_rank(); i < n; i += grid.size()) {
    sdata[block.thread_rank()] += g_i1data[i]*g_i2data[i];
  }
  // Reduce each block (called once per block)
  reduceBlock(sdata, block);
  // Write out the result to global memory
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

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
void reduceSum(int size, int threads, int numBlocks,
               const double *d_i1data,
               const double *d_i2data,
               double *d_odata){
  int smemSize = threads * sizeof(double);
  void *kernelArgs[] = {
    (void*)&d_i1data,
    (void*)&d_i2data,
    (void*)&d_odata,
    (void*)&size,
  };
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(numBlocks, 1, 1);
  cudaLaunchCooperativeKernel((void*)cuReduceSum,
                              dimGrid, dimBlock, kernelArgs, smemSize, NULL);

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
  sdata[block.thread_rank()] = g_idata[0];
  for (int i = grid.thread_rank(); i < n; i += grid.size()) {
    sdata[block.thread_rank()] = fmin(g_idata[i],sdata[block.thread_rank()]);
  }
  // Reduce each block (called once per block)
  reduceBlock(sdata, block);
  // Write out the result to global memory
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

void reduceMin(int size, int threads, int numBlocks,
               const double *d_idata,
               double *d_odata){  int smemSize = threads * sizeof(double);
  void *kernelArgs[] = {
    (void*)&d_idata,
    (void*)&d_odata,
    (void*)&size,
  };
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(numBlocks, 1, 1);
  cudaLaunchCooperativeKernel((void*)cuReduceMin,
                              dimGrid, dimBlock, kernelArgs, smemSize, NULL);
}
