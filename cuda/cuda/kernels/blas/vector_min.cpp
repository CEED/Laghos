// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
#include "../cuda.hpp"

// *****************************************************************************
#define CUDA_BLOCKSIZE 256

// *****************************************************************************
__global__ void cuKernelMin(const size_t N, double *gdsr, const double *x)
{
   __shared__ double s_min[CUDA_BLOCKSIZE];
   const size_t n = blockDim.x*blockIdx.x + threadIdx.x;
   if (n>=N) { return; }
   const size_t bid = blockIdx.x;
   const size_t tid = threadIdx.x;
   const size_t bbd = bid*blockDim.x;
   const size_t rid = bbd+tid;
   s_min[tid] = x[n];
   for (size_t workers=blockDim.x>>1; workers>0; workers>>=1)
   {
      __syncthreads();
      if (tid >= workers) { continue; }
      if (rid >= N) { continue; }
      const size_t dualTid = tid + workers;
      if (dualTid >= N) { continue; }
      const size_t rdd = bbd+dualTid;
      if (rdd >= N) { continue; }
      if (dualTid >= blockDim.x) { continue; }
      s_min[tid] = fmin(s_min[tid],s_min[dualTid]);
   }
   if (tid==0) { gdsr[bid] = s_min[0]; }
}

// *****************************************************************************
double cuVectorMin(const size_t N, const double *x)
{
   const size_t tpb = CUDA_BLOCKSIZE;
   const size_t blockSize = CUDA_BLOCKSIZE;
   const size_t gridSize = (N+blockSize-1)/blockSize;
   const size_t min_sz = (N%tpb)==0? (N/tpb) : (1+N/tpb);
   const size_t bytes = min_sz*sizeof(double);
   static double *h_min = NULL;
   if (!h_min) { h_min = (double*)calloc(min_sz,sizeof(double)); }
   static CUdeviceptr gdsr = (CUdeviceptr) NULL;
   if (!gdsr) { cuMemAlloc(&gdsr,bytes); }
   cuKernelMin<<<gridSize,blockSize>>>(N, (double*)gdsr, x);
   cuMemcpy((CUdeviceptr)h_min,(CUdeviceptr)gdsr,bytes);
   double min = HUGE_VAL;
   for (size_t i=0; i<min_sz; i+=1) { min = fmin(min,h_min[i]); }
   return min;
}


// *****************************************************************************
double vector_min(const int N, const double* __restrict x)
{
   return cuVectorMin(N, x);
}

