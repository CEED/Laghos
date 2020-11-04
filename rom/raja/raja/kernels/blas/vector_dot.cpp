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
#include "../raja.hpp"

// *****************************************************************************
#define BLOCKSIZE 256

// *****************************************************************************
__global__ void KernelDot(const size_t N, double *gdsr,
                            const double *x, const double *y)
{
   __shared__ double s_dot[BLOCKSIZE];
   const size_t n = blockDim.x*blockIdx.x + threadIdx.x;
   if (n>=N) { return; }
   const size_t bid = blockIdx.x;
   const size_t tid = threadIdx.x;
   const size_t bbd = bid*blockDim.x;
   const size_t rid = bbd+tid;
   s_dot[tid] = x[n] * y[n];
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
      s_dot[tid] += s_dot[dualTid];
   }
   if (tid==0) { gdsr[bid] = s_dot[0]; }
}

// *****************************************************************************
double gpuVectorDot(const size_t N, const double *x, const double *y)
{
   const size_t tpb = BLOCKSIZE;
   const size_t blockSize = BLOCKSIZE;
   const size_t gridSize = (N+blockSize-1)/blockSize;
   const size_t dot_sz = (N%tpb)==0? (N/tpb) : (1+N/tpb);
   const size_t bytes = dot_sz*sizeof(double);
   static double *h_dot = NULL;
   if (!h_dot) { h_dot = (double*)calloc(dot_sz,sizeof(double)); }

#if defined(RAJA_ENABLE_CUDA)
   static CUdeviceptr gdsr = (CUdeviceptr) NULL;
   if (!gdsr) { cuMemAlloc(&gdsr,bytes); }
   KernelDot<<<gridSize,blockSize>>>(N, (double*)gdsr, x, y);
   cuMemcpy((CUdeviceptr)h_dot,(CUdeviceptr)gdsr,bytes);
#elif defined(RAJA_ENABLE_HIP)
   static void* gdsr = (void*) NULL;
   if (!gdsr) { hipMalloc(&gdsr,bytes); }
   hipLaunchKernelGGL((KernelDot),dim3(gridSize),dim3(blockSize), 0, 0,
                           N, (double*)gdsr, x, y);
   hipMemcpy((void*)h_dot,(void*)gdsr,bytes, hipMemcpyDeviceToHost);
#endif

   double dot = 0.0;
   for (size_t i=0; i<dot_sz; i+=1) { dot += h_dot[i]; }
   return dot;
}

// *****************************************************************************
double vector_dot(const int N,
                  const double* __restrict x,
                  const double* __restrict y)
{
   if (mfem::rconfig::Get().Cuda() || mfem::rconfig::Get().Hip())
   {
      return gpuVectorDot(N,x,y);
   }
   ReduceDecl(Sum,dot,0.0);
   ReduceForall(i,N,dot += x[i]*y[i];);
   return dot;
}
