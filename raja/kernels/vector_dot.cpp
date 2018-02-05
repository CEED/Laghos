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
#include "raja.hpp"

// *****************************************************************************
#ifdef __NVCC__
#include <cub/cub.cuh>
/*__attribute__((unused))
static double cu_vector_dot(const int N,
                            const double* __restrict vec1,
                            const double* __restrict vec2) {
  unsigned int v=N;
  unsigned int nBitInN=0;
  for(;v;nBitInN++) v&=v-1;
  const int nBytes = nBitInN*sizeof(double);
  double h_dot[nBitInN];
  double *d_dot;
  checkCudaErrors(cuMemAlloc((CUdeviceptr*)&d_dot, nBytes));
  for(int i=0;i<nBitInN;i+=1) h_dot[i]=0.0;
  checkCudaErrors(cuMemcpyHtoD((CUdeviceptr)d_dot,h_dot,nBytes));
  for(unsigned int k=1, v=N,vof7=0,kof7=0;v;v>>=1,k<<=1){
    if (!(v&1)) continue;
    reduceSum(k,&vec1[vof7],&vec2[vof7],&d_dot[kof7]);
    kof7++;
    vof7+=k;
  }
  checkCudaErrors(cuMemcpyDtoH(h_dot,(CUdeviceptr)d_dot,nBytes));
  cudaFree(d_dot);
  for(int i=1;i<nBitInN;i+=1)
    h_dot[0]+=h_dot[i];
  //printf("\033[33m[vector_dot] %.14e\033[m\n",h_dot[0]);
  return h_dot[0];
  }*/

// *****************************************************************************
__attribute__((unused))
static double cub_vector_dot(const int N,
                             const double* __restrict vec1,
                             const double* __restrict vec2) {
  double h_dot;
  static double *d_dot = NULL;
  if (!d_dot)
    checkCudaErrors(cuMemAlloc((CUdeviceptr*)&d_dot, 1*sizeof(double)));
  
  static void *d_storage = NULL;
  static size_t storage_bytes = 0;
  if (!d_storage){
    cub::DeviceReduce::Dot(d_storage, storage_bytes, vec1, vec2, d_dot, N);
    cudaMalloc(&d_storage, storage_bytes);
  }
  //printf(" \033[32;1m%d\033[m", storage_bytes); fflush(stdout);
  cub::DeviceReduce::Dot(d_storage, storage_bytes, vec1, vec2, d_dot, N);
  checkCudaErrors(cuMemcpyDtoH(&h_dot,(CUdeviceptr)d_dot,1*sizeof(double)));
  return h_dot;
}
#endif // __NVCC__

// *****************************************************************************
double vector_dot(const int N,
                  const double* __restrict vec1,
                  const double* __restrict vec2) {
#ifdef __NVCC__
  if (cuda) return cub_vector_dot(N,vec1,vec2);
#endif
  ReduceDecl(Sum,dot,0.0);
  ReduceForall(i,N,dot += vec1[i]*vec2[i];);
  return dot;
}
