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
#include <sys/time.h>

double vector_dot(const int N,
                  const double* __restrict vec1,
                  const double* __restrict vec2) {
  //dbg();
#if !defined(__NVCC__)
  ReduceDecl(Sum,dot,0.0);
  ReduceForall(i,N,dot += vec1[i]*vec2[i];);
  //printf("\033[33m[vector_dot] %.14e\033[m\n",dot);
  return dot;
#else
  unsigned int v=N;
  unsigned int nBitInN=0;
  for(;v;nBitInN++) v&=v-1;
  const int nBytes = nBitInN*sizeof(double);
  double h_dot[nBitInN];
  double *d_dot; cudaMalloc(&d_dot, nBytes); cudaDeviceSynchronize();

  // flush dot results 
  for(int i=0;i<nBitInN;i+=1) h_dot[i]=0.0;
  checkCudaErrors(cudaMemcpy(d_dot,h_dot,nBytes,cudaMemcpyHostToDevice));

  for(unsigned int k=1, v=N,vof7=0,kof7=0;v;v>>=1,k<<=1){
    if (!(v&1)) continue;
    reduceSum(k,&vec1[vof7],&vec2[vof7],&d_dot[kof7]);
    kof7++;
    vof7+=k;
  }
  checkCudaErrors(cudaMemcpy(h_dot,d_dot,nBytes,cudaMemcpyDeviceToHost));
  cudaFree(d_dot);
  
  for(int i=1;i<nBitInN;i+=1)
    h_dot[0]+=h_dot[i];

  //printf("\033[33m[vector_dot] %.14e\033[m\n",h_dot[0]);
  return h_dot[0];
#endif
}
