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

double vector_min(const int N,
                  const double* __restrict vec) {
#if !defined(__NVCC__)
  ReduceDecl(Min,red,vec[0]);
  ReduceForall(i,N,red.min(vec[i]););
  return red;
#else
  //printf("\033[32;1m[vector_min] N=%d\033[m\n",N);
  unsigned int v=N;
  unsigned int nBitInN=0;
  for(;v;nBitInN++) v&=v-1;
  const int nBytes = nBitInN*sizeof(double);
  double *h_red=(double*)::malloc(nBytes);
  double *d_red; cudaMalloc(&d_red, nBytes);
   
  double h_vec0;
  checkCudaErrors(cudaMemcpy(&h_vec0,vec,1*sizeof(double),cudaMemcpyDeviceToHost));
  for(int i=0;i<nBitInN;i+=1) h_red[i] = h_vec0;
  checkCudaErrors(cudaMemcpy(d_red,h_red,nBytes,cudaMemcpyHostToDevice));

  for(unsigned int k=1, v=N,vof7=0,kof7=0;v;v>>=1,k<<=1){
    if (!(v&1)) continue;
    reduceMin(k,&vec[vof7],&d_red[kof7]);
    kof7++;
    vof7+=k;
  }
  
  checkCudaErrors(cudaMemcpy(h_red,d_red,nBytes,cudaMemcpyDeviceToHost));
  cudaFree(d_red);

  for(int i=1;i<nBitInN;i+=1)
    h_red[0]=h_red[0]<h_red[i]?h_red[0]:h_red[i];
  
  //printf("\033[32;1m[vector_min] done\n\033[m");
  return h_red[0];
#endif
}
