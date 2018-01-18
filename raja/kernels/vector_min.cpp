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
void reduceMin(int size, const double *d_idata, double *d_odata);

double vector_min(const int N,
                  const double* __restrict vec) {
#if defined(__RAJA__) || (!defined(__RAJA__)&&!defined(__NVCC__))
  ReduceDecl(Min,red,vec[0]);
  ReduceForall(i,N,red.min(vec[i]););
  return red;
#else
  unsigned int v=N;
  unsigned int nBitInN=0;
  for(;v;nBitInN++) v&=v-1;

  static double *red=NULL;
  if (!red) {
    cudaMallocManaged(&red, nBitInN*sizeof(double), cudaMemAttachGlobal);
    cudaDeviceSynchronize();
  }
  for(int i=0;i<nBitInN;i+=1) red[i]=vec[0];
  
  for(unsigned int k=1, v=N,vof7=0,kof7=0;v;v>>=1,k<<=1){
    if (!(v&1)) continue;
    reduceMin(k,&vec[vof7],&red[kof7]);
    kof7++;
    vof7+=k;
  }
  
  for(int i=1;i<nBitInN;i+=1)
    red[0]=red[0]<red[i]?red[0]:red[i];
  
  return red[0];
#endif
}
