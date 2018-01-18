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

void reduceSum(int,const double*,const double*,double*);

double vector_dot(const int N,
                  const double* __restrict vec1,
                  const double* __restrict vec2) {
#if defined(__RAJA__) || (!defined(__RAJA__)&&!defined(__NVCC__))
  ReduceDecl(Sum,dot,0.0);
  ReduceForall(i,N,dot += vec1[i]*vec2[i];);
  return dot;
#else
  unsigned int v=N;
  unsigned int nBitInN=0;
  //struct timeval st, et;
  for(;v;nBitInN++) v&=v-1;
  //printf("\n\t[vector_dot] %d bits in %d (0x%X)\n",nBitInN,N,N);

  static double *dot=NULL;
  if (!dot){
    //printf("cudaMallocManaged(dot)\n");
    //#warning should be size of block
    cudaMallocManaged(&dot, nBitInN*sizeof(double), cudaMemAttachGlobal);
    cudaDeviceSynchronize();
  }
  // flush dot results 
  for(int i=0;i<nBitInN;i+=1) dot[i]=0.0;
  
  //gettimeofday(&st, NULL);
  for(unsigned int k=1, v=N,vof7=0,kof7=0;v;v>>=1,k<<=1){
    if (!(v&1)) continue;
    //printf("[vector_dot] offset=%d, size=%d, launching dot[%d]=%f\n",vof7,k,kof7,dot[kof7]);
    reduceSum(k,&vec1[vof7],&vec2[vof7],&dot[kof7]);
    //printf("[vector_dot] got[%d]=%f\n",kof7,dot[kof7]);
    kof7++;
    vof7+=k;
  }
  //gettimeofday(&et, NULL);
  //const double alltime = ((et.tv_sec-st.tv_sec)*1000.0+(et.tv_usec-st.tv_usec)/1000.0);
  //printf("\033[32m[dot] reduceSum (%d) in \033[1m%12.6e(s)\n\033[m",N,alltime/1000.0);
  
  for(int i=1;i<nBitInN;i+=1){
    //printf("[vector_dot] dot[%d]=%f\n",i,dot[i]);
    dot[0]+=dot[i];
  }
  
  return dot[0];
#endif
}
