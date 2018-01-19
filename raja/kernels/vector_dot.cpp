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
#if !defined(__NVCC__)
  ReduceDecl(Sum,dot,0.0);
  ReduceForall(i,N,dot += vec1[i]*vec2[i];);
  return dot;
#else
  //struct timeval st, et;
  unsigned int v=N;
  unsigned int nBitInN=0;
  for(;v;nBitInN++) v&=v-1;
  const int nBytes = nBitInN*sizeof(double);
  double h_dot[nBitInN];
  double *d_dot; cudaMalloc(&d_dot, nBytes); cudaDeviceSynchronize();


  // flush dot results 
  for(int i=0;i<nBitInN;i+=1) h_dot[i]=0.0;
  checkCudaErrors(cudaMemcpy(d_dot,h_dot,nBytes,cudaMemcpyHostToDevice));

  //gettimeofday(&st, NULL);
  for(unsigned int k=1, v=N,vof7=0,kof7=0;v;v>>=1,k<<=1){
    if (!(v&1)) continue;
    //printf("[vector_dot] offset=%d, size=%d, launching dot[%d]=%f\n",vof7,k,kof7,dot[kof7]);
    reduceSum(k,&vec1[vof7],&vec2[vof7],&d_dot[kof7]);
    //printf("[vector_dot] got[%d]=%f\n",kof7,dot[kof7]);
    kof7++;
    vof7+=k;
  }
  checkCudaErrors(cudaMemcpy(h_dot,d_dot,nBytes,cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();
  cudaFree(d_dot);
  //gettimeofday(&et, NULL);
  //const double alltime = ((et.tv_sec-st.tv_sec)*1000.0+(et.tv_usec-st.tv_usec)/1000.0);
  //printf("\033[32m[dot] reduceSum (%d) in \033[1m%12.6e(s)\n\033[m",N,alltime/1000.0);
  
  for(int i=1;i<nBitInN;i+=1){
    //printf("[vector_dot] dot[%d]=%f\n",i,dot[i]);
    h_dot[0]+=h_dot[i];
  }

  //printf("\033[32m[dot] reduceSum[%d]=%f\n\033[m",N,h_dot[0]);
  //printf("\n");   fflush(stdout);
  return h_dot[0];
#endif
}
