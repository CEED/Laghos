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
void reduceMinV(int size, const double *d_idata, double *d_odata);
void reduceMinN(int size, const double *d_idata, double *d_odata);

double vector_min(const int N,
                  const double* __restrict vec) {
#if defined(__RAJA__) || (!defined(__RAJA__)&&!defined(__NVCC__))
#warning ReduceDecl MIN
  ReduceDecl/*Raja*/(Min,red,vec[0]);
  forall/*Raja*/(i,N,red.min(vec[i]););
#else
#warning pure CUDA min
  static double *red=NULL;
  if (!red) {
    printf("cudaMallocManaged(red)\n");
    //#warning should be size of block
    cudaMallocManaged(&red, 1*sizeof(double), cudaMemAttachGlobal);
    cudaDeviceSynchronize();
  }
  red[0]=vec[0];
  
  #warning HC N power of 2
  printf("N=%d => %d",N,64);
  reduceMinV(64,vec,&red[0]);
#endif
  //printf("[vector_min] red=%f\n",red[0]);
  return red;

  
/*
  assert(isPowerOfTwo(N)==0);
  static double *red=NULL;
  if (!red) {
    printf("cudaMallocManaged(red)\n");
    #warning should be size of block
    cudaMallocManaged(&red, 1*sizeof(double), cudaMemAttachGlobal);
    cudaDeviceSynchronize();
  }
  *red=vec[0];
  reduceMinV(N,vec,red);
#endif
  //printf("[vector_min] red=%f\n",red);
  return red[0];
*/
}
