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
/*void reduceSum(int size,
               const double *d_i1data,
               const double *d_i2data,
               double *d_odata);

void reduceSumN(int size,
               const double *d_i1data,
               const double *d_i2data,
               double *d_odata);
*/
double vector_dot(const int N,
                  const double* __restrict vec1,
                  const double* __restrict vec2) {
//#if defined(__RAJA__) || defined(__NVCC__)
//#warning ReduceDecl DOT
  ReduceDecl(Sum,dot,0.0);
  ReduceForall(i,N,dot += vec1[i]*vec2[i];);
  return dot;
/*#else
#warning pure CUDA dot
  static double *dot;
  if (!dot){
    //printf("cudaMallocManaged(dot)\n");
    //#warning should be size of block
    cudaMallocManaged(&dot, 1*sizeof(double), cudaMemAttachGlobal);
    cudaDeviceSynchronize();
  }
  dot[0]=0.0;
  
  #warning HC N power of 2
  //printf("N=%d => %d",N,16);
  reduceSum(16,vec1,vec2,&dot[0]);
  
  //printf("[vector_dot] dot=%f\n",dot[0]);
  return dot[0];
#endif
  */
}
