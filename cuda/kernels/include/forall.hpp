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
#ifndef LAGHOS_CUDA_KERNELS_FORALL
#define LAGHOS_CUDA_KERNELS_FORALL

// *****************************************************************************
#define ELEMENT_BATCH 10
#define M2_ELEMENT_BATCH 32

#define A2_ELEMENT_BATCH 1
#define A2_QUAD_BATCH 1

#define A3_ELEMENT_BATCH 1
#define A3_QUAD_BATCH 1

// *****************************************************************************
#define kernel __global__
#define share __shared__
#define sync __syncthreads();
#define exclusive_inc
#define exclusive_decl
#define exclusive_reset
#define exclusive_set(name,idx) name[idx]
#define exclusive(type,name,size) type name[size]
const int CUDA_BLOCK_SIZE = 256;
#define cuKer(name,end,...) name ## 0<<<((end+256-1)/256),256>>>(end,__VA_ARGS__)
#define cuLaunchKer(name,args) {                                      \
    cuLaunchKernel(name ## 0,                                         \
                   ((end+256-1)/256),1,1,                             \
                   256,1,1,                                           \
                   0,0,                                               \
                   args);                                             \
      }
#define cuKerGBS(name,grid,block,end,...) name ## 0<<<grid,block>>>(end,__VA_ARGS__)
#define call0p(name,id,grid,blck,...)                               \
  printf("\033[32;1m[call0] name=%s grid:%d, block:%d\033[m\n",#name,grid,blck); \
  call[id]<<<grid,blck>>>(__VA_ARGS__)
#define call0(name,id,grid,blck,...) call[id]<<<grid,blck>>>(__VA_ARGS__)
#define ReduceDecl(type,var,ini) double var=ini;
#define ReduceForall(i,max,body) 

#endif // LAGHOS_CUDA_KERNELS_FORALL
