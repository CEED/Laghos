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
#ifndef LAGHOS_HIP_KERNELS_FORALL
#define LAGHOS_HIP_KERNELS_FORALL

// *****************************************************************************
#define HIP_BLOCK_SIZE 256

#define ELEMENT_BATCH 10
#define M2_ELEMENT_BATCH 32

// *****************************************************************************
#define kernel __global__
#define share __shared__
#define sync __syncthreads();
// *****************************************************************************
#define hipKer(name,end,...)                                             \
   hipLaunchKernelGGL((name ## 0), dim3((end+HIP_BLOCK_SIZE-1)/HIP_BLOCK_SIZE),   \
               dim3(HIP_BLOCK_SIZE), 0, 0, end,__VA_ARGS__)
#define hipKerGBS(name,grid,block,end,...) hipLaunchKernelGGL((name ## 0), dim3(grid), dim3(block), 0, 0, end,__VA_ARGS__)
#define call0(id,grid,blck,...) hipLaunchKernelGGL((call[id]), dim3(grid), dim3(blck), 0, 0, __VA_ARGS__)

#endif // LAGHOS_HIP_KERNELS_FORALL
