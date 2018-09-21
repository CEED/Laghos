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
#ifndef MFEM_KERNELS_FORALL
#define MFEM_KERNELS_FORALL

// *****************************************************************************
#ifdef __NVCC__
#define kernel __global__
#define share __shared__
#define sync __syncthreads()
#define CUDA_BLOCK_SIZE 256
#define exeKernel(id,grid,blck,...) call[id]<<<grid,blck>>>(__VA_ARGS__)

#else // ***********************************************************************

#define sync
#define share
#define kernel
#define forall(i,max,body) for(int i=0;i<max;i++){body}
#define exeKernel(id,grid,blck,...) call[id](__VA_ARGS__)

#endif //__NVCC__

#endif // MFEM_KERNELS_FORALL
