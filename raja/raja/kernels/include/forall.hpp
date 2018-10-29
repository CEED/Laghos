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
#ifndef LAGHOS_RAJA_KERNELS_FORALL
#define LAGHOS_RAJA_KERNELS_FORALL

// *****************************************************************************
#define ELEMENT_BATCH 10
#define M2_ELEMENT_BATCH 32

#define A2_ELEMENT_BATCH 1
#define A2_QUAD_BATCH 1

#define A3_ELEMENT_BATCH 1
#define A3_QUAD_BATCH 1

// *****************************************************************************
#define sync
#define share
#define kernel

#define exclusive_inc
#define exclusive_decl
#define exclusive_reset
#define exclusive_set(name,idx) name[idx]
#define exclusive(type,name,size) type name[size]

const int CUDA_BLOCK_SIZE = 256;

#define cu_device __device__
#define cu_exec RAJA::cuda_exec<CUDA_BLOCK_SIZE>
#define cu_reduce RAJA::cuda_reduce<CUDA_BLOCK_SIZE>

#define sq_device __host__
#define sq_exec RAJA::seq_exec
#define sq_reduce RAJA::seq_reduce

#define ReduceDecl(type,var,ini) \
  RAJA::Reduce ## type<sq_reduce, RAJA::Real_type> var(ini);
#define ReduceForall(i,max,body) \
  RAJA::forall<sq_exec>(0,max,[=]sq_device(RAJA::Index_type i) {body});

#define forall(i,max,body)                                              \
  if (mfem::rconfig::Get().Cuda())                                      \
    RAJA::forall<cu_exec>(0,max,[=]cu_device(RAJA::Index_type i) {body}); \
  else                                                                  \
    RAJA::forall<sq_exec>(0,max,[=]sq_device(RAJA::Index_type i) {body});

#define forallS(i,max,step,body) {assert(false);forall(i,max,body)}
#define call0(name,id,grid,blck,...) call[id](__VA_ARGS__)
#define cuKerGBS(name,grid,block,end,...) name ## 0(end,__VA_ARGS__)
#define cuKer(name,end,...) name ## 0(end,__VA_ARGS__)

#endif // LAGHOS_RAJA_KERNELS_FORALL
