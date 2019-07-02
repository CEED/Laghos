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
#if defined(RAJA_ENABLE_CUDA)
#define CUDA_BLOCK_SIZE 256

#define cu_device __device__
#define cu_exec RAJA::cuda_exec<CUDA_BLOCK_SIZE>
#define cu_reduce RAJA::cuda_reduce<CUDA_BLOCK_SIZE>

#define sq_device __host__
#define sq_exec RAJA::seq_exec
#define sq_reduce RAJA::seq_reduce

#define ReduceDecl(type,var,ini) \
  RAJA::Reduce ## type<sq_reduce, RAJA::Real_type> var(ini);
#define ReduceForall(i,max,body) \
  RAJA::forall<sq_exec>(RAJA::RangeSegment(0, max),[=]sq_device(RAJA::Index_type i) {body});

#define forall(i,max,body)                                              \
   if (mfem::rconfig::Get().Cuda())                                     \
      RAJA::forall<cu_exec>(RAJA::RangeSegment(0, max),[=]cu_device(RAJA::Index_type i) {body}); \
   else                                                                 \
      RAJA::forall<sq_exec>(RAJA::RangeSegment(0, max),[=]sq_device(RAJA::Index_type i) {body});

#elif defined(RAJA_ENABLE_HIP)
#define HIP_BLOCK_SIZE 256

#define hip_device __device__
#define hip_exec RAJA::hip_exec<HIP_BLOCK_SIZE>
#define hip_reduce RAJA::hip_reduce<HIP_BLOCK_SIZE>

#define sq_device __host__
#define sq_exec RAJA::seq_exec
#define sq_reduce RAJA::seq_reduce

#define ReduceDecl(type,var,ini) \
  RAJA::Reduce ## type<sq_reduce, RAJA::Real_type> var(ini);
#define ReduceForall(i,max,body) \
  RAJA::forall<sq_exec>(RAJA::RangeSegment(0, max),[=]sq_device(RAJA::Index_type i) {body});

#define forall(i,max,body)                                              \
   if (mfem::rconfig::Get().Hip())                                     \
      RAJA::forall<hip_exec>(RAJA::RangeSegment(0, max),[=]hip_device(RAJA::Index_type i) {body}); \
   else                                                                 \
      RAJA::forall<sq_exec>(RAJA::RangeSegment(0, max),[=]sq_device(RAJA::Index_type i) {body});

#endif

#endif // LAGHOS_RAJA_KERNELS_FORALL
