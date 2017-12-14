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

// RAJA ************************************************************************
#ifdef __NVCC__
#include "cuda.h"
#include "RAJA/RAJA.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#include "RAJA/index/RangeSegment.hpp"
#define ReduceCapture =
#ifdef USE_CUDA
#warning RAJA CUDA
   const int CUDA_BLOCK_SIZE = 512;
#  define device __device__
#  define exec RAJA::cuda_exec<CUDA_BLOCK_SIZE>
#  define cu_exec RAJA::cuda_exec<CUDA_BLOCK_SIZE>
#  define reduce RAJA::cuda_reduce<CUDA_BLOCK_SIZE>
#else
#warning RAJA SEQ
#  define device __host__
#  define exec RAJA::seq_exec
#  define reduce RAJA::seq_reduce
#endif // USE_CUDA

// RAJA reduce *****************************************************************
#define ReduceDecl(type,var,ini) \
  RAJA::Reduce ## type<reduce, RAJA::Real_type> var(ini);

// RAJA forall *****************************************************************
template <typename T>
void forall(RAJA::Index_type max, T&& body) {
  RAJA::forall<exec>(0,max,[=]device(RAJA::Index_type i) {body(i);});
}

#define FORALL(i,end,body)                                              \
  RAJA::forall<RAJA::seq_exec>(0,end,[=](RAJA::Index_type i) {body});   \
  RAJA::forall<cu_exec>(0,end,[=]__device__(RAJA::Index_type i) {body});

/*
// https://stackoverflow.com/questions/44868369/how-to-immediately-invoke-a-c-lambda
template<class Callable>
auto operator+(decltype(invoke) const&, Callable c) -> decltype(c()) {
    return c();
}
*/

#else // __NVCC__

#define device
#define ReduceCapture &

// STD reduce ******************************************************************
class ReduceSum{
private:
  double s;
public:
  inline ReduceSum(double d):s(d){}
  inline operator double() { return s; }
  inline ReduceSum& operator +=(const double d) { return *this = (s+d); }
};
class ReduceMin{
private:
  double m;
public:
  inline ReduceMin(double d):m(d){}
  inline operator double() { return m; }
  inline ReduceMin& min(const double d) { return *this=(m<d)?m:d; }
};
#define ReduceDecl(type,var,ini) Reduce##type var(ini);

// STD forall ******************************************************************
template <typename T>
void forall(int max, T&& body) { for(int i=0;i<max;i++) body(i); }
#define FORALL(i,max,body) for(int i=0;i<max;i++){body(i);}

#endif // __NVCC__

#endif // LAGHOS_RAJA_KERNELS_FORALL
