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
#ifndef LAGHOS_KERNEL_DEFINITIONS
#define LAGHOS_KERNEL_DEFINITIONS

#include <math.h>
#include <stdbool.h>
#include <assert.h>

// Offsets *********************************************************************
#define   ijN(i,j,N) (i)+(N)*(j)
#define  ijkN(i,j,k,N) (i)+(N)*((j)+(N)*(k))
#define ijklN(i,j,k,l,N) (i)+(N)*((j)+(N)*((k)+(N)*(l)))

#define    ijNMt(i,j,N,M,t) (t)?((i)+(N)*(j)):((j)+(M)*(i))
#define    ijkNM(i,j,k,N,M) (i)+(N)*((j)+(M)*(k))
#define   _ijkNM(i,j,k,N,M) (j)+(N)*((k)+(M)*(i))
#define   ijklNM(i,j,k,l,N,M) (i)+(N)*((j)+(N)*((k)+(M)*(l)))
#define  _ijklNM(i,j,k,l,N,M)  (j)+(N)*((k)+(N)*((l)+(M)*(i)))
#define  ijklmNM(i,j,k,l,m,N,M) (i)+(N)*((j)+(N)*((k)+(M)*((l)+(M)*(m))))
#define _ijklmNM(i,j,k,l,m,N,M) (j)+(N)*((k)+(N)*((l)+(N)*((m)+(M)*(i))))
#define ijklmnNM(i,j,k,l,m,n,N,M) (i)+(N)*((j)+(N)*((k)+(M)*((l)+(M)*((m)+(M)*(n)))))

// RAJA ************************************************************************
#ifdef USE_RAJA
#include "RAJA/RAJA.hpp"
#define capture =
#ifdef USE_CUDA
#  define _device_ __device__
   const int CUDA_BLOCK_SIZE = 512;
#  define exec RAJA::cuda_exec<CUDA_BLOCK_SIZE>
#  define reduce RAJA::cuda_reduce<CUDA_BLOCK_SIZE>
#else
#  define _device_
#  define exec RAJA::seq_exec
#  define reduce RAJA::seq_reduce
#endif // USE_CUDA

// Reduce **********************************************************************
#define ReduceDecl(type,var,ini) \
  RAJA::Reduce ## type<reduce, RAJA::Real_type> var(ini);
// RAJA forall *****************************************************************
template <typename T>
void forall(RAJA::Index_type max, T&& body) {
  RAJA::forall<exec>(0,max,[=]_device_(RAJA::Index_type i) {
    body(i);
  });
}
#else // USE_RAJA
#define capture &
#define reduce
#define _device_
template <typename T>
void forall(int max, T&& body) { for(int i=0;i<max;i++) body(i); }
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
#endif // USE_RAJA

#endif // LAGHOS_KERNEL_DEFINITIONS
