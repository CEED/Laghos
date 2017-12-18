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
#ifndef LAGHOS_RAJA_MALLOC
#define LAGHOS_RAJA_MALLOC

// *****************************************************************************
#ifdef RAJA_USE_SIMPOOL
#ifdef __NVCC__
#define MFEM_USE_CUDAUM __NVCC__
#endif // __NVCC__
#include "simpool.hpp"
typedef DynamicPoolAllocator<Allocator> PoolType;
#endif // RAJA_USE_SIMPOOL

// ***************************************************************************
extern bool is_managed;

namespace mfem {

// *****************************************************************************
template<class T> struct rmalloc{
  // ***************************************************************************
  void* _new(size_t n) {
    rdbg("+]\033[m");
#ifdef RAJA_USE_SIMPOOL
    return PoolType::getInstance().allocate(n*sizeof(T));
#else
    if (!is_managed) return new T[n];
#ifdef __NVCC__
    void *ptr;
    cudaMallocManaged(&ptr, n*sizeof(T), cudaMemAttachGlobal);
    return ptr;
#endif // __NVCC__
    // We come here when the user requests a manager,
    // but has compiled the code without NVCC
    assert(false);
#endif // RAJA_USE_SIMPOOL
  }
  // ***************************************************************************
  void _delete(void *ptr) {
    rdbg("-]\033[m");
#ifdef RAJA_USE_SIMPOOL
    if (ptr) PoolType::getInstance().deallocate(ptr);
#else
    if (!is_managed) delete[] static_cast<T*>(ptr);
#ifdef __NVCC__
    else cudaFree(ptr);
#endif // __NVCC__
    ptr = nullptr;
#endif // RAJA_USE_SIMPOOL
  }
};

} // mfem

#endif // LAGHOS_RAJA_MALLOC
