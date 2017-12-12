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
#include "cuda.h"

// *****************************************************************************
template <typename T,bool = false> class rmanaged;

// CPU *************************************************************************
template<typename T> class rmanaged<T,false> {
public:
  void* operator new(size_t n) {
    //assert(false);
    printf("+]\033[m");fflush(stdout);
    return new T[n];
  }
  void operator delete(void *ptr) {
    //assert(false);
    //if (!ptr) return;
    //if (ptr==NULL) return;
    printf("-]\033[m");fflush(stdout);
    delete[] static_cast<T*>(ptr);
    ptr = nullptr;
  }
};

// GPU *************************************************************************
#ifdef USE_CUDA
template<typename T> class rmanaged<T,true> {
public:
  void* operator new(size_t n) {
    void *ptr;
    cudaMallocManaged(&ptr, n*zsizeof(T),cudaMemAttachSingle);
    cudaDeviceSynchronize();
    return ptr;
  }

  void operator delete(void *ptr) {
    cudaDeviceSynchronize();
    cudaFree(ptr);
    ptr = nullptr;
  }
};
#endif
