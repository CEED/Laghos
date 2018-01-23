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

// ***************************************************************************
extern bool is_managed;

namespace mfem {

// *****************************************************************************
template<class T> struct rmalloc{
  // ***************************************************************************
  static void* HoDNew(size_t n) {
#ifdef __NVCC__
    void *ptr;
    //cudaMalloc(&ptr, n*sizeof(T));
    cuMemAlloc((CUdeviceptr*)&ptr, n*sizeof(T));
    //cudaDeviceSynchronize();
    return ptr;
#else // __NVCC__
    return new T[n];
#endif 
  }

  // ***************************************************************************
  void* _new(size_t n) {
    //rdbg("+]\033[m");
    if (!is_managed) return new T[n];
#ifdef __NVCC__
    void *ptr;
    //cudaMalloc(&ptr, n*sizeof(T));
    cuMemAlloc((CUdeviceptr*)&ptr, n*sizeof(T));
    //cudaDeviceSynchronize();
    return ptr;
#endif // __NVCC__
    // We come here when the user requests a manager,
    // but has compiled the code without NVCC
    assert(false);
  }
  
  // ***************************************************************************
  void _delete(void *ptr) {
    //rdbg("-]\033[m");
    if (!is_managed) delete[] static_cast<T*>(ptr);
#ifdef __NVCC__
    else {
      //cudaDeviceSynchronize();
      //cudaFree(ptr);
      cuMemFree((CUdeviceptr)ptr);
    }
#endif // __NVCC__
    ptr = nullptr;
  }
};

} // mfem

#endif // LAGHOS_RAJA_MALLOC
