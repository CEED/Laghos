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

#ifndef MFEM_LAGHOS_QUPDATE_ALLOC
#define MFEM_LAGHOS_QUPDATE_ALLOC

namespace mfem
{

// ***************************************************************************
template<class T> struct qmalloc {

   // *************************************************************************
   inline void* operator new (size_t n, bool lock_page = false) {
      dbp("+]\033[m");
#ifdef __NVCC__
      void *ptr = NULL;
      if (lock_page)
         checkCudaErrors(cuMemHostAlloc(&ptr, n*sizeof(T),
                                        CU_MEMHOSTALLOC_PORTABLE));
      else
      {
         if (n==0) { n=1; }
         checkCudaErrors(cuMemAlloc((CUdeviceptr*)&ptr, n*sizeof(T)));
      }
      return ptr;
#else
      return ::new T[n];
#endif
   }

   // ***************************************************************************
   inline void operator delete (void *ptr) {
      dbp("-]\033[m");
#ifdef __NVCC__
      cuMemFree((CUdeviceptr)ptr); // or cuMemFreeHost if page_locked was used
#else
      if (ptr) ::delete[] static_cast<T*>(ptr);
#endif // __NVCC__
      ptr = nullptr;
   }
};

} // mfem

#endif // MFEM_LAGHOS_QUPDATE_ALLOC
