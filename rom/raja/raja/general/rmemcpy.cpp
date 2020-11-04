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
#include "../raja.hpp"

namespace mfem
{

// *************************************************************************
void* rmemcpy::rHtoH(void *dest, const void *src, std::size_t bytes,
                     const bool async)
{
   if (bytes==0) { return dest; }
   assert(src); assert(dest);
   std::memcpy(dest,src,bytes);
   return dest;
}

// *************************************************************************
void* rmemcpy::rHtoD(void *dest, const void *src, std::size_t bytes,
                     const bool async)
{
   if (bytes==0) { return dest; }
   assert(src); assert(dest);
#if defined(RAJA_ENABLE_CUDA)
   if (!rconfig::Get().Cuda()) { return std::memcpy(dest,src,bytes); }
   if (!rconfig::Get().Uvm())
   {
      cuMemcpyHtoD((CUdeviceptr)dest,src,bytes);
   }
   else { cuMemcpy((CUdeviceptr)dest,(CUdeviceptr)src,bytes); }
#elif defined(RAJA_ENABLE_HIP)
   if (!rconfig::Get().Hip()) { return std::memcpy(dest,src,bytes); }

   hipMemcpy(dest,src,bytes,hipMemcpyHostToDevice);
#endif
   return dest;
}

// ***************************************************************************
void* rmemcpy::rDtoH(void *dest, const void *src, std::size_t bytes,
                     const bool async)
{
   if (bytes==0) { return dest; }
   assert(src); assert(dest);
#if defined(RAJA_ENABLE_CUDA)
   if (!rconfig::Get().Cuda()) { return std::memcpy(dest,src,bytes); }
   if (!rconfig::Get().Uvm())
   {
      cuMemcpyDtoH(dest,(CUdeviceptr)src,bytes);
   }
   else { cuMemcpy((CUdeviceptr)dest,(CUdeviceptr)src,bytes); }
#elif defined(RAJA_ENABLE_HIP)
   if (!rconfig::Get().Hip()) { return std::memcpy(dest,src,bytes); }

   hipMemcpy(dest,src,bytes,hipMemcpyDeviceToHost);
#endif
   return dest;
}

// ***************************************************************************
void* rmemcpy::rDtoD(void *dest, const void *src, std::size_t bytes,
                     const bool async)
{
   if (bytes==0) { return dest; }
   assert(src); assert(dest);
#if defined(RAJA_ENABLE_CUDA)
   if (!rconfig::Get().Cuda()) { return std::memcpy(dest,src,bytes); }
   if (!rconfig::Get().Uvm())
   {
      if (!async)
      {
         cuMemcpyDtoD((CUdeviceptr)dest,(CUdeviceptr)src,bytes);
      }
      else
      {
         const CUstream s = *rconfig::Get().Stream();
         cuMemcpyDtoDAsync((CUdeviceptr)dest,(CUdeviceptr)src,bytes,s);
      }
   }
   else { cuMemcpy((CUdeviceptr)dest,(CUdeviceptr)src,bytes); }
#elif defined(RAJA_ENABLE_HIP)
   if (!rconfig::Get().Hip()) { return std::memcpy(dest,src,bytes); }

   if (!async)
   {
      hipMemcpy(dest,src,bytes,hipMemcpyDeviceToDevice);
   }
   else
   {
      const hipStream_t s = *rconfig::Get().Stream();
      hipMemcpyAsync(dest,src,bytes,hipMemcpyDeviceToDevice,s);
   }
#endif
   return dest;
}

} // mfem
