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
#include <mpi-ext.h>
#include <unistd.h>

namespace mfem
{


// ***************************************************************************
bool isNvidiaCudaMpsDaemonRunning(void)
{
   const char *command="pidof -s nvidia-cuda-mps-control>/dev/null";
   return system(command)==0;
}

// ***************************************************************************
void computeCapabilityOfTheDevice(const int mpi_rank,
#if defined(RAJA_ENABLE_CUDA)
                                  const CUdevice cuDevice,
#elif defined(RAJA_ENABLE_HIP)
                                  const hipDevice_t hipDevice,
#endif
                                  const int device)
{
   char name[128];
   int major, minor;
#if defined(RAJA_ENABLE_CUDA)
   cuDeviceGetName(name, 128, cuDevice);
   cuDeviceComputeCapability(&major, &minor, device);
#elif defined(RAJA_ENABLE_HIP)
   hipDeviceGetName(name, 128, hipDevice);
   hipDeviceComputeCapability(&major, &minor, device);
#endif
   printf("\033[32m[laghos] Rank_%d => Device_%d (%s:sm_%d.%d)\033[m\n",
          mpi_rank, device, name, major, minor);
}

// ***************************************************************************
// *   Setup
// ***************************************************************************
void rconfig::Setup(const int _mpi_rank,
                    const int _mpi_size,
                    const bool _cuda,
                    const bool _hip,
                    const bool _uvm,
                    const bool _aware,
                    const bool _hcpo,
                    const bool _sync)
{
   mpi_rank=_mpi_rank;
   mpi_size=_mpi_size;

   // Get the number of devices with compute capability greater or equal to 2.0
   // Can be changed wuth CUDA_VISIBLE_DEVICES
#if defined(RAJA_ENABLE_CUDA)
   cudaGetDeviceCount(&gpu_count);
#elif defined(RAJA_ENABLE_HIP)
   hipGetDeviceCount(&gpu_count);
#endif
   cuda=_cuda;
   hip=_hip;
   uvm=_uvm;
   aware=_aware;
   hcpo=_hcpo;
   sync=_sync;

   // Check for Enforced Kernel Synchronization
   if (Sync() && Root())
   {
      printf("\033[32m[laghos] \033[31;1mEnforced Kernel Synchronization!\033[m\n");
   }

   // Check if MPI is CUDA/HIP aware
   if (Root())
      printf("\033[32m[laghos] MPI %s GPU aware\033[m\n",
             aware?"\033[1mIS":"is \033[31;1mNOT\033[32m");

   if (Root())
   {
      printf("\033[32m[laghos] GPU device count: %i\033[m\n", gpu_count);
   }

   // Initializes the driver API
   // Must be called before any other function from the driver API
   // Currently, the Flags parameter must be 0.
   const unsigned int Flags = 0; // parameter must be 0

   // Returns properties for the selected device
   const int device = Mps()?0:(mpi_rank%gpu_count);
   // Check if we have enough devices for all ranks
   assert(device<gpu_count);

   // Get a handle to our compute device
#if defined(RAJA_ENABLE_CUDA)
   cuInit(Flags);
   cuDeviceGet(&cuDevice,device);
   computeCapabilityOfTheDevice(mpi_rank,cuDevice,device);

   // Get the properties of the device
   struct cudaDeviceProp properties;
   cudaGetDeviceProperties(&properties, device);
   maxXGridSize=properties.maxGridSize[0];
   maxXThreadsDim=properties.maxThreadsDim[0];

   // Create our context
   cuCtxCreate(&cuContext, CU_CTX_SCHED_AUTO, cuDevice);
   hStream=new CUstream;
   cuStreamCreate(hStream, CU_STREAM_DEFAULT);

#elif defined(RAJA_ENABLE_HIP)
   hipInit(Flags);
   hipDeviceGet(&hipDevice,device);
   computeCapabilityOfTheDevice(mpi_rank,hipDevice,device);

   // Get the properties of the device
   struct hipDeviceProp_t properties;
   hipGetDeviceProperties(&properties, device);
   maxXGridSize=properties.maxGridSize[0];
   maxXThreadsDim=properties.maxThreadsDim[0];

   // Create our context
   hStream=new hipStream_t;
   hipStreamCreate(hStream);
#endif
}

// ***************************************************************************
bool rconfig::IAmAlone()
{
   return mpi_size==1;
}

// ***************************************************************************
bool rconfig::GeomNeedsUpdate(const int sequence)
{
   assert(sequence==0);
   return (sequence!=0);
}

// ***************************************************************************
bool rconfig::DoHostConformingProlongationOperator()
{
   return ((Cuda()||Hip()))?hcpo:true;
}

} // namespace mfem
