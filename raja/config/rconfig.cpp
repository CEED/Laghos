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

namespace mfem {

  
  // ***************************************************************************
  bool isNvidiaCudaMpsDaemonRunning(void){
    const char *command="pidof -s nvidia-cuda-mps-control>/dev/null";
    return system(command)==0;
  }

#ifdef __NVCC__
  // ***************************************************************************
  void computeCapabilityOfTheDevice(const int mpi_rank,
                                    const CUdevice cuDevice,
                                    const int device){
    char name[128];
    int major, minor;
    cuDeviceGetName(name, 128, cuDevice);
    cuDeviceComputeCapability(&major, &minor, device);
    printf("\033[32m[laghos] Rank_%d => Device_%d (%s:sm_%d.%d)\033[m\n",
           mpi_rank, device, name, major, minor);
  }
#endif

  // ***************************************************************************
  // *   Setup
  // ***************************************************************************
  void rconfig::Setup(const int _mpi_rank,
                      const int _mpi_size,
                      const bool _cuda,
                      const bool _uvm,
                      const bool _share,
                      const bool _occa,
                      const bool _sync,
                      const bool _dot,
                      const int rs_levels){
    mpi_rank=_mpi_rank;
    mpi_size=_mpi_size;
    aware=(MPIX_Query_cuda_support()==1)?true:false;
    mps = isNvidiaCudaMpsDaemonRunning();
#ifdef __NVCC__
    // Get the number of devices with compute capability greater or equal to 2.0
    // Can be changed wuth CUDA_VISIBLE_DEVICES
    cuCheck(cudaGetDeviceCount(&gpu_count));
#endif
    cuda=_cuda;
    uvm=_uvm;
    share=_share;
    occa=_occa;
    sync=_sync;

#if defined(__NVCC__) 
    
    // Check for Enforced Kernel Synchronization
    if (Sync() && Root())
      printf("\033[32m[laghos] \033[31;1mEnforced Kernel Synchronization!\033[m\n");
    
    // Check if MPI is CUDA aware
    if (Root())
      printf("\033[32m[laghos] MPI %s CUDA aware\033[m\n",
             aware?"\033[1mIS":"is \033[31;1mNOT\033[32m");
            
    if (Root())
      printf("\033[32m[laghos] CUDA device count: %i\033[m\n", gpu_count);   

    if (Mps() && Root())
      printf("\033[32m[laghos] \033[32;1mMPS daemon\033[m\033[m\n");
    if (!Mps() && Root())
      printf("\033[32m[laghos] \033[31;1mNo MPS daemon\033[m\n");
   
    // Initializes the driver API
    // Must be called before any other function from the driver API
    // Currently, the Flags parameter must be 0. 
    const unsigned int Flags = 0; // parameter must be 0
    cuInit(Flags);
    
    // Returns properties for the selected device
    const int device = Mps()?0:(mpi_rank%gpu_count);
    // Check if we have enough devices for all ranks
    assert(device<gpu_count);
    
    // Get a handle to our compute device
    cuDeviceGet(&cuDevice,device);
    computeCapabilityOfTheDevice(mpi_rank,cuDevice,device);
    
    // Create our context
    cuCtxCreate(&cuContext, CU_CTX_SCHED_AUTO, cuDevice);
#endif
    if (_dot){
      dotTest(rs_levels);
      MPI_Finalize();
      exit(0);
    }
  }

  // ***************************************************************************
  bool rconfig::IAmAlone() {
    if (Occa()) return false;
    return mpi_size==1;
  }

  // ***************************************************************************
  bool rconfig::GeomNeedsUpdate(const int sequence) {
    if (Occa()) return true;
    assert(sequence==0);
    return (sequence!=0);
  }

  // ***************************************************************************
  bool rconfig::DoHostConformingProlongationOperator() {
    if (Occa()) return true;
    return (Cuda())?false:true;
  }
  
} // namespace mfem
