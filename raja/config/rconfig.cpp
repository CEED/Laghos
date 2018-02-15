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
  
  // ***************************************************************************
  void rconfig::Setup(const int _mpi_rank,
                      const int _mpi_size,
                      const bool _cuda,
                      const bool _uvm,
                      const bool _share,
                      const bool _like_occa){
    mpi_rank=_mpi_rank;
    mpi_size=_mpi_size;
    uvm=_uvm;
    cuda=_cuda;
    share=_share;
    like_occa=_like_occa;
    cuda_aware=(MPIX_Query_cuda_support()==1)?true:false;

#if defined(__NVCC__) 
    int nb_gpu=0;
    
    { // Check if there is CUDA aware support
      if (mpi_rank==0)
        printf("\033[32m[laghos] MPI %s CUDA aware\033[m\n",
               cuda_aware?"\033[1mIS":"is \033[31;1mNOT\033[32m");
    }
        
    { 
      // Returns the number of devices with compute capability greater or equal to 2.0
      // Can be changed wuth CUDA_VISIBLE_DEVICES
      checkCudaErrors(cudaGetDeviceCount(&nb_gpu));
      if (mpi_rank==0)
        printf("\033[32m[laghos] CUDA device count: %i\033[m\n", nb_gpu);
    }

    CUdevice cuDevice;
    CUcontext cuContext;
    const bool mpsRunning = isNvidiaCudaMpsDaemonRunning();
    const int device = mpsRunning?0:(mpi_rank%nb_gpu);
    
    // Check if we have enough devices for all ranks
    assert(device<nb_gpu);

    if (mpi_rank==0)
      if (mpsRunning)
        printf("\033[32m[laghos] \033[32;1mMPS daemon\033[m\033[32m => \033[32;1m#%d\033[m\n", device);
      else
        printf("\033[32m[laghos] \033[31;1mNo MPS daemon\033[m\n");

   
    // Initializes the driver API
    // Must be called before any other function from the driver API
    // Currently, the Flags parameter must be 0. 
    const unsigned int Flags = 0; // parameter must be 0
    cuInit(Flags);
    
    // Returns properties for the selected device
    cuDeviceGet(&cuDevice,device);

    { // Check the compute capability of the device
      char name[128];
      int major, minor;
      cuDeviceGetName(name, 128, cuDevice);
      cuDeviceComputeCapability(&major, &minor, device);
      printf("\033[32m[laghos] Rank_%d => Device_%d (%s:sm_%d.%d)\033[m\n",
             mpi_rank, device, name, major, minor);
    }
    
    // Create our context
    cuCtxCreate(&cuContext, CU_CTX_SCHED_AUTO, cuDevice);
#endif
  }

  // ***************************************************************************
  bool rconfig::IAmAlone() {
    //#warning Faking MPI    return false;
    if (like_occa) return false;
    return mpi_size==1;
  }

  // ***************************************************************************
  bool rconfig::NeedUpdate(const int sequence) {
    //if (like_occa) return false;
    if (like_occa) return true;
    assert(sequence==0);
    return (sequence!=0);
  }

  // ***************************************************************************
  bool rconfig::DoHostConformingProlongationOperator() {
    if (like_occa) return true;
    return (cuda)?false:true;
  }
  
} // namespace mfem
