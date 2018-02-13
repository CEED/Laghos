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

namespace mfem {

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
#if defined(__NVCC__) 
    CUdevice cuDevice;
    CUcontext cuContext;
#warning device tied to 0
    const int device = 0;//mpi_rank%mpi_size; // We still use the same device for now // mpi_rank;
    
    // Initializes the driver API
    // Must be called before any other function from the driver API
    // Currently, the Flags parameter must be 0. 
    const unsigned int Flags = 0; // parameter must be 0
    cuInit(Flags);
    
    // Returns properties for the selected device
    cuDeviceGet(&cuDevice,device);

    { // Check if we have enough devices for all ranks
      int gpu_n;
      checkCudaErrors(cudaGetDeviceCount(&gpu_n));
      if (mpi_rank==0)
        printf("\033[32m[laghos] CUDA device count: %i\033[m\n", gpu_n);
      //#warning NO assert(gpu_n>=mpi_size)
      //assert(gpu_n>=mpi_size);
    }
    
    //MPI_Barrier(MPI_COMM_WORLD);

    { // Check the compute capability of the device
      char name[128];
      int major, minor;
      cuDeviceGetName(name, 128, cuDevice);
      cuDeviceComputeCapability(&major, &minor, device);
      printf("\033[32m[laghos] Rank_%d => Device_%d (%s:sm_%d.%d)\033[m\n",
             mpi_rank, device, name, major, minor);
    }
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
