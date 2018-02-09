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
  bool rconfig::setupDevice(MPI_Session &mpi){
#if defined(__NVCC__)
    CUdevice cuDevice;
    CUcontext cuContext;
    const int mpi_rank = mpi.WorldRank();
    const int mpi_size = mpi.WorldSize();
    const int device = mpi_rank;
    
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
      if (mpi.Root())
        printf("\033[32m[laghos] CUDA device count: %i\033[m\n", gpu_n);
      assert(gpu_n>=mpi_size);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

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
    return true;
  }

  // ***************************************************************************
  bool rconfig::IAmAlone() {
    if (like_occa) return false;
    return world_size==1;
  }

  // ***************************************************************************
  bool rconfig::NeedUpdate(Mesh &mesh) {
    //if (like_occa) return true;
    assert(mesh.GetSequence()==0);
    return (mesh.GetSequence()!=0);
  }

  // ***************************************************************************
  bool rconfig::DoHostConformingProlongationOperator() {
    //if (like_occa) return false;
    //return (cuda)?false:true;
    return false; // On force pour l'instant
  }

} // namespace mfem
