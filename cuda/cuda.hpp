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
#ifndef LAGHOS_CUDA
#define LAGHOS_CUDA

// stdincs *********************************************************************
#include <stdio.h>
#include <stdarg.h>
#include <assert.h>

// CUDA ************************************************************************
#include <cuda.h>

// MFEM/fem  *******************************************************************
#include "fem/gridfunc.hpp"
#include "general/communication.hpp"
#include "fem/pfespace.hpp"

// LAGHOS/cuda/config **********************************************************
#include "./config/config.hpp"

// LAGHOS/cuda/general *********************************************************
#include "./general/memcpy.hpp"
#include "./general/malloc.hpp"
#include "./general/array.hpp"
#include "./general/table.hpp"
#include "./general/commd.hpp"

// LAGHOS/cuda/linalg **********************************************************
#include "./linalg/vector.hpp"
#include "./linalg/operator.hpp"
#include "./linalg/ode.hpp"
#include "./linalg/solvers.hpp"

// LAGHOS/cuda/kernels *********************************************************
#include "./kernels/include/kernels.hpp"

// LAGHOS/cuda/fem *************************************************************
#include "./fem/conform.hpp"
#include "./fem/prolong.hpp"
#include "./fem/restrict.hpp"
#include "./fem/fespace.hpp"
#include "./fem/bilinearform.hpp"
#include "./fem/cuGridfunc.hpp"
#include "./fem/bilininteg.hpp"

#endif // LAGHOS_CUDA

