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
#ifndef LAGHOS_RAJA
#define LAGHOS_RAJA

// stdincs *********************************************************************
#include <stdio.h>
#include <stdarg.h>
#include <assert.h>

// *****************************************************************************
#include "RAJA/RAJA.hpp"

// MFEM/fem  *******************************************************************
#include "fem/gridfunc.hpp"
#include "general/communication.hpp"
#include "fem/pfespace.hpp"

// LAGHOS/raja/config **********************************************************
#include "config/rconfig.hpp"

// LAGHOS/raja/general *********************************************************
#include "general/rmemcpy.hpp"
#include "general/rmalloc.hpp"
#include "general/rarray.hpp"
#include "general/rtable.hpp"
#include "general/rcommd.hpp"

// LAGHOS/raja/linalg **********************************************************
#include "linalg/rvector.hpp"
#include "linalg/roperator.hpp"
#include "linalg/rode.hpp"
#include "linalg/rsolvers.hpp"

// LAGHOS/raja/kernels *********************************************************
#include "kernels/include/kernels.hpp"

// LAGHOS/raja/fem *************************************************************
#include "fem/rconform.hpp"
#include "fem/rprolong.hpp"
#include "fem/rrestrict.hpp"
#include "fem/rfespace.hpp"
#include "fem/rbilinearform.hpp"
#include "fem/rgridfunc.hpp"
#include "fem/rbilininteg.hpp"

#endif // LAGHOS_RAJA

