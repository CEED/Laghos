// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See file LICENSE for details.
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

#include "occa.hpp"

namespace mfem {
  namespace hydrodynamics {
    enum ProblemOption {
      vortex      = 0,
      sedov       = 1,
      shockTube   = 2,
      triplePoint = 3
    };

    enum ODESolverOption {
      ForwardEuler = 1,
      RK2          = 2,
      RK3          = 3,
      RK4          = 4,
      RK6          = 6
    };

    static occa::properties GetProblemProperties() {
      occa::properties props;
      props["defines/VORTEX_PROBLEM"] = vortex;
      props["defines/SEDOV_PROBLEM"] = sedov;
      props["defines/SHOCK_TUBE_PROBLEM"] = shockTube;
      props["defines/TRIPLE_POINT_PROBLEM"] = triplePoint;
      return props;
    }
  }
}
