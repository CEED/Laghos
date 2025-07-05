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

#ifndef MFEM_ANALYTICAL_SURFACE
#define MFEM_ANALYTICAL_SURFACE

#include "mfem.hpp"
#include "general/forall.hpp"
#include "linalg/dtensor.hpp"
//#include "Line.hpp"
#include "Circle.hpp"

namespace mfem
{

  class AnalyticalSurface
  {
    
  protected:
    int geometryType;
    ParGridFunction alpha;
    AnalyticalGeometricShape *geometry;
    ParMesh *pmesh;
    L2_FECollection L2FEC_0;
    ParFiniteElementSpace L2_fes_0;
    
  public:
  AnalyticalSurface(int geometryType, ParMesh *pmesh);
  void SetupElementStatus();
  void ResetData();
  ParGridFunction& GetAlpha();
  AnalyticalGeometricShape& GetAnalyticalGeometricShape();
  void ComputeDistanceAndNormal(const Vector& x_ip, Vector& dist, Vector& tn) const;
  ~AnalyticalSurface();
  };
}
#endif // MFEM_LAGHOS
