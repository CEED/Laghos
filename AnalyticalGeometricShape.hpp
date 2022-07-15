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

#ifndef MFEM_ANALYTICAL_GEOMETRIC_SHAPE
#define MFEM_ANALYTICAL_GEOMETRIC_SHAPE

#include "mfem.hpp"

namespace mfem{

  class AnalyticalGeometricShape{

  protected:
    ParFiniteElementSpace &H1;
    ParFiniteElementSpace &L2;
    ParMesh *pmesh;
    
public:
    /// Element type related to shifted boundaries (not interfaces).
   /// For more than 1 level-set, we set the marker to CUT+level_set_index
   /// to discern between different level-sets.
    enum SBElementType {OUTSIDE = 0, INSIDE = 1, CUT = 2};

  AnalyticalGeometricShape(ParFiniteElementSpace &h1_fes, ParFiniteElementSpace &l2_fes);
  virtual void SetupElementStatus(Array<int> &elemStatus, Array<int> &ess_inactive) = 0;
    virtual void SetupFaceTags(Array<int> &elemStatus, Array<int> &faceTags, Array<int> &initialBoundaryFaceTags, int maxBTag) = 0;
  virtual void ComputeDistanceAndNormalAtQuadraturePoints(const IntegrationRule &b_ir, Array<int> &elemStatus, Array<int> &faceTags, DenseMatrix &quadratureDistance, DenseMatrix &quadratureTrueNormal) = 0;

  virtual  ~AnalyticalGeometricShape();
  };
}
#endif // MFEM_LAGHOS
