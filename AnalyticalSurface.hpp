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
#include "Line.hpp"

namespace mfem
{

  class AnalyticalSurface
  {
    
  protected:
    int geometryType;
    ParFiniteElementSpace &H1;
    ParFiniteElementSpace &L2;
    
    ParMesh *pmesh;
    // Integration rule for all assemblies.
    const IntegrationRule &b_ir;
  
  //    Array<int> nodalStatus;
    Array<int> elementalStatus;
    Array<int> faceTags;
    Array<int> initialBoundaryFaceTags;
    Array<int> initialElementTags;
    Array<int> ess_edofs;
    int maxBoundaryTag;

  //    Array<int> isFaceIntersectedStatus;
  
  //  Vector nodalDistance;
    //  Vector nodalTrueNormal;
  
    DenseMatrix quadratureDistance;
    DenseMatrix quadratureTrueNormal;
    DenseMatrix quadratureDistance_BF;
    DenseMatrix quadratureTrueNormal_BF;

    AnalyticalGeometricShape *geometry;

  
  public:
  AnalyticalSurface(int geometryType, ParFiniteElementSpace &h1_fes, ParFiniteElementSpace &l2_fes);
  void SetupNodeStatus();
  void SetupElementStatus();
  void SetupFaceTags();
  void ComputeDistanceAndNormalAtQuadraturePoints();
  void ComputeDistanceAndNormalAtCoordinates(const Vector &x, Vector &D, Vector &tN);
  void ResetData();
  Array<int>& GetEss_Vdofs();
  Array<int>& GetFace_Tags();
  Array<int>& GetElement_Status();
  const DenseMatrix& GetQuadratureDistance();
  const DenseMatrix& GetQuadratureTrueNormal();
  const DenseMatrix& GetQuadratureDistance_BF();
  const DenseMatrix& GetQuadratureTrueNormal_BF();
    
  ~AnalyticalSurface();
  };
}
#endif // MFEM_LAGHOS
