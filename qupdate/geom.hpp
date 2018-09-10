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

#ifndef MFEM_LAGHOS_QUPDATE_GEOM
#define MFEM_LAGHOS_QUPDATE_GEOM

namespace mfem {

namespace hydrodynamics {

// ***************************************************************************
// * qGeometry
// ***************************************************************************
class qGeometry {
public:
   ~qGeometry();
   qarray<int> eMap;
   qarray<double> meshNodes;
   qarray<double> J, invJ, detJ;
   static const int Jacobian    = (1 << 0);
   static const int JacobianInv = (1 << 1);
   static const int JacobianDet = (1 << 2);
   static qGeometry* Get(FiniteElementSpace&,
                         const IntegrationRule&);
   static void ReorderByVDim(GridFunction& nodes);
   static void ReorderByNodes(GridFunction& nodes);
};

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_LAGHOS_QUPDATE_GEOM
