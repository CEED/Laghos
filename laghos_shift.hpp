// Copyright (c) 2017, LawrenceA Livermore National Security, LLC. Produced at
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

#ifndef MFEM_LAGHOS_SHIFT
#define MFEM_LAGHOS_SHIFT

#include "mfem.hpp"
#include "laghos_assembly.hpp"
#include "AnalyticalSurface.hpp"

namespace mfem
{

namespace hydrodynamics
{

void shift_shape(const ParFiniteElementSpace &pfes_e_const,
                 const ParFiniteElementSpace &pfes_p,
                 int e_id,
                 const IntegrationPoint &ip, const Vector &dist,
		 int nterms, Vector &shape_shift);

void get_shifted_value(const ParGridFunction &g, int e_id,
                       const IntegrationPoint &ip, const Vector &dist,
                       int nterms, Vector &shifted_vec);

  
class SBM_BoundaryVectorMassIntegrator : public VectorMassIntegrator
{
protected:
   const ParFiniteElementSpace &H1; 
   const AnalyticalGeometricShape& geom;
   int num_taylor = 1;
  int int_order;
public:
   /// The given MatrixCoefficient fully couples the vector components, i.e.,
   /// the local (dof x vdim) matrices have no zero blocks.
  SBM_BoundaryVectorMassIntegrator(MatrixCoefficient &mc, const ParFiniteElementSpace &H1, const AnalyticalGeometricShape& geom, int num_taylor, int int_order)
    : VectorMassIntegrator(mc), H1(H1), geom(geom), num_taylor(num_taylor), int_order(int_order) { }

   /// Expected use is with BilinearForm::AddBdrFaceIntegrator(), where @a el1
   /// is for the volumetric neighbor of the boundary face, @a el2 is not used.
   void AssembleFaceMatrix(const FiniteElement &el1,
                           const FiniteElement &el2,
                           FaceElementTransformations &Tr,
                           DenseMatrix &elmat) override;
};

class SBM_BoundaryMixedForceIntegrator : public BilinearFormIntegrator
{
protected:
   VectorCoefficient &Q_ibp;
   VectorCoefficient &Q_pen;
   const ParFiniteElementSpace &H1; 
   const AnalyticalGeometricShape& geom;
   int num_taylor = 1;
  int int_order;
public:
   SBM_BoundaryMixedForceIntegrator(VectorCoefficient &vc_ibp, VectorCoefficient &vc_pen, const ParFiniteElementSpace &H1, const AnalyticalGeometricShape& geom, int num_taylor, int int_order) :  Q_ibp(vc_ibp), Q_pen(vc_pen), H1(H1), geom(geom), num_taylor(num_taylor), int_order(int_order) { }

   /// Expected use is with MixedBilinearForm::AddBdrFaceIntegrator(), where
   /// @a el1 and @a el2 are for the (mixed) volumetric neighbor of the face.
  void AssembleFaceMatrix(const FiniteElement &trial_fe,
                          const FiniteElement &test_fe,
                          FaceElementTransformations &Tr,
                          DenseMatrix &elmat) override;
};


class SBM_BoundaryMixedForceTIntegrator : public BilinearFormIntegrator
{
protected:
   VectorCoefficient &Q_ibp;
   VectorCoefficient &Q_pen;
   const ParFiniteElementSpace &H1; 
   const AnalyticalGeometricShape& geom;
   int num_taylor = 1;
  int int_order;
public:
  SBM_BoundaryMixedForceTIntegrator(VectorCoefficient &vc_ibp, VectorCoefficient &vc_pen, const ParFiniteElementSpace &H1, const AnalyticalGeometricShape& geom, int num_taylor, int int_order) : Q_ibp(vc_ibp), Q_pen(vc_pen), H1(H1), geom(geom), num_taylor(num_taylor), int_order(int_order) { }

   /// Expected use is with MixedBilinearForm::AddBdrFaceIntegrator(), where
   /// @a el1 and @a el2 are for the (mixed) volumetric neighbor of the face.
  void AssembleFaceMatrix(const FiniteElement &trial_fe,
                          const FiniteElement &test_fe,
                          FaceElementTransformations &Tr,
                          DenseMatrix &elmat) override;
};


} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_LAGHOS_SHIFT
