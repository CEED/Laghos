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

#ifndef MFEM_LAGHOS_MASS_INTEGRATORS
#define MFEM_LAGHOS_MASS_INTEGRATORS

#include "mfem.hpp"
#include "laghos_assembly.hpp"

namespace mfem
{

namespace hydrodynamics
{
  
class InteriorVectorMassIntegrator : public BilinearFormIntegrator
{
protected:
   Coefficient &Q;
   int order;
   Vector shape;
   DenseMatrix partelmat;

public:
   /// The given MatrixCoefficient fully couples the vector components, i.e.,
   /// the local (dof x vdim) matrices have no zero blocks.
  InteriorVectorMassIntegrator(Coefficient &Q, const IntegrationRule *ir, int order)
    : BilinearFormIntegrator(ir), Q(Q), order(order) { }

   virtual void AssembleElementMatrix(const FiniteElement &el,
			      ElementTransformation &Tr,
			      DenseMatrix &elmat);
};

class BoundaryVectorMassIntegratorV2 : public BilinearFormIntegrator
{
protected:
   Coefficient &Q;
   int order;
   Vector shape;
   DenseMatrix partelmat;
   double wall_bc_penalty;
   double C_I;
   double perimeter;
public:
   /// The given MatrixCoefficient fully couples the vector components, i.e.,
   /// the local (dof x vdim) matrices have no zero blocks.
  BoundaryVectorMassIntegratorV2(Coefficient &Q, const IntegrationRule *ir, int order, double wall_bc_penalty, double C_I, double perimeter)
    : BilinearFormIntegrator(ir), Q(Q), order(order), wall_bc_penalty(wall_bc_penalty), C_I(C_I), perimeter(perimeter) { }

   void AssembleFaceMatrix(const FiniteElement &el1,
			   const FiniteElement &el2,
			   FaceElementTransformations &Tr,
			   DenseMatrix &elmat) override;
};

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_LAGHOS_SHIFT
