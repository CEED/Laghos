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

#ifndef MFEM_LAGHOS_ASSEMBLY_KERNELS
#define MFEM_LAGHOS_ASSEMBLY_KERNELS

#ifdef MFEM_USE_MPI

#include <memory>
#include <iostream>

#include "general/okina.hpp"

namespace mfem
{

namespace hydrodynamics
{

// *****************************************************************************
class kForcePAOperator : public AbcForcePAOperator
{
private:
   const int dim, nzones;
   QuadratureData *quad_data;
   const ParFiniteElementSpace &h1fes, &l2fes;
   const ElemRestriction &h1restrict, &l2restrict;
   const IntegrationRule &integ_rule, &ir1D;
   const int D1D, Q1D;
   const int L1D, H1D;
   const int h1sz, l2sz;
   const DofToQuad *l2D2Q, *h1D2Q;
   mutable mfem::Vector gVecL2, gVecH1;
public:
   kForcePAOperator(QuadratureData*,
                    ParFiniteElementSpace&,
                    ParFiniteElementSpace&,
                    const IntegrationRule&);
   void Mult(const Vector&, Vector&) const;
   void MultTranspose(const Vector&, Vector&) const;
};

// *****************************************************************************
class kMassPAOperator : public AbcMassPAOperator
{
private:
   Coefficient &Q;
   const int dim, nzones;
   QuadratureData *quad_data;
   ParFiniteElementSpace &pfes;
   FiniteElementSpace *fes;
   const IntegrationRule &ir;
   int ess_tdofs_count;
   mfem::Array<int> ess_tdofs;
   ParBilinearForm *paBilinearForm;
   Operator *massOperator;
   mutable mfem::Vector distX;
   Tensors1D *tensors1D;
public:
   kMassPAOperator(Coefficient &q,
                   QuadratureData*,
                   ParFiniteElementSpace&,
                   const IntegrationRule&,
                   Tensors1D *t1D);
   void SetEssentialTrueDofs(mfem::Array<int>&);
   void EliminateRHS(mfem::Vector&);
   virtual void Mult(const mfem::Vector&, mfem::Vector&) const;
   
   virtual void ComputeDiagonal2D(Vector &diag) const {
      const TensorBasisElement *fe_H1 =
         dynamic_cast<const TensorBasisElement *>(fes->GetFE(0));
      const Array<int> &dof_map = fe_H1->GetDofMap();
      const DenseMatrix &HQs = tensors1D->HQshape1D;
      
      const int ndof1D = HQs.Height(), nqp1D = HQs.Width(), nqp = nqp1D * nqp1D;
      Vector dz(ndof1D * ndof1D);
      DenseMatrix HQ(ndof1D, nqp1D), D(dz.GetData(), ndof1D, ndof1D);
      Array<int> dofs;

      diag.SetSize(height);
      diag = 0.0;

      // Squares of the shape functions at all quadrature points.
      DenseMatrix HQs_sq(ndof1D, nqp1D);
      for (int i = 0; i < ndof1D; i++)
      {
         for (int k = 0; k < nqp1D; k++)
         {
            HQs_sq(i, k) = HQs(i, k) * HQs(i, k);
         }
      }

      for (int z = 0; z < nzones; z++)
      {
         DenseMatrix QQ(quad_data->rho0DetJ0w.GetData() + z*nqp, nqp1D, nqp1D);

         // HQ_i1_k2 = HQs_i1_k1^2 QQ_k1_k2    -- contract in x direction.
         // Y_i1_i2  = HQ_i1_k2    HQs_i2_k2^2 -- contract in y direction.
         mfem::Mult(HQs_sq, QQ, HQ);
         MultABt(HQ, HQs_sq, D);

         // Transfer from the tensor structure numbering to mfem's H1 numbering.
         fes->GetElementDofs(z, dofs);
         for (int j = 0; j < dz.Size(); j++)
         {
            diag[dofs[dof_map[j]]] += dz[j];
         }
      }

      for (int i = 0; i < height / 2; i++)
      {
         diag(i + height / 2) = diag(i);
      }
   }
   virtual void ComputeDiagonal3D(Vector&) const { mfem_error("ComputeDiagonal3D Not implemented!"); };
   virtual const Operator *GetProlongation() const
   { return pfes.GetProlongationMatrix(); }
   virtual const Operator *GetRestriction() const
   { return pfes.GetRestrictionMatrix(); }
};

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_LAGHOS_ASSEMBLY_KERNELS
