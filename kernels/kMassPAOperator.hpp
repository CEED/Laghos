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

#ifndef MFEM_LAGHOS_KERNEL_MASS_PA_OPERATOR_HPP
#define MFEM_LAGHOS_KERNEL_MASS_PA_OPERATOR_HPP

#include "mfem.hpp"

#ifdef MFEM_USE_MPI

#include <memory>
#include <iostream>

namespace mfem
{

namespace hydrodynamics
{

// *****************************************************************************
class kMassPAOperator : public AbcMassPAOperator
{
private:
   const int dim, nzones;
   QuadratureData *quad_data;
   ParFiniteElementSpace &fes;
   const IntegrationRule &ir;
   int ess_tdofs_count;
   mfem::Array<int> ess_tdofs;
   kernels::kBilinearForm *bilinearForm;
   Operator *massOperator;
   mutable mfem::Vector distX;
public:
   kMassPAOperator(QuadratureData*,
                   ParFiniteElementSpace&,
                   const IntegrationRule&);
   virtual void Setup();
   void SetEssentialTrueDofs(mfem::Array<int>&);
   void EliminateRHS(mfem::Vector&);
   virtual void Mult(const mfem::Vector&, mfem::Vector&) const;
   virtual void ComputeDiagonal2D(Vector&) const {};
   virtual void ComputeDiagonal3D(Vector&) const {};
   virtual const Operator *GetProlongation() const
   { return fes.GetProlongationMatrix(); }
   virtual const Operator *GetRestriction() const
   { return fes.GetRestrictionMatrix(); }
};
  

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_LAGHOS_KERNEL_MASS_PA_OPERATOR_HPP
