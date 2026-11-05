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

#include "laghos_assembly.hpp"

namespace mfem
{

namespace hydrodynamics
{

MassPAOperator::MassPAOperator(ParFiniteElementSpace &pfes,
                               const IntegrationRule &ir,
                               Coefficient &Q) :
   Operator(pfes.GetTrueVSize()),
   comm(pfes.GetParMesh()->GetComm()),
   dim(pfes.GetMesh()->Dimension()),
   pabf(&pfes),
   ess_tdofs_count(0),
   ess_tdofs(0)
{
   if (dim > 1)
   {
      pabf.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   pabf.AddDomainIntegrator(new mfem::MassIntegrator(Q, &ir));
   pabf.Assemble();
   pabf.FormSystemMatrix(mfem::Array<int>(), mass);
}

void MassPAOperator::SetEssentialTrueDofs(Array<int> &dofs)
{
   ess_tdofs_count = dofs.Size();
   if (ess_tdofs.Size() == 0)
   {
      int ess_tdofs_sz;
      MPI_Allreduce(&ess_tdofs_count,&ess_tdofs_sz, 1, MPI_INT, MPI_SUM, comm);
      MFEM_ASSERT(ess_tdofs_sz > 0, "ess_tdofs_sz should be positive!");
      ess_tdofs.SetSize(ess_tdofs_sz);
   }
   if (ess_tdofs_count == 0) { return; }
   ess_tdofs = dofs;
}

void MassPAOperator::EliminateRHS(Vector &b) const
{
   if (ess_tdofs_count > 0) { b.SetSubVector(ess_tdofs, 0.0); }
}

void MassPAOperator::Mult(const Vector &x, Vector &y) const
{
   mass->Mult(x, y);
   if (ess_tdofs_count > 0) { y.SetSubVector(ess_tdofs, 0.0); }
}

void MassPAOperator::FullAddMult(const Vector &x, Vector &y) const
{
   mass->AddMult(x, y);
}

} // namespace hydrodynamics

} // namespace mfem
