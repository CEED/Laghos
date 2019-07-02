// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
#include "../hip.hpp"

namespace mfem
{

// ***************************************************************************
// * HipBilinearForm
// ***************************************************************************
HipBilinearForm::HipBilinearForm(HipFiniteElementSpace* fes) :
   HipOperator(fes->GetVSize(),fes->GetVSize()),
   mesh(fes->GetMesh()),
   trialFes(fes),
   testFes(fes),
   localX(mesh->GetNE() * trialFes->GetLocalDofs() * trialFes->GetVDim()),
   localY(mesh->GetNE() * testFes->GetLocalDofs() * testFes->GetVDim()) {}

// ***************************************************************************
HipBilinearForm::~HipBilinearForm() { }

// ***************************************************************************
// Adds new Domain Integrator.
void HipBilinearForm::AddDomainIntegrator(HipIntegrator* i)
{
   AddIntegrator(i, DomainIntegrator);
}

// Adds new Boundary Integrator.
void HipBilinearForm::AddBoundaryIntegrator(HipIntegrator* i)
{
   AddIntegrator(i, BoundaryIntegrator);
}

// Adds new interior Face Integrator.
void HipBilinearForm::AddInteriorFaceIntegrator(HipIntegrator* i)
{
   AddIntegrator(i, InteriorFaceIntegrator);
}

// Adds new boundary Face Integrator.
void HipBilinearForm::AddBoundaryFaceIntegrator(HipIntegrator* i)
{
   AddIntegrator(i, BoundaryFaceIntegrator);
}

// Adds Integrator based on HipIntegratorType
void HipBilinearForm::AddIntegrator(HipIntegrator* i,
                                     const HipIntegratorType itype)
{
   assert(i);
   i->SetupIntegrator(*this, itype);
   integrators.push_back(i);
}

// ***************************************************************************
void HipBilinearForm::Assemble()
{
   const int integratorCount = (int) integrators.size();
   for (int i = 0; i < integratorCount; ++i)
   {
      integrators[i]->Assemble();
   }
}

// ***************************************************************************
void HipBilinearForm::FormLinearSystem(const Array<int>& constraintList,
                                        HipVector& x, HipVector& b,
                                        HipOperator*& Aout,
                                        HipVector& X, HipVector& B,
                                        int copy_interior)
{
   FormOperator(constraintList, Aout);
   InitRHS(constraintList, x, b, Aout, X, B, copy_interior);
}

// ***************************************************************************
void HipBilinearForm::FormOperator(const Array<int>& constraintList,
                                    HipOperator*& Aout)
{
   const HipOperator* trialP = trialFes->GetProlongationOperator();
   const HipOperator* testP  = testFes->GetProlongationOperator();
   HipOperator *rap = this;
   if (trialP) { rap = new HipRAPOperator(*testP, *this, *trialP); }
   Aout = new HipConstrainedOperator(rap, constraintList, rap!=this);
}

// ***************************************************************************
void HipBilinearForm::InitRHS(const Array<int>& constraintList,
                               const HipVector& x, const HipVector& b,
                               HipOperator* A,
                               HipVector& X, HipVector& B,
                               int copy_interior)
{
   const HipOperator* P = trialFes->GetProlongationOperator();
   const HipOperator* R = trialFes->GetRestrictionOperator();
   if (P)
   {
      // Variational restriction with P
      B.SetSize(P->Width());
      P->MultTranspose(b, B);
      X.SetSize(R->Height());
      R->Mult(x, X);
   }
   else
   {
      // rap, X and B point to the same data as this, x and b
      X.SetSize(x.Size(),x);
      B.SetSize(b.Size(),b);
   }
   HipConstrainedOperator* cA = static_cast<HipConstrainedOperator*>(A);
   if (cA)
   {
      cA->EliminateRHS(X, B);
   }
   else
   {
      mfem_error("HipBilinearForm::InitRHS expects an HipConstrainedOperator");
   }
}

// ***************************************************************************
void HipBilinearForm::Mult(const HipVector& x, HipVector& y) const
{
   trialFes->GlobalToLocal(x, localX);
   localY = 0;
   const int integratorCount = (int) integrators.size();
   for (int i = 0; i < integratorCount; ++i)
   {
      integrators[i]->MultAdd(localX, localY);
   }
   testFes->LocalToGlobal(localY, y);
}

// ***************************************************************************
void HipBilinearForm::MultTranspose(const HipVector& x, HipVector& y) const
{
   testFes->GlobalToLocal(x, localX);
   localY = 0;
   const int integratorCount = (int) integrators.size();
   for (int i = 0; i < integratorCount; ++i)
   {
      integrators[i]->MultTransposeAdd(localX, localY);
   }
   trialFes->LocalToGlobal(localY, y);
}

// ***************************************************************************
void HipBilinearForm::RecoverFEMSolution(const HipVector& X,
                                          const HipVector& b,
                                          HipVector& x)
{
   const HipOperator *P = this->GetProlongation();
   if (P)
   {
      // Apply conforming prolongation
      x.SetSize(P->Height());
      P->Mult(X, x);
   }
   // Otherwise X and x point to the same data
}


// ***************************************************************************
// * HipConstrainedOperator
// ***************************************************************************
HipConstrainedOperator::HipConstrainedOperator(HipOperator* A_,
                                                 const Array<int>& constraintList_,
                                                 bool own_A_) :
   HipOperator(A_->Height(), A_->Width())
{
   Setup(A_, constraintList_, own_A_);
}

void HipConstrainedOperator::Setup(HipOperator* A_,
                                    const Array<int>& constraintList_,
                                    bool own_A_)
{
   A = A_;
   own_A = own_A_;
   constraintIndices = constraintList_.Size();
   if (constraintIndices)
   {
      constraintList.allocate(constraintIndices);
   }
   z.SetSize(height);
   w.SetSize(height);
}

void HipConstrainedOperator::EliminateRHS(const HipVector& x,
                                           HipVector& b) const
{
   w = 0.0;
   A->Mult(w, z);
   b -= z;
}

void HipConstrainedOperator::Mult(const HipVector& x, HipVector& y) const
{
   if (constraintIndices == 0)
   {
      A->Mult(x, y);
      return;
   }
   z = x;
   A->Mult(z, y);
}

} // mfem
