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
#include "../cuda.hpp"

namespace mfem
{

// ***************************************************************************
// * CudaBilinearForm
// ***************************************************************************
CudaBilinearForm::CudaBilinearForm(CudaFiniteElementSpace* fes) :
   CudaOperator(fes->GetVSize(),fes->GetVSize()),
   mesh(fes->GetMesh()),
   trialFes(fes),
   testFes(fes),
   localX(mesh->GetNE() * trialFes->GetLocalDofs() * trialFes->GetVDim()),
   localY(mesh->GetNE() * testFes->GetLocalDofs() * testFes->GetVDim()) {}

// ***************************************************************************
CudaBilinearForm::~CudaBilinearForm() { }

// ***************************************************************************
// Adds new Domain Integrator.
void CudaBilinearForm::AddDomainIntegrator(CudaIntegrator* i)
{
   AddIntegrator(i, DomainIntegrator);
}

// Adds new Boundary Integrator.
void CudaBilinearForm::AddBoundaryIntegrator(CudaIntegrator* i)
{
   AddIntegrator(i, BoundaryIntegrator);
}

// Adds new interior Face Integrator.
void CudaBilinearForm::AddInteriorFaceIntegrator(CudaIntegrator* i)
{
   AddIntegrator(i, InteriorFaceIntegrator);
}

// Adds new boundary Face Integrator.
void CudaBilinearForm::AddBoundaryFaceIntegrator(CudaIntegrator* i)
{
   AddIntegrator(i, BoundaryFaceIntegrator);
}

// Adds Integrator based on CudaIntegratorType
void CudaBilinearForm::AddIntegrator(CudaIntegrator* i,
                                     const CudaIntegratorType itype)
{
   assert(i);
   i->SetupIntegrator(*this, itype);
   integrators.push_back(i);
}

// ***************************************************************************
void CudaBilinearForm::Assemble()
{
   const int integratorCount = (int) integrators.size();
   for (int i = 0; i < integratorCount; ++i)
   {
      integrators[i]->Assemble();
   }
}

// ***************************************************************************
void CudaBilinearForm::FormLinearSystem(const Array<int>& constraintList,
                                        CudaVector& x, CudaVector& b,
                                        CudaOperator*& Aout,
                                        CudaVector& X, CudaVector& B,
                                        int copy_interior)
{
   FormOperator(constraintList, Aout);
   InitRHS(constraintList, x, b, Aout, X, B, copy_interior);
}

// ***************************************************************************
void CudaBilinearForm::FormOperator(const Array<int>& constraintList,
                                    CudaOperator*& Aout)
{
   const CudaOperator* trialP = trialFes->GetProlongationOperator();
   const CudaOperator* testP  = testFes->GetProlongationOperator();
   CudaOperator *rap = this;
   if (trialP) { rap = new CudaRAPOperator(*testP, *this, *trialP); }
   Aout = new CudaConstrainedOperator(rap, constraintList, rap!=this);
}

// ***************************************************************************
void CudaBilinearForm::InitRHS(const Array<int>& constraintList,
                               const CudaVector& x, const CudaVector& b,
                               CudaOperator* A,
                               CudaVector& X, CudaVector& B,
                               int copy_interior)
{
   const CudaOperator* P = trialFes->GetProlongationOperator();
   const CudaOperator* R = trialFes->GetRestrictionOperator();
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
   CudaConstrainedOperator* cA = static_cast<CudaConstrainedOperator*>(A);
   if (cA)
   {
      cA->EliminateRHS(X, B);
   }
   else
   {
      mfem_error("CudaBilinearForm::InitRHS expects an CudaConstrainedOperator");
   }
}

// ***************************************************************************
void CudaBilinearForm::Mult(const CudaVector& x, CudaVector& y) const
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
void CudaBilinearForm::MultTranspose(const CudaVector& x, CudaVector& y) const
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
void CudaBilinearForm::RecoverFEMSolution(const CudaVector& X,
                                          const CudaVector& b,
                                          CudaVector& x)
{
   const CudaOperator *P = this->GetProlongation();
   if (P)
   {
      // Apply conforming prolongation
      x.SetSize(P->Height());
      P->Mult(X, x);
   }
   // Otherwise X and x point to the same data
}


// ***************************************************************************
// * CudaConstrainedOperator
// ***************************************************************************
CudaConstrainedOperator::CudaConstrainedOperator(CudaOperator* A_,
                                                 const Array<int>& constraintList_,
                                                 bool own_A_) :
   CudaOperator(A_->Height(), A_->Width())
{
   Setup(A_, constraintList_, own_A_);
}

void CudaConstrainedOperator::Setup(CudaOperator* A_,
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

void CudaConstrainedOperator::EliminateRHS(const CudaVector& x,
                                           CudaVector& b) const
{
   w = 0.0;
   A->Mult(w, z);
   b -= z;
}

void CudaConstrainedOperator::Mult(const CudaVector& x, CudaVector& y) const
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
