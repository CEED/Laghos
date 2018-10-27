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

namespace mfem {

// ***************************************************************************
// * CudaBilinearForm
// ***************************************************************************
CudaBilinearForm::CudaBilinearForm(CudaFiniteElementSpace* fes) :
  CudaOperator(fes->GetVSize(),fes->GetVSize()),
  mesh(fes->GetMesh()),
  trialFes(fes),
  testFes(fes),
  localX(mesh->GetNE() * trialFes->GetLocalDofs() * trialFes->GetVDim()),
  localY(mesh->GetNE() * testFes->GetLocalDofs() * testFes->GetVDim()){}

  // ***************************************************************************
  CudaBilinearForm::~CudaBilinearForm(){ }
  
// ***************************************************************************
// Adds new Domain Integrator.
void CudaBilinearForm::AddDomainIntegrator(CudaIntegrator* i) {
  push(SteelBlue); 
  AddIntegrator(i, DomainIntegrator);
  pop();
}

// Adds new Boundary Integrator.
void CudaBilinearForm::AddBoundaryIntegrator(CudaIntegrator* i) {
  push(SteelBlue);
  AddIntegrator(i, BoundaryIntegrator);
  pop();
}

// Adds new interior Face Integrator.
void CudaBilinearForm::AddInteriorFaceIntegrator(CudaIntegrator* i) {
  push(SteelBlue);
  AddIntegrator(i, InteriorFaceIntegrator);
  pop();
}

// Adds new boundary Face Integrator.
void CudaBilinearForm::AddBoundaryFaceIntegrator(CudaIntegrator* i) {
  push(SteelBlue);
  AddIntegrator(i, BoundaryFaceIntegrator);
  pop();
}

// Adds Integrator based on CudaIntegratorType
void CudaBilinearForm::AddIntegrator(CudaIntegrator* i,
                                     const CudaIntegratorType itype) {
  push(SteelBlue);
  assert(i);
  i->SetupIntegrator(*this, itype);
  integrators.push_back(i);
  pop();
}

// ***************************************************************************
void CudaBilinearForm::Assemble() {
  push(SteelBlue);
  const int integratorCount = (int) integrators.size();
  for (int i = 0; i < integratorCount; ++i) {
    integrators[i]->Assemble();
  }
  pop();
}

// ***************************************************************************
void CudaBilinearForm::FormLinearSystem(const Array<int>& constraintList,
                                        CudaVector& x, CudaVector& b,
                                        CudaOperator*& Aout,
                                        CudaVector& X, CudaVector& B,
                                        int copy_interior) {
  push(SteelBlue);
  FormOperator(constraintList, Aout);
  InitRHS(constraintList, x, b, Aout, X, B, copy_interior);
  pop();
}

// ***************************************************************************
void CudaBilinearForm::FormOperator(const Array<int>& constraintList,
                                    CudaOperator*& Aout) {
  push(SteelBlue);
  const CudaOperator* trialP = trialFes->GetProlongationOperator();
  const CudaOperator* testP  = testFes->GetProlongationOperator();
  CudaOperator *rap = this;
  if (trialP) { rap = new CudaRAPOperator(*testP, *this, *trialP); }
  Aout = new CudaConstrainedOperator(rap, constraintList, rap!=this);
  pop();
}

// ***************************************************************************
void CudaBilinearForm::InitRHS(const Array<int>& constraintList,
                               const CudaVector& x, const CudaVector& b,
                               CudaOperator* A,
                               CudaVector& X, CudaVector& B,
                               int copy_interior) {
  push(SteelBlue);
  const CudaOperator* P = trialFes->GetProlongationOperator();
  const CudaOperator* R = trialFes->GetRestrictionOperator();
  if (P) {
    // Variational restriction with P
    B.SetSize(P->Width());
    P->MultTranspose(b, B);
    X.SetSize(R->Height());
    R->Mult(x, X);
  } else {
    // rap, X and B point to the same data as this, x and b
    X.SetSize(x.Size(),x);
    B.SetSize(b.Size(),b);
  }
  CudaConstrainedOperator* cA = static_cast<CudaConstrainedOperator*>(A);
  if (cA) {
    cA->EliminateRHS(X, B);
  } else {
    mfem_error("CudaBilinearForm::InitRHS expects an CudaConstrainedOperator");
  }
  pop();
}

// ***************************************************************************
void CudaBilinearForm::Mult(const CudaVector& x, CudaVector& y) const {
  push(SteelBlue);
  trialFes->GlobalToLocal(x, localX);
  localY = 0;
  const int integratorCount = (int) integrators.size();
  for (int i = 0; i < integratorCount; ++i) {
    integrators[i]->MultAdd(localX, localY);
  }
  testFes->LocalToGlobal(localY, y);
  pop();
}

// ***************************************************************************
void CudaBilinearForm::MultTranspose(const CudaVector& x, CudaVector& y) const {
  push(SteelBlue);
  testFes->GlobalToLocal(x, localX);
  localY = 0;
  const int integratorCount = (int) integrators.size();
  for (int i = 0; i < integratorCount; ++i) {
    integrators[i]->MultTransposeAdd(localX, localY);
  }
  trialFes->LocalToGlobal(localY, y);
  pop();
}

// ***************************************************************************
void CudaBilinearForm::RecoverFEMSolution(const CudaVector& X,
                                          const CudaVector& b,
                                          CudaVector& x) {
  push(SteelBlue);
  const CudaOperator *P = this->GetProlongation();
  if (P)
  {
    // Apply conforming prolongation
    x.SetSize(P->Height());
    P->Mult(X, x);
  }
  // Otherwise X and x point to the same data
  pop();
}


// ***************************************************************************
// * CudaConstrainedOperator
// ***************************************************************************
CudaConstrainedOperator::CudaConstrainedOperator(CudaOperator* A_,
                                                 const Array<int>& constraintList_,
                                                 bool own_A_) :
  CudaOperator(A_->Height(), A_->Width()) {
  push(SteelBlue);
  Setup(A_, constraintList_, own_A_);
  pop();
}

void CudaConstrainedOperator::Setup(CudaOperator* A_,
                                    const Array<int>& constraintList_,
                                    bool own_A_) {
  push(SteelBlue);
  A = A_;
  own_A = own_A_;
  constraintIndices = constraintList_.Size();
  if (constraintIndices) {
    constraintList.allocate(constraintIndices);
  }
  z.SetSize(height);
  w.SetSize(height);
  pop();
}

void CudaConstrainedOperator::EliminateRHS(const CudaVector& x,
                                           CudaVector& b) const {
  push(SteelBlue);
  w = 0.0;
  A->Mult(w, z);
  b -= z;
  pop();
}

void CudaConstrainedOperator::Mult(const CudaVector& x, CudaVector& y) const {
  push(SteelBlue);
  if (constraintIndices == 0) {
    A->Mult(x, y);
    pop();
    return;
  }
  z = x;
  A->Mult(z, y); // roperator.hpp:76
  pop();
}

} // mfem
