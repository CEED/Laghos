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
#include "raja.hpp"

namespace mfem {

// ***************************************************************************
// * RajaBilinearForm
// ***************************************************************************
RajaBilinearForm::RajaBilinearForm(RajaFiniteElementSpace* fes) :
  Operator(fes->GetVSize(),fes->GetVSize()),
  mesh(fes->GetMesh()),
  trialFes(fes),
  testFes(fes),
  localX(mesh->GetNE() * trialFes->GetLocalDofs() * trialFes->GetVDim()),
  localY(mesh->GetNE() * testFes->GetLocalDofs() * testFes->GetVDim())
{ }

// ***************************************************************************
// Adds new Domain Integrator.
void RajaBilinearForm::AddDomainIntegrator(RajaIntegrator* i) {
  AddIntegrator(i, DomainIntegrator);
}

// Adds new Boundary Integrator.
void RajaBilinearForm::AddBoundaryIntegrator(RajaIntegrator* i) {
  AddIntegrator(i, BoundaryIntegrator);
}

// Adds new interior Face Integrator.
void RajaBilinearForm::AddInteriorFaceIntegrator(RajaIntegrator* i) {
  AddIntegrator(i, InteriorFaceIntegrator);
}

// Adds new boundary Face Integrator.
void RajaBilinearForm::AddBoundaryFaceIntegrator(RajaIntegrator* i) {
  AddIntegrator(i, BoundaryFaceIntegrator);
}

// Adds Integrator based on RajaIntegratorType
void RajaBilinearForm::AddIntegrator(RajaIntegrator* i,
                                     const RajaIntegratorType itype) {
  assert(i);
  i->SetupIntegrator(*this, itype);
  integrators.push_back(i);
}

// ***************************************************************************
void RajaBilinearForm::Assemble() {
  const int integratorCount = (int) integrators.size();
  for (int i = 0; i < integratorCount; ++i) {
    integrators[i]->Assemble();
  }
}

// ***************************************************************************
void RajaBilinearForm::FormLinearSystem(const Array<int>& constraintList,
                                        RajaVector& x, RajaVector& b,
                                        Operator*& Aout,
                                        RajaVector& X, RajaVector& B,
                                        int copy_interior) {
  FormOperator(constraintList, Aout);
  InitRHS(constraintList, x, b, Aout, X, B, copy_interior);
}

// ***************************************************************************
void RajaBilinearForm::FormOperator(const Array<int>& constraintList,
                                    Operator*& Aout) {
  const Operator* trialP = trialFes->GetProlongationOperator();
  const Operator* testP  = testFes->GetProlongationOperator();
  Operator* rap = this;
  if (trialP) { rap = new RajaRAPOperator(*testP, *this, *trialP); }
  Aout = new RajaConstrainedOperator(rap, constraintList, rap!=this);
}

// ***************************************************************************
void RajaBilinearForm::InitRHS(const Array<int>& constraintList,
                               const RajaVector& x, const RajaVector& b,
                               Operator* A,
                               RajaVector& X, RajaVector& B,
                               int copy_interior) {
  const Operator* P = trialFes->GetProlongationOperator();
  const Operator* R = trialFes->GetRestrictionOperator();
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
  RajaConstrainedOperator* cA = static_cast<RajaConstrainedOperator*>(A);
  if (cA) {
    cA->EliminateRHS(X, B);
  } else {
    mfem_error("RajaBilinearForm::InitRHS expects an RajaConstrainedOperator");
  }
}

// ***************************************************************************
void RajaBilinearForm::Mult(const RajaVector& x, RajaVector& y) const {
  trialFes->GlobalToLocal(x, localX);
  localY = 0;
  const int integratorCount = (int) integrators.size();
  for (int i = 0; i < integratorCount; ++i) {
    integrators[i]->MultAdd(localX, localY);
  }
  testFes->LocalToGlobal(localY, y);
}

// ***************************************************************************
void RajaBilinearForm::MultTranspose(const RajaVector& x, RajaVector& y) const {
  testFes->GlobalToLocal(x, localX);
  localY = 0;
  const int integratorCount = (int) integrators.size();
  for (int i = 0; i < integratorCount; ++i) {
    integrators[i]->MultTransposeAdd(localX, localY);
  }
  trialFes->LocalToGlobal(localY, y);
}

// ***************************************************************************
void RajaBilinearForm::RecoverFEMSolution(const RajaVector& X,
    const RajaVector& b,
    RajaVector& x) {
  TRecoverFEMSolution<RajaVector> (X, b, x);
}


// ***************************************************************************
// * RajaConstrainedOperator
// ***************************************************************************
RajaConstrainedOperator::RajaConstrainedOperator(Operator* A_,
    const Array<int>& constraintList_,
    bool own_A_) :
  Operator(A_->Height(), A_->Width()) {
  Setup(A_, constraintList_, own_A_);
}

void RajaConstrainedOperator::Setup(Operator* A_,
                                    const Array<int>& constraintList_,
                                    bool own_A_) {
  A = A_;
  own_A = own_A_;
  constraintIndices = constraintList_.Size();
  if (constraintIndices) {
    constraintList.allocate(constraintIndices);
  }
  z.SetSize(height);
  w.SetSize(height);
}

void RajaConstrainedOperator::EliminateRHS(const RajaVector& x,
    RajaVector& b) const {
  w = 0.0;
  A->Mult(w, z);
  b -= z;
}

void RajaConstrainedOperator::Mult(const RajaVector& x, RajaVector& y) const {
  if (constraintIndices == 0) {
    A->Mult(x, y);
    return;
  }
  z = x;
  A->Mult(z, y);
}

} // mfem
