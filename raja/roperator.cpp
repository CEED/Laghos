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
#include "raja.hpp"

namespace mfem {
  
  void RajaOperator::FormLinearSystem(const Array<int> &ess_tdof_list,
                                      RajaVector &x, RajaVector &b,
                                      RajaOperator* &Aout, RajaVector &X, RajaVector &B,
                                      int copy_interior)
  {
    const RajaOperator *P = this->GetProlongation();
    const RajaOperator *R = this->GetRestriction();
    RajaOperator *rap;
    if (P)
    {
      // Variational restriction with P
      B.SetSize(P->Width());
      P->MultTranspose(b, B);
      X.SetSize(R->Height());
      R->Mult(x, X);
      rap = new RajaRAPOperator(*P, *this, *P);
    }
    else
    {
      // rap, X and B point to the same data as this, x and b
#warning
      //X.NewDataAndSize(x.GetData(), x.Size());
      //B.NewDataAndSize(b.GetData(), b.Size());
      rap = this;
    }
    
#warning
    //if (!copy_interior) { X.SetSubVectorComplement(ess_tdof_list, 0.0); }

    // Impose the boundary conditions through a ConstrainedOperator, which owns
    // the rap operator when P and R are non-trivial
    RajaConstrainedOperator *A = new RajaConstrainedOperator(rap, ess_tdof_list,
                                                     rap != this);
    A->EliminateRHS(X, B);
    Aout = A;
  }

  
  void RajaOperator::RecoverFEMSolution(const RajaVector &X,
                                        const RajaVector &b,
                                        RajaVector &x)
  {
    const RajaOperator *P = this->GetProlongation();
    if (P)
    {
      // Apply conforming prolongation
      x.SetSize(P->Height());
      P->Mult(X, x);
    }
    // Otherwise X and x point to the same data
  }


  void RajaOperator::PrintMatlab(std::ostream & out, int n, int m) const  {
  }

} // mfem
