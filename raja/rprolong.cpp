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
  // * RajaProlongationOperator
  // ***************************************************************************
  RajaProlongationOperator::RajaProlongationOperator
  (const RajaConformingProlongationOperator* Op):
    RajaOperator(Op->Height(), Op->Width()),pmat(Op){}
  
  // ***************************************************************************
  void RajaProlongationOperator::Mult(const RajaVector& x,
                                      RajaVector& y) const {
    push();
    if (rconfig::IAmAlone()){
      y=x;
      pop();
      return;
    }
    push(hostX:D2H,Red);
    const Vector hostX=x;//D2H
    pop(); 
    
    push(hostY);
    Vector hostY(y.Size());
    pop();
    
    push(pmat->Mult,Blue);
    //pmat->Mult(x, y); // RajaConformingProlongationOperator::Mult
    pmat->h_Mult(hostX, hostY); // fem/pfespace.cpp:2675
    pop();
    
    push(hostY:H2D,Yellow);
    y=hostY;//H2D
    pop();
    pop();
  }

  // ***************************************************************************
  void RajaProlongationOperator::MultTranspose(const RajaVector& x,
                                               RajaVector& y) const {
    push();
    if (rconfig::IAmAlone()){
      y=x;
      pop();
      return;
    }
    push(hostX:D2H,Red);
    const Vector hostX=x;//D2H
    pop();

    push(hostY);
    Vector hostY(y.Size());
    pop();

    push(pmat->MultT,Blue);
    //pmat->MultTranspose(x, y); // RajaConformingProlongationOperator::MultTranspose
    pmat->h_MultTranspose(hostX, hostY);
    pop();
    
    push(hostY:H2D,Yellow);
    y=hostY;//H2D
    pop();
    pop();
  }

} // namespace mfem
