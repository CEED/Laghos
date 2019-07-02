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
// * HipProlongationOperator
// ***************************************************************************
HipProlongationOperator::HipProlongationOperator
(const HipConformingProlongationOperator* Op):
   HipOperator(Op->Height(), Op->Width()),pmat(Op) {}

// ***************************************************************************
void HipProlongationOperator::Mult(const HipVector& x,
                                    HipVector& y) const
{
   if (rconfig::Get().IAmAlone())
   {
      y=x;
      return;
   }
   if (!rconfig::Get().DoHostConformingProlongationOperator())
   {
      pmat->d_Mult(x, y);
      return;
   }
   const Vector hostX=x;//D2H
   Vector hostY(y.Size());
   pmat->h_Mult(hostX, hostY);
   y=hostY;//H2D
}

// ***************************************************************************
void HipProlongationOperator::MultTranspose(const HipVector& x,
                                             HipVector& y) const
{
   if (rconfig::Get().IAmAlone())
   {
      y=x;
      return;
   }
   if (!rconfig::Get().DoHostConformingProlongationOperator())
   {
      pmat->d_MultTranspose(x, y);
      return;
   }
   const Vector hostX=x;
   Vector hostY(y.Size());
   pmat->h_MultTranspose(hostX, hostY);
   y=hostY;//H2D
}

} // namespace mfem
