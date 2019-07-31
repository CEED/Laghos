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
#include "../cuda.hpp"

namespace mfem
{

// ***************************************************************************
// * CudaProlongationOperator
// ***************************************************************************
CudaProlongationOperator::CudaProlongationOperator
(const CudaConformingProlongationOperator* Op):
   CudaOperator(Op->Height(), Op->Width()),pmat(Op) {}

// ***************************************************************************
void CudaProlongationOperator::Mult(const CudaVector& x,
                                    CudaVector& y) const
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
void CudaProlongationOperator::MultTranspose(const CudaVector& x,
                                             CudaVector& y) const
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
