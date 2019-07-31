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
void CudaGridFunction::ToQuad(const IntegrationRule& ir,
                              CudaVector& quadValues)
{
   const FiniteElement& fe = *(fes.GetFE(0));
   const int dim  = fe.GetDim();
   const int vdim = fes.GetVDim();
   const int elements = fes.GetNE();
   const int numQuad  = ir.GetNPoints();
   const CudaDofQuadMaps* maps = CudaDofQuadMaps::Get(fes, ir);
   const int quad1D  = IntRules.Get(Geometry::SEGMENT,ir.GetOrder()).GetNPoints();
   const int dofs1D =fes.GetFE(0)->GetOrder() + 1;
   quadValues.SetSize(numQuad * elements);
   if (rconfig::Get().Share())
   {
      rGridFuncToQuadS(dim,vdim,dofs1D,quad1D,elements,
                       maps->dofToQuad,
                       fes.GetLocalToGlobalMap(),
                       ptr(),
                       quadValues);
   }
   else
      rGridFuncToQuad(dim,vdim,dofs1D,quad1D,elements,
                      maps->dofToQuad,
                      fes.GetLocalToGlobalMap(),
                      ptr(),
                      quadValues);
}

} // mfem
