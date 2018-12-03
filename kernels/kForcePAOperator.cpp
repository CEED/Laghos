// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project (17-SC-20-SC)
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include "../laghos_assembly.hpp"
#include "kForcePAOperator.hpp"
#include "kernels.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{

namespace hydrodynamics
{

// *****************************************************************************
kForcePAOperator::kForcePAOperator(QuadratureData *qd,
                                   ParFiniteElementSpace &h1f,
                                   ParFiniteElementSpace &l2f,
                                   const IntegrationRule &ir) :
   AbcForcePAOperator(/**h1f.GetTrueVLayout()*/),
   dim(h1f.GetMesh()->Dimension()),
   nzones(h1f.GetMesh()->GetNE()),
   quad_data(qd),
   h1fes(h1f),
   l2fes(l2f),
   h1k(*(new kFiniteElementSpace(static_cast<FiniteElementSpace*>(&h1f)))),
   l2k(*(new kFiniteElementSpace(static_cast<FiniteElementSpace*>(&l2f)))),
   integ_rule(ir),
   ir1D(IntRules.Get(Geometry::SEGMENT, integ_rule.GetOrder())),
   NUM_DOFS_1D(h1fes.GetFE(0)->GetOrder()+1),
   NUM_QUAD_1D(ir1D.GetNPoints()),
   L2_DOFS_1D(l2fes.GetFE(0)->GetOrder()+1),
   H1_DOFS_1D(h1fes.GetFE(0)->GetOrder()+1),
   h1sz(h1fes.GetVDim() * h1fes.GetFE(0)->GetDof() * nzones),
   l2sz(l2fes.GetFE(0)->GetDof() * nzones),
   l2D2Q(kDofQuadMaps::Get(l2fes, integ_rule)),
   h1D2Q(kDofQuadMaps::Get(h1fes, integ_rule)),
   gVecL2(h1sz),
   gVecH1(l2sz)
{
   push();
   gVecL2.SetSize(l2sz);//ng.MakeLayout(l2sz));
   gVecH1.SetSize(h1sz);//ng.MakeLayout(h1sz));
   pop();
}

// *****************************************************************************
void kForcePAOperator::Mult(const mfem::Vector &vecL2,
                            mfem::Vector &vecH1) const {
   push();   
   //const kernels::Vector &rVecL2 = vecL2.Get_PVector()->As<const kernels::Vector>();
   //mfem::Vector &rgVecL2 = gVecL2.Get_PVector()->As<kernels::Vector>();
   //const kernels::Vector &rgVecH1 = gVecH1.Get_PVector()->As<const kernels::Vector>();
   //kernels::Vector &rVecH1 = vecH1.Get_PVector()->As<kernels::Vector>();
   l2k.GlobalToLocal(/*rV*/vecL2, /*rgV*/gVecL2);
   dbg("rForceMult");
   rForceMult(dim,
              NUM_DOFS_1D,
              NUM_QUAD_1D,
              L2_DOFS_1D,
              H1_DOFS_1D,
              nzones,
              l2D2Q->dofToQuad,
              h1D2Q->quadToDof,
              h1D2Q->quadToDofD,
              quad_data->stressJinvT.Data(),
              gVecL2,//(const double*)rgVecL2.KernelsMem().ptr(),
              gVecH1);//(double*)rgVecH1.KernelsMem().ptr());
   dbg("done");
   h1k.LocalToGlobal(/*r*/gVecH1, /*rV*/vecH1);
   //pop();
}

// *************************************************************************
void kForcePAOperator::MultTranspose(const mfem::Vector &vecH1,
                                     mfem::Vector &vecL2) const {
   push();
   //const kernels::Vector &rVecH1 = vecH1.Get_PVector()->As<const kernels::Vector>();
   //kernels::Vector &rgVecH1 = gVecH1.Get_PVector()->As<kernels::Vector>();
   //const kernels::Vector &rgVecL2 = gVecL2.Get_PVector()->As<const kernels::Vector>();
   //kernels::Vector &rVecL2 = vecL2.Get_PVector()->As<kernels::Vector>();
   h1k.GlobalToLocal(/*rV*/vecH1, /*rgV*/gVecH1);
   dbg("rForceMultTranspose");
   rForceMultTranspose(dim,
                       NUM_DOFS_1D,
                       NUM_QUAD_1D,
                       L2_DOFS_1D,
                       H1_DOFS_1D,
                       nzones,
                       l2D2Q->quadToDof,
                       h1D2Q->dofToQuad,
                       h1D2Q->dofToQuadD,
                       (const double*)quad_data->stressJinvT.Data(),
                       gVecH1,//(const double*)rgVecH1.KernelsMem().ptr(),
                       gVecL2);//(double*)rgVecL2.KernelsMem().ptr());
   l2k.LocalToGlobal(/*r*/gVecL2, /*rV*/vecL2);
   //pop();
}

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI
