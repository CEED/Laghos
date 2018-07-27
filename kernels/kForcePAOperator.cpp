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
kForcePAOperator::kForcePAOperator(const QuadratureData *qd,
                                   ParFiniteElementSpace &h1f,
                                   ParFiniteElementSpace &l2f,
                                   const IntegrationRule &ir,
                                   const bool engine) :
   dim(h1f.GetMesh()->Dimension()),
   nzones(h1f.GetMesh()->GetNE()),
   quad_data(qd),
   h1fes(h1f),
   l2fes(l2f),
   h1k(*h1fes.Get_PFESpace().As<kernels::KernelsFiniteElementSpace>()),
   l2k(*l2fes.Get_PFESpace().As<kernels::KernelsFiniteElementSpace>()),
   integ_rule(ir),
   ir1D(IntRules.Get(Geometry::SEGMENT, integ_rule.GetOrder())),
   NUM_DOFS_1D(h1fes.GetFE(0)->GetOrder()+1),
   NUM_QUAD_1D(ir1D.GetNPoints()),
   L2_DOFS_1D(l2fes.GetFE(0)->GetOrder()+1),
   H1_DOFS_1D(h1fes.GetFE(0)->GetOrder()+1),
   h1sz(h1fes.GetVDim() * h1fes.GetFE(0)->GetDof() * nzones),
   l2sz(l2fes.GetFE(0)->GetDof() * nzones),
   h1D2Q(kernels::KernelsDofQuadMaps::Get(h1fes, integ_rule)),
   l2D2Q(kernels::KernelsDofQuadMaps::Get(l2fes, integ_rule)),
   gVecL2(h1sz),
   gVecH1(l2sz)
{
   if (!engine) return;
   const Engine &ng = l2f.GetMesh()->GetEngine();
   gVecL2.Resize(ng.MakeLayout(l2sz));
   gVecH1.Resize(ng.MakeLayout(h1sz));
}

// *****************************************************************************
void kForcePAOperator::Mult(const mfem::Vector &vecL2,
                            mfem::Vector &vecH1) const {
   push();
   const kernels::Vector rVecL2 = vecL2.Get_PVector()->As<const kernels::Vector>();
   kernels::Vector rgVecL2 = gVecL2.Get_PVector()->As<kernels::Vector>();
   kernels::Vector rVecH1 = vecH1.Get_PVector()->As<kernels::Vector>();
   kernels::Vector rgVecH1 = gVecH1.Get_PVector()->As<kernels::Vector>();
   l2k.GlobalToLocal(rVecL2, rgVecL2);
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
              (const double*)rgVecL2.KernelsMem().ptr(),
              (double*)rgVecH1.KernelsMem().ptr());
   h1k.LocalToGlobal(rgVecH1, rVecH1);
   pop();
}

// *************************************************************************
void kForcePAOperator::MultTranspose(const mfem::Vector &vecH1,
                                     mfem::Vector &vecL2) const {
   push();
   // vecH1 & vecL2 are now on the host, wrap them with k*
   Vector kVecH1(h1fes.GetVLayout());
   Vector kVecL2(l2fes.GetVLayout());
   // push vecH1's data down to the device
   kVecH1.PushData(vecH1.GetData());
   const kernels::Vector rVecH1 = kVecH1.Get_PVector()->As<const kernels::Vector>();
   kernels::Vector rgVecH1 = gVecH1.Get_PVector()->As<kernels::Vector>();
   kernels::Vector rgVecL2 = gVecL2.Get_PVector()->As<kernels::Vector>();
   kernels::Vector rVecL2 = kVecL2.Get_PVector()->As<kernels::Vector>();
   h1k.GlobalToLocal(rVecH1, rgVecH1);
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
                       (const double*)rgVecH1.KernelsMem().ptr(),
                       (double*)rgVecL2.KernelsMem().ptr());
   dbg("LocalToGlobal");
   l2k.LocalToGlobal(rgVecL2, rVecL2);
   // back to the host argument
   vecL2 = kVecL2;
   pop();
}

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI
