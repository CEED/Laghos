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
#include "kForceOperator.hpp"
#include "backends/kernels/kernels.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{

namespace hydrodynamics
{

// *****************************************************************************
kForceOperator::kForceOperator(ParFiniteElementSpace &h1f,
                               ParFiniteElementSpace &l2f,
                               const IntegrationRule &ir,
                               const QuadratureData *qd,
                               const bool engine)
   : Operator(l2f.GetTrueVSize(), h1f.GetTrueVSize()),
     dim(h1f.GetMesh()->Dimension()),
     nzones(h1f.GetMesh()->GetNE()),
     h1fes(h1f),
     l2fes(l2f),
     integ_rule(ir),
     quad_data(qd),
     gVecL2(l2fes.GetFE(0)->GetDof() * nzones),
     gVecH1(h1fes.GetVDim() * h1fes.GetFE(0)->GetDof() * nzones) {
   if (!engine) return;
   // push down to device the two vectors gVecL2 & gVecH1
   const Engine &ng = l2f.GetMesh()->GetEngine();
   gVecL2.Resize(ng.MakeLayout(l2fes.GetFE(0)->GetDof() * nzones));
   gVecH1.Resize(ng.MakeLayout(h1fes.GetVDim() *h1fes.GetFE(0)->GetDof() * nzones));
   //GetParFESpace
   h1D2Q = kernels::KernelsDofQuadMaps::Get(h1fes, integ_rule);
   l2D2Q = kernels::KernelsDofQuadMaps::Get(l2fes, integ_rule);
}
  
// *****************************************************************************
kForceOperator::~kForceOperator(){}


// *************************************************************************
void kForceOperator::Mult(const mfem::Vector &vecL2,
                          mfem::Vector &vecH1) const {
   push();
   const kernels::KernelsFiniteElementSpace &rl2 = *l2fes.Get_PFESpace().As<kernels::KernelsFiniteElementSpace>();
   const kernels::KernelsFiniteElementSpace &rh1 = *h1fes.Get_PFESpace().As<kernels::KernelsFiniteElementSpace>();
   const kernels::Vector rVecL2 = vecL2.Get_PVector()->As<const kernels::Vector>();
   kernels::Vector rgVecL2 = gVecL2.Get_PVector()->As<kernels::Vector>();
   kernels::Vector rVecH1 = vecH1.Get_PVector()->As<kernels::Vector>();
   kernels::Vector rgVecH1 = gVecH1.Get_PVector()->As<kernels::Vector>();
   dbg("GlobalToLocal");
   rl2.GlobalToLocal(rVecL2, rgVecL2);
   //dbg("rgVecL2:\n"); rgVecL2.Print();
   const int NUM_DOFS_1D = h1fes.GetFE(0)->GetOrder()+1;
   const IntegrationRule &ir1D = IntRules.Get(Geometry::SEGMENT, integ_rule.GetOrder());
   const int NUM_QUAD_1D  = ir1D.GetNPoints();
   const int L2_DOFS_1D = l2fes.GetFE(0)->GetOrder()+1;
   const int H1_DOFS_1D = h1fes.GetFE(0)->GetOrder()+1;
   dbg("rForceMult");
   dbg("rForceMult: dim=%d, NUM_DOFS_1D=%d, NUM_QUAD_1D=%d, nzones=%d",dim,NUM_DOFS_1D,NUM_QUAD_1D, nzones);
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
   dbg("LocalToGlobal");
   rh1.LocalToGlobal(rgVecH1, rVecH1);
   pop();
}

// *************************************************************************
void kForceOperator::MultTranspose(const Vector &vecH1,
                                   Vector &vecL2) const {
   push();
   // vecH1 & vecL2 are now on the host, wrap them with k*
   Vector kVecH1(h1fes.GetVLayout());
   Vector kVecL2(l2fes.GetVLayout());
   // push vecH1's data down to the device
   kVecH1.PushData(vecH1.GetData());
   // switch to backend mode
   const kernels::KernelsFiniteElementSpace &rl2 = *l2fes.Get_PFESpace().As<kernels::KernelsFiniteElementSpace>();
   const kernels::KernelsFiniteElementSpace &rh1 = *h1fes.Get_PFESpace().As<kernels::KernelsFiniteElementSpace>();
   const kernels::Vector rVecH1 = kVecH1.Get_PVector()->As<const kernels::Vector>();
   kernels::Vector rgVecH1 = gVecH1.Get_PVector()->As<kernels::Vector>();
   kernels::Vector rgVecL2 = gVecL2.Get_PVector()->As<kernels::Vector>();
   kernels::Vector rVecL2 = kVecL2.Get_PVector()->As<kernels::Vector>();
   // **************************************************************************
   dbg("GlobalToLocal");
   rh1.GlobalToLocal(rVecH1, rgVecH1);
   // **************************************************************************
   const int NUM_DOFS_1D = h1fes.GetFE(0)->GetOrder()+1;
   const IntegrationRule &ir1D = IntRules.Get(Geometry::SEGMENT, integ_rule.GetOrder());
   const int NUM_QUAD_1D  = ir1D.GetNPoints();
   const int L2_DOFS_1D = l2fes.GetFE(0)->GetOrder()+1;
   const int H1_DOFS_1D = h1fes.GetFE(0)->GetOrder()+1;
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
   // **************************************************************************
   dbg("LocalToGlobal");
   rl2.LocalToGlobal(rgVecL2, rVecL2);
   // back to the host argument
   vecL2 = kVecL2;
   pop();
}

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI
