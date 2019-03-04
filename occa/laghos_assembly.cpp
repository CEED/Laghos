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

#include "laghos_assembly.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

MpiOstream mpiout;

namespace mfem
{
namespace hydrodynamics
{
QuadratureData::QuadratureData(int dim,
                               int elements,
                               int nqp)
{
   Setup(occa::getDevice(), dim, elements, nqp);
}

QuadratureData::QuadratureData(occa::device device_,
                               int dim,
                               int elements,
                               int nqp)
{
   Setup(device_, dim, elements, nqp);
}

void QuadratureData::Setup(occa::device device_,
                           int dim,
                           int elements,
                           int nqp)
{
   device = device_;

   rho0DetJ0w.SetSize(device, nqp * elements);
   stressJinvT.SetSize(device, dim * dim * nqp * elements);
   dtEst.SetSize(device, nqp * elements);
}

void DensityIntegrator::AssembleRHSElementVect(const FiniteElement &fe,
                                               ElementTransformation &Tr,
                                               const IntegrationRule &integ_rule,
                                               Vector &rho0DetJ0w,
                                               Vector &elvect)
{
   const int ip_cnt = integ_rule.GetNPoints();
   Vector shape(fe.GetDof());

   elvect.SetSize(fe.GetDof());
   elvect = 0.0;

   for (int q = 0; q < ip_cnt; q++)
   {
      fe.CalcShape(integ_rule.IntPoint(q), shape);
      // Note that rhoDetJ = rho0DetJ0.
      shape *= rho0DetJ0w(Tr.ElementNo*ip_cnt + q);
      elvect += shape;
   }
}

OccaMassOperator::OccaMassOperator(OccaFiniteElementSpace &fes_,
                                   const IntegrationRule &integ_rule_,
                                   QuadratureData *quad_data_)
   : Operator(fes_.GetTrueVSize()),
     device(occa::getDevice()),
     fes(fes_),
     integ_rule(integ_rule_),
     bilinearForm(&fes),
     quad_data(quad_data_),
     x_gf(device, &fes),
     y_gf(device, &fes) {}

OccaMassOperator::OccaMassOperator(occa::device device_,
                                   OccaFiniteElementSpace &fes_,
                                   const IntegrationRule &integ_rule_,
                                   QuadratureData *quad_data_)
   : Operator(fes_.GetTrueVSize()),
     device(device_),
     fes(fes_),
     integ_rule(integ_rule_),
     bilinearForm(&fes),
     quad_data(quad_data_),
     x_gf(device, &fes),
     y_gf(device, &fes) {}

void OccaMassOperator::Setup()
{
   dim = fes.GetMesh()->Dimension();
   elements = fes.GetMesh()->GetNE();

   ess_tdofs_count = 0;

   OccaMassIntegrator &massInteg = *(new OccaMassIntegrator());
   massInteg.SetIntegrationRule(integ_rule);
   massInteg.SetOperator(quad_data->rho0DetJ0w);

   bilinearForm.AddDomainIntegrator(&massInteg);
   bilinearForm.Assemble();

   bilinearForm.FormOperator(Array<int>(), massOperator);
}

void OccaMassOperator::SetEssentialTrueDofs(Array<int> &dofs)
{
   ess_tdofs_count = dofs.Size();
   if (ess_tdofs_count == 0)
   {
      return;
   }
   if (ess_tdofs.size<int>() < ess_tdofs_count)
   {
      ess_tdofs = device.malloc(ess_tdofs_count * sizeof(int),
                                dofs.GetData());
   }
   else
   {
      ess_tdofs.copyFrom(dofs.GetData(),
                         ess_tdofs_count * sizeof(int));
   }
}

void OccaMassOperator::Mult(const OccaVector &x, OccaVector &y) const
{
   distX = x;
   if (ess_tdofs_count)
   {
      distX.SetSubVector(ess_tdofs, 0.0, ess_tdofs_count);
   }

   massOperator->Mult(distX, y);

   if (ess_tdofs_count)
   {
      y.SetSubVector(ess_tdofs, 0.0, ess_tdofs_count);
   }
}

void OccaMassOperator::EliminateRHS(OccaVector &b)
{
   if (ess_tdofs_count)
   {
      b.SetSubVector(ess_tdofs, 0.0, ess_tdofs_count);
   }
}

OccaForceOperator::OccaForceOperator(OccaFiniteElementSpace &h1fes_,
                                     OccaFiniteElementSpace &l2fes_,
                                     const IntegrationRule &integ_rule_,
                                     QuadratureData *quad_data_)
   : Operator(l2fes_.GetTrueVSize(), h1fes_.GetTrueVSize()),
     device(occa::getDevice()),
     dim(h1fes_.GetMesh()->Dimension()),
     elements(h1fes_.GetMesh()->GetNE()),
     h1fes(h1fes_),
     l2fes(l2fes_),
     integ_rule(integ_rule_),
     quad_data(quad_data_),
     gVecL2(device, l2fes.GetLocalDofs() * elements),
     gVecH1(device, h1fes.GetVDim() * h1fes.GetLocalDofs() * elements) {}

OccaForceOperator::OccaForceOperator(occa::device device_,
                                     OccaFiniteElementSpace &h1fes_,
                                     OccaFiniteElementSpace &l2fes_,
                                     const IntegrationRule &integ_rule_,
                                     QuadratureData *quad_data_)
   : Operator(l2fes_.GetTrueVSize(), h1fes_.GetTrueVSize()),
     device(device_),
     dim(h1fes_.GetMesh()->Dimension()),
     elements(h1fes_.GetMesh()->GetNE()),
     h1fes(h1fes_),
     l2fes(l2fes_),
     integ_rule(integ_rule_),
     quad_data(quad_data_),
     gVecL2(device, l2fes.GetLocalDofs() * elements),
     gVecH1(device, h1fes.GetVDim() * h1fes.GetLocalDofs() * elements) {}

void OccaForceOperator::Setup()
{
   occa::properties h1Props, l2Props, props;
   SetProperties(h1fes, integ_rule, h1Props);
   SetProperties(l2fes, integ_rule, l2Props);

   props = h1Props;
   props["defines/L2_DOFS_1D"] = l2Props["defines/NUM_DOFS_1D"];
   props["defines/H1_DOFS_1D"] = h1Props["defines/NUM_DOFS_1D"];

   multKernel = device.buildKernel("occa://laghos/force.okl",
                                   stringWithDim("Mult", dim),
                                   props);

   multTransposeKernel = device.buildKernel("occa://laghos/force.okl",
                                            stringWithDim("MultTranspose", dim),
                                            props);

   h1D2Q = OccaDofQuadMaps::Get(device, h1fes, integ_rule);
   l2D2Q = OccaDofQuadMaps::Get(device, l2fes, integ_rule);
}

void OccaForceOperator::Mult(const OccaVector &vecL2, OccaVector &vecH1) const
{
   l2fes.GlobalToLocal(vecL2, gVecL2);

   multKernel(elements,
              l2D2Q.dofToQuad,
              h1D2Q.quadToDof,
              h1D2Q.quadToDofD,
              quad_data->stressJinvT,
              gVecL2,
              gVecH1);

   h1fes.LocalToGlobal(gVecH1, vecH1);
}

void OccaForceOperator::MultTranspose(const OccaVector &vecH1,
                                      OccaVector &vecL2) const
{
   h1fes.GlobalToLocal(vecH1, gVecH1);

   multTransposeKernel(elements,
                       l2D2Q.quadToDof,
                       h1D2Q.dofToQuad,
                       h1D2Q.dofToQuadD,
                       quad_data->stressJinvT,
                       gVecH1,
                       gVecL2);

   l2fes.LocalToGlobal(gVecL2, vecL2);
}
} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI
