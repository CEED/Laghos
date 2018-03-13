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

namespace mfem
{

namespace hydrodynamics
{

QuadratureData::QuadratureData(int dim,
                               int nzones,
                               int nqp)
{ Setup(dim, nzones, nqp); }


void QuadratureData::Setup(int dim,
                           int nzones,
                           int nqp)
{
  push(Wheat);
   rho0DetJ0w.SetSize(nqp * nzones);
   stressJinvT.SetSize(dim * dim * nqp * nzones);
   dtEst.SetSize(nqp * nzones);
   pop();
}

void DensityIntegrator::AssembleRHSElementVect(const FiniteElement &fe,
                                               ElementTransformation &Tr,
                                               Vector &elvect)
{
   push(Wheat);
   const int ip_cnt = integ_rule.GetNPoints();
   Vector shape(fe.GetDof());
   Vector rho0DetJ0w = quad_data.rho0DetJ0w;
   
   elvect.SetSize(fe.GetDof());
   elvect = 0.0;

   for (int q = 0; q < ip_cnt; q++)
   {
      fe.CalcShape(integ_rule.IntPoint(q), shape);
      // Note that rhoDetJ = rho0DetJ0.
      shape *= rho0DetJ0w(Tr.ElementNo*ip_cnt + q);
      elvect += shape;
   }
   pop();
}

// *****************************************************************************
RajaMassOperator::RajaMassOperator(RajaFiniteElementSpace &fes_,
                                   const IntegrationRule &integ_rule_,
                                   QuadratureData *quad_data_)
   : RajaOperator(fes_.GetTrueVSize()),
     fes(fes_),
     integ_rule(integ_rule_),
     ess_tdofs_count(0),
     bilinearForm(&fes),
     quad_data(quad_data_),
     x_gf(fes),
     y_gf(fes) {}

// *****************************************************************************
RajaMassOperator::~RajaMassOperator(){
}

// *****************************************************************************
void RajaMassOperator::Setup()
{
   push(Wheat);
   dim=fes.GetMesh()->Dimension();
   nzones=fes.GetMesh()->GetNE();
   RajaMassIntegrator &massInteg = *(new RajaMassIntegrator());
   massInteg.SetIntegrationRule(integ_rule);
   massInteg.SetOperator(quad_data->rho0DetJ0w);
   bilinearForm.AddDomainIntegrator(&massInteg);
   bilinearForm.Assemble();
   bilinearForm.FormOperator(Array<int>(), massOperator);
   pop();
}

// *************************************************************************
void RajaMassOperator::SetEssentialTrueDofs(Array<int> &dofs)
{
  push(Wheat);
  dbg("\n\033[33;1m[SetEssentialTrueDofs] dofs.Size()=%d\033[m",dofs.Size());
  ess_tdofs_count = dofs.Size();
  
  if (ess_tdofs.Size()==0){
#ifdef MFEM_USE_MPI
    int global_ess_tdofs_count;
    const MPI_Comm comm = fes.GetParMesh()->GetComm();
    MPI_Allreduce(&ess_tdofs_count,&global_ess_tdofs_count,
                  1, MPI_INT, MPI_SUM, comm);
    assert(global_ess_tdofs_count>0);
    ess_tdofs.allocate(global_ess_tdofs_count);
#else
    assert(ess_tdofs_count>0);
    ess_tdofs.allocate(ess_tdofs_count);
#endif
  }else assert(ess_tdofs_count<=ess_tdofs.Size());

  assert(ess_tdofs>0);
  
  if (ess_tdofs_count == 0) { pop(); return; }
  
  {
    push(H2D:ess_tdofs,Red);
    assert(ess_tdofs_count>0);
    assert(dofs.GetData());
    rHtoD(ess_tdofs.ptr(),dofs.GetData(),ess_tdofs_count*sizeof(int));
    pop();
  }
  pop();
}

// *****************************************************************************
void RajaMassOperator::EliminateRHS(RajaVector &b)
{
  push(Wheat);
  if (ess_tdofs_count > 0)
    b.SetSubVector(ess_tdofs, 0.0, ess_tdofs_count);
  pop();
}

// *************************************************************************
void RajaMassOperator::Mult(const RajaVector &x, RajaVector &y) const
{
  push(Wheat);
   
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
  pop();
}

  
// *****************************************************************************
// * RajaForceOperator
// *****************************************************************************
RajaForceOperator::RajaForceOperator(RajaFiniteElementSpace &h1fes_,
                                     RajaFiniteElementSpace &l2fes_,
                                     const IntegrationRule &integ_rule_,
                                     const QuadratureData *quad_data_)
   : RajaOperator(l2fes_.GetTrueVSize(), h1fes_.GetTrueVSize()),
     dim(h1fes_.GetMesh()->Dimension()),
     nzones(h1fes_.GetMesh()->GetNE()),
     h1fes(h1fes_),
     l2fes(l2fes_),
     integ_rule(integ_rule_),
     quad_data(quad_data_),
     gVecL2(l2fes.GetLocalDofs() * nzones),
     gVecH1(h1fes.GetVDim() * h1fes.GetLocalDofs() * nzones) { }
  
// *****************************************************************************
RajaForceOperator::~RajaForceOperator(){}

// *************************************************************************
void RajaForceOperator::Setup()
{
   h1D2Q = RajaDofQuadMaps::Get(h1fes, integ_rule);
   l2D2Q = RajaDofQuadMaps::Get(l2fes, integ_rule);
}

// *************************************************************************
void RajaForceOperator::Mult(const RajaVector &vecL2,
                             RajaVector &vecH1) const {
   push(Wheat);
   l2fes.GlobalToLocal(vecL2, gVecL2);
   const int NUM_DOFS_1D = h1fes.GetFE(0)->GetOrder()+1;
   const IntegrationRule &ir1D = IntRules.Get(Geometry::SEGMENT, integ_rule.GetOrder());
   const int NUM_QUAD_1D  = ir1D.GetNPoints();
   const int L2_DOFS_1D = l2fes.GetFE(0)->GetOrder()+1;
   const int H1_DOFS_1D = h1fes.GetFE(0)->GetOrder()+1;
   if (rconfig::Get().Share())
     rForceMultS(dim,
                 NUM_DOFS_1D,
                 NUM_QUAD_1D,
                 L2_DOFS_1D,
                 H1_DOFS_1D,
                 nzones,
                 l2D2Q->dofToQuad,
                 h1D2Q->quadToDof,
                 h1D2Q->quadToDofD,
                 quad_data->stressJinvT,
                 gVecL2,
                 gVecH1);
   else
     rForceMult(dim,
                NUM_DOFS_1D,
                NUM_QUAD_1D,
                L2_DOFS_1D,
                H1_DOFS_1D,
                nzones,
                l2D2Q->dofToQuad,
                h1D2Q->quadToDof,
                h1D2Q->quadToDofD,
                quad_data->stressJinvT,
                gVecL2,
                gVecH1);
   h1fes.LocalToGlobal(gVecH1, vecH1);
   pop();
}

// *************************************************************************
void RajaForceOperator::MultTranspose(const RajaVector &vecH1,
                                      RajaVector &vecL2) const {
   push(Wheat);
   h1fes.GlobalToLocal(vecH1, gVecH1);
   const int NUM_DOFS_1D = h1fes.GetFE(0)->GetOrder()+1;
   const IntegrationRule &ir1D = IntRules.Get(Geometry::SEGMENT, integ_rule.GetOrder());
   const int NUM_QUAD_1D  = ir1D.GetNPoints();
   const int L2_DOFS_1D = l2fes.GetFE(0)->GetOrder()+1;
   const int H1_DOFS_1D = h1fes.GetFE(0)->GetOrder()+1;
   if (rconfig::Get().Share())
     rForceMultTransposeS(dim,
                          NUM_DOFS_1D,
                          NUM_QUAD_1D,
                          L2_DOFS_1D,
                          H1_DOFS_1D,
                          nzones,
                          l2D2Q->quadToDof,
                          h1D2Q->dofToQuad,
                          h1D2Q->dofToQuadD,
                          quad_data->stressJinvT,
                          gVecH1,
                          gVecL2);
   else
     rForceMultTranspose(dim,
                         NUM_DOFS_1D,
                         NUM_QUAD_1D,
                         L2_DOFS_1D,
                         H1_DOFS_1D,
                         nzones,
                         l2D2Q->quadToDof,
                         h1D2Q->dofToQuad,
                         h1D2Q->dofToQuadD,
                         quad_data->stressJinvT,
                         gVecH1,
                         gVecL2);
   l2fes.LocalToGlobal(gVecL2, vecL2);
   pop();
}

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI
