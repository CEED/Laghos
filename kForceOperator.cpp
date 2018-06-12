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
#include "kForceOperator.hpp"
#include "backends/raja/raja.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{

namespace hydrodynamics
{

kForceOperator::kForceOperator(ParFiniteElementSpace &h1f,
                               ParFiniteElementSpace &l2f,
                               const IntegrationRule &ir,
                               const QuadratureData *qd)
   : Operator(l2f.GetTrueVSize(), h1f.GetTrueVSize()),
     dim(h1f.GetMesh()->Dimension()),
     nzones(h1f.GetMesh()->GetNE()),
     h1fes(*h1f.Get_PFESpace().As<raja::RajaFiniteElementSpace>()),
     l2fes(*l2f.Get_PFESpace().As<raja::RajaFiniteElementSpace>()),
     integ_rule(ir),
     quad_data(qd),
     gVecL2(l2fes.GetFE(0)->GetDof() * nzones),
     gVecH1(h1fes.GetVDim() * h1fes.GetFE(0)->GetDof() * nzones) {
   // push down to device the two vectors gVecL2 & gVecH1
   const Engine &engine = l2f.GetMesh()->GetEngine();
   gVecL2.Resize(engine.MakeLayout(l2fes.GetFE(0)->GetDof() * nzones));
   gVecH1.Resize(engine.MakeLayout(h1fes.GetVDim() * h1fes.GetFE(0)->GetDof() * nzones));
}
  
// *****************************************************************************
kForceOperator::~kForceOperator(){}

// *************************************************************************
void kForceOperator::Setup()
{
   h1D2Q = raja::RajaDofQuadMaps::Get(h1fes.GetParFESpace(), integ_rule);
   l2D2Q = raja::RajaDofQuadMaps::Get(l2fes.GetParFESpace(), integ_rule);
}

// *************************************************************************
void kForceOperator::Mult(const mfem::Vector &vecL2,
                          mfem::Vector &vecH1) const {
   push();
   //const DFiniteElementSpace &dl2 = l2fes.Get_PFESpace();
   //const mfem::Engine &eng = l2fes.GetMesh()->GetEngine();
   //const raja::RajaFiniteElementSpace *rl2 = dl2.As<raja::RajaFiniteElementSpace>();
   //const raja::RajaFiniteElementSpace *rl2 = l2fes.Get_PFESpace().As<raja::RajaFiniteElementSpace>();
   //const raja::RajaFiniteElementSpace *rh1 = h1fes.Get_PFESpace().As<raja::RajaFiniteElementSpace>();
   const raja::Vector rVecL2 = vecL2.Get_PVector()->As<const raja::Vector>();
   raja::Vector rgVecL2 = gVecL2.Get_PVector()->As<raja::Vector>();
   raja::Vector rVecH1 = vecH1.Get_PVector()->As<raja::Vector>();
   raja::Vector rgVecH1 = gVecH1.Get_PVector()->As<raja::Vector>();
   dbg("GlobalToLocal");
   l2fes.GlobalToLocal(rVecL2, rgVecL2);
   const int NUM_DOFS_1D = h1fes.GetFE(0)->GetOrder()+1;
   const IntegrationRule &ir1D = IntRules.Get(Geometry::SEGMENT, integ_rule.GetOrder());
   const int NUM_QUAD_1D  = ir1D.GetNPoints();
   const int L2_DOFS_1D = l2fes.GetFE(0)->GetOrder()+1;
   const int H1_DOFS_1D = h1fes.GetFE(0)->GetOrder()+1;
   dbg("rForceMult: dim=%d, NUM_DOFS_1D=%d, NUM_QUAD_1D=%d, nzones=%d",dim,NUM_DOFS_1D,NUM_QUAD_1D, nzones);
   assert(l2D2Q->dofToQuad);
   assert(h1D2Q->quadToDof);
   assert(h1D2Q->quadToDofD);
   assert(quad_data->stressJinvT.Data());
   assert(rgVecL2.RajaMem());
   assert(rgVecH1.RajaMem());

#define NUM_DIM 2
#define NUM_QUAD NUM_QUAD_1D*NUM_QUAD_1D
// this: (i,j,q,e,D,Q) i*D*D*Q + (e*Q+q) + j*D*D*D*Q
//#define ijklNM(i,j,q,e,D,Q) i*D*D*Q + (e*Q+q) + j*D*D*D*Q
#define     ijklNM(i,j,k,l,N,M) (i)+(N)*((j)+(N)*((k)+(M)*(l)))
      
#define new_ijklNM(i,j,k,l,N,M) (k)+(M)*((l)+(N*N)*((i)+(N)*j))
/*   
   double *stressJinvT = quad_data->stressJinvT.Data();
   for(int el=0;el<nzones;el+=1){
      printf("\nElem #%d",el);
      for (int q = 0; q < NUM_QUAD; ++q) {
         printf("\n\tQ #%d ",q);
         for (int i = 0; i < NUM_DIM; ++i) {
            for (int j = 0; j < NUM_DIM; ++j) {
               printf(" %f", stressJinvT[new_ijklNM(i,j,q,el,NUM_DIM,NUM_QUAD)]);
            }
         }
      }
   }
   
   for(int el=0;el<nzones;el+=1){
      printf("\nElem #%d",el);
      for (int q = 0; q < NUM_QUAD; ++q) {
         printf("\n\tQ #%d: ",q);
         for (int i = 0; i < NUM_DIM; ++i) {
            for (int j = 0; j < NUM_DIM; ++j) {
               printf(" %f", stressJinvT[ijklNM(i,j,q,el,NUM_DIM,NUM_QUAD)]);
            }
         }
      }
      }
*/
   /*
   printf("\nElem #%d",el);
   for (int q = 0; q < NUM_QUAD; ++q) {
   printf("\n\tQ #%d: ",q);
   printf(" %f %f %f %f",
   stressJinvT[ijklNM(0,0,q,el,NUM_DIM,NUM_QUAD)],
   stressJinvT[ijklNM(1,0,q,el,NUM_DIM,NUM_QUAD)],
                stressJinvT[ijklNM(0,1,q,el,NUM_DIM,NUM_QUAD)],
                stressJinvT[ijklNM(1,1,q,el,NUM_DIM,NUM_QUAD)]);
      }
      }*/
   
   rForceMult(dim,
              NUM_DOFS_1D,
              NUM_QUAD_1D,
              L2_DOFS_1D,
              H1_DOFS_1D,
              nzones,
              l2D2Q->dofToQuad,
              h1D2Q->quadToDof,
              h1D2Q->quadToDofD,
              (const double*)quad_data->stressJinvT.Data(),
              (const double*)rgVecL2.RajaMem().ptr(),
              (double*)rgVecH1.RajaMem().ptr());
   dbg("LocalToGlobal");
   h1fes.LocalToGlobal(rgVecH1, rVecH1);
   pop();
}

// *************************************************************************
void kForceOperator::MultTranspose(const Vector &vecH1,
                                   Vector &vecL2) const {
   push();
   const raja::Vector rVecH1 = vecH1.Get_PVector()->As<const raja::Vector>();
   raja::Vector rgVecH1 = gVecH1.Get_PVector()->As<raja::Vector>();
   raja::Vector rgVecL2 = gVecL2.Get_PVector()->As<raja::Vector>();
   raja::Vector rVecL2 = vecL2.Get_PVector()->As<const raja::Vector>();
   dbg("GlobalToLocal");
   h1fes.GlobalToLocal(rVecH1, rgVecH1);
   const int NUM_DOFS_1D = h1fes.GetFE(0)->GetOrder()+1;
   const IntegrationRule &ir1D = IntRules.Get(Geometry::SEGMENT, integ_rule.GetOrder());
   const int NUM_QUAD_1D  = ir1D.GetNPoints();
   const int L2_DOFS_1D = l2fes.GetFE(0)->GetOrder()+1;
   const int H1_DOFS_1D = h1fes.GetFE(0)->GetOrder()+1;
   dbg("rForceMultTranspose: dim=%d, NUM_DOFS_1D=%d, NUM_QUAD_1D=%d, nzones=%d",dim,NUM_DOFS_1D,NUM_QUAD_1D, nzones);
   rForceMultTranspose(dim,
                       NUM_DOFS_1D,
                       NUM_QUAD_1D,
                       L2_DOFS_1D,
                       H1_DOFS_1D,
                       nzones,
                       l2D2Q->quadToDof,
                       h1D2Q->dofToQuad,
                       h1D2Q->dofToQuadD,
                       (double*)quad_data->stressJinvT.Data(),
                       rgVecH1.RajaMem(),
                       rgVecL2.RajaMem());
   dbg("LocalToGlobal");
   l2fes.LocalToGlobal(rgVecL2, rVecL2);
   pop();
}

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI
