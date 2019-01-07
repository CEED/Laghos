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
#include "laghos_assembly_kernels.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{

namespace hydrodynamics
{

// *****************************************************************************
// * Kernel MASS Operator
// *****************************************************************************
kMassPAOperator::kMassPAOperator(QuadratureData *qd_,
                                 ParFiniteElementSpace &pfes_,
                                 const IntegrationRule &ir_) :
   AbcMassPAOperator(pfes_.GetTrueVSize()),
   dim(pfes_.GetMesh()->Dimension()),
   nzones(pfes_.GetMesh()->GetNE()),
   quad_data(qd_),
   pfes(pfes_),
   fes(static_cast<FiniteElementSpace*>(&pfes_)),
   ir(ir_),
   ess_tdofs_count(0),
   ess_tdofs(0),
   paBilinearForm(new mfem::ParBilinearForm(&pfes,
                                            AssemblyLevel::PARTIAL,
                                            pfes.GetMesh()->GetNE())),
   massOperator(NULL)
{ }

// *****************************************************************************
void kMassPAOperator::Setup()
{
   // PAMassIntegrator Setup
   mfem::PAMassIntegrator *paMassInteg = new mfem::PAMassIntegrator();
   // No setup, it is done in PABilinearForm::Assemble
   //paMassInteg->Setup(fes,&ir);
   assert(ir);
   paMassInteg->SetIntegrationRule(ir); // in NonlinearFormIntegrator
   // Add mass integretor to PA bilinear form
   paBilinearForm->AddDomainIntegrator(paMassInteg);
   paBilinearForm->Assemble();
   // Setup has to be done before, which is done in ->Assemble above
   paMassInteg->SetOperator(quad_data->rho0DetJ0w);
   paBilinearForm->FormSystemOperator(mfem::Array<int>(), massOperator);
}

// *************************************************************************
void kMassPAOperator::SetEssentialTrueDofs(mfem::Array<int> &dofs)
{
   ess_tdofs_count = dofs.Size();
   if (ess_tdofs.Size()==0){
      int global_ess_tdofs_count;
      const MPI_Comm comm = pfes.GetParMesh()->GetComm();
      MPI_Allreduce(&ess_tdofs_count,&global_ess_tdofs_count,
                    1, MPI_INT, MPI_SUM, comm);
      assert(global_ess_tdofs_count>0);
      ess_tdofs.SetSize(global_ess_tdofs_count);
   }
   if (ess_tdofs_count == 0)
   {
      return;
   }
   ess_tdofs = dofs;
}

// *****************************************************************************
void kMassPAOperator::EliminateRHS(mfem::Vector &b)
{
   if (ess_tdofs_count > 0){
      b.SetSubVector(ess_tdofs, 0.0);
   }
}

// *************************************************************************
void kMassPAOperator::Mult(const mfem::Vector &x,
                                 mfem::Vector &y) const
{   
   if (distX.Size()!=x.Size()) {
      distX.SetSize(x.Size());
   }
   assert(distX.Size()==x.Size());
   distX = x;
   if (ess_tdofs_count)
   {
      distX.SetSubVector(ess_tdofs, 0.0);
   }
   massOperator->Mult(distX, y);
   if (ess_tdofs_count)
   {
      y.SetSubVector(ess_tdofs, 0.0);
   }
}

// *****************************************************************************
// * Kernel FORCE Operator
// *****************************************************************************
kForcePAOperator::kForcePAOperator(QuadratureData *qd,
                                   ParFiniteElementSpace &h1f,
                                   ParFiniteElementSpace &l2f,
                                   const IntegrationRule &ir) :
   AbcForcePAOperator(),
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
   gVecL2.SetSize(l2sz);
   gVecH1.SetSize(h1sz);
}

// *****************************************************************************
template<const int NUM_DIM,
         const int NUM_DOFS_1D,
         const int NUM_QUAD_1D,
         const int L2_DOFS_1D,
         const int H1_DOFS_1D> static
void kForceMult2D(const int numElements,
                  const double* L2DofToQuad,
                  const double* H1QuadToDof,
                  const double* H1QuadToDofD,
                  const double* stressJinvT,
                  const double* e,
                  double* v) {
  const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
  MFEM_FORALL(el,numElements,
  {
    double e_xy[NUM_QUAD_2D];
    for (int i = 0; i < NUM_QUAD_2D; ++i) {
      e_xy[i] = 0;
    }
    for (int dy = 0; dy < L2_DOFS_1D; ++dy) {
      double e_x[NUM_QUAD_1D];
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        e_x[qy] = 0;
      }
      for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
        const double r_e = e[ijkN(dx,dy,el,L2_DOFS_1D)];
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          e_x[qx] += L2DofToQuad[ijN(qx,dx,NUM_QUAD_1D)] * r_e;
        }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        const double wy = L2DofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          e_xy[ijN(qx,qy,NUM_QUAD_1D)] += wy * e_x[qx];
        }
      }
    }
    for (int c = 0; c < 2; ++c) {
      for (int dy = 0; dy < H1_DOFS_1D; ++dy) {
        for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
          v[_ijklNM(c,dx,dy,el,NUM_DOFS_1D,numElements)] = 0.0;
        }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        double Dxy[H1_DOFS_1D];
        double xy[H1_DOFS_1D];
        for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
          Dxy[dx] = 0.0;
          xy[dx]  = 0.0;
        }
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
           const double esx = e_xy[ijN(qx,qy,NUM_QUAD_1D)]  *
             stressJinvT[__ijklmNM(0,c,qx,qy,el,NUM_DIM,NUM_QUAD_1D,numElements)];
           const double esy = e_xy[ijN(qx,qy,NUM_QUAD_1D)] *
             stressJinvT[__ijklmNM(1,c,qx,qy,el,NUM_DIM,NUM_QUAD_1D,numElements)];
          for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
            Dxy[dx] += esx * H1QuadToDofD[ijN(dx,qx,H1_DOFS_1D)];
            xy[dx]  += esy * H1QuadToDof[ijN(dx,qx,H1_DOFS_1D)];
          }
        }
        for (int dy = 0; dy < H1_DOFS_1D; ++dy) {
          const double wy  = H1QuadToDof[ijN(dy,qy,H1_DOFS_1D)];
          const double wDy = H1QuadToDofD[ijN(dy,qy,H1_DOFS_1D)];
          for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
            v[_ijklNM(c,dx,dy,el,NUM_DOFS_1D,numElements)] += wy* Dxy[dx] + wDy*xy[dx];
          }
        }
      }
    }
  });
}

// *****************************************************************************
#define TkForceMult2D(d,D,Q,L,H)                                        \
   template void kForceMult2D<d,D,Q,L,H>(const int numElements,         \
                                         const double* L2DofToQuad,     \
                                         const double* H1QuadToDof,     \
                                         const double* H1QuadToDofD,    \
                                         const double* stressJinvT,     \
                                         const double* e,               \
                                         double* v)
TkForceMult2D(2,2,2,1,2);
TkForceMult2D(2,3,4,2,3);
TkForceMult2D(2,4,6,3,4);
TkForceMult2D(2,5,8,4,5);
TkForceMult2D(2,6,10,5,6);
TkForceMult2D(2,7,12,6,7);
TkForceMult2D(2,8,14,7,8);
TkForceMult2D(2,9,16,8,9);
TkForceMult2D(2,10,18,9,10);
TkForceMult2D(2,11,20,10,11);
TkForceMult2D(2,12,22,11,12);
TkForceMult2D(2,13,24,12,13);
TkForceMult2D(2,14,26,13,14);
TkForceMult2D(2,15,28,14,15);
TkForceMult2D(2,16,30,15,16);
TkForceMult2D(2,17,32,16,17);

// *****************************************************************************
template<const int NUM_DIM,
         const int NUM_DOFS_1D,
         const int NUM_QUAD_1D,
         const int L2_DOFS_1D,
         const int H1_DOFS_1D> static
void kForceMult3D(const int numElements,
                  const double* L2DofToQuad,
                  const double* H1QuadToDof,
                  const double* H1QuadToDofD,
                  const double* stressJinvT,
                  const double* e,
                  double* v) {
  const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
  const int NUM_QUAD_3D = NUM_QUAD_1D*NUM_QUAD_1D*NUM_QUAD_1D;
  MFEM_FORALL(el,numElements,
  {
    double e_xyz[NUM_QUAD_3D];
    for (int i = 0; i < NUM_QUAD_3D; ++i) {
      e_xyz[i] = 0;
    }
    for (int dz = 0; dz < L2_DOFS_1D; ++dz) {
      double e_xy[NUM_QUAD_2D];
      for (int i = 0; i < NUM_QUAD_2D; ++i) {
        e_xy[i] = 0;
      }
      for (int dy = 0; dy < L2_DOFS_1D; ++dy) {
        double e_x[NUM_QUAD_1D];
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          e_x[qy] = 0;
        }
        for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
          const double r_e = e[ijklN(dx,dy,dz,el,L2_DOFS_1D)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            e_x[qx] += L2DofToQuad[ijN(qx,dx,NUM_QUAD_1D)] * r_e;
          }
        }
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          const double wy = L2DofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            e_xy[ijN(qx,qy,NUM_QUAD_1D)] += wy * e_x[qx];
          }
        }
      }
      for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
        const double wz = L2DofToQuad[ijN(qz,dz,NUM_QUAD_1D)];
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            e_xyz[ijkN(qx,qy,qz,NUM_QUAD_1D)] += wz * e_xy[ijN(qx,qy,NUM_QUAD_1D)];
          }
        }
      }
    }
    for (int c = 0; c < 3; ++c) {
      for (int dz = 0; dz < H1_DOFS_1D; ++dz) {
        for (int dy = 0; dy < H1_DOFS_1D; ++dy) {
          for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
            v[_ijklmNM(c,dx,dy,dz,el,NUM_DOFS_1D,numElements)] = 0;
          }
        }
      }
      for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
        double Dxy_x[H1_DOFS_1D * H1_DOFS_1D];
        double xDy_y[H1_DOFS_1D * H1_DOFS_1D];
        double xy_z[H1_DOFS_1D * H1_DOFS_1D] ;
        for (int d = 0; d < (H1_DOFS_1D * H1_DOFS_1D); ++d) {
          Dxy_x[d] = xDy_y[d] = xy_z[d] = 0;
        }
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          double Dx_x[H1_DOFS_1D];
          double x_y[H1_DOFS_1D];
          double x_z[H1_DOFS_1D];
          for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
            Dx_x[dx] = x_y[dx] = x_z[dx] = 0;
          }
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            const double r_e = e_xyz[ijkN(qx,qy,qz,NUM_QUAD_1D)];
            const double esx = r_e * stressJinvT[__ijxyzeDQE(0,c,qx,qy,qz,el,NUM_DIM,NUM_QUAD_1D,numElements)];
            const double esy = r_e * stressJinvT[__ijxyzeDQE(1,c,qx,qy,qz,el,NUM_DIM,NUM_QUAD_1D,numElements)];
            const double esz = r_e * stressJinvT[__ijxyzeDQE(2,c,qx,qy,qz,el,NUM_DIM,NUM_QUAD_1D,numElements)];
            for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
              Dx_x[dx] += esx * H1QuadToDofD[ijN(dx,qx,H1_DOFS_1D)];
              x_y[dx]  += esy * H1QuadToDof[ijN(dx,qx,H1_DOFS_1D)];
              x_z[dx]  += esz * H1QuadToDof[ijN(dx,qx,H1_DOFS_1D)];
            }
          }
          for (int dy = 0; dy < H1_DOFS_1D; ++dy) {
            const double wy  = H1QuadToDof[ijN(dy,qy,H1_DOFS_1D)];
            const double wDy = H1QuadToDofD[ijN(dy,qy,H1_DOFS_1D)];
            for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
              Dxy_x[ijN(dx,dy,H1_DOFS_1D)] += Dx_x[dx] * wy;
              xDy_y[ijN(dx,dy,H1_DOFS_1D)] += x_y[dx]  * wDy;
              xy_z[ijN(dx,dy,H1_DOFS_1D)]  += x_z[dx]  * wy;
            }
          }
        }
        for (int dz = 0; dz < H1_DOFS_1D; ++dz) {
          const double wz  = H1QuadToDof[ijN(dz,qz,H1_DOFS_1D)];
          const double wDz = H1QuadToDofD[ijN(dz,qz,H1_DOFS_1D)];
          for (int dy = 0; dy < H1_DOFS_1D; ++dy) {
            for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
              v[_ijklmNM(c,dx,dy,dz,el,NUM_DOFS_1D,numElements)] +=
                ((Dxy_x[ijN(dx,dy,H1_DOFS_1D)] * wz) +
                 (xDy_y[ijN(dx,dy,H1_DOFS_1D)] * wz) +
                 (xy_z[ijN(dx,dy,H1_DOFS_1D)]  * wDz));
            }
          }
        }
      }
    }
  });
}

// *****************************************************************************
#define TForceMult3D(d,D,Q,L,H)                                        \
   template void kForceMult3D<d,D,Q,L,H>(const int numElements,        \
                                         const double* L2DofToQuad,    \
                                         const double* H1QuadToDof,    \
                                         const double* H1QuadToDofD,   \
                                         const double* stressJinvT,    \
                                         const double* e,              \
                                         double* v)
TForceMult3D(3,2,2,1,2);
TForceMult3D(3,3,4,2,3);
TForceMult3D(3,4,6,3,4);
TForceMult3D(3,5,8,4,5);
TForceMult3D(3,6,10,5,6);
TForceMult3D(3,7,12,6,7);
TForceMult3D(3,8,14,7,8);
TForceMult3D(3,9,16,8,9);
TForceMult3D(3,10,18,9,10);
TForceMult3D(3,11,20,10,11);
TForceMult3D(3,12,22,11,12);
TForceMult3D(3,13,24,12,13);
TForceMult3D(3,14,26,13,14);
TForceMult3D(3,15,28,14,15);
TForceMult3D(3,16,30,15,16);
TForceMult3D(3,17,32,16,17);

// *****************************************************************************
typedef void (*fForceMult)(const int numElements,
                           const double* L2QuadToDof,
                           const double* H1DofToQuad,
                           const double* H1DofToQuadD,
                           const double* stressJinvT,
                           const double* e,
                           double* v);

// *****************************************************************************
static void kForceMult(const int NUM_DIM,
                       const int NUM_DOFS_1D,
                       const int NUM_QUAD_1D,
                       const int L2_DOFS_1D,
                       const int H1_DOFS_1D,
                       const int nzones,
                       const double* L2QuadToDof,
                       const double* H1DofToQuad,
                       const double* H1DofToQuadD,
                       const double* stressJinvT,
                       const double* e,
                       double* v)
{
   assert(NUM_QUAD_1D==2*(NUM_DOFS_1D-1));
   assert(NUM_DOFS_1D==H1_DOFS_1D);
   assert(L2_DOFS_1D==NUM_DOFS_1D-1);
   const unsigned int id = ((NUM_DIM)<<4)|(NUM_DOFS_1D-2);
   assert(LOG2(NUM_DIM)<=4);
   assert(LOG2(NUM_DOFS_1D-2)<=4);
   static std::unordered_map<unsigned long long, fForceMult> call = {
      {0x20,&kForceMult2D<2,2,2,1,2>},
      {0x21,&kForceMult2D<2,3,4,2,3>},
      {0x22,&kForceMult2D<2,4,6,3,4>},
      {0x23,&kForceMult2D<2,5,8,4,5>},
      {0x24,&kForceMult2D<2,6,10,5,6>},
      {0x25,&kForceMult2D<2,7,12,6,7>},
      {0x26,&kForceMult2D<2,8,14,7,8>},
      {0x27,&kForceMult2D<2,9,16,8,9>},
      {0x28,&kForceMult2D<2,10,18,9,10>},
      {0x29,&kForceMult2D<2,11,20,10,11>},
      {0x2A,&kForceMult2D<2,12,22,11,12>},
      {0x2B,&kForceMult2D<2,13,24,12,13>},
      {0x2C,&kForceMult2D<2,14,26,13,14>},
      {0x2D,&kForceMult2D<2,15,28,14,15>},
      {0x2E,&kForceMult2D<2,16,30,15,16>},
      {0x2F,&kForceMult2D<2,17,32,16,17>},
      // 3D
      {0x30,&kForceMult3D<3,2,2,1,2>},
      {0x31,&kForceMult3D<3,3,4,2,3>},
      {0x32,&kForceMult3D<3,4,6,3,4>},
      {0x33,&kForceMult3D<3,5,8,4,5>},
      {0x34,&kForceMult3D<3,6,10,5,6>},
      {0x35,&kForceMult3D<3,7,12,6,7>},
      {0x36,&kForceMult3D<3,8,14,7,8>},
      {0x37,&kForceMult3D<3,9,16,8,9>},
      {0x38,&kForceMult3D<3,10,18,9,10>},
      {0x39,&kForceMult3D<3,11,20,10,11>},
      {0x3A,&kForceMult3D<3,12,22,11,12>},
      {0x3B,&kForceMult3D<3,13,24,12,13>},
      {0x3C,&kForceMult3D<3,14,26,13,14>},
      {0x3D,&kForceMult3D<3,15,28,14,15>},
      {0x3E,&kForceMult3D<3,16,30,15,16>},
      {0x3F,&kForceMult3D<3,17,32,16,17>},
   };
   if (!call[id]){
      printf("\n[rForceMult] id \033[33m0x%X\033[m ",id);
      fflush(stdout);
   }
   assert(call[id]);  
   GET_CONST_ADRS(L2QuadToDof);
   GET_CONST_ADRS(H1DofToQuad);
   GET_CONST_ADRS(H1DofToQuadD);
   GET_CONST_ADRS(stressJinvT);
   GET_CONST_ADRS(e);
   GET_ADRS(v);
   call[id](nzones,
            d_L2QuadToDof,
            d_H1DofToQuad,
            d_H1DofToQuadD,
            d_stressJinvT,
            d_e, d_v);
}

// *****************************************************************************
void kForcePAOperator::Mult(const mfem::Vector &vecL2,
                            mfem::Vector &vecH1) const {
   l2k.GlobalToLocal(vecL2, gVecL2);
   kForceMult(dim,
              NUM_DOFS_1D,
              NUM_QUAD_1D,
              L2_DOFS_1D,
              H1_DOFS_1D,
              nzones,
              l2D2Q->dofToQuad,
              h1D2Q->quadToDof,
              h1D2Q->quadToDofD,
              quad_data->stressJinvT.Data(),
              gVecL2,
              gVecH1);
   h1k.LocalToGlobal(gVecH1, vecH1);
}

// *****************************************************************************
template<const int NUM_DIM,
         const int NUM_DOFS_1D,
         const int NUM_QUAD_1D,
         const int L2_DOFS_1D,
         const int H1_DOFS_1D> static
void kForceMultTranspose2D(const int numElements,
                           const double* L2QuadToDof,
                           const double* H1DofToQuad,
                           const double* H1DofToQuadD,
                           const double* stressJinvT,
                           const double* v,
                           double* e) {
   const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
  MFEM_FORALL(el,numElements,
  {
    double vStress[NUM_QUAD_2D];
    for (int i = 0; i < NUM_QUAD_2D; ++i) {
      vStress[i] = 0;
    }
    for (int c = 0; c < NUM_DIM; ++c) {
      double v_Dxy[NUM_QUAD_2D];
      double v_xDy[NUM_QUAD_2D];
      for (int i = 0; i < NUM_QUAD_2D; ++i) {
        v_Dxy[i] = v_xDy[i] = 0;
      }
      for (int dy = 0; dy < H1_DOFS_1D; ++dy) {
        double v_x[NUM_QUAD_1D];
        double v_Dx[NUM_QUAD_1D];
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          v_x[qx] = v_Dx[qx] = 0;
        }

        for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
          const double r_v = v[_ijklNM(c,dx,dy,el,NUM_DOFS_1D,numElements)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            v_x[qx]  += r_v * H1DofToQuad[ijN(qx,dx,NUM_QUAD_1D)];
            v_Dx[qx] += r_v * H1DofToQuadD[ijN(qx,dx,NUM_QUAD_1D)];
          }
        }
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          const double wy  = H1DofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
          const double wDy = H1DofToQuadD[ijN(qy,dy,NUM_QUAD_1D)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            v_Dxy[ijN(qx,qy,NUM_QUAD_1D)] += v_Dx[qx] * wy;
            v_xDy[ijN(qx,qy,NUM_QUAD_1D)] += v_x[qx]  * wDy;
          }
        }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          vStress[ijN(qx,qy,NUM_QUAD_1D)] +=
            ((v_Dxy[ijN(qx,qy,NUM_QUAD_1D)] *
              stressJinvT[__ijklmNM(0,c,qx,qy,el,NUM_DIM,NUM_QUAD_1D,numElements)]) +
             (v_xDy[ijN(qx,qy,NUM_QUAD_1D)] *
              stressJinvT[__ijklmNM(1,c,qx,qy,el,NUM_DIM,NUM_QUAD_1D,numElements)]));
        }
      }
    }
    for (int dy = 0; dy < L2_DOFS_1D; ++dy) {
      for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
        e[ijkN(dx,dy,el,L2_DOFS_1D)] = 0;
      }
    }
    for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
      double e_x[L2_DOFS_1D];
      for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
        e_x[dx] = 0;
      }
      for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
        const double r_v = vStress[ijN(qx,qy,NUM_QUAD_1D)];
        for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
          e_x[dx] += r_v * L2QuadToDof[ijN(dx,qx,L2_DOFS_1D)];
        }
      }
      for (int dy = 0; dy < L2_DOFS_1D; ++dy) {
        const double w = L2QuadToDof[ijN(dy,qy,L2_DOFS_1D)];
        for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
          e[ijkN(dx,dy,el,L2_DOFS_1D)] += e_x[dx] * w;
        }
      }
    }
  });
}

// *****************************************************************************
#define TForceMultTranspose2D(d,D,Q,L,H)                                \
   template void kForceMultTranspose2D<d,D,Q,L,H>(const int numElements, \
                                                  const double* L2QuadToDof, \
                                                  const double* H1DofToQuad, \
                                                  const double* H1DofToQuadD, \
                                                  const double* stressJinvT, \
                                                  const double* v,      \
                                                  double* e)
TForceMultTranspose2D(2,2,2,1,2);
TForceMultTranspose2D(2,3,4,2,3);
TForceMultTranspose2D(2,4,6,3,4);
TForceMultTranspose2D(2,5,8,4,5);
TForceMultTranspose2D(2,6,10,5,6);
TForceMultTranspose2D(2,7,12,6,7);
TForceMultTranspose2D(2,8,14,7,8);
TForceMultTranspose2D(2,9,16,8,9);
TForceMultTranspose2D(2,10,18,9,10);
TForceMultTranspose2D(2,11,20,10,11);
TForceMultTranspose2D(2,12,22,11,12);
TForceMultTranspose2D(2,13,24,12,13);
TForceMultTranspose2D(2,14,26,13,14);
TForceMultTranspose2D(2,15,28,14,15);
TForceMultTranspose2D(2,16,30,15,16);
TForceMultTranspose2D(2,17,32,16,17);

// *****************************************************************************
template<const int NUM_DIM,
         const int NUM_DOFS_1D,
         const int NUM_QUAD_1D,
         const int L2_DOFS_1D,
         const int H1_DOFS_1D> static
void kForceMultTranspose3D(const int numElements,
                           const double* L2QuadToDof,
                           const double* H1DofToQuad,
                           const double* H1DofToQuadD,
                           const double* stressJinvT,
                           const double* v,
                           double* e) {
   const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
  const int NUM_QUAD_3D = NUM_QUAD_1D*NUM_QUAD_1D*NUM_QUAD_1D;
  MFEM_FORALL(el,numElements,
  {
    double vStress[NUM_QUAD_3D];
    for (int i = 0; i < NUM_QUAD_3D; ++i) {
      vStress[i] = 0;
    }
    for (int c = 0; c < NUM_DIM; ++c) {
      for (int dz = 0; dz < H1_DOFS_1D; ++dz) {
        double Dxy_x[NUM_QUAD_2D];
        double xDy_y[NUM_QUAD_2D];
        double xy_z[NUM_QUAD_2D] ;
        for (int i = 0; i < NUM_QUAD_2D; ++i) {
          Dxy_x[i] = xDy_y[i] = xy_z[i] = 0;
        }
        for (int dy = 0; dy < H1_DOFS_1D; ++dy) {
          double Dx_x[NUM_QUAD_1D];
          double x_y[NUM_QUAD_1D];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            Dx_x[qx] = x_y[qx] = 0;
          }
          for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
            const double r_v = v[_ijklmNM(c,dx,dy,dz,el,NUM_DOFS_1D,numElements)];
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
              Dx_x[qx] += r_v * H1DofToQuadD[ijN(qx,dx,NUM_QUAD_1D)];
              x_y[qx]  += r_v * H1DofToQuad[ijN(qx,dx,NUM_QUAD_1D)];
            }
          }
          for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
            const double wy  = H1DofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
            const double wDy = H1DofToQuadD[ijN(qy,dy,NUM_QUAD_1D)];
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
              Dxy_x[ijN(qx,qy,NUM_QUAD_1D)] += Dx_x[qx] * wy;
              xDy_y[ijN(qx,qy,NUM_QUAD_1D)] += x_y[qx]  * wDy;
              xy_z[ijN(qx,qy,NUM_QUAD_1D)]  += x_y[qx]  * wy;
            }
          }
        }
        for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
          const double wz  = H1DofToQuad[ijN(qz,dz,NUM_QUAD_1D)];
          const double wDz = H1DofToQuadD[ijN(qz,dz,NUM_QUAD_1D)];
          for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
              vStress[ijkN(qx,qy,qz,NUM_QUAD_1D)] +=
                 ((Dxy_x[ijN(qx,qy,NUM_QUAD_1D)]*wz * stressJinvT[__ijxyzeDQE(0,c,qx,qy,qz,el,NUM_DIM,NUM_QUAD_1D,numElements)]) +
                  (xDy_y[ijN(qx,qy,NUM_QUAD_1D)]*wz * stressJinvT[__ijxyzeDQE(1,c,qx,qy,qz,el,NUM_DIM,NUM_QUAD_1D,numElements)]) +
                  (xy_z[ijN(qx,qy,NUM_QUAD_1D)] *wDz * stressJinvT[__ijxyzeDQE(2,c,qx,qy,qz,el,NUM_DIM,NUM_QUAD_1D,numElements)]));
            }
          }
        }
      }
    }
    for (int dz = 0; dz < L2_DOFS_1D; ++dz) {
      for (int dy = 0; dy < L2_DOFS_1D; ++dy) {
        for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
          e[ijklN(dx,dy,dz,el,L2_DOFS_1D)] = 0;
        }
      }
    }
    for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
      double e_xy[L2_DOFS_1D * L2_DOFS_1D];
      for (int d = 0; d < (L2_DOFS_1D * L2_DOFS_1D); ++d) {
        e_xy[d] = 0;
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        double e_x[L2_DOFS_1D];
        for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
          e_x[dx] = 0;
        }
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          const double r_v = vStress[ijkN(qx,qy,qz,NUM_QUAD_1D)];
          for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
            e_x[dx] += r_v * L2QuadToDof[ijN(dx,qx,L2_DOFS_1D)];
          }
        }
        for (int dy = 0; dy < L2_DOFS_1D; ++dy) {
          const double w = L2QuadToDof[ijN(dy,qy,L2_DOFS_1D)];
          for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
            e_xy[ijN(dx,dy,L2_DOFS_1D)] += e_x[dx] * w;
          }
        }
      }
      for (int dz = 0; dz < L2_DOFS_1D; ++dz) {
        const double w = L2QuadToDof[ijN(dz,qz,L2_DOFS_1D)];
        for (int dy = 0; dy < L2_DOFS_1D; ++dy) {
          for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
            e[ijklN(dx,dy,dz,el,L2_DOFS_1D)] += w * e_xy[ijN(dx,dy,L2_DOFS_1D)];
          }
        }
      }
    }
  });
}

// *****************************************************************************
#define TForceMultTranspose3D(d,D,Q,L,H)                                        \
   template void kForceMultTranspose3D<d,D,Q,L,H>(const int numElements, \
                                                  const double* L2QuadToDof, \
                                                  const double* H1DofToQuad, \
                                                  const double* H1DofToQuadD, \
                                                  const double* stressJinvT, \
                                                  const double* v,      \
                                                  double* e)
TForceMultTranspose3D(3,2,2,1,2);
TForceMultTranspose3D(3,3,4,2,3);
TForceMultTranspose3D(3,4,6,3,4);
TForceMultTranspose3D(3,5,8,4,5);
TForceMultTranspose3D(3,6,10,5,6);
TForceMultTranspose3D(3,7,12,6,7);
TForceMultTranspose3D(3,8,14,7,8);
TForceMultTranspose3D(3,9,16,8,9);
TForceMultTranspose3D(3,10,18,9,10);
TForceMultTranspose3D(3,11,20,10,11);
TForceMultTranspose3D(3,12,22,11,12);
TForceMultTranspose3D(3,13,24,12,13);
TForceMultTranspose3D(3,14,26,13,14);
TForceMultTranspose3D(3,15,28,14,15);
TForceMultTranspose3D(3,16,30,15,16);
TForceMultTranspose3D(3,17,32,16,17);

// *****************************************************************************
typedef void (*fForceMultTranspose)(const int numElements,
                                    const double* L2QuadToDof,
                                    const double* H1DofToQuad,
                                    const double* H1DofToQuadD,
                                    const double* stressJinvT,
                                    const double* v,
                                    double* e);

// *****************************************************************************
static void rForceMultTranspose(const int NUM_DIM,
                                const int NUM_DOFS_1D,
                                const int NUM_QUAD_1D,
                                const int L2_DOFS_1D,
                                const int H1_DOFS_1D,
                                const int nzones,
                                const double* L2QuadToDof,
                                const double* H1DofToQuad,
                                const double* H1DofToQuadD,
                                const double* stressJinvT,
                                const double* v,
                                double* e)
{
   assert(NUM_DOFS_1D==H1_DOFS_1D);
   assert(L2_DOFS_1D==NUM_DOFS_1D-1);
   assert(NUM_QUAD_1D==2*(NUM_DOFS_1D-1));
   assert(NUM_QUAD_1D==2*(NUM_DOFS_1D-1));
   const unsigned int id = ((NUM_DIM)<<4)|(NUM_DOFS_1D-2);
   static std::unordered_map<unsigned long long, fForceMultTranspose> call = {
      // 2D
      {0x20,&kForceMultTranspose2D<2,2,2,1,2>},
      {0x21,&kForceMultTranspose2D<2,3,4,2,3>},
      {0x22,&kForceMultTranspose2D<2,4,6,3,4>},
      {0x23,&kForceMultTranspose2D<2,5,8,4,5>},
      {0x24,&kForceMultTranspose2D<2,6,10,5,6>},
      {0x25,&kForceMultTranspose2D<2,7,12,6,7>},
      {0x26,&kForceMultTranspose2D<2,8,14,7,8>},
      {0x27,&kForceMultTranspose2D<2,9,16,8,9>},
      {0x28,&kForceMultTranspose2D<2,10,18,9,10>},
      {0x29,&kForceMultTranspose2D<2,11,20,10,11>},
      {0x2A,&kForceMultTranspose2D<2,12,22,11,12>},
      {0x2B,&kForceMultTranspose2D<2,13,24,12,13>},
      {0x2C,&kForceMultTranspose2D<2,14,26,13,14>},
      {0x2D,&kForceMultTranspose2D<2,15,28,14,15>},
      {0x2E,&kForceMultTranspose2D<2,16,30,15,16>},
      {0x2F,&kForceMultTranspose2D<2,17,32,16,17>},
      // 3D
      {0x30,&kForceMultTranspose3D<3,2,2,1,2>},
      {0x31,&kForceMultTranspose3D<3,3,4,2,3>},
      {0x32,&kForceMultTranspose3D<3,4,6,3,4>},
      {0x33,&kForceMultTranspose3D<3,5,8,4,5>},
      {0x34,&kForceMultTranspose3D<3,6,10,5,6>},
      {0x35,&kForceMultTranspose3D<3,7,12,6,7>},
      {0x36,&kForceMultTranspose3D<3,8,14,7,8>},
      {0x37,&kForceMultTranspose3D<3,9,16,8,9>},
      {0x38,&kForceMultTranspose3D<3,10,18,9,10>},
      {0x39,&kForceMultTranspose3D<3,11,20,10,11>},
      {0x3A,&kForceMultTranspose3D<3,12,22,11,12>},
      {0x3B,&kForceMultTranspose3D<3,13,24,12,13>},
      {0x3C,&kForceMultTranspose3D<3,14,26,13,14>},
      {0x3D,&kForceMultTranspose3D<3,15,28,14,15>},
      {0x3E,&kForceMultTranspose3D<3,16,30,15,16>},
      {0x3F,&kForceMultTranspose3D<3,17,32,16,17>},
   };
   if (!call[id]) {
      printf("\n[rForceMultTranspose] id \033[33m0x%X\033[m ",id);
      fflush(stdout);
   }
   assert(call[id]);
   GET_CONST_ADRS(L2QuadToDof);
   GET_CONST_ADRS(H1DofToQuad);
   GET_CONST_ADRS(H1DofToQuadD);
   GET_CONST_ADRS(stressJinvT);
   GET_CONST_ADRS(v);
   GET_ADRS(e);
   call[id](nzones,
            d_L2QuadToDof,
            d_H1DofToQuad,
            d_H1DofToQuadD,
            d_stressJinvT,
            d_v,
            d_e);
}

// *************************************************************************
void kForcePAOperator::MultTranspose(const mfem::Vector &vecH1,
                                     mfem::Vector &vecL2) const {
   h1k.GlobalToLocal(vecH1, gVecH1);
   rForceMultTranspose(dim,
                       NUM_DOFS_1D,
                       NUM_QUAD_1D,
                       L2_DOFS_1D,
                       H1_DOFS_1D,
                       nzones,
                       l2D2Q->quadToDof,
                       h1D2Q->dofToQuad,
                       h1D2Q->dofToQuadD,
                       quad_data->stressJinvT.Data(),
                       gVecH1,
                       gVecL2);
   l2k.LocalToGlobal(gVecL2, vecL2);
}

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI
