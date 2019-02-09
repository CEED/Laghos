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
#include "laghos_kernels.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{

namespace hydrodynamics
{

// *****************************************************************************
// * Kernel MASS Operator
// *****************************************************************************
kMassPAOperator::kMassPAOperator(Coefficient &q,
                                 QuadratureData *qd_,
                                 ParFiniteElementSpace &pfes_,
                                 const IntegrationRule &ir_) :
   AbcMassPAOperator(pfes_.GetTrueVSize()),
   Q(q),
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
   paBilinearForm->AddDomainIntegrator(new mfem::MassIntegrator(Q,&ir));
   paBilinearForm->Assemble();
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
   h1k(*(new FiniteElementSpaceExtension(*static_cast<FiniteElementSpace*>(&h1f)))),
   l2k(*(new FiniteElementSpaceExtension(*static_cast<FiniteElementSpace*>(&l2f)))),
   integ_rule(ir),
   ir1D(IntRules.Get(Geometry::SEGMENT, integ_rule.GetOrder())),
   D1D(h1fes.GetFE(0)->GetOrder()+1),
   Q1D(ir1D.GetNPoints()),
   L1D(l2fes.GetFE(0)->GetOrder()+1),
   H1D(h1fes.GetFE(0)->GetOrder()+1),
   h1sz(h1fes.GetVDim() * h1fes.GetFE(0)->GetDof() * nzones),
   l2sz(l2fes.GetFE(0)->GetDof() * nzones),
   l2D2Q(DofToQuad::Get(l2fes, integ_rule)),
   h1D2Q(DofToQuad::Get(h1fes, integ_rule)),
   gVecL2(h1sz),
   gVecH1(l2sz)
{
   gVecL2.SetSize(l2sz);
   gVecH1.SetSize(h1sz);
}

// *****************************************************************************
template<const int DIM,
         const int D1D,
         const int Q1D,
         const int L1D,
         const int H1D> static
void kForceMult2D(const int E,
                  const double* L2DofToQuad,
                  const double* H1QuadToDof,
                  const double* H1QuadToDofD,
                  const double* stressJinvT,
                  const double* e,
                  double* v) {
   const int Q2D = Q1D*Q1D;
   MFEM_FORALL(el,E,
   {
      double e_xy[Q2D];
      for (int i = 0; i < Q2D; ++i) {
         e_xy[i] = 0;
      }
      for (int dy = 0; dy < L1D; ++dy) {
         double e_x[Q1D];
         for (int qy = 0; qy < Q1D; ++qy) {
            e_x[qy] = 0;
         }
         for (int dx = 0; dx < L1D; ++dx) {
            const double r_e = e[ijkN(dx,dy,el,L1D)];
            for (int qx = 0; qx < Q1D; ++qx) {
               e_x[qx] += L2DofToQuad[ijN(qx,dx,Q1D)] * r_e;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy) {
            const double wy = L2DofToQuad[ijN(qy,dy,Q1D)];
            for (int qx = 0; qx < Q1D; ++qx) {
               e_xy[ijN(qx,qy,Q1D)] += wy * e_x[qx];
            }
         }
      }
      for (int c = 0; c < 2; ++c) {
         for (int dy = 0; dy < H1D; ++dy) {
            for (int dx = 0; dx < H1D; ++dx) {
               v[jkliNM(c,dx,dy,el,D1D,E)] = 0.0;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy) {
            double Dxy[H1D];
            double xy[H1D];
            for (int dx = 0; dx < H1D; ++dx) {
               Dxy[dx] = 0.0;
               xy[dx]  = 0.0;
            }
            for (int qx = 0; qx < Q1D; ++qx) {
               const double esx = e_xy[ijN(qx,qy,Q1D)]  *
                  stressJinvT[xyeijDQE(0,c,qx,qy,el,DIM,Q1D,E)];
               const double esy = e_xy[ijN(qx,qy,Q1D)] *
                  stressJinvT[xyeijDQE(1,c,qx,qy,el,DIM,Q1D,E)];
               for (int dx = 0; dx < H1D; ++dx) {
                  Dxy[dx] += esx * H1QuadToDofD[ijN(dx,qx,H1D)];
                  xy[dx]  += esy * H1QuadToDof[ijN(dx,qx,H1D)];
               }
            }
            for (int dy = 0; dy < H1D; ++dy) {
               const double wy  = H1QuadToDof[ijN(dy,qy,H1D)];
               const double wDy = H1QuadToDofD[ijN(dy,qy,H1D)];
               for (int dx = 0; dx < H1D; ++dx) {
                  v[jkliNM(c,dx,dy,el,D1D,E)] += wy* Dxy[dx] + wDy*xy[dx];
               }
            }
         }
      }
   });
}

// *****************************************************************************
template<const int DIM,
         const int D1D,
         const int Q1D,
         const int L1D,
         const int H1D> static
void kForceMult3D(const int E,
                  const double* L2DofToQuad,
                  const double* H1QuadToDof,
                  const double* H1QuadToDofD,
                  const double* stressJinvT,
                  const double* e,
                  double* v) {
   const int Q2D = Q1D*Q1D;
   const int Q3D = Q1D*Q1D*Q1D;
   MFEM_FORALL(el,E,
   {
      double e_xyz[Q3D];
      for (int i = 0; i < Q3D; ++i) {
         e_xyz[i] = 0;
      }
      for (int dz = 0; dz < L1D; ++dz) {
         double e_xy[Q2D];
         for (int i = 0; i < Q2D; ++i) {
            e_xy[i] = 0;
         }
         for (int dy = 0; dy < L1D; ++dy) {
            double e_x[Q1D];
            for (int qy = 0; qy < Q1D; ++qy) {
               e_x[qy] = 0;
            }
            for (int dx = 0; dx < L1D; ++dx) {
               const double r_e = e[ijklN(dx,dy,dz,el,L1D)];
               for (int qx = 0; qx < Q1D; ++qx) {
                  e_x[qx] += L2DofToQuad[ijN(qx,dx,Q1D)] * r_e;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy) {
               const double wy = L2DofToQuad[ijN(qy,dy,Q1D)];
               for (int qx = 0; qx < Q1D; ++qx) {
                  e_xy[ijN(qx,qy,Q1D)] += wy * e_x[qx];
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz) {
            const double wz = L2DofToQuad[ijN(qz,dz,Q1D)];
            for (int qy = 0; qy < Q1D; ++qy) {
               for (int qx = 0; qx < Q1D; ++qx) {
                  e_xyz[ijkN(qx,qy,qz,Q1D)] += wz * e_xy[ijN(qx,qy,Q1D)];
               }
            }
         }
      }
      for (int c = 0; c < 3; ++c) {
         for (int dz = 0; dz < H1D; ++dz) {
            for (int dy = 0; dy < H1D; ++dy) {
               for (int dx = 0; dx < H1D; ++dx) {
                  v[jklmiNM(c,dx,dy,dz,el,D1D,E)] = 0;
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz) {
            double Dxy_x[H1D * H1D];
            double xDy_y[H1D * H1D];
            double xy_z[H1D * H1D] ;
            for (int d = 0; d < (H1D * H1D); ++d) {
               Dxy_x[d] = xDy_y[d] = xy_z[d] = 0;
            }
            for (int qy = 0; qy < Q1D; ++qy) {
               double Dx_x[H1D];
               double x_y[H1D];
               double x_z[H1D];
               for (int dx = 0; dx < H1D; ++dx) {
                  Dx_x[dx] = x_y[dx] = x_z[dx] = 0;
               }
               for (int qx = 0; qx < Q1D; ++qx) {
                  const double r_e = e_xyz[ijkN(qx,qy,qz,Q1D)];
                  const double esx = r_e * stressJinvT[xyzeijDQE(0,c,qx,qy,qz,el,DIM,Q1D,E)];
                  const double esy = r_e * stressJinvT[xyzeijDQE(1,c,qx,qy,qz,el,DIM,Q1D,E)];
                  const double esz = r_e * stressJinvT[xyzeijDQE(2,c,qx,qy,qz,el,DIM,Q1D,E)];
                  for (int dx = 0; dx < H1D; ++dx) {
                     Dx_x[dx] += esx * H1QuadToDofD[ijN(dx,qx,H1D)];
                     x_y[dx]  += esy * H1QuadToDof[ijN(dx,qx,H1D)];
                     x_z[dx]  += esz * H1QuadToDof[ijN(dx,qx,H1D)];
                  }
               }
               for (int dy = 0; dy < H1D; ++dy) {
                  const double wy  = H1QuadToDof[ijN(dy,qy,H1D)];
                  const double wDy = H1QuadToDofD[ijN(dy,qy,H1D)];
                  for (int dx = 0; dx < H1D; ++dx) {
                     Dxy_x[ijN(dx,dy,H1D)] += Dx_x[dx] * wy;
                     xDy_y[ijN(dx,dy,H1D)] += x_y[dx]  * wDy;
                     xy_z[ijN(dx,dy,H1D)]  += x_z[dx]  * wy;
                  }
               }
            }
            for (int dz = 0; dz < H1D; ++dz) {
               const double wz  = H1QuadToDof[ijN(dz,qz,H1D)];
               const double wDz = H1QuadToDofD[ijN(dz,qz,H1D)];
               for (int dy = 0; dy < H1D; ++dy) {
                  for (int dx = 0; dx < H1D; ++dx) {
                     v[jklmiNM(c,dx,dy,dz,el,D1D,E)] +=
                        ((Dxy_x[ijN(dx,dy,H1D)] * wz) +
                         (xDy_y[ijN(dx,dy,H1D)] * wz) +
                         (xy_z[ijN(dx,dy,H1D)]  * wDz));
                  }
               }
            }
         }
      }
   });
}

// *****************************************************************************
typedef void (*fForceMult)(const int E,
                           const double* L2QuadToDof,
                           const double* H1DofToQuad,
                           const double* H1DofToQuadD,
                           const double* stressJinvT,
                           const double* e,
                           double* v);

// *****************************************************************************
static void kForceMult(const int DIM,
                       const int D1D,
                       const int Q1D,
                       const int L1D,
                       const int H1D,
                       const int nzones,
                       const double* L2QuadToDof,
                       const double* H1DofToQuad,
                       const double* H1DofToQuadD,
                       const double* stressJinvT,
                       const double* e,
                       double* v)
{
   assert(Q1D==2*(D1D-1));
   assert(D1D==H1D);
   assert(L1D==D1D-1);
   const unsigned int id = ((DIM)<<4)|(D1D-2);
   assert(LOG2(DIM)<=4);
   assert(LOG2(D1D-2)<=4);
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
   GET_CONST_PTR(L2QuadToDof);
   GET_CONST_PTR(H1DofToQuad);
   GET_CONST_PTR(H1DofToQuadD);
   GET_CONST_PTR(stressJinvT);
   GET_CONST_PTR(e);
   GET_PTR(v);
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
   l2k.L2E(vecL2, gVecL2);
   kForceMult(dim,
              D1D,
              Q1D,
              L1D,
              H1D,
              nzones,
              l2D2Q->B,
              h1D2Q->Bt,
              h1D2Q->Gt,
              quad_data->stressJinvT.Data(),
              gVecL2,
              gVecH1);
   h1k.E2L(gVecH1, vecH1);
}

// *****************************************************************************
template<const int DIM,
         const int D1D,
         const int Q1D,
         const int L1D,
         const int H1D> static
void kForceMultTranspose2D(const int E,
                           const double* L2QuadToDof,
                           const double* H1DofToQuad,
                           const double* H1DofToQuadD,
                           const double* stressJinvT,
                           const double* v,
                           double* e) {
   const int Q2D = Q1D*Q1D;
   MFEM_FORALL(el,E,
   {
      double vStress[Q2D];
      for (int i = 0; i < Q2D; ++i) {
         vStress[i] = 0;
      }
      for (int c = 0; c < DIM; ++c) {
         double v_Dxy[Q2D];
         double v_xDy[Q2D];
         for (int i = 0; i < Q2D; ++i) {
            v_Dxy[i] = v_xDy[i] = 0;
         }
         for (int dy = 0; dy < H1D; ++dy) {
            double v_x[Q1D];
            double v_Dx[Q1D];
            for (int qx = 0; qx < Q1D; ++qx) {
               v_x[qx] = v_Dx[qx] = 0;
            }

            for (int dx = 0; dx < H1D; ++dx) {
               const double r_v = v[jkliNM(c,dx,dy,el,D1D,E)];
               for (int qx = 0; qx < Q1D; ++qx) {
                  v_x[qx]  += r_v * H1DofToQuad[ijN(qx,dx,Q1D)];
                  v_Dx[qx] += r_v * H1DofToQuadD[ijN(qx,dx,Q1D)];
               }
            }
            for (int qy = 0; qy < Q1D; ++qy) {
               const double wy  = H1DofToQuad[ijN(qy,dy,Q1D)];
               const double wDy = H1DofToQuadD[ijN(qy,dy,Q1D)];
               for (int qx = 0; qx < Q1D; ++qx) {
                  v_Dxy[ijN(qx,qy,Q1D)] += v_Dx[qx] * wy;
                  v_xDy[ijN(qx,qy,Q1D)] += v_x[qx]  * wDy;
               }
            }
         }
         for (int qy = 0; qy < Q1D; ++qy) {
            for (int qx = 0; qx < Q1D; ++qx) {
               vStress[ijN(qx,qy,Q1D)] +=
                  ((v_Dxy[ijN(qx,qy,Q1D)] *
                    stressJinvT[xyeijDQE(0,c,qx,qy,el,DIM,Q1D,E)]) +
                   (v_xDy[ijN(qx,qy,Q1D)] *
                    stressJinvT[xyeijDQE(1,c,qx,qy,el,DIM,Q1D,E)]));
            }
         }
      }
      for (int dy = 0; dy < L1D; ++dy) {
         for (int dx = 0; dx < L1D; ++dx) {
            e[ijkN(dx,dy,el,L1D)] = 0;
         }
      }
      for (int qy = 0; qy < Q1D; ++qy) {
         double e_x[L1D];
         for (int dx = 0; dx < L1D; ++dx) {
            e_x[dx] = 0;
         }
         for (int qx = 0; qx < Q1D; ++qx) {
            const double r_v = vStress[ijN(qx,qy,Q1D)];
            for (int dx = 0; dx < L1D; ++dx) {
               e_x[dx] += r_v * L2QuadToDof[ijN(dx,qx,L1D)];
            }
         }
         for (int dy = 0; dy < L1D; ++dy) {
            const double w = L2QuadToDof[ijN(dy,qy,L1D)];
            for (int dx = 0; dx < L1D; ++dx) {
               e[ijkN(dx,dy,el,L1D)] += e_x[dx] * w;
            }
         }
      }
   });
}

// *****************************************************************************
template<const int DIM,
         const int D1D,
         const int Q1D,
         const int L1D,
         const int H1D> static
void kForceMultTranspose3D(const int E,
                           const double* L2QuadToDof,
                           const double* H1DofToQuad,
                           const double* H1DofToQuadD,
                           const double* stressJinvT,
                           const double* v,
                           double* e) {
   const int Q2D = Q1D*Q1D;
   const int Q3D = Q1D*Q1D*Q1D;
   MFEM_FORALL(el,E,
   {
      double vStress[Q3D];
      for (int i = 0; i < Q3D; ++i) {
         vStress[i] = 0;
      }
      for (int c = 0; c < DIM; ++c) {
         for (int dz = 0; dz < H1D; ++dz) {
            double Dxy_x[Q2D];
            double xDy_y[Q2D];
            double xy_z[Q2D] ;
            for (int i = 0; i < Q2D; ++i) {
               Dxy_x[i] = xDy_y[i] = xy_z[i] = 0;
            }
            for (int dy = 0; dy < H1D; ++dy) {
               double Dx_x[Q1D];
               double x_y[Q1D];
               for (int qx = 0; qx < Q1D; ++qx) {
                  Dx_x[qx] = x_y[qx] = 0;
               }
               for (int dx = 0; dx < H1D; ++dx) {
                  const double r_v = v[jklmiNM(c,dx,dy,dz,el,D1D,E)];
                  for (int qx = 0; qx < Q1D; ++qx) {
                     Dx_x[qx] += r_v * H1DofToQuadD[ijN(qx,dx,Q1D)];
                     x_y[qx]  += r_v * H1DofToQuad[ijN(qx,dx,Q1D)];
                  }
               }
               for (int qy = 0; qy < Q1D; ++qy) {
                  const double wy  = H1DofToQuad[ijN(qy,dy,Q1D)];
                  const double wDy = H1DofToQuadD[ijN(qy,dy,Q1D)];
                  for (int qx = 0; qx < Q1D; ++qx) {
                     Dxy_x[ijN(qx,qy,Q1D)] += Dx_x[qx] * wy;
                     xDy_y[ijN(qx,qy,Q1D)] += x_y[qx]  * wDy;
                     xy_z[ijN(qx,qy,Q1D)]  += x_y[qx]  * wy;
                  }
               }
            }
            for (int qz = 0; qz < Q1D; ++qz) {
               const double wz  = H1DofToQuad[ijN(qz,dz,Q1D)];
               const double wDz = H1DofToQuadD[ijN(qz,dz,Q1D)];
               for (int qy = 0; qy < Q1D; ++qy) {
                  for (int qx = 0; qx < Q1D; ++qx) {
                     const int ix = xyzeijDQE(0,c,qx,qy,qz,el,DIM,Q1D,E);
                     const double sx = stressJinvT[ix];
                     vStress[ijkN(qx,qy,qz,Q1D)] +=
                        ((Dxy_x[ijN(qx,qy,Q1D)] * wz * sx) +
                         (xDy_y[ijN(qx,qy,Q1D)] * wz *
                          stressJinvT[xyzeijDQE(1,c,qx,qy,qz,el,DIM,Q1D,E)]) +
                         (xy_z[ijN(qx,qy,Q1D)] * wDz *
                          stressJinvT[xyzeijDQE(2,c,qx,qy,qz,el,DIM,Q1D,E)]));
                  }
               }
            }
         }
      }
      for (int dz = 0; dz < L1D; ++dz) {
         for (int dy = 0; dy < L1D; ++dy) {
            for (int dx = 0; dx < L1D; ++dx) {
               e[ijklN(dx,dy,dz,el,L1D)] = 0;
            }
         }
      }
      for (int qz = 0; qz < Q1D; ++qz) {
         double e_xy[L1D * L1D];
         for (int d = 0; d < (L1D * L1D); ++d) {
            e_xy[d] = 0;
         }
         for (int qy = 0; qy < Q1D; ++qy) {
            double e_x[L1D];
            for (int dx = 0; dx < L1D; ++dx) {
               e_x[dx] = 0;
            }
            for (int qx = 0; qx < Q1D; ++qx) {
               const double r_v = vStress[ijkN(qx,qy,qz,Q1D)];
               for (int dx = 0; dx < L1D; ++dx) {
                  e_x[dx] += r_v * L2QuadToDof[ijN(dx,qx,L1D)];
               }
            }
            for (int dy = 0; dy < L1D; ++dy) {
               const double w = L2QuadToDof[ijN(dy,qy,L1D)];
               for (int dx = 0; dx < L1D; ++dx) {
                  e_xy[ijN(dx,dy,L1D)] += e_x[dx] * w;
               }
            }
         }
         for (int dz = 0; dz < L1D; ++dz) {
            const double w = L2QuadToDof[ijN(dz,qz,L1D)];
            for (int dy = 0; dy < L1D; ++dy) {
               for (int dx = 0; dx < L1D; ++dx) {
                  e[ijklN(dx,dy,dz,el,L1D)] += w * e_xy[ijN(dx,dy,L1D)];
               }
            }
         }
      }
   });
}

// *****************************************************************************
typedef void (*fForceMultTranspose)(const int E,
                                    const double* L2QuadToDof,
                                    const double* H1DofToQuad,
                                    const double* H1DofToQuadD,
                                    const double* stressJinvT,
                                    const double* v,
                                    double* e);

// *****************************************************************************
static void rForceMultTranspose(const int DIM,
                                const int D1D,
                                const int Q1D,
                                const int L1D,
                                const int H1D,
                                const int nzones,
                                const double* L2QuadToDof,
                                const double* H1DofToQuad,
                                const double* H1DofToQuadD,
                                const double* stressJinvT,
                                const double* v,
                                double* e)
{
   assert(D1D==H1D);
   assert(L1D==D1D-1);
   assert(Q1D==2*(D1D-1));
   assert(Q1D==2*(D1D-1));
   const unsigned int id = ((DIM)<<4)|(D1D-2);
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
   GET_CONST_PTR(L2QuadToDof);
   GET_CONST_PTR(H1DofToQuad);
   GET_CONST_PTR(H1DofToQuadD);
   GET_CONST_PTR(stressJinvT);
   GET_CONST_PTR(v);
   GET_PTR(e);
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
   h1k.L2E(vecH1, gVecH1);
   rForceMultTranspose(dim,
                       D1D,
                       Q1D,
                       L1D,
                       H1D,
                       nzones,
                       l2D2Q->Bt,
                       h1D2Q->B,
                       h1D2Q->G,
                       quad_data->stressJinvT.Data(),
                       gVecH1,
                       gVecL2);
   l2k.E2L(gVecL2, vecL2);
}

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI
