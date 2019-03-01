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
#include "linalg/device.hpp"

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
   paBilinearForm->AddDomainIntegrator(new mfem::PAMassIntegrator(Q,&ir));
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
   h1k(*(new E2LOperator(*static_cast<FiniteElementSpace*>(&h1f)))),
   l2k(*(new E2LOperator(*static_cast<FiniteElementSpace*>(&l2f)))),
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
void kForceMult2D(const int NE,
                  const double* _B,
                  const double* _Bt,
                  const double* _Gt,
                  const double* _sJit,
                  const double* _e,
                  double* _v) {
   const DeviceMatrix L2B(_B, Q1D,L1D);
   const DeviceMatrix H1Bt(_Bt, H1D,Q1D);
   const DeviceMatrix H1Gt(_Gt, H1D,Q1D);
   const DeviceTensor<5> sJit(_sJit, Q1D,Q1D,NE,2,2);
   const DeviceTensor<3> energy(_e, L1D, L1D, NE);
   DeviceTensor<4> velocity(_v, D1D,D1D,NE,2);
   MFEM_FORALL(e, NE,
   {
      double e_xy[Q1D][Q1D];
      for (int qx = 0; qx < Q1D; ++qx) {
         for (int qy = 0; qy < Q1D; ++qy) {
            e_xy[qx][qy] = 0.0;
         }
      }
      for (int dy = 0; dy < L1D; ++dy) {
         double e_x[Q1D];
         for (int qy = 0; qy < Q1D; ++qy) {
            e_x[qy] = 0.0;
         }
         for (int dx = 0; dx < L1D; ++dx) {
            const double r_e = energy(dx,dy,e);
            for (int qx = 0; qx < Q1D; ++qx) {
               e_x[qx] += L2B(qx,dx) * r_e;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy) {
            const double wy = L2B(qy,dy);
            for (int qx = 0; qx < Q1D; ++qx) {
               e_xy[qx][qy] += wy * e_x[qx];
            }
         }
      }
      for (int c = 0; c < 2; ++c) {
         for (int dy = 0; dy < H1D; ++dy) {
            for (int dx = 0; dx < H1D; ++dx) {
               velocity(dx,dy,e,c) = 0.0;
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
               const double esx = e_xy[qx][qy] * sJit(qx,qy,e,0,c);
               const double esy = e_xy[qx][qy] * sJit(qx,qy,e,1,c);
               for (int dx = 0; dx < H1D; ++dx) {
                  Dxy[dx] += esx * H1Gt(dx,qx);
                  xy[dx]  += esy * H1Bt(dx,qx);
               }
            }
            for (int dy = 0; dy < H1D; ++dy) {
               const double wy  = H1Bt(dy,qy);
               const double wDy = H1Gt(dy,qy);
               for (int dx = 0; dx < H1D; ++dx) {
                  velocity(dx,dy,e,c) += wy* Dxy[dx] + wDy*xy[dx];
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
void kForceMult3D(const int NE,
                  const double* _B,
                  const double* _Bt,
                  const double* _Gt,
                  const double* _sJit,
                  const double* _e,
                  double* _v) {
   const DeviceMatrix L2B(_B, Q1D,L1D);
   const DeviceMatrix H1Bt(_Bt, H1D,Q1D);
   const DeviceMatrix H1Gt(_Gt, H1D,Q1D);
   const DeviceTensor<6> sJit(_sJit, Q1D,Q1D,Q1D,NE,3,3);
   const DeviceTensor<4> energy(_e, L1D, L1D, L1D, NE);
   DeviceTensor<5> velocity(_v, D1D,D1D,D1D,NE,3);
   MFEM_FORALL(e, NE,
   {
      double e_xyz[Q1D][Q1D][Q1D];
      for (int qx = 0; qx < Q1D; ++qx) {
         for (int qy = 0; qy < Q1D; ++qy) {
            for (int qz = 0; qz < Q1D; ++qz) {
               e_xyz[qx][qy][qz] = 0.0;
            }
         }
      }
      for (int dz = 0; dz < L1D; ++dz) {
         double e_xy[Q1D][Q1D];
         for (int qx = 0; qx < Q1D; ++qx) {
            for (int qy = 0; qy < Q1D; ++qy) {
               e_xy[qx][qy] = 0.0;
            }
         }
         for (int dy = 0; dy < L1D; ++dy) {
            double e_x[Q1D];
            for (int qx = 0; qx < Q1D; ++qx) {
               e_x[qx] = 0.0;
            }
            for (int dx = 0; dx < L1D; ++dx) {
               const double r_e = energy(dx,dy,dz,e);
               for (int qx = 0; qx < Q1D; ++qx) {
                  e_x[qx] += L2B(qx,dx) * r_e;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy) {
               const double wy = L2B(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx) {
                  e_xy[qx][qy] += wy * e_x[qx];
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz) {
            const double wz = L2B(qz,dz);
            for (int qy = 0; qy < Q1D; ++qy) {
               for (int qx = 0; qx < Q1D; ++qx) {
                  e_xyz[qx][qy][qz] += wz * e_xy[qx][qy];
               }
            }
         }
      }
      for (int c = 0; c < 3; ++c) {
         for (int dz = 0; dz < H1D; ++dz) {
            for (int dy = 0; dy < H1D; ++dy) {
               for (int dx = 0; dx < H1D; ++dx) {
                  velocity(dx,dy,dz,e,c) = 0.0;
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz) {
            double Dxy_x[H1D][H1D];
            double xDy_y[H1D][H1D];
            double xy_z[H1D][H1D] ;
            for (int dx = 0; dx < H1D; ++dx) {
               for (int dy = 0; dy < H1D; ++dy) {
                  Dxy_x[dx][dy] = xDy_y[dx][dy] = xy_z[dx][dy] = 0.0;
               }
            }
            for (int qy = 0; qy < Q1D; ++qy) {
               double Dx_x[H1D];
               double x_y[H1D];
               double x_z[H1D];
               for (int dx = 0; dx < H1D; ++dx) {
                  Dx_x[dx] = x_y[dx] = x_z[dx] = 0.0;
               }
               for (int qx = 0; qx < Q1D; ++qx) {
                  const double r_e = e_xyz[qx][qy][qz];
                  const double esx = r_e * sJit(qx,qy,qz,e,0,c);
                  const double esy = r_e * sJit(qx,qy,qz,e,1,c);
                  const double esz = r_e * sJit(qx,qy,qz,e,2,c);
                  for (int dx = 0; dx < H1D; ++dx) {
                     Dx_x[dx] += esx * H1Gt(dx,qx);
                     x_y[dx]  += esy * H1Bt(dx,qx);
                     x_z[dx]  += esz * H1Bt(dx,qx);
                  }
               }
               for (int dy = 0; dy < H1D; ++dy) {
                  const double wy  = H1Bt(dy,qy);
                  const double wDy = H1Gt(dy,qy);
                  for (int dx = 0; dx < H1D; ++dx) {
                     Dxy_x[dx][dy] += Dx_x[dx] * wy;
                     xDy_y[dx][dy] += x_y[dx]  * wDy;
                     xy_z[dx][dy]  += x_z[dx]  * wy;
                  }
               }
            }
            for (int dz = 0; dz < H1D; ++dz) {
               const double wz  = H1Bt(dz,qz);
               const double wDz = H1Gt(dz,qz);
               for (int dy = 0; dy < H1D; ++dy) {
                  for (int dx = 0; dx < H1D; ++dx) {
                     velocity(dx,dy,dz,e,c) +=
                        ((Dxy_x[dx][dy] * wz) +
                         (xDy_y[dx][dy] * wz) +
                         (xy_z[dx][dy] * wDz));
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
                       const int NE,
                       const double* B,
                       const double* Bt,
                       const double* Gt,
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
      {0x21,&kForceMult2D<2,3,4,2,3>},
      {0x31,&kForceMult3D<3,3,4,2,3>},
   };
   if (!call[id]){
      printf("\n[rForceMult] id \033[33m0x%X\033[m ",id);
      fflush(stdout);
   }
   call[id](NE, B, Bt, Gt, stressJinvT, e, v);
}

// *****************************************************************************
void kForcePAOperator::Mult(const mfem::Vector &x,
                            mfem::Vector &y) const {
   l2k.Mult(x, gVecL2);
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
   h1k.MultTranspose(gVecH1, y);
}

// *****************************************************************************
template<const int DIM,
         const int D1D,
         const int Q1D,
         const int L1D,
         const int H1D> static
void kForceMultTranspose2D(const int NE,
                           const double* _Bt,
                           const double* _B,
                           const double* _G,
                           const double* _sJit,
                           const double* _v,
                           double* _e) {
   const DeviceMatrix L2Bt(_Bt, L1D,Q1D);
   const DeviceMatrix H1B(_B, Q1D,H1D);
   const DeviceMatrix H1G(_G, Q1D,H1D);
   const DeviceTensor<5> sJit(_sJit, Q1D,Q1D,NE,2,2);
   const DeviceTensor<4> velocity(_v, D1D,D1D,NE,2);
   DeviceTensor<3> energy(_e, L1D, L1D, NE);
   MFEM_FORALL(e, NE,
   {
      double vStress[Q1D][Q1D];
      for (int qx = 0; qx < Q1D; ++qx) {
         for (int qy = 0; qy < Q1D; ++qy) {
            vStress[qx][qy] = 0.0;
         }
      }
      for (int c = 0; c < DIM; ++c) {
         double v_Dxy[Q1D][Q1D];
         double v_xDy[Q1D][Q1D];
         for (int qx = 0; qx < Q1D; ++qx) {
            for (int qy = 0; qy < Q1D; ++qy) {
            v_Dxy[qx][qy] = v_xDy[qx][qy] = 0.0;
            }
         }
         for (int dy = 0; dy < H1D; ++dy) {
            double v_x[Q1D];
            double v_Dx[Q1D];
            for (int qx = 0; qx < Q1D; ++qx) {
               v_x[qx] = v_Dx[qx] = 0.0;
            }

            for (int dx = 0; dx < H1D; ++dx) {
               const double r_v = velocity(dx,dy,e,c);
               for (int qx = 0; qx < Q1D; ++qx) {
                  v_x[qx]  += r_v * H1B(qx,dx);
                  v_Dx[qx] += r_v * H1G(qx,dx);
               }
            }
            for (int qy = 0; qy < Q1D; ++qy) {
               const double wy  = H1B(qy,dy);
               const double wDy = H1G(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx) {
                  v_Dxy[qx][qy] += v_Dx[qx] * wy;
                  v_xDy[qx][qy] += v_x[qx]  * wDy;
               }
            }
         }
         for (int qy = 0; qy < Q1D; ++qy) {
            for (int qx = 0; qx < Q1D; ++qx) {
               const double sJitx = sJit(qx,qy,e,0,c);
               const double sJity = sJit(qx,qy,e,1,c);
               vStress[qx][qy] +=
                  v_Dxy[qx][qy] * sJitx + v_xDy[qx][qy] * sJity;
            }
         }
      }
      for (int dy = 0; dy < L1D; ++dy) {
         for (int dx = 0; dx < L1D; ++dx) {
            energy(dx,dy,e) = 0.0;
         }
      }
      for (int qy = 0; qy < Q1D; ++qy) {
         double e_x[L1D];
         for (int dx = 0; dx < L1D; ++dx) {
            e_x[dx] = 0.0;
         }
         for (int qx = 0; qx < Q1D; ++qx) {
            const double r_v = vStress[qx][qy];
            for (int dx = 0; dx < L1D; ++dx) {
               e_x[dx] += r_v * L2Bt(dx,qx);
            }
         }
         for (int dy = 0; dy < L1D; ++dy) {
            const double w = L2Bt(dy,qy);
            for (int dx = 0; dx < L1D; ++dx) {
               energy(dx,dy,e) += e_x[dx] * w;
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
void kForceMultTranspose3D(const int NE,
                           const double* _Bt,
                           const double* _B,
                           const double* _G,
                           const double* _sJit,
                           const double* _v,
                           double* _e) {
   const DeviceMatrix L2Bt(_Bt, L1D,Q1D);
   const DeviceMatrix H1B(_B, Q1D,H1D);
   const DeviceMatrix H1G(_G, Q1D,H1D);
   const DeviceTensor<6> sJit(_sJit, Q1D,Q1D,Q1D,NE,3,3);
   const DeviceTensor<5> velocity(_v, D1D,D1D,D1D,NE,3);
   DeviceTensor<4> energy(_e, L1D,L1D,L1D,NE);
   MFEM_FORALL(e,NE,
   {
      double vStress[Q1D][Q1D][Q1D];
      for (int qx = 0; qx < Q1D; ++qx) {
         for (int qy = 0; qy < Q1D; ++qy) {
            for (int qz = 0; qz < Q1D; ++qz) {
               vStress[qx][qy][qz] = 0.0;
            }
         }
      }
      for (int c = 0; c < DIM; ++c) {
         for (int dz = 0; dz < H1D; ++dz) {
            double Dxy_x[Q1D][Q1D];
            double xDy_y[Q1D][Q1D];
            double xy_z[Q1D][Q1D];
            for (int qx = 0; qx < Q1D; ++qx) {
               for (int qy = 0; qy < Q1D; ++qy) {
                  Dxy_x[qx][qy] = xDy_y[qx][qy] = xy_z[qx][qy] = 0.0;
               }
            }
            for (int dy = 0; dy < H1D; ++dy) {
               double Dx_x[Q1D];
               double x_y[Q1D];
               for (int qx = 0; qx < Q1D; ++qx) {
                  Dx_x[qx] = x_y[qx] = 0;
               }
               for (int dx = 0; dx < H1D; ++dx) {
                  const double r_v = velocity(dx,dy,dz,e,c);
                  for (int qx = 0; qx < Q1D; ++qx) {
                     Dx_x[qx] += r_v * H1G(qx,dx);
                     x_y[qx]  += r_v * H1B(qx,dx);
                  }
               }
               for (int qy = 0; qy < Q1D; ++qy) {
                  const double wy  = H1B(qy,dy);
                  const double wDy = H1G(qy,dy);
                  for (int qx = 0; qx < Q1D; ++qx) {
                     Dxy_x[qx][qy] += Dx_x[qx] * wy;
                     xDy_y[qx][qy] += x_y[qx]  * wDy;
                     xy_z[qx][qy]  += x_y[qx]  * wy;
                  }
               }
            }
            for (int qz = 0; qz < Q1D; ++qz) {
               const double wz  = H1B(qz,dz);
               const double wDz = H1G(qz,dz);
               for (int qy = 0; qy < Q1D; ++qy) {
                  for (int qx = 0; qx < Q1D; ++qx) {
                     vStress[qx][qy][qz] +=
                        ((Dxy_x[qx][qy] * wz * sJit(qx,qy,qz,e,0,c)) +
                         (xDy_y[qx][qy] * wz * sJit(qx,qy,qz,e,1,c)) +
                         (xy_z[qx][qy] * wDz * sJit(qx,qy,qz,e,2,c)));
                  }
               }
            }
         }
      }
      for (int dz = 0; dz < L1D; ++dz) {
         for (int dy = 0; dy < L1D; ++dy) {
            for (int dx = 0; dx < L1D; ++dx) {
               energy(dx,dy,dz,e) = 0.0;
            }
         }
      }
      for (int qz = 0; qz < Q1D; ++qz) {
         double e_xy[L1D][L1D];
         for (int dx = 0; dx < L1D; ++dx) {
            for (int dy = 0; dy < L1D; ++dy) {
               e_xy[dx][dy] = 0.0;
            }
         }
         for (int qy = 0; qy < Q1D; ++qy) {
            double e_x[L1D];
            for (int dx = 0; dx < L1D; ++dx) {
               e_x[dx] = 0.0;
            }
            for (int qx = 0; qx < Q1D; ++qx) {
               const double r_v = vStress[qx][qy][qz];
               for (int dx = 0; dx < L1D; ++dx) {
                  e_x[dx] += r_v * L2Bt(dx,qx);
               }
            }
            for (int dy = 0; dy < L1D; ++dy) {
               const double w = L2Bt(dy,qy);
               for (int dx = 0; dx < L1D; ++dx) {
                  e_xy[dx][dy] += e_x[dx] * w;
               }
            }
         }
         for (int dz = 0; dz < L1D; ++dz) {
            const double w = L2Bt(dz,qz);
            for (int dy = 0; dy < L1D; ++dy) {
               for (int dx = 0; dx < L1D; ++dx) {
                  energy(dx,dy,dz,e) += w * e_xy[dx][dy];
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
      {0x21,&kForceMultTranspose2D<2,3,4,2,3>},
      {0x31,&kForceMultTranspose3D<3,3,4,2,3>},
   };
   if (!call[id]) {
      printf("\n[rForceMultTranspose] id \033[33m0x%X\033[m ",id);
      fflush(stdout);
   }
   assert(call[id]);
   call[id](nzones,
            L2QuadToDof,
            H1DofToQuad,
            H1DofToQuadD,
            stressJinvT,
            v,
            e);
}

// *************************************************************************
void kForcePAOperator::MultTranspose(const mfem::Vector &x,
                                     mfem::Vector &y) const {
   h1k.Mult(x, gVecH1);
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
   l2k.MultTranspose(gVecL2, y);
}

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI
