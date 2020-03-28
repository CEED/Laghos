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
#include <unordered_map>

namespace mfem
{

namespace hydrodynamics
{

void DensityIntegrator::AssembleRHSElementVect(const FiniteElement &fe,
                                               ElementTransformation &Tr,
                                               Vector &elvect)
{
   const int nqp = IntRule->GetNPoints();
   Vector shape(fe.GetDof());
   elvect.SetSize(fe.GetDof());
   elvect = 0.0;
   for (int q = 0; q < nqp; q++)
   {
      fe.CalcShape(IntRule->IntPoint(q), shape);
      // Note that rhoDetJ = rho0DetJ0.
      shape *= qdata.rho0DetJ0w(Tr.ElementNo*nqp + q);
      elvect += shape;
   }
}

void ForceIntegrator::AssembleElementMatrix2(const FiniteElement &trial_fe,
                                             const FiniteElement &test_fe,
                                             ElementTransformation &Tr,
                                             DenseMatrix &elmat)
{
   const int e = Tr.ElementNo;
   const int nqp = IntRule->GetNPoints();
   const int dim = trial_fe.GetDim();
   const int h1dofs_cnt = test_fe.GetDof();
   const int l2dofs_cnt = trial_fe.GetDof();
   elmat.SetSize(h1dofs_cnt*dim, l2dofs_cnt);
   elmat = 0.0;
   DenseMatrix vshape(h1dofs_cnt, dim), loc_force(h1dofs_cnt, dim);
   Vector shape(l2dofs_cnt), Vloc_force(loc_force.Data(), h1dofs_cnt*dim);
   for (int q = 0; q < nqp; q++)
   {
      const IntegrationPoint &ip = IntRule->IntPoint(q);
      // Form stress:grad_shape at the current point.
      test_fe.CalcDShape(ip, vshape);
      for (int i = 0; i < h1dofs_cnt; i++)
      {
         for (int vd = 0; vd < dim; vd++) // Velocity components.
         {
            loc_force(i, vd) = 0.0;
            for (int gd = 0; gd < dim; gd++) // Gradient components.
            {
               const int eq = e*nqp + q;
               const double stressJinvT = qdata.stressJinvT(vd)(eq, gd);
               loc_force(i, vd) +=  stressJinvT * vshape(i,gd);
            }
         }
      }
      trial_fe.CalcShape(ip, shape);
      AddMultVWt(Vloc_force, shape, elmat);
   }
}

MassPAOperator::MassPAOperator(FiniteElementSpace &fes,
                               const IntegrationRule &ir,
                               Coefficient &Q) :
   Operator(fes.GetTrueVSize()),
   dim(fes.GetMesh()->Dimension()),
   NE(fes.GetMesh()->GetNE()),
   vsize(fes.GetVSize()),
   pabf(&fes),
   ess_tdofs_count(0),
   ess_tdofs(0)
{
   pabf.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   pabf.AddDomainIntegrator(new mfem::MassIntegrator(Q, &ir));
   pabf.Assemble();
   pabf.FormSystemMatrix(mfem::Array<int>(), mass);
}

void MassPAOperator::SetEssentialTrueDofs(Array<int> &dofs)
{
   ess_tdofs_count = dofs.Size();
   if (ess_tdofs.Size() == 0)
   {
      ess_tdofs.SetSize(ess_tdofs_count);
   }
   if (ess_tdofs_count == 0) { return; }
   ess_tdofs = dofs;
}

void MassPAOperator::EliminateRHS(Vector &b) const
{
   if (ess_tdofs_count > 0) { b.SetSubVector(ess_tdofs, 0.0); }
}

void MassPAOperator::Mult(const Vector &x, Vector &y) const
{
   mass->Mult(x, y);
   if (ess_tdofs_count > 0) { y.SetSubVector(ess_tdofs, 0.0); }
}

ForcePAOperator::ForcePAOperator(const QuadratureData &qdata,
                                 FiniteElementSpace &h1,
                                 FiniteElementSpace &l2,
                                 const IntegrationRule &ir) :
   Operator(),
   dim(h1.GetMesh()->Dimension()),
   NE(h1.GetMesh()->GetNE()),
   qdata(qdata),
   H1(h1),
   L2(l2),
   H1R(H1.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC)),
   L2R(L2.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC)),
   ir1D(IntRules.Get(Geometry::SEGMENT, ir.GetOrder())),
   D1D(H1.GetFE(0)->GetOrder()+1),
   Q1D(ir1D.GetNPoints()),
   L1D(L2.GetFE(0)->GetOrder()+1),
   H1sz(H1.GetVDim() * H1.GetFE(0)->GetDof() * NE),
   L2sz(L2.GetFE(0)->GetDof() * NE),
   L2D2Q(&L2.GetFE(0)->GetDofToQuad(ir, DofToQuad::TENSOR)),
   H1D2Q(&H1.GetFE(0)->GetDofToQuad(ir, DofToQuad::TENSOR)),
   X(L2sz), Y(H1sz) { }

template<int DIM, int D1D, int Q1D, int L1D, int NBZ = 1> static
void ForceMult2D(const int NE,
                 const Array<double> &B_,
                 const Array<double> &Bt_,
                 const Array<double> &Gt_,
                 const DenseTensor &sJit_,
                 const Vector &x, Vector &y)
{
   auto b = Reshape(B_.Read(), Q1D, L1D);
   auto bt = Reshape(Bt_.Read(), D1D, Q1D);
   auto gt = Reshape(Gt_.Read(), D1D, Q1D);
   const double *StressJinvT = Read(sJit_.GetMemory(), Q1D*Q1D*NE*DIM*DIM);
   auto sJit = Reshape(StressJinvT, Q1D, Q1D, NE, DIM, DIM);
   auto energy = Reshape(x.Read(), L1D, L1D, NE);
   const double eps1 = std::numeric_limits<double>::epsilon();
   const double eps2 = eps1*eps1;
   auto velocity = Reshape(y.Write(), D1D, D1D, DIM, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, 1,
   {
      const int z = MFEM_THREAD_ID(z);

      MFEM_SHARED double B[Q1D][L1D];
      MFEM_SHARED double Bt[D1D][Q1D];
      MFEM_SHARED double Gt[D1D][Q1D];

      MFEM_SHARED double Ez[NBZ][L1D][L1D];
      double (*E)[L1D] = (double (*)[L1D])(Ez + z);

      MFEM_SHARED double LQz[2][NBZ][D1D][Q1D];
      double (*LQ0)[Q1D] = (double (*)[Q1D])(LQz[0] + z);
      double (*LQ1)[Q1D] = (double (*)[Q1D])(LQz[1] + z);

      MFEM_SHARED double QQz[3][NBZ][Q1D][Q1D];
      double (*QQ)[Q1D] = (double (*)[Q1D])(QQz[0] + z);
      double (*QQ0)[Q1D] = (double (*)[Q1D])(QQz[1] + z);
      double (*QQ1)[Q1D] = (double (*)[Q1D])(QQz[2] + z);

      if (z == 0)
      {
         MFEM_FOREACH_THREAD(q,x,Q1D)
         {
            MFEM_FOREACH_THREAD(l,y,Q1D)
            {
               if (l < L1D) { B[q][l] = b(q,l); }
               if (l < D1D) { Bt[l][q] = bt(l,q); }
               if (l < D1D) { Gt[l][q] = gt(l,q); }
            }
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(lx,x,L1D)
      {
         MFEM_FOREACH_THREAD(ly,y,L1D)
         {
            E[lx][ly] = energy(lx,ly,e);
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(ly,y,L1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u = 0.0;
            for (int lx = 0; lx < L1D; ++lx)
            {
               u += B[qx][lx] * E[lx][ly];
            }
            LQ0[ly][qx] = u;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double u = 0.0;
            for (int ly = 0; ly < L1D; ++ly)
            {
               u += B[qy][ly] * LQ0[ly][qx];
            }
            QQ[qy][qx] = u;
         }
      }
      MFEM_SYNC_THREAD;

      for (int c = 0; c < DIM; ++c)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               const double esx = QQ[qy][qx] * sJit(qx,qy,e,0,c);
               const double esy = QQ[qy][qx] * sJit(qx,qy,e,1,c);
               QQ0[qy][qx] = esx;
               QQ1[qy][qx] = esy;
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               double u = 0.0;
               double v = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  u += Gt[dx][qx] * QQ0[qy][qx];
                  v += Bt[dx][qx] * QQ1[qy][qx];
               }
               LQ0[dx][qy] = u;
               LQ1[dx][qy] = v;
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               double u = 0.0;
               double v = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  u += LQ0[dx][qy] * Bt[dy][qy];
                  v += LQ1[dx][qy] * Gt[dy][qy];
               }
               velocity(dx,dy,c,e) = u + v;
            }
         }
         MFEM_SYNC_THREAD;
      }
      for (int c = 0; c < DIM; ++c)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               const double v = velocity(dx,dy,c,e);
               if (fabs(v) < eps2)
               {
                  velocity(dx,dy,c,e) = 0.0;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

template<int DIM, int D1D, int Q1D, int L1D> static
void ForceMult3D(const int NE,
                 const Array<double> &B_,
                 const Array<double> &Bt_,
                 const Array<double> &Gt_,
                 const DenseTensor &sJit_,
                 const Vector &x, Vector &y)
{
   auto b = Reshape(B_.Read(), Q1D, L1D);
   auto bt = Reshape(Bt_.Read(), D1D, Q1D);
   auto gt = Reshape(Gt_.Read(), D1D, Q1D);
   const double *StressJinvT = Read(sJit_.GetMemory(), Q1D*Q1D*Q1D*NE*DIM*DIM);
   auto sJit = Reshape(StressJinvT, Q1D, Q1D, Q1D, NE, DIM, DIM);
   auto energy = Reshape(x.Read(), L1D, L1D, L1D, NE);
   const double eps1 = std::numeric_limits<double>::epsilon();
   const double eps2 = eps1*eps1;
   auto velocity = Reshape(y.Write(), D1D, D1D, D1D, DIM, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int z = MFEM_THREAD_ID(z);

      MFEM_SHARED double B[Q1D][L1D];
      MFEM_SHARED double Bt[D1D][Q1D];
      MFEM_SHARED double Gt[D1D][Q1D];

      MFEM_SHARED double E[L1D][L1D][L1D];

      MFEM_SHARED double sm0[3][Q1D*Q1D*Q1D];
      MFEM_SHARED double sm1[3][Q1D*Q1D*Q1D];

      double (*MMQ0)[D1D][Q1D] = (double (*)[D1D][Q1D]) (sm0+0);
      double (*MMQ1)[D1D][Q1D] = (double (*)[D1D][Q1D]) (sm0+1);
      double (*MMQ2)[D1D][Q1D] = (double (*)[D1D][Q1D]) (sm0+2);

      double (*MQQ0)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm1+0);
      double (*MQQ1)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm1+1);
      double (*MQQ2)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm1+2);

      MFEM_SHARED double QQQ[Q1D][Q1D][Q1D];
      double (*QQQ0)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm0+0);
      double (*QQQ1)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm0+1);
      double (*QQQ2)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm0+2);

      if (z == 0)
      {
         MFEM_FOREACH_THREAD(q,x,Q1D)
         {
            MFEM_FOREACH_THREAD(l,y,Q1D)
            {
               if (l < L1D) { B[q][l] = b(q,l); }
               if (l < D1D) { Bt[l][q] = bt(l,q); }
               if (l < D1D) { Gt[l][q] = gt(l,q); }
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(lx,x,L1D)
      {
         MFEM_FOREACH_THREAD(ly,y,L1D)
         {
            MFEM_FOREACH_THREAD(lz,z,L1D)
            {
               E[lx][ly][lz] = energy(lx,ly,lz,e);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(lz,z,L1D)
      {
         MFEM_FOREACH_THREAD(ly,y,L1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0;
               for (int lx = 0; lx < L1D; ++lx)
               {
                  u += B[qx][lx] * E[lx][ly][lz];
               }
               MMQ0[lz][ly][qx] = u;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(lz,z,L1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0;
               for (int ly = 0; ly < L1D; ++ly)
               {
                  u += B[qy][ly] * MMQ0[lz][ly][qx];
               }
               MQQ0[lz][qy][qx] = u;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0;
               for (int lz = 0; lz < L1D; ++lz)
               {
                  u += B[qz][lz] * MQQ0[lz][qy][qx];
               }
               QQQ[qz][qy][qx] = u;
            }
         }
      }
      MFEM_SYNC_THREAD;
      for (int c = 0; c < 3; ++c)
      {
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  const double esx = QQQ[qz][qy][qx] * sJit(qx,qy,qz,e,0,c);
                  const double esy = QQQ[qz][qy][qx] * sJit(qx,qy,qz,e,1,c);
                  const double esz = QQQ[qz][qy][qx] * sJit(qx,qy,qz,e,2,c);
                  QQQ0[qz][qy][qx] = esx;
                  QQQ1[qz][qy][qx] = esy;
                  QQQ2[qz][qy][qx] = esz;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(hx,x,D1D)
               {
                  double u = 0.0;
                  double v = 0.0;
                  double w = 0.0;
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     u += Gt[hx][qx] * QQQ0[qz][qy][qx];
                     v += Bt[hx][qx] * QQQ1[qz][qy][qx];
                     w += Bt[hx][qx] * QQQ2[qz][qy][qx];
                  }
                  MQQ0[hx][qy][qz] = u;
                  MQQ1[hx][qy][qz] = v;
                  MQQ2[hx][qy][qz] = w;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(hy,y,D1D)
            {
               MFEM_FOREACH_THREAD(hx,x,D1D)
               {
                  double u = 0.0;
                  double v = 0.0;
                  double w = 0.0;
                  for (int qy = 0; qy < Q1D; ++qy)
                  {
                     u += MQQ0[hx][qy][qz] * Bt[hy][qy];
                     v += MQQ1[hx][qy][qz] * Gt[hy][qy];
                     w += MQQ2[hx][qy][qz] * Bt[hy][qy];
                  }
                  MMQ0[hx][hy][qz] = u;
                  MMQ1[hx][hy][qz] = v;
                  MMQ2[hx][hy][qz] = w;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(hz,z,D1D)
         {
            MFEM_FOREACH_THREAD(hy,y,D1D)
            {
               MFEM_FOREACH_THREAD(hx,x,D1D)
               {
                  double u = 0.0;
                  double v = 0.0;
                  double w = 0.0;
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     u += MMQ0[hx][hy][qz] * Bt[hz][qz];
                     v += MMQ1[hx][hy][qz] * Bt[hz][qz];
                     w += MMQ2[hx][hy][qz] * Gt[hz][qz];
                  }
                  velocity(hx,hy,hz,c,e) = u + v + w;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
      for (int c = 0; c < 3; ++c)
      {
         MFEM_FOREACH_THREAD(hz,z,D1D)
         {
            MFEM_FOREACH_THREAD(hy,y,D1D)
            {
               MFEM_FOREACH_THREAD(hx,x,D1D)
               {
                  const double v = velocity(hx,hy,hz,c,e);
                  if (fabs(v) < eps2)
                  {
                     velocity(hx,hy,hz,c,e) = 0.0;
                  }
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

typedef void (*fForceMult)(const int E,
                           const Array<double> &B,
                           const Array<double> &Bt,
                           const Array<double> &Gt,
                           const DenseTensor &stressJinvT,
                           const Vector &X, Vector &Y);

static void ForceMult(const int DIM, const int D1D, const int Q1D,
                      const int L1D, const int H1D, const int NE,
                      const Array<double> &B,
                      const Array<double> &Bt,
                      const Array<double> &Gt,
                      const DenseTensor &stressJinvT,
                      const Vector &e,
                      Vector &v)
{
   MFEM_VERIFY(D1D==H1D, "D1D!=H1D");
   MFEM_VERIFY(L1D==D1D-1,"L1D!=D1D-1");
   const int id = ((DIM)<<8)|(D1D)<<4|(Q1D);
   static std::unordered_map<int, fForceMult> call =
   {
      // 2D
      {0x234,&ForceMult2D<2,3,4,2>},
      {0x246,&ForceMult2D<2,4,6,3>},
      {0x258,&ForceMult2D<2,5,8,4>},
      // 3D
      {0x334,&ForceMult3D<3,3,4,2>},
      {0x346,&ForceMult3D<3,4,6,3>},
      {0x358,&ForceMult3D<3,5,8,4>},
   };
   if (!call[id])
   {
      mfem::out << "Unknown kernel 0x" << std::hex << id << std::endl;
      MFEM_ABORT("Unknown kernel");
   }
   call[id](NE, B, Bt, Gt, stressJinvT, e, v);
}

void ForcePAOperator::Mult(const Vector &x, Vector &y) const
{
   if (L2R) { L2R->Mult(x, X); }
   else { X = x; }
   ForceMult(dim, D1D, Q1D, L1D, D1D, NE,
             L2D2Q->B, H1D2Q->Bt, H1D2Q->Gt,
             qdata.stressJinvT, X, Y);
   H1R->MultTranspose(Y, y);
}

template<int DIM, int D1D, int Q1D, int L1D, int NBZ = 1> static
void ForceMultTranspose2D(const int NE,
                          const Array<double> &Bt_,
                          const Array<double> &B_,
                          const Array<double> &G_,
                          const DenseTensor &sJit_,
                          const Vector &x, Vector &y)
{
   auto b = Reshape(B_.Read(), Q1D, D1D);
   auto g = Reshape(G_.Read(), Q1D, D1D);
   auto bt = Reshape(Bt_.Read(), L1D, Q1D);
   const double *StressJinvT = Read(sJit_.GetMemory(), Q1D*Q1D*NE*DIM*DIM);
   auto sJit = Reshape(StressJinvT, Q1D, Q1D, NE, DIM, DIM);
   auto velocity = Reshape(x.Read(), D1D, D1D, DIM, NE);
   auto energy = Reshape(y.Write(), L1D, L1D, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int z = MFEM_THREAD_ID(z);

      MFEM_SHARED double Bt[L1D][Q1D];
      MFEM_SHARED double B[Q1D][D1D];
      MFEM_SHARED double G[Q1D][D1D];

      MFEM_SHARED double Vz[NBZ][D1D*D1D];
      double (*V)[D1D] = (double (*)[D1D])(Vz + z);

      MFEM_SHARED double DQz[DIM][NBZ][D1D*Q1D];
      double (*DQ0)[Q1D] = (double (*)[Q1D])(DQz[0] + z);
      double (*DQ1)[Q1D] = (double (*)[Q1D])(DQz[1] + z);

      MFEM_SHARED double QQz[3][NBZ][Q1D*Q1D];
      double (*QQ)[Q1D] = (double (*)[Q1D])(QQz[0] + z);
      double (*QQ0)[Q1D] = (double (*)[Q1D])(QQz[1] + z);
      double (*QQ1)[Q1D] = (double (*)[Q1D])(QQz[2] + z);

      MFEM_SHARED double QLz[NBZ][Q1D*L1D];
      double (*QL)[L1D] = (double (*)[L1D]) (QLz + z);

      if (z == 0)
      {
         MFEM_FOREACH_THREAD(q,x,Q1D)
         {
            MFEM_FOREACH_THREAD(h,y,Q1D)
            {
               if (h < D1D) { B[q][h] = b(q,h); }
               if (h < D1D) { G[q][h] = g(q,h); }
               const int l = h;
               if (l < L1D) { Bt[l][q] = bt(l,q); }
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            QQ[qy][qx] = 0.0;
         }
      }
      MFEM_SYNC_THREAD;

      for (int c = 0; c < DIM; ++c)
      {

         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               V[dx][dy] = velocity(dx,dy,c,e);
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0;
               double v = 0.0;
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const double input = V[dx][dy];
                  u += B[qx][dx] * input;
                  v += G[qx][dx] * input;
               }
               DQ0[dy][qx] = u;
               DQ1[dy][qx] = v;
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0;
               double v = 0.0;
               for (int dy = 0; dy < D1D; ++dy)
               {
                  u += DQ1[dy][qx] * B[qy][dy];
                  v += DQ0[dy][qx] * G[qy][dy];
               }
               QQ0[qy][qx] = u;
               QQ1[qy][qx] = v;
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               const double esx = QQ0[qy][qx] * sJit(qx,qy,e,0,c);
               const double esy = QQ1[qy][qx] * sJit(qx,qy,e,1,c);
               QQ[qy][qx] += esx + esy;
            }
         }
         MFEM_SYNC_THREAD;
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(lx,x,L1D)
         {
            double u = 0.0;
            for (int qx = 0; qx < Q1D; ++qx)
            {
               u += QQ[qy][qx] * Bt[lx][qx];
            }
            QL[qy][lx] = u;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(ly,y,L1D)
      {
         MFEM_FOREACH_THREAD(lx,x,L1D)
         {
            double u = 0.0;
            for (int qy = 0; qy < Q1D; ++qy)
            {
               u += QL[qy][lx] * Bt[ly][qy];
            }
            energy(lx,ly,e) = u;
         }
      }
      MFEM_SYNC_THREAD;
   });
}

template<int DIM, int D1D, int Q1D, int L1D> static
void ForceMultTranspose3D(const int NE,
                          const Array<double> &Bt_,
                          const Array<double> &B_,
                          const Array<double> &G_,
                          const DenseTensor &sJit_,
                          const Vector &v_,
                          Vector &e_)
{
   auto b = Reshape(B_.Read(), Q1D, D1D);
   auto g = Reshape(G_.Read(), Q1D, D1D);
   auto bt = Reshape(Bt_.Read(), L1D, Q1D);
   const double *StressJinvT = Read(sJit_.GetMemory(), Q1D*Q1D*Q1D*NE*DIM*DIM);
   auto sJit = Reshape(StressJinvT, Q1D, Q1D, Q1D, NE, DIM, DIM);
   auto velocity = Reshape(v_.Read(), D1D, D1D, D1D, DIM, NE);
   auto energy = Reshape(e_.Write(), L1D, L1D, L1D, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int z = MFEM_THREAD_ID(z);

      MFEM_SHARED double Bt[L1D][Q1D];
      MFEM_SHARED double B[Q1D][D1D];
      MFEM_SHARED double G[Q1D][D1D];

      MFEM_SHARED double sm0[3][Q1D*Q1D*Q1D];
      MFEM_SHARED double sm1[3][Q1D*Q1D*Q1D];
      double (*V)[D1D][D1D]    = (double (*)[D1D][D1D]) (sm0+0);
      double (*MMQ0)[D1D][Q1D] = (double (*)[D1D][Q1D]) (sm0+1);
      double (*MMQ1)[D1D][Q1D] = (double (*)[D1D][Q1D]) (sm0+2);

      double (*MQQ0)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm1+0);
      double (*MQQ1)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm1+1);
      double (*MQQ2)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm1+2);

      double (*QQQ0)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm0+0);
      double (*QQQ1)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm0+1);
      double (*QQQ2)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm0+2);

      MFEM_SHARED double QQQ[Q1D][Q1D][Q1D];

      if (z == 0)
      {
         MFEM_FOREACH_THREAD(q,x,Q1D)
         {
            MFEM_FOREACH_THREAD(h,y,Q1D)
            {
               if (h < D1D) { B[q][h] = b(q,h); }
               if (h < D1D) { G[q][h] = g(q,h); }
               const int l = h;
               if (l < L1D) { Bt[l][q] = bt(l,q); }
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               QQQ[qz][qy][qx] = 0.0;
            }
         }
      }
      MFEM_SYNC_THREAD;

      for (int c = 0; c < DIM; ++c)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dz,z,D1D)
               {
                  V[dx][dy][dz] = velocity(dx,dy,dz,c,e);
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  double u = 0.0;
                  double v = 0.0;
                  for (int dx = 0; dx < D1D; ++dx)
                  {
                     const double input = V[dx][dy][dz];
                     u += G[qx][dx] * input;
                     v += B[qx][dx] * input;
                  }
                  MMQ0[dz][dy][qx] = u;
                  MMQ1[dz][dy][qx] = v;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  double u = 0.0;
                  double v = 0.0;
                  double w = 0.0;
                  for (int dy = 0; dy < D1D; ++dy)
                  {
                     u += MMQ0[dz][dy][qx] * B[qy][dy];
                     v += MMQ1[dz][dy][qx] * G[qy][dy];
                     w += MMQ1[dz][dy][qx] * B[qy][dy];
                  }
                  MQQ0[dz][qy][qx] = u;
                  MQQ1[dz][qy][qx] = v;
                  MQQ2[dz][qy][qx] = w;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  double u = 0.0;
                  double v = 0.0;
                  double w = 0.0;
                  for (int dz = 0; dz < D1D; ++dz)
                  {
                     u += MQQ0[dz][qy][qx] * B[qz][dz];
                     v += MQQ1[dz][qy][qx] * B[qz][dz];
                     w += MQQ2[dz][qy][qx] * G[qz][dz];
                  }
                  QQQ0[qz][qy][qx] = u;
                  QQQ1[qz][qy][qx] = v;
                  QQQ2[qz][qy][qx] = w;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  const double esx = QQQ0[qz][qy][qx] * sJit(qx,qy,qz,e,0,c);
                  const double esy = QQQ1[qz][qy][qx] * sJit(qx,qy,qz,e,1,c);
                  const double esz = QQQ2[qz][qy][qx] * sJit(qx,qy,qz,e,2,c);
                  QQQ[qz][qy][qx] += esx + esy + esz;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(lx,x,L1D)
            {
               double u = 0.0;
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  u += QQQ[qz][qy][qx] * Bt[lx][qx];
               }
               MQQ0[qz][qy][lx] = u;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(ly,y,L1D)
         {
            MFEM_FOREACH_THREAD(lx,x,L1D)
            {
               double u = 0.0;
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  u += MQQ0[qz][qy][lx] * Bt[ly][qy];
               }
               MMQ0[qz][ly][lx] = u;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(lz,z,L1D)
      {
         MFEM_FOREACH_THREAD(ly,y,L1D)
         {
            MFEM_FOREACH_THREAD(lx,x,L1D)
            {
               double u = 0.0;
               for (int qz = 0; qz < Q1D; ++qz)
               {
                  u += MMQ0[qz][ly][lx] * Bt[lz][qz];
               }
               energy(lx,ly,lz,e) = u;
            }
         }
      }
      MFEM_SYNC_THREAD;
   });
}

typedef void (*fForceMultTranspose)(const int NE,
                                    const Array<double> &Bt,
                                    const Array<double> &B,
                                    const Array<double> &G,
                                    const DenseTensor &sJit,
                                    const Vector &X, Vector &Y);

static void ForceMultTranspose(const int DIM, const int D1D, const int Q1D,
                               const int L1D, const int NE,
                               const Array<double> &L2Bt,
                               const Array<double> &H1B,
                               const Array<double> &H1G,
                               const DenseTensor &stressJinvT,
                               const Vector &v,
                               Vector &e)
{
   // DIM, D1D, Q1D, L1D(=D1D-1)
   MFEM_VERIFY(L1D==D1D-1, "L1D!=D1D-1");
   const int id = ((DIM)<<8)|(D1D)<<4|(Q1D);
   static std::unordered_map<int, fForceMultTranspose> call =
   {
      {0x234,&ForceMultTranspose2D<2,3,4,2>},
      {0x246,&ForceMultTranspose2D<2,4,6,3>},
      {0x258,&ForceMultTranspose2D<2,5,8,4>},
      {0x334,&ForceMultTranspose3D<3,3,4,2>},
      {0x346,&ForceMultTranspose3D<3,4,6,3>},
      {0x358,&ForceMultTranspose3D<3,5,8,4>}
   };
   if (!call[id])
   {
      mfem::out << "Unknown kernel 0x" << std::hex << id << std::endl;
      MFEM_ABORT("Unknown kernel");
   }
   call[id](NE, L2Bt, H1B, H1G, stressJinvT, v, e);
}

void ForcePAOperator::MultTranspose(const Vector &x, Vector &y) const
{
   H1R->Mult(x, Y);
   ForceMultTranspose(dim, D1D, Q1D, L1D, NE,
                      L2D2Q->Bt, H1D2Q->B, H1D2Q->G,
                      qdata.stressJinvT, Y, X);
   if (L2R) { L2R->MultTranspose(X, y); }
   else { y = X; }
}

} // namespace hydrodynamics

} // namespace mfem
