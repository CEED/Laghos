#pragma once

#include "general/forall.hpp"
#include "linalg/dtensor.hpp"
#include "fem/kernels.hpp"
#include "linalg/kernels.hpp"

namespace mfem
{

namespace internal
{

namespace quadrature_interpolator
{

template<QVectorLayout Q_LAYOUT, bool GRAD_PHYS,
         int T_VDIM = 0, int T_D1D = 0, int T_Q1D = 0>
static void Derivatives3D(const int NE,
                          const real_t *b_,
                          const real_t *g_,
                          const real_t *j_,
                          const real_t *x_,
                          real_t *y_,
                          const int sdim = 3,
                          const int vdim = 0,
                          const int d1d = 0,
                          const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int VDIM = T_VDIM ? T_VDIM : vdim;

   const auto b = Reshape(b_, Q1D, D1D);
   const auto g = Reshape(g_, Q1D, D1D);
   const auto j = Reshape(j_, Q1D, Q1D, Q1D, 3, 3, NE);
   const auto x = Reshape(x_, D1D, D1D, D1D, VDIM, NE);
   auto y = Q_LAYOUT == QVectorLayout:: byNODES ?
            Reshape(y_, Q1D, Q1D, Q1D, VDIM, 3, NE):
            Reshape(y_, VDIM, 3, Q1D, Q1D, Q1D, NE);

   constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_INTERP_1D;
   constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_INTERP_1D;

   Vector vm0(3*MQ1*MQ1*MQ1), vm1(3*MQ1*MQ1*MQ1);
   vm0.UseDevice(true), vm1.UseDevice(true);
   auto gm0 = Reshape(vm0.Write(), 3, MQ1*MQ1*MQ1);
   auto gm1 = Reshape(vm1.Write(), 3, MQ1*MQ1*MQ1);

   mfem::forall_3D(NE,
                   Q1D>8 ? 8 : Q1D,
                   Q1D>8 ? 8 : Q1D,
                   Q1D>8 ? 8 : Q1D,
                   [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_INTERP_1D;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_INTERP_1D;

      MFEM_SHARED real_t BG[2][MQ1*MD1];
      kernels::internal::LoadBG<MD1,MQ1>(D1D,Q1D,b,g,BG);
      DeviceMatrix B(BG[0], D1D, Q1D);
      DeviceMatrix G(BG[1], D1D, Q1D);

      //MFEM_SHARED real_t sm0[3][MQ1*MQ1*MQ1];
      //MFEM_SHARED real_t sm1[3][MQ1*MQ1*MQ1];
      DeviceTensor<3> X(&gm0(2,0), D1D, D1D, D1D);
      DeviceTensor<3> DDQ0(&gm0(0,0), D1D, D1D, Q1D);
      DeviceTensor<3> DDQ1(&gm0(1,0), D1D, D1D, Q1D);
      DeviceTensor<3> DQQ0(&gm1(0,0), D1D, Q1D, Q1D);
      DeviceTensor<3> DQQ1(&gm1(1,0), D1D, Q1D, Q1D);
      DeviceTensor<3> DQQ2(&gm1(2,0), D1D, Q1D, Q1D);

      for (int c = 0; c < VDIM; ++c)
      {
         kernels::internal::LoadX(e,D1D,c,x,X);
         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  real_t u = 0.0;
                  real_t v = 0.0;
                  for (int dx = 0; dx < D1D; ++dx)
                  {
                     const real_t input = X(dx,dy,dz);
                     u += input * B(dx,qx);
                     v += input * G(dx,qx);
                  }
                  DDQ0(dz,dy,qx) = u;
                  DDQ1(dz,dy,qx) = v;
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
                  real_t u = 0.0;
                  real_t v = 0.0;
                  real_t w = 0.0;
                  for (int dy = 0; dy < D1D; ++dy)
                  {
                     u += DDQ1(dz,dy,qx) * B(dy,qy);
                     v += DDQ0(dz,dy,qx) * G(dy,qy);
                     w += DDQ0(dz,dy,qx) * B(dy,qy);
                  }
                  DQQ0(dz,qy,qx) = u;
                  DQQ1(dz,qy,qx) = v;
                  DQQ2(dz,qy,qx) = w;
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
                  real_t u = 0.0;
                  real_t v = 0.0;
                  real_t w = 0.0;
                  for (int dz = 0; dz < D1D; ++dz)
                  {
                     u += DQQ0(dz,qy,qx) * B(dz,qz);
                     v += DQQ1(dz,qy,qx) * B(dz,qz);
                     w += DQQ2(dz,qy,qx) * G(dz,qz);
                  }
                  if (GRAD_PHYS)
                  {
                     real_t Jloc[9], Jinv[9];
                     for (int col = 0; col < 3; col++)
                     {
                        for (int row = 0; row < 3; row++)
                        {
                           Jloc[row+3*col] = j(qx,qy,qz,row,col,e);
                        }
                     }
                     kernels::CalcInverse<3>(Jloc, Jinv);
                     const real_t U = Jinv[0]*u + Jinv[1]*v + Jinv[2]*w;
                     const real_t V = Jinv[3]*u + Jinv[4]*v + Jinv[5]*w;
                     const real_t W = Jinv[6]*u + Jinv[7]*v + Jinv[8]*w;
                     u = U; v = V; w = W;
                  }
                  if (Q_LAYOUT == QVectorLayout::byVDIM)
                  {
                     y(c,0,qx,qy,qz,e) = u;
                     y(c,1,qx,qy,qz,e) = v;
                     y(c,2,qx,qy,qz,e) = w;
                  }
                  if (Q_LAYOUT == QVectorLayout::byNODES)
                  {
                     y(qx,qy,qz,c,0,e) = u;
                     y(qx,qy,qz,c,1,e) = v;
                     y(qx,qy,qz,c,2,e) = w;
                  }
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

} // namespace quadrature_interpolator

} // namespace internal

} // namespace mfem

