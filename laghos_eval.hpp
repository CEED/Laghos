#pragma once

#include "general/forall.hpp"
#include "linalg/dtensor.hpp"
#include "fem/kernels.hpp"
#include "fem/fespace.hpp"

namespace mfem
{

namespace internal
{

namespace quadrature_interpolator
{

// Template compute kernel for Values in 3D: tensor product version.
template<QVectorLayout Q_LAYOUT,
         int T_VDIM = 0, int T_D1D = 0, int T_Q1D = 0>
static void Values3D(const int NE,
                     const real_t *b_,
                     const real_t *x_,
                     real_t *y_,
                     const int vdim = 0,
                     const int d1d = 0,
                     const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int VDIM = T_VDIM ? T_VDIM : vdim;

   const auto b = Reshape(b_, Q1D, D1D);
   const auto x = Reshape(x_, D1D, D1D, D1D, VDIM, NE);
   auto y = Q_LAYOUT == QVectorLayout:: byNODES ?
            Reshape(y_, Q1D, Q1D, Q1D, VDIM, NE):
            Reshape(y_, VDIM, Q1D, Q1D, Q1D, NE);

   constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_INTERP_1D;
   constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_INTERP_1D;
   constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;

   Vector vm0(MDQ*MDQ*MDQ), vm1(MDQ*MDQ*MDQ);
   vm0.UseDevice(true), vm1.UseDevice(true);
   auto gm0 = vm0.Write(), gm1 = vm1.Write();

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
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;

      MFEM_SHARED real_t sB[MQ1*MD1];
      // MFEM_SHARED real_t sm0[MDQ*MDQ*MDQ];
      // MFEM_SHARED real_t sm1[MDQ*MDQ*MDQ];

      kernels::internal::LoadB<MD1,MQ1>(D1D,Q1D,b,sB);

      ConstDeviceMatrix B(sB, D1D,Q1D);
      DeviceCube DDD(gm0, MD1,MD1,MD1);
      DeviceCube DDQ(gm1, MD1,MD1,MQ1);
      DeviceCube DQQ(gm0, MD1,MQ1,MQ1);
      DeviceCube QQQ(gm1, MQ1,MQ1,MQ1);

      for (int c = 0; c < VDIM; c++)
      {
         kernels::internal::LoadX(e,D1D,c,x,DDD);
         kernels::internal::EvalX(D1D,Q1D,B,DDD,DDQ);
         kernels::internal::EvalY(D1D,Q1D,B,DDQ,DQQ);
         kernels::internal::EvalZ(D1D,Q1D,B,DQQ,QQQ);
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  const real_t u = QQQ(qz,qy,qx);
                  if (Q_LAYOUT == QVectorLayout::byVDIM) { y(c,qx,qy,qz,e) = u; }
                  if (Q_LAYOUT == QVectorLayout::byNODES) { y(qx,qy,qz,c,e) = u; }
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