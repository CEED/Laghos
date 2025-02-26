#pragma once

// #include "fem/quadinterpolator.hpp"
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

template<int T_D1D = 0, int T_Q1D = 0, bool SMEM = true>
static void Det3D(const int NE,
                  const real_t *b,
                  const real_t *g,
                  const real_t *x,
                  real_t *y,
                  const int d1d = 0,
                  const int q1d = 0,
                  Vector *d_buff = nullptr) // used only with SMEM = false
{
   constexpr int DIM = 3;
   static constexpr int GRID = SMEM ? 0 : 128;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   const auto B = Reshape(b, Q1D, D1D);
   const auto G = Reshape(g, Q1D, D1D);
   const auto X = Reshape(x, D1D, D1D, D1D, DIM, NE);
   auto Y = Reshape(y, Q1D, Q1D, Q1D, NE);

   real_t *GM = nullptr;
   if (!SMEM)
   {
      const DeviceDofQuadLimits &limits = DeviceDofQuadLimits::Get();
      const int max_q1d = T_Q1D ? T_Q1D : limits.MAX_Q1D;
      const int max_d1d = T_D1D ? T_D1D : limits.MAX_D1D;
      const int max_qd = std::max(max_q1d, max_d1d);
      const int mem_size = max_qd * max_qd * max_qd * 9;
      d_buff->SetSize(2*mem_size*GRID);
      GM = d_buff->Write();
   }

   static constexpr int MQ1 =
      T_Q1D ? T_Q1D : (SMEM ? DofQuadLimits::MAX_DET_1D : DofQuadLimits::MAX_Q1D);
   static constexpr int MD1 =
      T_D1D ? T_D1D : (SMEM ? DofQuadLimits::MAX_DET_1D : DofQuadLimits::MAX_D1D);
   static constexpr int MDQ = MQ1 > MD1 ? MQ1 : MD1;
   static constexpr int MSZ = MDQ * MDQ * MDQ * 9;

   Vector vm0(MSZ), vm1(MSZ);
   vm0.UseDevice(true), vm1.UseDevice(true);
   auto GM0 = vm0.Write(), GM1 = vm1.Write();

   mfem::forall_3D(NE,
                   Q1D>8 ? 8 : Q1D,
                   Q1D>8 ? 8 : Q1D,
                   Q1D>8 ? 8 : Q1D,
                   [=] MFEM_HOST_DEVICE (int e)
   {
      static constexpr int MQ1 = T_Q1D ? T_Q1D :
                                 (SMEM ? DofQuadLimits::MAX_DET_1D : DofQuadLimits::MAX_Q1D);
      static constexpr int MD1 = T_D1D ? T_D1D :
                                 (SMEM ? DofQuadLimits::MAX_DET_1D : DofQuadLimits::MAX_D1D);
      static constexpr int MDQ = MQ1 > MD1 ? MQ1 : MD1;
      static constexpr int MSZ = MDQ * MDQ * MDQ * 9;

      const int bid = MFEM_BLOCK_ID(x);
      MFEM_SHARED real_t BG[2][MQ1*MD1];
      //   MFEM_SHARED real_t SM0[SMEM?MSZ:1];
      //   MFEM_SHARED real_t SM1[SMEM?MSZ:1];
      real_t *lm0 = SMEM ? GM0 : GM + MSZ*bid;
      real_t *lm1 = SMEM ? GM1 : GM + MSZ*(GRID+bid);
      real_t (*DDD)[MD1*MD1*MD1] = (real_t (*)[MD1*MD1*MD1]) (lm0);
      real_t (*DDQ)[MD1*MD1*MQ1] = (real_t (*)[MD1*MD1*MQ1]) (lm1);
      real_t (*DQQ)[MD1*MQ1*MQ1] = (real_t (*)[MD1*MQ1*MQ1]) (lm0);
      real_t (*QQQ)[MQ1*MQ1*MQ1] = (real_t (*)[MQ1*MQ1*MQ1]) (lm1);

      kernels::internal::LoadX<MD1>(e,D1D,X,DDD);
      kernels::internal::LoadBG<MD1,MQ1>(D1D,Q1D,B,G,BG);

      kernels::internal::GradX<MD1,MQ1>(D1D,Q1D,BG,DDD,DDQ);
      kernels::internal::GradY<MD1,MQ1>(D1D,Q1D,BG,DDQ,DQQ);
      kernels::internal::GradZ<MD1,MQ1>(D1D,Q1D,BG,DQQ,QQQ);

      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               real_t J[9];
               kernels::internal::PullGrad<MQ1>(Q1D, qx,qy,qz, QQQ, J);
               Y(qx,qy,qz,e) = kernels::Det<3>(J);
            }
         }
      }
   });
}

} // namespace quadrature_interpolator

} // namespace internal

} // namespace mfem

