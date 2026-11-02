// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
#pragma once

#include "laghos_hydro.hpp"
#include "linalg/kernels.hpp"

namespace mfem::hydrodynamics
{

// ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
using dscalar_t =
   std::conditional_t<mfem::UseEnzyme, real_t,
   mfem::future::dual<real_t, real_t>>;

template <int DIM, typename T = real_t>
using vecd_t = tensor<T, DIM>;

template <int DIM, typename T = real_t>
using matd_t = tensor<T, DIM, DIM>;

// ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
template <typename T> MFEM_HOST_DEVICE inline
void ComputeMaterialProperties(const real_t &gamma, const T &rho,
                               const T &E, T &p, T &cs)
{
   p = (gamma - 1.0) * rho * E;
   cs = sqrt(gamma * (gamma - 1.0) * E);
}

// ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
// Smooth transition between 0 and 1 for x in [-eps, eps].
template <typename T> MFEM_HOST_DEVICE inline
T smooth_step_01(const T x, const real_t eps)
{
   const T y = (x + eps) / (2.0 * eps);
   if (y < 0.0) { return T{0.0}; }
   if (y > 1.0) { return T{1.0}; }
   return (3.0 - 2.0 * y) * y * y;
}

// ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
// Singular value i of a 3x3 matrix, in decreasing order, from the closed-form
// eigenvalues of A^T A
template <typename T> MFEM_HOST_DEVICE
T calcsv(const future::tensor<T, 3, 3> A, const int i)
{
   using std::acos, std::cos, std::fabs, std::fmax, std::fmin, std::sqrt;
   double mult;
   T a00, a01, a02, a10, a11, a12, a20, a21, a22;

   a00 = A(0, 0); a01 = A(0, 1); a02 = A(0, 2);
   a10 = A(1, 0); a11 = A(1, 1); a12 = A(1, 2);
   a20 = A(2, 0); a21 = A(2, 1); a22 = A(2, 2);

   {
      // The scaling factor is a power of 2, without derivative
      double d_max = 0.0;
      for (int m = 0; m < 3; m++)
      {
         for (int n = 0; n < 3; n++)
         {
            const double a_mn = fabs(future::get_value(A(m, n)));
            if (d_max < a_mn) { d_max = a_mn; }
         }
      }
      future::GetScalingFactor(d_max, mult);
   }

   a00 = a00 / mult; a01 = a01 / mult; a02 = a02 / mult;
   a10 = a10 / mult; a11 = a11 / mult; a12 = a12 / mult;
   a20 = a20 / mult; a21 = a21 / mult; a22 = a22 / mult;

   // Build B = A^T A (symmetric positive semidefinite)
   const T b00 = a00*a00 + a10*a10 + a20*a20;
   const T b01 = a00*a01 + a10*a11 + a20*a21;
   const T b02 = a00*a02 + a10*a12 + a20*a22;
   const T b11 = a01*a01 + a11*a11 + a21*a21;
   const T b12 = a01*a02 + a11*a12 + a21*a22;
   const T b22 = a02*a02 + a12*a12 + a22*a22;

   T e0, e1, e2;

   // Closed-form eigenvalues for symmetric 3x3 matrix
   // See eigenvalue algorithm - stable trigonometric form
   const T p1 = b01*b01 + b02*b02 + b12*b12;
   if (p1 == 0.0)
   {
      // B is diagonal
      e0 = b00;
      e1 = b11;
      e2 = b22;
   }
   else
   {
      const T q = (b00 + b11 + b22) / 3.0;

      const T c00 = b00 - q;
      const T c11 = b11 - q;
      const T c22 = b22 - q;

      const T p2 = c00*c00 + c11*c11 + c22*c22 + 2.0*p1;
      const T p = sqrt(p2 / 6.0);

      // C = (1/p) * (B - q I)
      const T invp = 1.0 / p;
      const T c01 = b01 * invp;
      const T c02 = b02 * invp;
      const T c12 = b12 * invp;
      const T cc00 = c00 * invp;
      const T cc11 = c11 * invp;
      const T cc22 = c22 * invp;

      // r = det(C) / 2
      const T detC =
         cc00*(cc11*cc22 - c12*c12) -
         c01*(c01*cc22 - c12*c02) +
         c02*(c01*c12 - cc11*c02);
      T r = 0.5 * detC;

      // Clamp r into [-1, 1] to avoid NaNs from acos due to roundoff, the
      // bounds of repeated eigenvalues where acos has no derivative included
      if (r <= -1.0) { r = -1.0; }
      if (r >=  1.0) { r =  1.0; }

      const T phi = acos(r) / 3.0;

      // These are the eigenvalues of B, ordered as e0 >= e1 >= e2 in exact arithmetic
      e0 = q + 2.0*p*cos(phi);
      e2 = q + 2.0*p*cos(phi + 2.0*M_PI/3.0);
      e1 = 3.0*q - e0 - e2;
   }

   // Enforce non-negativity (B is PSD, but roundoff can make tiny negatives)
   if (e0 < 0.0) { e0 = 0.0; }
   if (e1 < 0.0) { e1 = 0.0; }
   if (e2 < 0.0) { e2 = 0.0; }

   // Select requested eigenvalue by sorting (robust even in nearly-degenerate cases)
   const T emax = fmax(e0, fmax(e1, e2));
   const T emin = fmin(e0, fmin(e1, e2));
   const T emid = (e0 + e1 + e2) - emax - emin;

   const T chosen = (i == 0) ? emax : ((i == 1) ? emid : emin);
   // Singular A, where the square root has no derivative
   if (chosen == 0.0) { return chosen * mult; }
   return sqrt(chosen) * mult;
}

// ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
template <int DIM, typename T>
MFEM_HOST_DEVICE inline
matd_t<DIM, T> qdata_setup(const matd_t<DIM, T> &dvdxi,
                           const real_t &rho0,
                           const matd_t<DIM> &J0,
                           const matd_t<DIM, T> &J,
                           const real_t &gamma,
                           const T &E,
                           const real_t &w,
                           const real_t &h0,
                           const real_t &order_v,
                           const real_t &cfl,
                           const bool &use_viscosity,
                           const int &viscosity_type,
                           const real_t &visc_q1,
                           const real_t &visc_q2,
                           real_t &dt_est)
{
   using matd = matd_t<DIM, T>;

   T p, cs;
   const T detJ = det(J);
   const matd invJ = inv(J);
   matd stress{};
   const T rho = rho0 * det(J0) / detJ;
   // The stock Laghos viscosity (type 1) also uses the stock clamp of E
   const bool stock_av = use_viscosity && viscosity_type == 1;
   const auto smoothmax = [](const real_t a, const T b, const real_t k = 1e-6)
   {
      return 0.5 * (a + b + sqrt((a-b)*(a-b) + k*k));
   };
   const T Ez = stock_av ? fmax(0.0, E) : smoothmax(0.0, E, 1e-6);
   T dt_visc_coeff{0.0};

   ComputeMaterialProperties(gamma, rho, Ez, p, cs);

   for (int d = 0; d < DIM; d++) { stress(d, d) = -p; }

   if (use_viscosity)
   {
      // y0 is value at x=0.
      auto softstep7 = [](const T &width, const real_t y0, const T x)
      {
         const T shift = 2.0 * width * atanh(1.0 - 2.0 * y0);
         return 0.5 + 0.5 * tanh(0.5 * (x-shift) / width);
      };

      // Diffuse the kink at |x| ~ 0, activates as |x| approaches eps
      auto softabs = [](const T &eps, const T x)
      {
         return hypot(x, eps);
      };

      if (viscosity_type == 1)
      {
         // Stock Laghos tensor viscosity, see QUpdateBody in laghos_solver.cpp
         // (without the vorticity scaling, as for the stock Sedov problem)
         // The symmetric velocity gradient eigenvector of the smallest
         // eigenvalue gives the direction of maximal compression, which
         // defines the length scale h from the initial one h0
         const matd sgrad_v = sym(dvdxi * invJ);
         real_t sgrad_v_data[DIM*DIM], eig_val[DIM], eig_vec[DIM*DIM];
         for (int j = 0; j < DIM; j++)
         {
            for (int i = 0; i < DIM; i++)
            {
               sgrad_v_data[i + DIM*j] = future::get_value(sgrad_v(i, j));
            }
         }
         if constexpr (DIM == 1)
         {
            eig_val[0] = sgrad_v_data[0];
            eig_vec[0] = 1.0;
         }
         else { kernels::CalcEigenvalues<DIM>(sgrad_v_data, eig_val, eig_vec); }
         vecd_t<DIM, T> compr_dir;
         for (int k = 0; k < DIM; k++) { compr_dir(k) = eig_vec[k]; }
         // Measure of maximal compression
         T mu{eig_val[0]};
         if constexpr (future::is_dual_number_v<T>)
         {
            // Tangents of the eigenpair (l0, v0) of S from the eigenpairs
            // (lk, vk): dl0 = v0.dS.v0, dv0 = sum vk.dS.v0 / (l0 - lk) vk,
            // without the terms of the eigenvalues equal to l0
            auto vdSv = [&](const int a, const int b)
            {
               real_t vdSv_ab = 0.0;
               for (int i = 0; i < DIM; i++)
               {
                  for (int j = 0; j < DIM; j++)
                  {
                     vdSv_ab += eig_vec[a*DIM + i] * sgrad_v(i, j).gradient *
                                eig_vec[b*DIM + j];
                  }
               }
               return vdSv_ab;
            };
            mu.gradient = vdSv(0, 0);
            for (int k = 1; k < DIM; k++)
            {
               if (eig_val[k] == eig_val[0]) { continue; }
               const real_t c = vdSv(k, 0) / (eig_val[0] - eig_val[k]);
               for (int i = 0; i < DIM; i++)
               {
                  compr_dir(i).gradient += c * eig_vec[k*DIM + i];
               }
            }
         }
         const vecd_t<DIM, T> ph_dir = (J * inv(J0)) * compr_dir;
         const T h = h0 * norm(ph_dir) / norm(compr_dir);
         // Quadratic term, then a smooth "if (mu < 0) += 0.5 rho h cs"
         constexpr real_t eps = 1e-12;
         T visc_coeff = 2.0 * rho * h * h * fabs(mu);
         visc_coeff += 0.5 * rho * h * cs *
                       (1.0 - smooth_step_01(mu - 2.0 * eps, eps));
         stress += visc_coeff * sgrad_v;
         dt_visc_coeff = visc_coeff;
      }
      else if (viscosity_type == 2)
      {
         // No artificial viscosity term
      }
      else if (viscosity_type == 21 || viscosity_type == 22 ||
               viscosity_type == 4)
      {
         MFEM_ABORT_KERNEL("works only in 2D!");
      }
      else if (viscosity_type == 7)
      {
         // Smoother viscosity for well-defined gradients.

         const auto dvdx = dvdxi * inv(J);
         const auto h = h0 * pow(det(J) / det(J0), 1.0 / static_cast<real_t>(DIM));
         const auto delta_v = h * tr(dvdx);

         // Coarses mesh h for the domain.
         const real_t h_coarse = 1.0;
         const T dv_scale = 2.0 * h / h_coarse * cs;

         // Smooth activation switch.
         // Motivation: in compression, delta_v < 0.
         //             in shocks, delta_v starts to be comparable to cs.
         // -> this viscosity becomes noticable when delta_v > dv_scale.
         const auto psi = softstep7(dv_scale, 0.5, -delta_v);

         // Smooths the kink at |delta_v| ~ 0 with correct units scaling.
         // The smoothing becomes active when |delta_v| approaches dv_scale.
         const auto abs_delta_v = softabs(dv_scale, delta_v);

         const auto mu = rho * h * (visc_q2 * abs_delta_v + psi * visc_q1 * cs);
         stress += mu * sym(dvdx);

         dt_visc_coeff = mu;
      }
      else { MFEM_ABORT_KERNEL("wrong viscosity option"); }
   }

   if (rho < 0.0)
   {
      dt_est = 0.0;
   }

   if (detJ < 0.0)
   {
      // This will force repetition of the step with smaller dt.
      dt_est = 0.0;
   }
   else
   {
      // The dt estimate is a plain value side output
      const real_t sv = future::get_value(calcsv(J, DIM-1));

      const real_t hmin = sv / static_cast<real_t>(order_v);
      const real_t ihmin = 1.0 / hmin;
      const real_t irhoihminsq = ihmin * ihmin / future::get_value(rho);
      const real_t idt = future::get_value(cs) / hmin +
                         2.5 * future::get_value(dt_visc_coeff) * irhoihminsq;

      if (idt > 0.0)
      {
         dt_est = fmin(dt_est, cfl / idt);
      }
   }

   matd stressJiT = stress * transpose(invJ) * detJ * w;
   return stressJiT;
}

// ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
// Derivative kernels of the Jacobian blocks: matrix-free actions and the
// assembled matrices of the AMG preconditioner
constexpr auto JACOBIAN_KERNELS = future::DerivativeKernels::Action |
                                  future::DerivativeKernels::AssembleMatrix;

} // namespace mfem::hydrodynamics
