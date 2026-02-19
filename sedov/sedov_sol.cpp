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

#include "sedov_sol.hpp"

#include "adaptive_quad.hpp"
#include "bisect.hpp"

#include <algorithm>
#include <cmath>

#include <iostream>

SedovSol::SedovSol(int dim_, double gamma_, double rho_0_, double blast_energy_,
                   double omega_)
   : dim(dim_), gamma(gamma_), rho_0(rho_0_), omega(omega_),
     blast_energy(blast_energy_)
{
   a = (dim + 2 - omega) * (gamma + 1) * 0.25;
   b = (gamma + 1) / (gamma - 1);
   c = (dim + 2 - omega) * gamma * 0.5;
   d = ((dim + 2 - omega) * (gamma + 1) /
        ((dim + 2 - omega) * (gamma + 1) - 2 * (2 + dim * (gamma - 1))));
   e = (2 + dim * (gamma - 1)) * 0.5;

   alpha0 = 2. / (dim + 2 - omega);
   alpha2 = -(gamma - 1) / (2 * (gamma - 1) + dim - gamma * omega);
   alpha1 =
      ((dim + 2 - omega) * gamma / (2 + dim * (gamma - 1)) *
       (2 * (dim * (2 - gamma) - omega) / (gamma * pow((dim + 2 - omega), 2)) -
        alpha2));
   alpha3 = (dim - omega) / (2 * (gamma - 1) + dim - dim * omega);
   alpha4 =
      (dim + 2 - omega) * (dim - omega) * alpha1 / (dim * (2 - gamma) - omega);
   alpha5 = (omega * (1 + gamma) - 2 * dim) / (dim * (2 - gamma) - omega);

   V0 = 2. / ((dim + 2 - omega) * gamma);
   Vv = 2. / (dim + 2 - omega);
   V2 = 4. / ((dim + 2 - omega) * (gamma + 1));
   Vs = 2. / ((gamma - 1) * dim + 2);

   if (V2 == Vs)
   {
      // singular
      alpha = (gamma + 1) / (gamma - 1) * pow(2, dim) /
              pow(dim * ((gamma - 1) * dim + 2), 2);
      if (dim > 1)
      {
         alpha *= M_PI;
      }
   }
   else
   {
      // standard or vacuum
      auto Vmin = std::min(V0, Vv);
      auto J1_integrand = [dim = dim, gamma = gamma, alpha0 = alpha0,
                               alpha1 = alpha1, alpha2 = alpha2, alpha3 = alpha3,
                               alpha4 = alpha4, alpha5 = alpha5, a = a, b = b, c = c,
                               d = d, e = e, omega = omega](double V)
      {
         return -(gamma + 1) / (gamma - 1) * pow(V, 2) *
                (alpha0 / V + alpha2 * c / (c * V - 1) -
                 alpha1 * e / (1 - e * V)) *
                pow((pow((a * V), alpha0) * pow((b * (c * V - 1)), alpha2) *
                     pow((d * (1 - e * V)), alpha1)),
                    (-(dim + 2 - omega))) *
                pow((b * (c * V - 1)), alpha3) * pow((d * (1 - e * V)), alpha4) *
                pow((b * (1 - c * V / gamma)), alpha5);
      };
      scalar_error_functor err_fun;
      err_fun.eps_abs = 1.49e-15;
      err_fun.eps_rel = 1.49e-15;
      auto J1 = gk21_integrate(J1_integrand, err_fun, Vmin, V2, 20, 64);

      auto J2_integrand = [dim = dim, gamma = gamma, alpha0 = alpha0,
                               alpha1 = alpha1, alpha2 = alpha2, alpha3 = alpha3,
                               alpha4 = alpha4, alpha5 = alpha5, a = a, b = b, c = c,
                               d = d, e = e, omega = omega](double V)
      {
         double denom = 1 - c * V;
         if (fabs(denom) <= 1e-15)
         {
            denom = std::copysign(1e-15, denom);
         }
         return -(gamma + 1) / (2 * gamma) * pow(V, 2) * (c * V - gamma) / denom *
                (alpha0 / V + alpha2 * c / -denom - alpha1 * e / (1 - e * V)) *
                pow(pow(a * V, alpha0) * pow(b * (c * V - 1), alpha2) *
                    pow(d * (1 - e * V), alpha1),
                    -(dim + 2 - omega)) *
                pow((b * (c * V - 1)), alpha3) * pow((d * (1 - e * V)), alpha4) *
                pow((b * (1 - c * V / gamma)), alpha5);

      };
      auto J2 = gk21_integrate(J2_integrand, err_fun, Vmin, V2, 20, 64);
      double I1 = pow(2, dim - 2) * J1;
      double I2 = pow(2, (dim - 1)) / (gamma - 1) * J2;
      if (dim > 1)
      {
         I1 *= M_PI;
         I2 *= M_PI;
      }
      alpha = I1 + I2;
   }
}

void SedovSol::SetTime(double t_)
{
   t = t_;

   r2 = pow((blast_energy / (alpha * rho_0)), (1. / (dim + 2 - omega))) *
        pow(t, (2. / (dim + 2 - omega)));
   U = (2 / (dim + 2 - omega)) * (r2 / t);
   rho1 = rho_0 * pow(r2, -omega);
   rho2 = ((gamma + 1) / (gamma - 1)) * rho1;
   v2 = (2 / (gamma + 1)) * U;
   p2 = (2 / (gamma + 1)) * rho1 * U * U;
}

void SedovSol::EvalSol(double r, double &rho, double &v, double &P) const
{
   if (r >= r2)
   {
      // pre-shock state
      rho = rho_0 * pow(r, -omega);
      v = 0;
      P = 0;
      return;
   }
   // post-shock state
   if (V2 == Vs)
   {
      // singular
      rho = rho2 * pow((r / r2), (dim - 2));
      v = v2 * r / r2;
      P = p2 * pow((r / r2), dim);
   }
   else
   {
      // find V(r)
      auto x1 = [&](double V) { return a * V; };
      auto x2 = [&](double V) { return b * (c * V - 1); };
      auto x3 = [&](double V) { return d * (1 - e * V); };
      auto x4 = [&](double V) { return b * (1 - c * V / gamma); };
      auto lmbda = [&](double V)
      {
         return pow(x1(V), -alpha0) * pow(x2(V), -alpha2) * pow(x3(V), -alpha1);
      };
      auto f = [&](double V) { return x1(V) * lmbda(V); };
      auto g = [&](double V)
      {
         return pow(x1(V), alpha0 * omega) *
                pow(x2(V), (alpha3 + alpha2 * omega)) *
                pow(x3(V), (alpha4 + alpha1 * omega)) * pow(x4(V), alpha5);
      };
      auto h = [&](double V)
      {
         return pow(x1(V), (alpha0 * dim)) *
                pow(x3(V), (alpha4 + alpha1 * (omega - 2))) *
                pow(x4(V), (1 + alpha5));
      };
      double V;
      if (V2 < Vs)
      {
         // standard
         V = bisection([&](double V_) { return r2 * lmbda(V_) - r; }, V0, V2);
      }
      else
      {
         // vacuum
         V = bisection([&](double V_) { return r2 * lmbda(V_) - r; }, Vv, V2);
         double r_vacuum = r2 * lmbda(Vv);
         if (r <= r_vacuum)
         {
            // vacuum part
            rho = 0;
            v = 0;
            P = 0;
            return;
         }
      }
      rho = rho2 * g(V);
      v = v2 * f(V);
      P = p2 * h(V);
   }
}
