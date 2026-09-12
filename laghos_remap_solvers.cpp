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

#include "laghos_remap_solvers.hpp"
#include "laghos_remap.hpp"

namespace mfem
{
namespace ale
{
namespace geom_consistent_solvers
{
void GeomConsODESolver::Init(TimeDependentGeomConsOperator &f_)
{
    ODESolver::Init(f_);
    f = &f_;
}

void ForwardEulerSolver::Init(TimeDependentGeomConsOperator &f_)
{
    GeomConsODESolver::Init(f_);
    dU.SetSize(f->Width());
}

void ForwardEulerSolver::Step(Vector &U, real_t &t, real_t &dt)
{
    Vector K;

    // Solve for the flux
    f->SetTime(t);
    f->ImplicitSolveFlux(dt, flux);

    // Explicit step
    f->MultConserv(flux, U, K, dU);

    // Limit step
    f->LimitUpdate(dt, U, K, dU);

    // Update state
    U.Add(dt, dU);
    t += dt;
}

void RKSolver::ConstructD()
{
   // Convert high-order to Forward Euler factors
   d = new real_t[s*(s+1)/2];

   const real_t *a_n = a; // new coeff line
   const real_t *a_o = a; // old coeff line
   int i_o = -1; // old stage
   real_t c_o = 0.; // old time fraction

   for (int i = 0; i < s; i++)
   {
      const real_t c_n = (i<s-1)?(c[i]):(1.); // new time fraction
      const real_t dc = c_n - c_o; // time fraction diff
      real_t *di = d + i*(i+1)/2;

      for (int j = 0; j < i; j++)
      {
         const real_t a_oj = (j<=i_o)?(a_o[j]):(0.); // old coeff
         const real_t m = (a_n[j] - a_oj) / dc; // old HO update coeff
         if (m == 0.)
         {
            di[j] = 0.;
            continue;
         }
         // Express j-th HO update by Forward Euler updates
         const real_t *dj = d + j*(j+1)/2;
         const real_t dij = m / dj[j];
         for (int k = 0; k < j; k++)
         {
            di[k] -= dj[k] * dij;
         }
         di[j] = dij;
      }
      di[i] = a_n[i] / dc;

      // Update stage

      const double c_next = (i < s-2)?(c[i+1]):(1.);
      if (c_next > c_n)
      {
         i_o = i;
         c_o = c_n;
         a_o = a_n;
      }

      if (i < s-2)
      {
         a_n += i+1;
      }
      else
      {
         a_n = b;
      }
   }
}

RKSolver::RKSolver(int s_, const real_t a_[], const real_t b_[], const real_t c_[])
: s(s_), a(a_), b(b_), c(c_)
{
    dxs = new Vector[s];
    Ks = new Vector[s];
    fs = new ParGridFunction[s];
    ConstructD();
}

void RKSolver::Init(TimeDependentGeomConsOperator &f_)
{
    GeomConsODESolver::Init(f_);
    for (int i = 0; i < s; i++)
    {
       dxs[i].SetSize(f->Height());
    }
}

void RKSolver::Step(Vector &x, real_t &t, real_t &dt)
{
    // Update state
    f->SetTime(t);

    const real_t *a_n = a; // new coeff line
    const real_t *a_o = a; // old coeff line
    const real_t *di = d;
    int i_o = -1;
    real_t c_o = 0.;

    for(int i = 0; i < s; i++)
    {
        const real_t ci = (i<s-1)?(c[i]):(1.);
        const real_t dc = ci - c_o;
        const real_t dct = dc * dt;

        // Project fluxes

        f->ImplicitSolveFlux(dct, fs[i]);

        if(i > 0)
        {
            // Forward Euler -> high order
            for(int j = 0; j <= i_o; j++)
            {
                const real_t m = (a_o[j] - a_n[j]) / dc;
                fs[i].Add(m, fs[j]);
            }
            for(int j = i_o+1; j < i; j++)
            {
                const real_t m = - a_n[j] / dc;
                fs[i].Add(m, fs[j]);
            }
            fs[i] *= dc / a_n[i];
        }

        // Explicit step

        f->MultConserv(fs[i], x, Ks[i], dxs[i]);

        if(i > 0)
        {
            // High order -> Forward Euler
            dxs[i] *= di[i];
            Ks[i] *= di[i];
            for(int j = 0; j < i; j++)
            {
                dxs[i].Add(di[j], dxs[j]);
                Ks[i].Add(di[j], Ks[j]);
            }
        }

        // Limit step
    
        f->LimitUpdate(dct, x, Ks[i], dxs[i]);

        // Update state
        const real_t c_n = (i < s-2)?(c[i+1]):(1.);
        if(i >= s-1 || c_n > ci)
        {
            f->SetTime(t + ci*dt);
            x.Add(dct, dxs[i]);
            i_o = i;
            c_o = ci;
            a_o = a_n;
        }

        // Proceed to the next stage
        if(i < s-2)
            a_n += i+1;
        else
            a_n = b;
        di += i+1;
    }

    t += dt;
}

RKSolver::~RKSolver()
{
    delete[] fs;
    delete[] dxs;
    delete[] Ks;
    delete[] d;
}

//2-stage, 2nd order
const real_t RK2Solver::a[] = {.5};
const real_t RK2Solver::b[] = {0., 1.};
const real_t RK2Solver::c[] = {.5};

//3-stage, 3rd order
const real_t RK3Solver::a[] = {1./3., 0., 2./3.};
const real_t RK3Solver::b[] = {.25, 0., .75};
const real_t RK3Solver::c[] = {1./3., 2./3.};

//4-stage, 4th order for linear, 3rd for non-linear
//const real_t RK4Solver::a[] = {.25, 0., .5, 0., .25, .5};
//const real_t RK4Solver::b[] = {0., 2./3., -1./3., 2./3.};
//const real_t RK4Solver::c[] = {.25, .5, .75};
//4-stage, 4th order, non-equidistant
//const real_t RK4Solver::a[] = {.5, 0., .5, 0., 0., 1.};
//const real_t RK4Solver::b[] = {1./6., 2./6., 2./6., 1./6.};
//const real_t RK4Solver::c[] = {.5, .5, 1.};
//4-stage, 4th order, equidistant (except the last)
const real_t RK4Solver::a[] = {1./3., -1./3., 1., 1., -1., 1.};
const real_t RK4Solver::b[] = {1./8., 3./8., 3./8., 1./8.};
const real_t RK4Solver::c[] = {1./3., 2./3., 1.};

//6-stage, 5th order, equidistant (except the last)
const real_t RK6Solver::a[] = {.25, 1./8., 1./8., 0., -.5, 1., 3./16., 0., 0.,
                                  9./16., -3./7., 2./7., 12./7., -12./7., 8./7.
                                 };
const real_t RK6Solver::b[] = {7./90., 0., 32./90., 12./90., 32./90., 7./90.};
const real_t RK6Solver::c[] = {.25, .25, .5, .75, 1.};

} // geom_consistent_solvers
} // namespace ale
} // namespace mfem
