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
    dU_LO.SetSize(f->Width());
}

void ForwardEulerSolver::Step(Vector &U, real_t &t, real_t &dt)
{
    // Solve for the flux
    f->SetTime(t);
    f->ImplicitSolveFlux(dt, flux);

    // Explicit step
    f->MultConserv(flux, U, dU);
    f->MultConservLowOrder(flux, U, dU_LO);

    // Limit step
    f->LimitUpdate(dt, U, dU_LO, dU);

    // Update state
    U.Add(dt, dU);
    t += dt;
}
} // geom_consistent_solvers
} // namespace ale
} // namespace mfem
