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
void ForwardEulerSolver::Init(TimeDependentOperator &f_)
{
    MFEM_VERIFY(dynamic_cast<TimeDependentGeomConsOperator*>(&f_),
        "Not a geometrically consistent operator");
    ODESolver::Init(f_);
    dU.SetSize(f->Width());
}

void ForwardEulerSolver::Step(Vector &U, real_t &t, real_t &dt)
{
    auto *gcf = static_cast<TimeDependentGeomConsOperator*>(f);

    // Solve for the flux
    gcf->SetTime(t);
    gcf->ImplicitSolveFlux(dt, flux);

    // Explicit step
    gcf->MultConserv(flux, U, dU);

    // Limit step
    gcf->LimitUpdate(dt, U, dU);

    // Update state
    U.Add(dt, dU);
    t += dt;
}
} // geom_consistent_solvers
} // namespace ale
} // namespace mfem
