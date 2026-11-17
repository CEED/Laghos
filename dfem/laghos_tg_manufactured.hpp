// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.

#pragma once

#include <mfem.hpp>

namespace mfem::hydrodynamics
{

class LagrangianHydroOperator;

/// Run one fixed-step Taylor--Green manufactured-adjoint convergence sample
/// with RK2Avg or classical RK4 and print the initial-adjoint errors.
bool VerifyTaylorGreenManufacturedAdjoint(
   ParFiniteElementSpace &H1,
   ParFiniteElementSpace &L2,
   const Array<int> &essential_velocity_tdofs,
   Coefficient &rho0_coefficient,
   const ParGridFunction &rho0,
   const ParGridFunction &material,
   const BlockVector &initial_state,
   const IntegrationRule &ir,
   int qfunction_backend,
   int ode_solver_type,
   LagrangianHydroOperator &hydro,
   real_t dt,
   int num_steps,
   int temporal_refinements = 0);

} // namespace mfem::hydrodynamics
