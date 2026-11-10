// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.

#pragma once

#include <mfem.hpp>

namespace mfem::hydrodynamics
{

class LagrangianHydroOperator;

/// Verify the discrete adjoint of fixed-step RK2Avg Taylor--Green integration.
/// This check currently requires one MPI task and no viscosity.
bool VerifyTaylorGreenRK2AvgAdjoint(
   ParFiniteElementSpace &H1,
   ParFiniteElementSpace &L2,
   const Array<int> &essential_velocity_tdofs,
   Coefficient &rho0_coefficient,
   const ParGridFunction &rho0,
   const ParGridFunction &material,
   const BlockVector &initial_state,
   LagrangianHydroOperator &hydro,
   real_t dt,
   real_t fd_epsilon,
   int num_steps,
   int checkpoint_interval,
   bool finite_difference_check,
   bool visualization);

} // namespace mfem::hydrodynamics
