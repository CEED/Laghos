// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.

#pragma once

#include <mfem.hpp>

namespace mfem::hydrodynamics
{

class LagrangianHydroOperator;

/// Run one fixed-step RK2Avg Taylor--Green manufactured-adjoint convergence
/// sample and print the initial-adjoint errors.
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
   LagrangianHydroOperator &hydro,
   real_t dt,
   int num_steps);

} // namespace mfem::hydrodynamics
