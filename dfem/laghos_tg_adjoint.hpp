// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.

#pragma once

#include <mfem.hpp>

#include <functional>

namespace mfem::hydrodynamics
{

class LagrangianHydroOperator;

/// True-dof representation of the Lagrangian state or one of its covectors.
struct TaylorGreenAdjointState
{
   Vector x, v, e;

   TaylorGreenAdjointState() = default;
   TaylorGreenAdjointState(int h1_size, int l2_size) :
      x(h1_size), v(h1_size), e(l2_size) { }
};

using TaylorGreenObjectiveGradient =
   std::function<void(const TaylorGreenAdjointState &, real_t,
                      TaylorGreenAdjointState &)>;

/// Integrate Taylor--Green with fixed-step RK2Avg and backpropagate a
/// terminal objective plus a trapezoidal-in-time running objective.
void ComputeTaylorGreenRK2AvgAdjoint(
   ParFiniteElementSpace &H1,
   ParFiniteElementSpace &L2,
   const Array<int> &essential_velocity_tdofs,
   Coefficient &rho0_coefficient,
   const ParGridFunction &rho0,
   const ParGridFunction &material,
   const BlockVector &initial_state,
   LagrangianHydroOperator &hydro,
   real_t dt,
   int num_steps,
   const TaylorGreenObjectiveGradient &terminal_gradient,
   const TaylorGreenObjectiveGradient &running_gradient,
   TaylorGreenAdjointState &initial_gradient,
   TaylorGreenAdjointState *final_state = nullptr);

/// Verify the discrete adjoint of fixed-step RK2Avg Taylor--Green integration.
/// This check requires no viscosity.
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
