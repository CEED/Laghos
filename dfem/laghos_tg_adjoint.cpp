// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.

#include "laghos_tg_adjoint.hpp"

#include "laghos_qf.hpp"
#include "laghos_solver.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <vector>

namespace mfem::hydrodynamics
{

namespace
{

struct HydroState
{
   Vector x, v, e;

   HydroState() = default;
   HydroState(int h1_size, int l2_size) :
      x(h1_size), v(h1_size), e(l2_size) { }
};

real_t Dot(const HydroState &a, const HydroState &b)
{
   return a.x * b.x + a.v * b.v + a.e * b.e;
}

real_t Norml2(const HydroState &a)
{
   return std::sqrt(Dot(a, a));
}

void Copy(const HydroState &source, HydroState &destination)
{
   destination.x = source.x;
   destination.v = source.v;
   destination.e = source.e;
}

void Add(const HydroState &a, real_t alpha, const HydroState &b,
         HydroState &result)
{
   add(a.x, alpha, b.x, result.x);
   add(a.v, alpha, b.v, result.v);
   add(a.e, alpha, b.e, result.e);
}

class TaylorGreenModel
{
private:
   ParFiniteElementSpace &H1, &L2;
   const Array<int> &essential_velocity_tdofs;
   Vector rho0, initial_coordinates, material;

   DifferentiableOperator &momentum;
   DifferentiableOperator &energy;
   DifferentiableOperator &source;
   DifferentiableOperator &objective;
   mutable Vector objective_values;

   ParBilinearForm velocity_mass, energy_mass;
   HypreParMatrix velocity_mass_matrix, energy_mass_matrix;
   mutable CGSolver velocity_mass_solver, energy_mass_solver;
   mutable HypreSmoother velocity_mass_preconditioner,
           energy_mass_preconditioner;

   MultiVector StateInputFields(const HydroState &state) const
   {
      // QDataInputs order in laghos_qf.hpp.
      return MultiVector{state.v, rho0, initial_coordinates, state.x,
                         material, state.e};
   }

   MultiVector SourceInputFields(const HydroState &state) const
   {
      return MultiVector{state.x};
   }

   MultiVector ObjectiveInputFields(const HydroState &state) const
   {
      return MultiVector{state.v, rho0, initial_coordinates};
   }

   void SolveVelocityMass(const Vector &rhs, Vector &solution) const
   {
      Vector constrained_rhs(rhs);
      constrained_rhs.SetSubVector(essential_velocity_tdofs, 0.0);
      solution = 0.0;
      velocity_mass_solver.Mult(constrained_rhs, solution);
      solution.SetSubVector(essential_velocity_tdofs, 0.0);
   }

   void SolveEnergyMass(const Vector &rhs, Vector &solution) const
   {
      solution = 0.0;
      energy_mass_solver.Mult(rhs, solution);
   }

public:
   TaylorGreenModel(ParFiniteElementSpace &h1,
                    ParFiniteElementSpace &l2,
                    const Array<int> &essential_tdofs,
                    Coefficient &rho0_coefficient,
                    const ParGridFunction &rho0_gf,
                    const ParGridFunction &material_gf,
                    const Vector &initial_x,
                    LagrangianHydroOperator &hydro) :
      H1(h1), L2(l2), essential_velocity_tdofs(essential_tdofs),
      rho0(L2.GetTrueVSize()),
      initial_coordinates(H1.GetTrueVSize()),
      material(material_gf.ParFESpace()->GetTrueVSize()),
      momentum(*hydro.momentum_mf), energy(*hydro.energy_mf),
      source(*hydro.taylor_source_mf), objective(*hydro.kinetic_energy_mf),
      objective_values(L2.GetTrueVSize()),
      velocity_mass(&H1), energy_mass(&L2),
      velocity_mass_solver(H1.GetComm()), energy_mass_solver(L2.GetComm()),
      velocity_mass_preconditioner(), energy_mass_preconditioner()
   {
      rho0_gf.GetTrueDofs(rho0);
      material_gf.GetTrueDofs(material);
      initial_coordinates = initial_x;

      velocity_mass.SetAssemblyLevel(AssemblyLevel::LEGACY);
      velocity_mass.AddDomainIntegrator(
         new VectorMassIntegrator(rho0_coefficient, &hydro.ir));
      velocity_mass.Assemble();
      velocity_mass.Finalize();
      velocity_mass.FormSystemMatrix(essential_velocity_tdofs,
                                     velocity_mass_matrix);

      energy_mass.SetAssemblyLevel(AssemblyLevel::LEGACY);
      energy_mass.AddDomainIntegrator(
         new MassIntegrator(rho0_coefficient, &hydro.ir));
      energy_mass.Assemble();
      energy_mass.Finalize();
      Array<int> no_essential_dofs;
      energy_mass.FormSystemMatrix(no_essential_dofs, energy_mass_matrix);

      velocity_mass_preconditioner.SetType(HypreSmoother::Jacobi, 1);
      velocity_mass_solver.SetPreconditioner(velocity_mass_preconditioner);
      velocity_mass_solver.SetOperator(velocity_mass_matrix);
      velocity_mass_solver.SetRelTol(1e-13);
      velocity_mass_solver.SetAbsTol(0.0);
      velocity_mass_solver.SetMaxIter(1000);
      velocity_mass_solver.SetPrintLevel(-1);
      velocity_mass_solver.iterative_mode = false;

      energy_mass_preconditioner.SetType(HypreSmoother::Jacobi, 1);
      energy_mass_solver.SetPreconditioner(energy_mass_preconditioner);
      energy_mass_solver.SetOperator(energy_mass_matrix);
      energy_mass_solver.SetRelTol(1e-13);
      energy_mass_solver.SetAbsTol(0.0);
      energy_mass_solver.SetMaxIter(1000);
      energy_mass_solver.SetPrintLevel(-1);
      energy_mass_solver.iterative_mode = false;
   }

   void Residual(const HydroState &state, Vector &momentum_value,
                 Vector &energy_value) const
   {
      MultiVector inputs = StateInputFields(state);
      MultiVector momentum_output{momentum_value};
      momentum.Mult(inputs, momentum_output);

      MultiVector energy_output{energy_value};
      energy.Mult(inputs, energy_output);

      Vector source_value(L2.GetTrueVSize());
      MultiVector source_inputs = SourceInputFields(state);
      MultiVector source_output{source_value};
      source.Mult(source_inputs, source_output);
      energy_value += source_value;
   }

   void Rhs(const HydroState &state, HydroState &rhs) const
   {
      Vector momentum_value(H1.GetTrueVSize());
      Vector energy_value(L2.GetTrueVSize());
      Residual(state, momentum_value, energy_value);

      rhs.x = state.v;
      momentum_value.Neg();
      SolveVelocityMass(momentum_value, rhs.v);
      SolveEnergyMass(energy_value, rhs.e);
   }

   // RHS Jacobian-vector product used to propagate a direction forward.
   void RhsJvp(const HydroState &state, const HydroState &direction,
               HydroState &result) const
   {
      Vector dmomentum(H1.GetTrueVSize());
      Vector denergy(L2.GetTrueVSize());
      dmomentum = 0.0;
      denergy = 0.0;

      MultiVector inputs = StateInputFields(state);
      auto add_derivative = [&](DifferentiableOperator &op, int id,
                                const Vector &dfield, Vector &sum)
      {
         Vector part(sum.Size());
         MultiVector output{part};
         op.GetDerivative(id, inputs)->Mult(dfield, output);
         sum += part;
      };
      for (auto field :
           {
              std::pair{COORDINATES, &HydroState::x},
              std::pair{VELOCITY, &HydroState::v},
              std::pair{INTERNAL_ENERGY, &HydroState::e}
           })
      {
         add_derivative(momentum, field.first, direction.*field.second,
                        dmomentum);
         add_derivative(energy, field.first, direction.*field.second, denergy);
      }

      Vector dsource(L2.GetTrueVSize());
      MultiVector source_inputs = SourceInputFields(state);
      MultiVector source_output{dsource};
      source.GetDerivative(COORDINATES, source_inputs)->Mult(direction.x,
                                                             source_output);
      denergy += dsource;

      result.x = direction.v;
      dmomentum.Neg();
      SolveVelocityMass(dmomentum, result.v);
      SolveEnergyMass(denergy, result.e);
   }

   /// Backpropagate lambda_next through one Forward Euler step:
   /// lambda = lambda_next + dt * R'(state)^T lambda_next.
   void ComputeLambda(const HydroState &state,
                      const HydroState &lambda_next,
                      real_t dt,
                      HydroState &lambda) const
   {
      Vector momentum_lambda(H1.GetTrueVSize());
      Vector energy_lambda(L2.GetTrueVSize());
      SolveVelocityMass(lambda_next.v, momentum_lambda);
      momentum_lambda.Neg();
      SolveEnergyMass(lambda_next.e, energy_lambda);

      HydroState rhs_lambda(H1.GetTrueVSize(), L2.GetTrueVSize());
      rhs_lambda.x = 0.0;
      rhs_lambda.v = 0.0;
      rhs_lambda.e = 0.0;

      MultiVector inputs = StateInputFields(state);
      MultiVector momentum_output_lambda{momentum_lambda};
      MultiVector energy_output_lambda{energy_lambda};
      auto add_transpose = [&](DifferentiableOperator &op, int id,
                               MultiVector &output_lambda,
                               Vector &sum)
      {
         Vector part(sum.Size());
         MultiVector output{part};
         op.GetDerivative(id, inputs)->MultTranspose(output_lambda, output);
         sum += part;
      };
      for (auto field :
           {
              std::pair{COORDINATES, &HydroState::x},
              std::pair{VELOCITY, &HydroState::v},
              std::pair{INTERNAL_ENERGY, &HydroState::e}
           })
      {
         add_transpose(momentum, field.first, momentum_output_lambda,
                       rhs_lambda.*field.second);
         add_transpose(energy, field.first, energy_output_lambda,
                       rhs_lambda.*field.second);
      }

      Vector source_x(H1.GetTrueVSize());
      MultiVector source_inputs = SourceInputFields(state);
      MultiVector source_output_lambda{energy_lambda};
      MultiVector source_x_output{source_x};
      source.GetDerivative(COORDINATES, source_inputs)->MultTranspose(
         source_output_lambda, source_x_output);
      rhs_lambda.x += source_x;

      rhs_lambda.v += lambda_next.x;
      rhs_lambda.v.SetSubVector(essential_velocity_tdofs, 0.0);
      Add(lambda_next, dt, rhs_lambda, lambda);
   }

   real_t Objective(const HydroState &state) const
   {
      MultiVector inputs = ObjectiveInputFields(state);
      MultiVector output{objective_values};
      objective.Mult(inputs, output);
      const real_t local_value = objective_values.Sum();
      real_t value = 0.0;
      MPI_Allreduce(&local_value, &value, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, H1.GetComm());
      return value;
   }

   void ObjectiveGradient(const HydroState &state, HydroState &gradient) const
   {
      gradient.x = 0.0;
      gradient.v = 0.0;
      gradient.e = 0.0;

      MultiVector inputs = ObjectiveInputFields(state);
      Vector objective_lambda(L2.GetTrueVSize());
      objective_lambda = 1.0;
      MultiVector output_lambda{objective_lambda};
      MultiVector gradient_output{gradient.v};
      objective.GetDerivative(VELOCITY, inputs)->MultTranspose(
         output_lambda, gradient_output);
      gradient.v.SetSubVector(essential_velocity_tdofs, 0.0);
   }

   void ForwardTrajectory(const HydroState &initial, real_t dt, int num_steps,
                          std::vector<HydroState> &states) const
   {
      MFEM_VERIFY(num_steps > 0, "number of time steps must be positive");
      states.clear();
      states.reserve(num_steps + 1);
      for (int n = 0; n <= num_steps; n++)
      {
         states.emplace_back(H1.GetTrueVSize(), L2.GetTrueVSize());
      }
      Copy(initial, states[0]);

      HydroState rhs(H1.GetTrueVSize(), L2.GetTrueVSize());
      for (int n = 0; n < num_steps; n++)
      {
         Rhs(states[n], rhs);
         Add(states[n], dt, rhs, states[n + 1]);
      }
   }

   real_t TrajectoryObjective(const HydroState &initial, real_t dt,
                              int num_steps) const
   {
      std::vector<HydroState> states;
      ForwardTrajectory(initial, dt, num_steps, states);
      return Objective(states.back());
   }
};

void ExtractTrueState(ParFiniteElementSpace &H1, ParFiniteElementSpace &L2,
                      const BlockVector &state, HydroState &true_state)
{
   ParGridFunction x(&H1), v(&H1), e(&L2);
   Vector &monolithic_state = const_cast<BlockVector &>(state);
   const int h1_size = H1.GetVSize();
   x.MakeRef(&H1, monolithic_state, 0);
   v.MakeRef(&H1, monolithic_state, h1_size);
   e.MakeRef(&L2, monolithic_state, 2 * h1_size);
   x.SyncAliasMemory(monolithic_state);
   v.SyncAliasMemory(monolithic_state);
   e.SyncAliasMemory(monolithic_state);
   x.GetTrueDofs(true_state.x);
   v.GetTrueDofs(true_state.v);
   e.GetTrueDofs(true_state.e);
}

} // namespace

bool VerifyTaylorGreenForwardEulerAdjoint(
   ParFiniteElementSpace &H1,
   ParFiniteElementSpace &L2,
   const Array<int> &ess_tdofs_v,
   Coefficient &rho0_coefficient,
   const ParGridFunction &rho0,
   const ParGridFunction &material,
   const BlockVector &initial_state,
   LagrangianHydroOperator &hydro,
   real_t dt,
   real_t fd_epsilon)
{
   MFEM_VERIFY(H1.GetComm() == MPI_COMM_WORLD,
               "Taylor--Green adjoint check expects MPI_COMM_WORLD");
   MFEM_VERIFY(Mpi::WorldSize() == 1,
               "Taylor--Green adjoint check currently requires one MPI task");
   MFEM_VERIFY(H1.GetMesh()->Dimension() == 2,
               "Taylor--Green adjoint check currently requires a 2D mesh");
   MFEM_VERIFY(H1.GetVSize() == H1.GetTrueVSize() &&
               L2.GetVSize() == L2.GetTrueVSize(),
               "Taylor--Green adjoint check requires conforming one-rank spaces");
   MFEM_VERIFY(dt > 0.0 && fd_epsilon > 0.0,
               "dt and finite-difference epsilon must be positive");

   constexpr int num_steps = 5;

   HydroState state(H1.GetTrueVSize(), L2.GetTrueVSize());
   ExtractTrueState(H1, L2, initial_state, state);
   TaylorGreenModel model(H1, L2, ess_tdofs_v, rho0_coefficient, rho0,
                          material, state.x, hydro);

   HydroState rhs(H1.GetTrueVSize(), L2.GetTrueVSize());
   model.Rhs(state, rhs);

   BlockVector laghos_rhs(initial_state);
   hydro.ResetQuadratureData();
   hydro.Mult(initial_state, laghos_rhs);
   HydroState laghos_true_rhs(H1.GetTrueVSize(), L2.GetTrueVSize());
   ExtractTrueState(H1, L2, laghos_rhs, laghos_true_rhs);
   HydroState rhs_difference(H1.GetTrueVSize(), L2.GetTrueVSize());
   Add(rhs, -1.0, laghos_true_rhs, rhs_difference);
   const real_t rhs_reference_norm = std::max(Norml2(laghos_true_rhs), 1.0);
   const real_t rhs_relative_error = Norml2(rhs_difference) / rhs_reference_norm;
   const real_t rhs_x_error = rhs_difference.x.Norml2();
   const real_t rhs_v_error = rhs_difference.v.Norml2();
   const real_t rhs_e_error = rhs_difference.e.Norml2();

   std::vector<HydroState> states;
   model.ForwardTrajectory(state, dt, num_steps, states);

   HydroState initial_direction(H1.GetTrueVSize(), L2.GetTrueVSize());
   HydroState lambda_next(H1.GetTrueVSize(), L2.GetTrueVSize());
   initial_direction.x.Randomize(1001);
   initial_direction.v.Randomize(1002);
   initial_direction.e.Randomize(1003);
   initial_direction.v.SetSubVector(ess_tdofs_v, 0.0);
   lambda_next.x.Randomize(2001);
   lambda_next.v.Randomize(2002);
   lambda_next.e.Randomize(2003);
   const real_t direction_norm = Norml2(initial_direction);
   initial_direction.x /= direction_norm;
   initial_direction.v /= direction_norm;
   initial_direction.e /= direction_norm;

   HydroState direction(H1.GetTrueVSize(), L2.GetTrueVSize());
   Copy(initial_direction, direction);
   HydroState rhs_jvp(H1.GetTrueVSize(), L2.GetTrueVSize());
   HydroState direction_next(H1.GetTrueVSize(), L2.GetTrueVSize());
   HydroState lambda(H1.GetTrueVSize(), L2.GetTrueVSize());
   for (int n = 0; n < num_steps; n++)
   {
      model.RhsJvp(states[n], direction, rhs_jvp);
      Add(direction, dt, rhs_jvp, direction_next);
      Copy(direction_next, direction);
   }
   const real_t jvp_dot = Dot(lambda_next, direction);
   for (int n = num_steps - 1; n >= 0; n--)
   {
      model.ComputeLambda(states[n], lambda_next, dt, lambda);
      Copy(lambda, lambda_next);
   }
   const real_t vjp_dot = Dot(lambda_next, initial_direction);
   const real_t dot_scale = std::max({std::abs(jvp_dot), std::abs(vjp_dot),
                                      real_t(1.0)});
   const real_t transpose_relative_error =
      std::abs(jvp_dot - vjp_dot) / dot_scale;

   const real_t objective_value = model.Objective(states.back());
   HydroState lambda_final(H1.GetTrueVSize(), L2.GetTrueVSize());
   model.ObjectiveGradient(states.back(), lambda_final);

   HydroState lambda_0(H1.GetTrueVSize(), L2.GetTrueVSize());
   Copy(lambda_final, lambda_next);
   for (int n = num_steps - 1; n >= 0; n--)
   {
      model.ComputeLambda(states[n], lambda_next, dt, lambda_0);
      Copy(lambda_0, lambda_next);
   }

   HydroState plus(H1.GetTrueVSize(), L2.GetTrueVSize());
   HydroState minus(H1.GetTrueVSize(), L2.GetTrueVSize());
   Add(state, fd_epsilon, initial_direction, plus);
   Add(state, -fd_epsilon, initial_direction, minus);
   const real_t directional_fd =
      (model.TrajectoryObjective(plus, dt, num_steps) -
       model.TrajectoryObjective(minus, dt, num_steps)) /
      (2.0 * fd_epsilon);
   const real_t directional_adjoint = Dot(lambda_0, initial_direction);
   const real_t directional_scale =
      std::max({std::abs(directional_fd), std::abs(directional_adjoint),
                real_t(1.0)});
   const real_t directional_relative_error =
      std::abs(directional_fd - directional_adjoint) / directional_scale;

   HydroState finite_difference_gradient(H1.GetTrueVSize(), L2.GetTrueVSize());
   finite_difference_gradient.x = 0.0;
   finite_difference_gradient.v = 0.0;
   finite_difference_gradient.e = 0.0;

   auto finite_difference_block = [&](Vector HydroState::*member,
                                      const Array<int> *excluded)
   {
      Vector &gradient_block = finite_difference_gradient.*member;
      const int size = gradient_block.Size();
      Array<char> skip(size);
      skip = 0;
      if (excluded)
      {
         for (int i = 0; i < excluded->Size(); i++)
         {
            skip[(*excluded)[i]] = 1;
         }
      }
      for (int i = 0; i < size; i++)
      {
         if (skip[i]) { continue; }
         HydroState state_plus(H1.GetTrueVSize(), L2.GetTrueVSize());
         HydroState state_minus(H1.GetTrueVSize(), L2.GetTrueVSize());
         Copy(state, state_plus);
         Copy(state, state_minus);
         (state_plus.*member)[i] += fd_epsilon;
         (state_minus.*member)[i] -= fd_epsilon;
         gradient_block[i] =
            (model.TrajectoryObjective(state_plus, dt, num_steps) -
             model.TrajectoryObjective(state_minus, dt, num_steps)) /
            (2.0 * fd_epsilon);
      }
   };
   finite_difference_block(&HydroState::x, nullptr);
   finite_difference_block(&HydroState::v, &ess_tdofs_v);
   finite_difference_block(&HydroState::e, nullptr);

   HydroState gradient_difference(H1.GetTrueVSize(), L2.GetTrueVSize());
   Add(lambda_0, -1.0, finite_difference_gradient, gradient_difference);
   const real_t gradient_relative_error =
      Norml2(gradient_difference) /
      std::max(Norml2(finite_difference_gradient), real_t(1.0));
   const real_t gradient_max_error = std::max(
   {
      gradient_difference.x.Normlinf(), gradient_difference.v.Normlinf(),
      gradient_difference.e.Normlinf()
   });

   out << std::setprecision(12)
       << "\nTaylor--Green five-step Forward Euler adjoint verification\n"
       << "  dt                              = " << dt << '\n'
       << "  objective J(S5)                 = " << objective_value << '\n'
       << "  dFEM/Laghos RHS relative error  = " << rhs_relative_error << '\n'
       << "    position/velocity/energy      = " << rhs_x_error << " / "
       << rhs_v_error << " / " << rhs_e_error << '\n'
       << "    dFEM RHS block norms          = " << rhs.x.Norml2() << " / "
       << rhs.v.Norml2() << " / " << rhs.e.Norml2() << '\n'
       << "    Laghos RHS block norms        = " << laghos_true_rhs.x.Norml2()
       << " / " << laghos_true_rhs.v.Norml2() << " / "
       << laghos_true_rhs.e.Norml2() << '\n'
       << "  trajectory JVP/lambda error     = " << transpose_relative_error << '\n'
       << "  directional adjoint derivative  = " << directional_adjoint << '\n'
       << "  directional finite difference   = " << directional_fd << '\n'
       << "  directional relative error      = " << directional_relative_error << '\n'
       << "  full-gradient relative error    = " << gradient_relative_error << '\n'
       << "  full-gradient max abs error     = " << gradient_max_error << '\n';

   const bool passed = rhs_relative_error < 1e-8 &&
                       transpose_relative_error < 1e-10 &&
                       directional_relative_error < 5e-7 &&
                       gradient_relative_error < 5e-6;
   out << "  result                          = "
       << (passed ? "PASS" : "FAIL") << "\n\n";
   return passed;
}

} // namespace mfem::hydrodynamics
