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

using HydroState = TaylorGreenAdjointState;

struct RK2AvgStepData
{
   HydroState state, midpoint;
   Vector velocity_stage1, velocity_stage2;

   RK2AvgStepData(int h1_size, int l2_size) :
      state(h1_size, l2_size), midpoint(h1_size, l2_size),
      velocity_stage1(h1_size), velocity_stage2(h1_size) { }
};

real_t GlobalSum(real_t local_value, MPI_Comm comm)
{
   real_t global_value = 0.0;
   MPI_Allreduce(&local_value, &global_value, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
   return global_value;
}

real_t GlobalMax(real_t local_value, MPI_Comm comm)
{
   real_t global_value = 0.0;
   MPI_Allreduce(&local_value, &global_value, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_MAX, comm);
   return global_value;
}

real_t Dot(const HydroState &a, const HydroState &b, MPI_Comm comm)
{
   return GlobalSum(a.x * b.x + a.v * b.v + a.e * b.e, comm);
}

real_t Norml2(const HydroState &a, MPI_Comm comm)
{
   return std::sqrt(Dot(a, a, comm));
}

real_t Norml2(const Vector &a, MPI_Comm comm)
{
   return std::sqrt(GlobalSum(a * a, comm));
}

real_t Normlinf(const Vector &a, MPI_Comm comm)
{
   return GlobalMax(a.Normlinf(), comm);
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

void AddInPlace(HydroState &a, real_t alpha, const HydroState &b)
{
   a.x.Add(alpha, b.x);
   a.v.Add(alpha, b.v);
   a.e.Add(alpha, b.e);
}

class TaylorGreenModel
{
private:
   ParFiniteElementSpace &H1, &L2;
   const Array<int> &essential_velocity_tdofs;
   Vector rho0, initial_coordinates, material;

   DifferentiableOperator &momentum;
   DifferentiableOperator &energy;
   DifferentiableOperator &rk2avg_energy;
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

   MultiVector RK2AvgEnergyInputFields(const HydroState &state,
                                       const Vector &work_velocity) const
   {
      return MultiVector{state.v, rho0, initial_coordinates, state.x,
                         material, state.e, work_velocity};
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
      rk2avg_energy(*hydro.rk2avg_energy_mf),
      source(*hydro.taylor_source_mf), objective(*hydro.kinetic_energy_mf),
      objective_values(L2.GetTrueVSize()),
      velocity_mass(&H1), energy_mass(&L2),
      velocity_mass_solver(H1.GetComm()), energy_mass_solver(L2.GetComm()),
      velocity_mass_preconditioner(), energy_mass_preconditioner()
   {
      rho0_gf.GetTrueDofs(rho0);
      material_gf.GetTrueDofs(material);
      initial_coordinates = initial_x;
      objective.SetMultLevel(DifferentiableOperator::TVECTOR);

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

   void VelocityRate(const HydroState &state, Vector &rate) const
   {
      Vector momentum_value(H1.GetTrueVSize());
      MultiVector inputs = StateInputFields(state);
      MultiVector output{momentum_value};
      momentum.Mult(inputs, output);
      momentum_value.Neg();
      SolveVelocityMass(momentum_value, rate);
   }

   void VelocityJvp(const HydroState &state, const HydroState &direction,
                    Vector &result) const
   {
      Vector dmomentum(H1.GetTrueVSize());
      dmomentum = 0.0;
      MultiVector inputs = StateInputFields(state);
      for (auto field :
           {
              std::pair{COORDINATES, &HydroState::x},
              std::pair{VELOCITY, &HydroState::v},
              std::pair{INTERNAL_ENERGY, &HydroState::e}
           })
      {
         Vector part(dmomentum.Size());
         MultiVector output{part};
         momentum.GetDerivative(field.first, inputs)->Mult(
            direction.*field.second, output);
         dmomentum += part;
      }
      dmomentum.Neg();
      SolveVelocityMass(dmomentum, result);
   }

   void VelocityVjp(const HydroState &state, const Vector &rate_lambda,
                    HydroState &state_lambda) const
   {
      Vector momentum_lambda(H1.GetTrueVSize());
      SolveVelocityMass(rate_lambda, momentum_lambda);
      momentum_lambda.Neg();
      MultiVector inputs = StateInputFields(state);
      MultiVector output_lambda{momentum_lambda};
      for (auto field :
           {
              std::pair{COORDINATES, &HydroState::x},
              std::pair{VELOCITY, &HydroState::v},
              std::pair{INTERNAL_ENERGY, &HydroState::e}
           })
      {
         Vector part((state_lambda.*field.second).Size());
         MultiVector output{part};
         momentum.GetDerivative(field.first, inputs)->MultTranspose(
            output_lambda, output);
         state_lambda.*field.second += part;
      }
   }

   void EnergyRate(const HydroState &state, const Vector &work_velocity,
                   Vector &rate) const
   {
      Vector energy_value(L2.GetTrueVSize());
      MultiVector energy_inputs =
         RK2AvgEnergyInputFields(state, work_velocity);
      MultiVector energy_output{energy_value};
      rk2avg_energy.Mult(energy_inputs, energy_output);

      Vector source_value(L2.GetTrueVSize());
      MultiVector source_inputs = SourceInputFields(state);
      MultiVector source_output{source_value};
      source.Mult(source_inputs, source_output);
      energy_value += source_value;
      SolveEnergyMass(energy_value, rate);
   }

   void EnergyJvp(const HydroState &state, const Vector &work_velocity,
                  const HydroState &state_direction,
                  const Vector &work_velocity_direction,
                  Vector &result) const
   {
      Vector denergy(L2.GetTrueVSize());
      denergy = 0.0;
      MultiVector inputs = RK2AvgEnergyInputFields(state, work_velocity);
      for (auto field :
           {
              std::pair{COORDINATES, &HydroState::x},
              std::pair{VELOCITY, &HydroState::v},
              std::pair{INTERNAL_ENERGY, &HydroState::e}
           })
      {
         Vector part(denergy.Size());
         MultiVector output{part};
         rk2avg_energy.GetDerivative(field.first, inputs)->Mult(
            state_direction.*field.second, output);
         denergy += part;
      }
      {
         Vector part(denergy.Size());
         MultiVector output{part};
         rk2avg_energy.GetDerivative(RK2AVG_VELOCITY, inputs)->Mult(
            work_velocity_direction, output);
         denergy += part;
      }

      Vector dsource(L2.GetTrueVSize());
      MultiVector source_inputs = SourceInputFields(state);
      MultiVector source_output{dsource};
      source.GetDerivative(COORDINATES, source_inputs)->Mult(
         state_direction.x, source_output);
      denergy += dsource;
      SolveEnergyMass(denergy, result);
   }

   void EnergyVjp(const HydroState &state, const Vector &work_velocity,
                  const Vector &rate_lambda, HydroState &state_lambda,
                  Vector &work_velocity_lambda) const
   {
      Vector energy_lambda(L2.GetTrueVSize());
      SolveEnergyMass(rate_lambda, energy_lambda);
      MultiVector energy_output_lambda{energy_lambda};
      MultiVector inputs = RK2AvgEnergyInputFields(state, work_velocity);

      Vector work_part(work_velocity.Size());
      MultiVector work_output{work_part};
      rk2avg_energy.GetDerivative(RK2AVG_VELOCITY, inputs)->MultTranspose(
         energy_output_lambda, work_output);
      work_velocity_lambda += work_part;
      for (auto field :
           {
              std::pair{COORDINATES, &HydroState::x},
              std::pair{VELOCITY, &HydroState::v},
              std::pair{INTERNAL_ENERGY, &HydroState::e}
           })
      {
         Vector part((state_lambda.*field.second).Size());
         MultiVector output{part};
         rk2avg_energy.GetDerivative(field.first, inputs)->MultTranspose(
            energy_output_lambda, output);
         state_lambda.*field.second += part;
      }

      Vector source_x(H1.GetTrueVSize());
      MultiVector source_inputs = SourceInputFields(state);
      MultiVector source_x_output{source_x};
      source.GetDerivative(COORDINATES, source_inputs)->MultTranspose(
         energy_output_lambda, source_x_output);
      state_lambda.x += source_x;
   }

   void RK2AvgStep(const HydroState &state, real_t dt,
                   RK2AvgStepData &step, HydroState &next) const
   {
      const real_t half_dt = 0.5 * dt;
      Copy(state, step.state);

      Vector acceleration1(H1.GetTrueVSize());
      Vector energy_rate1(L2.GetTrueVSize());
      VelocityRate(state, acceleration1);
      add(state.v, half_dt, acceleration1, step.velocity_stage1);
      EnergyRate(state, step.velocity_stage1, energy_rate1);
      add(state.x, half_dt, step.velocity_stage1, step.midpoint.x);
      add(state.v, half_dt, acceleration1, step.midpoint.v);
      add(state.e, half_dt, energy_rate1, step.midpoint.e);

      Vector acceleration2(H1.GetTrueVSize());
      Vector energy_rate2(L2.GetTrueVSize());
      VelocityRate(step.midpoint, acceleration2);
      add(state.v, half_dt, acceleration2, step.velocity_stage2);
      EnergyRate(step.midpoint, step.velocity_stage2, energy_rate2);
      add(state.x, dt, step.velocity_stage2, next.x);
      add(state.v, dt, acceleration2, next.v);
      add(state.e, dt, energy_rate2, next.e);
   }

   void RK2AvgJvp(const RK2AvgStepData &step,
                  const HydroState &direction, real_t dt,
                  HydroState &direction_next) const
   {
      const real_t half_dt = 0.5 * dt;
      Vector acceleration1_direction(H1.GetTrueVSize());
      Vector velocity1_direction(H1.GetTrueVSize());
      Vector energy1_direction(L2.GetTrueVSize());
      VelocityJvp(step.state, direction, acceleration1_direction);
      add(direction.v, half_dt, acceleration1_direction,
          velocity1_direction);
      EnergyJvp(step.state, step.velocity_stage1, direction,
                velocity1_direction, energy1_direction);

      HydroState midpoint_direction(H1.GetTrueVSize(), L2.GetTrueVSize());
      add(direction.x, half_dt, velocity1_direction, midpoint_direction.x);
      add(direction.v, half_dt, acceleration1_direction,
          midpoint_direction.v);
      add(direction.e, half_dt, energy1_direction, midpoint_direction.e);

      Vector acceleration2_direction(H1.GetTrueVSize());
      Vector velocity2_direction(H1.GetTrueVSize());
      Vector energy2_direction(L2.GetTrueVSize());
      VelocityJvp(step.midpoint, midpoint_direction,
                  acceleration2_direction);
      add(direction.v, half_dt, acceleration2_direction,
          velocity2_direction);
      EnergyJvp(step.midpoint, step.velocity_stage2, midpoint_direction,
                velocity2_direction, energy2_direction);

      add(direction.x, dt, velocity2_direction, direction_next.x);
      add(direction.v, dt, acceleration2_direction, direction_next.v);
      add(direction.e, dt, energy2_direction, direction_next.e);
   }

   /// Backpropagate lambda_next through one complete RK2Avg step.
   void ComputeLambda(const RK2AvgStepData &step,
                      const HydroState &lambda_next, real_t dt,
                      HydroState &lambda) const
   {
      const real_t half_dt = 0.5 * dt;
      Copy(lambda_next, lambda);

      Vector velocity2_lambda(lambda_next.x);
      velocity2_lambda *= dt;
      Vector acceleration2_lambda(lambda_next.v);
      acceleration2_lambda *= dt;
      Vector energy2_lambda(lambda_next.e);
      energy2_lambda *= dt;

      HydroState midpoint_lambda(H1.GetTrueVSize(), L2.GetTrueVSize());
      midpoint_lambda.x = 0.0;
      midpoint_lambda.v = 0.0;
      midpoint_lambda.e = 0.0;
      EnergyVjp(step.midpoint, step.velocity_stage2, energy2_lambda,
                midpoint_lambda, velocity2_lambda);
      lambda.v += velocity2_lambda;
      acceleration2_lambda.Add(half_dt, velocity2_lambda);
      VelocityVjp(step.midpoint, acceleration2_lambda, midpoint_lambda);
      midpoint_lambda.v.SetSubVector(essential_velocity_tdofs, 0.0);

      AddInPlace(lambda, 1.0, midpoint_lambda);
      Vector velocity1_lambda(midpoint_lambda.x);
      velocity1_lambda *= half_dt;
      Vector acceleration1_lambda(midpoint_lambda.v);
      acceleration1_lambda *= half_dt;
      Vector energy1_lambda(midpoint_lambda.e);
      energy1_lambda *= half_dt;

      EnergyVjp(step.state, step.velocity_stage1, energy1_lambda,
                lambda, velocity1_lambda);
      lambda.v += velocity1_lambda;
      acceleration1_lambda.Add(half_dt, velocity1_lambda);
      VelocityVjp(step.state, acceleration1_lambda, lambda);
      lambda.v.SetSubVector(essential_velocity_tdofs, 0.0);
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
                          std::vector<HydroState> &states,
                          std::vector<RK2AvgStepData> &steps) const
   {
      MFEM_VERIFY(num_steps > 0, "number of time steps must be positive");
      states.clear();
      states.reserve(num_steps + 1);
      steps.clear();
      steps.reserve(num_steps);
      for (int n = 0; n <= num_steps; n++)
      {
         states.emplace_back(H1.GetTrueVSize(), L2.GetTrueVSize());
      }
      for (int n = 0; n < num_steps; n++)
      {
         steps.emplace_back(H1.GetTrueVSize(), L2.GetTrueVSize());
      }
      Copy(initial, states[0]);

      for (int n = 0; n < num_steps; n++)
      {
         RK2AvgStep(states[n], dt, steps[n], states[n + 1]);
      }
   }

   real_t TrajectoryObjective(const HydroState &initial, real_t dt,
                              int num_steps) const
   {
      std::vector<HydroState> states;
      std::vector<RK2AvgStepData> steps;
      ForwardTrajectory(initial, dt, num_steps, states, steps);
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

void ComputeTaylorGreenRK2AvgAdjoint(
   ParFiniteElementSpace &H1,
   ParFiniteElementSpace &L2,
   const Array<int> &ess_tdofs_v,
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
   TaylorGreenAdjointState *final_state)
{
   MFEM_VERIFY(H1.GetComm() == MPI_COMM_WORLD,
               "Taylor--Green adjoint integration expects MPI_COMM_WORLD");
   MFEM_VERIFY(H1.GetMesh()->Dimension() == 2,
               "Taylor--Green adjoint integration currently requires a 2D mesh");
   MFEM_VERIFY(dt > 0.0 && num_steps > 0,
               "time step and step count must be positive");

   HydroState state(H1.GetTrueVSize(), L2.GetTrueVSize());
   ExtractTrueState(H1, L2, initial_state, state);
   TaylorGreenModel model(H1, L2, ess_tdofs_v, rho0_coefficient, rho0,
                          material, state.x, hydro);

   std::vector<HydroState> states;
   std::vector<RK2AvgStepData> steps;
   model.ForwardTrajectory(state, dt, num_steps, states, steps);
   if (final_state) { Copy(states.back(), *final_state); }

   const real_t final_time = num_steps * dt;
   HydroState lambda_next(H1.GetTrueVSize(), L2.GetTrueVSize());
   terminal_gradient(states.back(), final_time, lambda_next);
   lambda_next.v.SetSubVector(ess_tdofs_v, 0.0);

   HydroState objective_gradient(H1.GetTrueVSize(), L2.GetTrueVSize());
   running_gradient(states.back(), final_time, objective_gradient);
   objective_gradient.v.SetSubVector(ess_tdofs_v, 0.0);
   AddInPlace(lambda_next, 0.5 * dt, objective_gradient);

   HydroState lambda(H1.GetTrueVSize(), L2.GetTrueVSize());
   for (int n = num_steps - 1; n >= 0; n--)
   {
      model.ComputeLambda(steps[n], lambda_next, dt, lambda);
      running_gradient(states[n], n * dt, objective_gradient);
      objective_gradient.v.SetSubVector(ess_tdofs_v, 0.0);
      AddInPlace(lambda, (n == 0 ? 0.5 : 1.0) * dt,
                 objective_gradient);
      Copy(lambda, lambda_next);
   }
   Copy(lambda_next, initial_gradient);
}

bool VerifyTaylorGreenRK2AvgAdjoint(
   ParFiniteElementSpace &H1,
   ParFiniteElementSpace &L2,
   const Array<int> &ess_tdofs_v,
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
   bool visualization)
{
   MFEM_VERIFY(H1.GetComm() == MPI_COMM_WORLD,
               "Taylor--Green adjoint check expects MPI_COMM_WORLD");
   MFEM_VERIFY(H1.GetMesh()->Dimension() == 2,
               "Taylor--Green adjoint check currently requires a 2D mesh");
   MFEM_VERIFY(dt > 0.0 && fd_epsilon > 0.0,
               "dt and finite-difference epsilon must be positive");
   MFEM_VERIFY(num_steps > 0 && checkpoint_interval > 0,
               "step count and checkpoint interval must be positive");

   MPI_Comm comm = H1.GetComm();
   int myid = 0, num_ranks = 1;
   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &num_ranks);

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
   const real_t rhs_reference_norm =
      std::max(Norml2(laghos_true_rhs, comm), real_t(1.0));
   const real_t rhs_relative_error =
      Norml2(rhs_difference, comm) / rhs_reference_norm;
   const real_t rhs_x_error = Norml2(rhs_difference.x, comm);
   const real_t rhs_v_error = Norml2(rhs_difference.v, comm);
   const real_t rhs_e_error = Norml2(rhs_difference.e, comm);
   const real_t rhs_x_norm = Norml2(rhs.x, comm);
   const real_t rhs_v_norm = Norml2(rhs.v, comm);
   const real_t rhs_e_norm = Norml2(rhs.e, comm);
   const real_t laghos_rhs_x_norm = Norml2(laghos_true_rhs.x, comm);
   const real_t laghos_rhs_v_norm = Norml2(laghos_true_rhs.v, comm);
   const real_t laghos_rhs_e_norm = Norml2(laghos_true_rhs.e, comm);

   std::vector<HydroState> states;
   std::vector<RK2AvgStepData> steps;
   model.ForwardTrajectory(state, dt, num_steps, states, steps);

   BlockVector checkpoint_state(initial_state);
   RK2AvgSolver checkpoint_solver;
   checkpoint_solver.Init(hydro);
   StateId checkpoint_step{0};
   real_t checkpoint_time = 0.0;
   real_t checkpoint_dt = dt;
   ODECheckpointer::StateParameters checkpoint_parameters{
      &checkpoint_step, &checkpoint_time, &checkpoint_dt
   };
   const std::size_t checkpoint_count =
      1 + (num_steps - 1) / checkpoint_interval;
   ODEMemoryBlockCheckpointer checkpoints(
      checkpoint_state, checkpoint_solver, checkpoint_parameters,
      checkpoint_count);
   IntervalSchedule schedule(StateId{num_steps},
                             StateId{checkpoint_interval});
   ODEStatePropagator propagator(checkpoints, checkpoint_solver);
   CheckpointController forward_controller(checkpoints, propagator);
   if (myid == 0) { out << "\nCheckpointed forward sweep\n"; }
   for (;;)
   {
      const CheckpointCommand command = schedule.Next();
      forward_controller.Execute(command);
      if (myid == 0 && command.action == CheckpointAction::Store)
      {
         out << "  store checkpoint at state " << command.to_step
             << ", t = " << checkpoint_time << '\n';
      }
      else if (myid == 0 && command.action == CheckpointAction::Advance)
      {
         out << "  forward step " << command.from_step << " -> "
             << command.to_step << ", t = " << checkpoint_time << '\n';
      }
      else if (command.action == CheckpointAction::Finished)
      {
         break;
      }
   }

   HydroState laghos_final_state(H1.GetTrueVSize(), L2.GetTrueVSize());
   ExtractTrueState(H1, L2, checkpoint_state, laghos_final_state);
   HydroState trajectory_difference(H1.GetTrueVSize(), L2.GetTrueVSize());
   Add(states.back(), -1.0, laghos_final_state, trajectory_difference);
   const real_t trajectory_relative_error =
      Norml2(trajectory_difference, comm) /
      std::max(Norml2(laghos_final_state, comm), real_t(1.0));

   HydroState initial_direction(H1.GetTrueVSize(), L2.GetTrueVSize());
   HydroState lambda_next(H1.GetTrueVSize(), L2.GetTrueVSize());
   initial_direction.x.Randomize(1001);
   initial_direction.v.Randomize(1002);
   initial_direction.e.Randomize(1003);
   initial_direction.v.SetSubVector(ess_tdofs_v, 0.0);
   lambda_next.x.Randomize(2001);
   lambda_next.v.Randomize(2002);
   lambda_next.e.Randomize(2003);
   lambda_next.v.SetSubVector(ess_tdofs_v, 0.0);
   const real_t direction_norm = Norml2(initial_direction, comm);
   initial_direction.x /= direction_norm;
   initial_direction.v /= direction_norm;
   initial_direction.e /= direction_norm;

   HydroState direction(H1.GetTrueVSize(), L2.GetTrueVSize());
   Copy(initial_direction, direction);
   HydroState direction_next(H1.GetTrueVSize(), L2.GetTrueVSize());
   HydroState lambda(H1.GetTrueVSize(), L2.GetTrueVSize());
   for (int n = 0; n < num_steps; n++)
   {
      model.RK2AvgJvp(steps[n], direction, dt, direction_next);
      Copy(direction_next, direction);
   }
   const real_t jvp_dot = Dot(lambda_next, direction, comm);
   for (int n = num_steps - 1; n >= 0; n--)
   {
      model.ComputeLambda(steps[n], lambda_next, dt, lambda);
      Copy(lambda, lambda_next);
   }
   const real_t vjp_dot = Dot(lambda_next, initial_direction, comm);
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
      model.ComputeLambda(steps[n], lambda_next, dt, lambda_0);
      Copy(lambda_0, lambda_next);
   }

   const std::size_t window_capacity =
      static_cast<std::size_t>(std::min(num_steps, checkpoint_interval) + 1);
   ODEMemorySnapshotCheckpointer window_store(
      checkpoint_state, checkpoint_solver, checkpoint_parameters,
      window_capacity);
   ExactCheckpointWindow window(window_store, window_capacity);
   CheckpointController reverse_controller(checkpoints, propagator, &window);

   HydroState checkpoint_lambda(H1.GetTrueVSize(), L2.GetTrueVSize());
   Copy(lambda_final, lambda_next);
   real_t replay_state_relative_error = 0.0;
   if (myid == 0) { out << "\nCheckpointed reverse sweep\n"; }
   for (int n = num_steps - 1; n >= 0; n--)
   {
      CheckpointInfo primary_origin, window_origin;
      const bool have_primary =
         checkpoints.FindAtOrBefore(StateId{n}, primary_origin);
      const bool have_window =
         window.FindAtOrBefore(StateId{n}, window_origin);
      MFEM_VERIFY(have_primary || have_window,
                  "no checkpoint is available for the reverse step");
      const bool use_primary = have_primary &&
                               (!have_window ||
                                primary_origin.state >= window_origin.state);
      const CheckpointInfo origin = use_primary ? primary_origin : window_origin;
      if (myid == 0)
      {
         out << "  reverse step " << n + 1 << " -> " << n << '\n'
             << "    ";
         if (origin.state == StateId{n})
         {
            out << (use_primary ? "restore checkpoint state "
                                : "restore replay-window state ")
                << origin.state << '\n';
         }
         else
         {
            out << (use_primary ? "restore checkpoint state "
                                : "restore replay-window state ")
                << origin.state << ", replay " << StateId{n} - origin.state
                << " step(s)\n";
         }
      }
      reverse_controller.RestoreState(StateId{n});
      hydro.UpdateMesh(checkpoint_state);
      hydro.ResetQuadratureData();

      HydroState restored_state(H1.GetTrueVSize(), L2.GetTrueVSize());
      ExtractTrueState(H1, L2, checkpoint_state, restored_state);
      HydroState restored_difference(H1.GetTrueVSize(), L2.GetTrueVSize());
      Add(restored_state, -1.0, states[n], restored_difference);
      replay_state_relative_error = std::max(
         replay_state_relative_error,
         Norml2(restored_difference, comm) /
         std::max(Norml2(states[n], comm), real_t(1.0)));

      RK2AvgStepData replay_step(H1.GetTrueVSize(), L2.GetTrueVSize());
      HydroState replay_next(H1.GetTrueVSize(), L2.GetTrueVSize());
      model.RK2AvgStep(restored_state, dt, replay_step, replay_next);
      HydroState replay_difference(H1.GetTrueVSize(), L2.GetTrueVSize());
      Add(replay_next, -1.0, states[n + 1], replay_difference);
      replay_state_relative_error = std::max(
         replay_state_relative_error,
         Norml2(replay_difference, comm) /
         std::max(Norml2(states[n + 1], comm), real_t(1.0)));

      model.ComputeLambda(replay_step, lambda_next, dt, checkpoint_lambda);
      Copy(checkpoint_lambda, lambda_next);
   }

   HydroState checkpoint_gradient_difference(H1.GetTrueVSize(),
                                              L2.GetTrueVSize());
   Add(checkpoint_lambda, -1.0, lambda_0, checkpoint_gradient_difference);
   const real_t checkpoint_gradient_relative_error =
      Norml2(checkpoint_gradient_difference, comm) /
      std::max(Norml2(lambda_0, comm), real_t(1.0));

   const real_t directional_adjoint = Dot(lambda_0, initial_direction, comm);
   real_t directional_fd = 0.0;
   real_t directional_relative_error = 0.0;
   real_t gradient_relative_error = 0.0;
   real_t gradient_max_error = 0.0;
   if (finite_difference_check)
   {
      if (myid == 0)
      {
         out << "\nFinite-difference verification\n"
             << "  directional check: run two complete forward trajectories\n"
             << std::flush;
      }
      HydroState plus(H1.GetTrueVSize(), L2.GetTrueVSize());
      HydroState minus(H1.GetTrueVSize(), L2.GetTrueVSize());
      Add(state, fd_epsilon, initial_direction, plus);
      Add(state, -fd_epsilon, initial_direction, minus);
      directional_fd =
         (model.TrajectoryObjective(plus, dt, num_steps) -
          model.TrajectoryObjective(minus, dt, num_steps)) /
         (2.0 * fd_epsilon);
      const real_t directional_scale =
         std::max({std::abs(directional_fd), std::abs(directional_adjoint),
                   real_t(1.0)});
      directional_relative_error =
         std::abs(directional_fd - directional_adjoint) / directional_scale;

      HydroState finite_difference_gradient(H1.GetTrueVSize(),
                                             L2.GetTrueVSize());
      finite_difference_gradient.x = 0.0;
      finite_difference_gradient.v = 0.0;
      finite_difference_gradient.e = 0.0;

      if (myid == 0)
      {
         out << "  full-gradient check: perturb each unconstrained initial degree "
             << "of freedom and run two complete forward trajectories\n"
             << std::flush;
      }

      auto finite_difference_block = [&](Vector HydroState::*member,
                                         const Array<int> *excluded)
      {
         Vector &gradient_block = finite_difference_gradient.*member;
         const int local_size = gradient_block.Size();
         Array<char> local_skip(local_size);
         local_skip = 0;
         if (excluded)
         {
            for (int i = 0; i < excluded->Size(); i++)
            {
               local_skip[(*excluded)[i]] = 1;
            }
         }

         // Every objective evaluation is collective. Visit the true dofs
         // rank by rank so that exactly one owned dof is perturbed while all
         // ranks execute the same pair of trajectories.
         for (int owner = 0; owner < num_ranks; owner++)
         {
            int owner_size = (myid == owner) ? local_size : 0;
            MPI_Bcast(&owner_size, 1, MPI_INT, owner, comm);
            std::vector<char> owner_skip(owner_size);
            if (myid == owner)
            {
               for (int i = 0; i < owner_size; i++)
               {
                  owner_skip[i] = local_skip[i];
               }
            }
            MPI_Bcast(owner_skip.data(), owner_size, MPI_CHAR, owner, comm);

            for (int i = 0; i < owner_size; i++)
            {
               if (owner_skip[i]) { continue; }
               HydroState state_plus(H1.GetTrueVSize(), L2.GetTrueVSize());
               HydroState state_minus(H1.GetTrueVSize(), L2.GetTrueVSize());
               Copy(state, state_plus);
               Copy(state, state_minus);
               if (myid == owner)
               {
                  (state_plus.*member)[i] += fd_epsilon;
                  (state_minus.*member)[i] -= fd_epsilon;
               }
               const real_t derivative =
                  (model.TrajectoryObjective(state_plus, dt, num_steps) -
                   model.TrajectoryObjective(state_minus, dt, num_steps)) /
                  (2.0 * fd_epsilon);
               if (myid == owner) { gradient_block[i] = derivative; }
            }
         }
      };
      finite_difference_block(&HydroState::x, nullptr);
      finite_difference_block(&HydroState::v, &ess_tdofs_v);
      finite_difference_block(&HydroState::e, nullptr);

      HydroState gradient_difference(H1.GetTrueVSize(), L2.GetTrueVSize());
      Add(lambda_0, -1.0, finite_difference_gradient, gradient_difference);
      gradient_relative_error =
         Norml2(gradient_difference, comm) /
         std::max(Norml2(finite_difference_gradient, comm), real_t(1.0));
      gradient_max_error = std::max(
      {
         Normlinf(gradient_difference.x, comm),
         Normlinf(gradient_difference.v, comm),
         Normlinf(gradient_difference.e, comm)
      });
   }
   else if (myid == 0)
   {
      out << "\nFinite-difference verification disabled; use -tga-fd to enable.\n";
   }

   if (myid == 0)
   {
      out << std::setprecision(12)
          << "\nTaylor--Green fixed-step RK2Avg adjoint verification\n"
          << "  steps                           = " << num_steps << '\n'
          << "  dt                              = " << dt << '\n'
          << "  objective J(S_N)                = " << objective_value << '\n'
          << "  dFEM/Laghos RHS relative error  = " << rhs_relative_error << '\n'
          << "    position/velocity/energy      = " << rhs_x_error << " / "
          << rhs_v_error << " / " << rhs_e_error << '\n'
          << "    dFEM RHS block norms          = " << rhs_x_norm << " / "
          << rhs_v_norm << " / " << rhs_e_norm << '\n'
          << "    Laghos RHS block norms        = "
          << laghos_rhs_x_norm << " / " << laghos_rhs_v_norm << " / "
          << laghos_rhs_e_norm << '\n'
          << "  dFEM/Laghos trajectory error    = "
          << trajectory_relative_error << '\n'
          << "  checkpoint interval/count       = " << checkpoint_interval
          << " / " << checkpoints.Size() << '\n'
          << "  checkpoint replay state error   = "
          << replay_state_relative_error << '\n'
          << "  checkpoint/full adjoint error   = "
          << checkpoint_gradient_relative_error << '\n'
          << "  trajectory JVP dot              = " << jvp_dot << '\n'
          << "  trajectory lambda dot           = " << vjp_dot << '\n'
          << "  trajectory JVP/lambda error     = " << transpose_relative_error << '\n'
          << "  directional adjoint derivative  = " << directional_adjoint << '\n';
      if (finite_difference_check)
      {
         out << "  directional finite difference   = " << directional_fd << '\n'
             << "  directional relative error      = "
             << directional_relative_error << '\n'
             << "  full-gradient relative error    = "
             << gradient_relative_error << '\n'
             << "  full-gradient max abs error     = " << gradient_max_error << '\n';
      }
      else
      {
         out << "  finite-difference checks         = disabled\n";
      }
   }

   bool passed = rhs_relative_error < 1e-8 &&
                 trajectory_relative_error < 1e-8 &&
                 replay_state_relative_error < 1e-8 &&
                 checkpoint_gradient_relative_error < 1e-8 &&
                 transpose_relative_error < 1e-10;
   if (finite_difference_check)
   {
      passed = passed && directional_relative_error < 5e-7 &&
               gradient_relative_error < 5e-6;
   }
   if (myid == 0)
   {
      out << "  result                          = "
          << (passed ? "PASS" : "FAIL") << "\n\n";
   }

   if (visualization)
   {
      ParGridFunction initial_energy_gradient(&L2);
      initial_energy_gradient.SetFromTrueDofs(checkpoint_lambda.e);
      socketstream gradient_socket;
      const char vishost[] = "localhost";
      constexpr int visport = 19916;
      VisualizeField(gradient_socket, vishost, visport,
                     initial_energy_gradient,
                     "Gradient of J with respect to initial energy",
                     0, 0, 700, 700);
   }
   return passed;
}

} // namespace mfem::hydrodynamics
