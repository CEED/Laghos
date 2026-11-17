// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.

#include "laghos_tg_lagrangian_manufactured.hpp"

#include "laghos_qf.hpp"
#include "laghos_solver.hpp"
#include "laghos_tg_adjoint.hpp"

#include <cmath>
#include <iomanip>
#include <memory>

namespace mfem::hydrodynamics
{

constexpr int LAGRANGIAN_MANUFACTURED_TIME = 22;
constexpr int LAGRANGIAN_MANUFACTURED_VALUE = 23;

template <int DIM>
struct TaylorGreenLagrangianObjectiveQFunction
{
   static_assert(DIM == 2,
                 "the manufactured Lagrangian adjoint is two-dimensional");
   using dvec = vecd_t<DIM, dscalar_t>;
   using dmat = matd_t<DIM, dscalar_t>;
   using vec = vecd_t<DIM>;
   using mat = matd_t<DIM>;

   QConstants qconstants;
   real_t final_time;
   bool terminal;

   MFEM_HOST_DEVICE inline
   void operator()(const dvec &x, const dmat &J, const dvec &v,
                   const dmat &dvdxi, const dscalar_t &energy,
                   const vec &initial_x, const mat &J0, const real_t &time,
                   const real_t &weight, dscalar_t &value) const
   {
      using std::cos;
      using std::sin;

      const real_t pia = M_PI * initial_x(0);
      const real_t pib = M_PI * initial_x(1);
      const real_t sa = sin(pia), ca = cos(pia);
      const real_t sb = sin(pib), cb = cos(pib);
      const real_t sa2 = sa * sa, sb2 = sb * sb;
      const real_t sa4 = sa2 * sa2, sb4 = sb2 * sb2;
      const real_t phi = sa4 * sa2 * sb4 * sb2;
      vec grad_phi;
      grad_phi(0) = 6.0 * M_PI * sa4 * sa * ca * sb4 * sb2;
      grad_phi(1) = 6.0 * M_PI * sb4 * sb * cb * sa4 * sa2;

      const real_t tau = time / final_time;
      const real_t lambda_x_factor = 1.0 + tau;
      const real_t lambda_v_factor = 1.0 + 2.0 * tau;
      const real_t lambda_e = (1.0 + 3.0 * tau) * phi;
      const real_t dt_lambda_x_factor = 1.0 / final_time;
      const real_t dt_lambda_v_factor = 2.0 / final_time;
      const real_t dt_lambda_e = 3.0 * phi / final_time;

      vec direction;
      direction(0) = 1.0;
      direction(1) = -1.0;

      const real_t material_measure = det(J0) * weight;
      value = 0.0;
      for (int i = 0; i < DIM; i++)
      {
         const real_t lambda_x = lambda_x_factor * phi * direction(i);
         const real_t dt_lambda_x =
            dt_lambda_x_factor * phi * direction(i);
         const real_t dt_lambda_v =
            dt_lambda_v_factor * phi * direction(i);
         value -= (dt_lambda_x * x(i) + dt_lambda_v * v(i) +
                   lambda_x * v(i)) * material_measure;
      }
      value -= dt_lambda_e * energy * material_measure;

      if (terminal)
      {
         // The state-dependent constants omitted above do not change the
         // objective gradient. Replace the running functional by its terminal
         // linear functional.
         value = lambda_e * energy * material_measure;
         for (int i = 0; i < DIM; i++)
         {
            const real_t lambda_x = lambda_x_factor * phi * direction(i);
            const real_t lambda_v = lambda_v_factor * phi * direction(i);
            value += (lambda_x * x(i) + lambda_v * v(i)) * material_measure;
         }
         return;
      }

      // The first Piola weak terms are evaluated in reference-element
      // coordinates. StressSetup returns sigma J^{-T} det(J) weight, which
      // is precisely the tensor contracted with a reference gradient.
      constexpr real_t rho0 = 1.0;
      constexpr real_t gamma = 5.0 / 3.0;
      const auto setup = StressSetup<DIM>(qconstants, dvdxi, rho0, J0, J,
                                          gamma, energy, weight);
      const auto stress_jit = future::get<0>(setup);

      mat grad_lambda_v_dxi;
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            grad_lambda_v_dxi(i, j) = 0.0;
            for (int k = 0; k < DIM; k++)
            {
               const real_t grad_lambda_v_ik =
                  lambda_v_factor * direction(i) * grad_phi(k);
               grad_lambda_v_dxi(i, j) += grad_lambda_v_ik * J0(k, j);
            }
         }
      }

      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            value += stress_jit(i, j) * grad_lambda_v_dxi(i, j);
            value -= lambda_e * stress_jit(i, j) * dvdxi(i, j);
         }
      }

      const dscalar_t source =
         3.0 / 8.0 * M_PI *
         (cos(3.0*M_PI*x(0)) * cos(M_PI*x(1)) -
          cos(M_PI*x(0)) * cos(3.0*M_PI*x(1)));
      value -= lambda_e * source * det(J) * weight;
   }
};

std::shared_ptr<DifferentiableOperator> CreateLagrangianObjective(
   ParFiniteElementSpace &H1, ParFiniteElementSpace &L2,
   const IntegrationRule &ir, const QConstants &qconstants,
   int qfunction_backend, real_t final_time, bool terminal)
{
   ParMesh &mesh = *H1.GetParMesh();
   Array<int> attributes(mesh.attributes.Max());
   attributes = 1;
   auto create = [&](auto backend)
   {
      using backend_t = decltype(backend);
      std::vector inputs
      {
         FieldDescriptor{COORDINATES, &H1},
         FieldDescriptor{VELOCITY, &H1},
         FieldDescriptor{INTERNAL_ENERGY, &L2},
         FieldDescriptor{COORDINATES0, &H1},
         FieldDescriptor{LAGRANGIAN_MANUFACTURED_TIME, &L2}
      };
      std::vector outputs
      {
         FieldDescriptor{LAGRANGIAN_MANUFACTURED_VALUE, &L2}
      };
      auto objective = std::make_shared<DifferentiableOperator>(
                          inputs, outputs, mesh);
      future::tuple kernel_inputs =
      {
         Value<COORDINATES>{}, Gradient<COORDINATES>{},
         Value<VELOCITY>{}, Gradient<VELOCITY>{},
         Value<INTERNAL_ENERGY>{},
         Value<COORDINATES0>{}, Gradient<COORDINATES0>{},
         Value<LAGRANGIAN_MANUFACTURED_TIME>{}, Weight{}
      };
      future::tuple kernel_outputs =
      {
         Value<LAGRANGIAN_MANUFACTURED_VALUE>{}
      };
      auto derivatives = std::integer_sequence<size_t, COORDINATES, VELOCITY,
           INTERNAL_ENERGY> {};
      TaylorGreenLagrangianObjectiveQFunction<2> qf
      {
         qconstants, final_time, terminal
      };
      objective->AddDomainIntegrator<
      backend_t, future::DerivativeKernels::ApplyTranspose>(
         qf, kernel_inputs, kernel_outputs, ir, attributes, derivatives);
      return objective;
   };
   MFEM_VERIFY(qfunction_backend == LOCAL_QF ||
               qfunction_backend == GLOBAL_QF,
               "unknown Q-function backend");
   return qfunction_backend == GLOBAL_QF ?
          create(future::GlobalQFBackend{}) :
          create(future::LocalQFBackend{});
}

class TaylorGreenLagrangianObjective
{
private:
   Vector initial_coordinates, time_values, output_ones;
   std::shared_ptr<DifferentiableOperator> terminal, running;

   MultiVector Inputs(const TaylorGreenAdjointState &state, real_t time)
   {
      time_values = time;
      return MultiVector{state.x, state.v, state.e,
                         initial_coordinates, time_values};
   }

   void Gradient(DifferentiableOperator &objective,
                 const TaylorGreenAdjointState &state, real_t time,
                 TaylorGreenAdjointState &gradient)
   {
      gradient.x = 0.0;
      gradient.v = 0.0;
      gradient.e = 0.0;
      MultiVector inputs = Inputs(state, time);
      MultiVector output_lambda{output_ones};
      for (auto field :
           {
              std::pair{COORDINATES, &TaylorGreenAdjointState::x},
              std::pair{VELOCITY, &TaylorGreenAdjointState::v},
              std::pair{INTERNAL_ENERGY, &TaylorGreenAdjointState::e}
           })
      {
         MultiVector output{gradient.*field.second};
         objective.GetDerivative(field.first, inputs)->MultTranspose(
            output_lambda, output);
      }
   }

public:
   TaylorGreenLagrangianObjective(
      ParFiniteElementSpace &H1, ParFiniteElementSpace &L2,
      const Vector &initial_x, const IntegrationRule &ir,
      const QConstants &qconstants, int qfunction_backend, real_t final_time) :
      initial_coordinates(initial_x), time_values(L2.GetTrueVSize()),
      output_ones(L2.GetTrueVSize()),
      terminal(CreateLagrangianObjective(H1, L2, ir, qconstants,
                                         qfunction_backend, final_time, true)),
      running(CreateLagrangianObjective(H1, L2, ir, qconstants,
                                        qfunction_backend, final_time, false))
   {
      output_ones = 1.0;
   }

   void TerminalGradient(const TaylorGreenAdjointState &state, real_t time,
                         TaylorGreenAdjointState &gradient)
   {
      Gradient(*terminal, state, time, gradient);
   }

   void RunningGradient(const TaylorGreenAdjointState &state, real_t time,
                        TaylorGreenAdjointState &gradient)
   {
      Gradient(*running, state, time, gradient);
   }
};

void ApplyLagrangianInverseMass(
   ParFiniteElementSpace &space, bool vector_space,
   const IntegrationRule &ir, const Array<int> &essential_dofs,
   const Vector &covector, Vector &field)
{
   ConstantCoefficient one(1.0);
   ParBilinearForm mass(&space);
   mass.SetAssemblyLevel(AssemblyLevel::LEGACY);
   if (vector_space)
   {
      mass.AddDomainIntegrator(new VectorMassIntegrator(one, &ir));
   }
   else
   {
      mass.AddDomainIntegrator(new MassIntegrator(one, &ir));
   }
   mass.Assemble();
   mass.Finalize();
   HypreParMatrix matrix;
   mass.FormSystemMatrix(essential_dofs, matrix);

   HypreSmoother preconditioner;
   preconditioner.SetType(HypreSmoother::Jacobi, 1);
   CGSolver solver(space.GetComm());
   solver.SetPreconditioner(preconditioner);
   solver.SetOperator(matrix);
   solver.SetRelTol(1e-13);
   solver.SetAbsTol(0.0);
   solver.SetMaxIter(1000);
   solver.SetPrintLevel(-1);
   solver.iterative_mode = false;
   field.SetSize(covector.Size());
   field = 0.0;
   solver.Mult(covector, field);
}

real_t LagrangianFieldDifferenceL2(
   ParFiniteElementSpace &space, bool vector_space,
   const IntegrationRule &ir, const Vector &a, const Vector &b)
{
   ConstantCoefficient one(1.0);
   ParBilinearForm mass(&space);
   mass.SetAssemblyLevel(AssemblyLevel::LEGACY);
   if (vector_space)
   {
      mass.AddDomainIntegrator(new VectorMassIntegrator(one, &ir));
   }
   else
   {
      mass.AddDomainIntegrator(new MassIntegrator(one, &ir));
   }
   mass.Assemble();
   mass.Finalize();
   Array<int> no_essential_dofs;
   HypreParMatrix matrix;
   mass.FormSystemMatrix(no_essential_dofs, matrix);

   Vector difference(a);
   difference -= b;
   Vector mass_difference(difference.Size());
   matrix.Mult(difference, mass_difference);
   const real_t local_norm_squared = difference * mass_difference;
   real_t global_norm_squared = 0.0;
   MPI_Allreduce(&local_norm_squared, &global_norm_squared, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_SUM, space.GetComm());
   return std::sqrt(std::max(global_norm_squared, real_t(0.0)));
}

void ExactInitialLagrangianVectorAdjoint(const Vector &a, Vector &lambda)
{
   const real_t phi = std::pow(std::sin(M_PI * a(0)), 6) *
                      std::pow(std::sin(M_PI * a(1)), 6);
   lambda.SetSize(2);
   lambda(0) = phi;
   lambda(1) = -phi;
}

real_t ExactInitialLagrangianEnergyAdjoint(const Vector &a)
{
   return std::pow(std::sin(M_PI * a(0)), 6) *
          std::pow(std::sin(M_PI * a(1)), 6);
}

bool VerifyTaylorGreenLagrangianManufacturedAdjoint(
   ParFiniteElementSpace &H1,
   ParFiniteElementSpace &L2,
   const Array<int> &ess_tdofs_v,
   Coefficient &rho0_coefficient,
   const ParGridFunction &rho0,
   ParGridFunction &material,
   const BlockVector &initial_state,
   const IntegrationRule &ir,
   const QConstants &qconstants,
   int qfunction_backend,
   int ode_solver_type,
   LagrangianHydroOperator &hydro,
   real_t dt,
   int num_steps,
   int temporal_refinements)
{
   MFEM_VERIFY(dt > 0.0 && num_steps > 0,
               "manufactured-adjoint time step and step count must be positive");
   MFEM_VERIFY(temporal_refinements >= 0,
               "number of temporal refinements must be nonnegative");
   MFEM_VERIFY(ode_solver_type == 4 || ode_solver_type == 7,
               "manufactured adjoint supports only RK4 (4) and RK2Avg (7)");
   const real_t final_time = dt * num_steps;
   const int h1_size = H1.GetTrueVSize();
   const int l2_size = L2.GetTrueVSize();

   Vector initial_x(h1_size);
   {
      ParGridFunction x(&H1);
      Vector &state = const_cast<BlockVector &>(initial_state);
      x.MakeRef(&H1, state, 0);
      x.SyncAliasMemory(state);
      x.GetTrueDofs(initial_x);
   }

   TaylorGreenLagrangianObjective objective(
      H1, L2, initial_x, ir, qconstants,
      qfunction_backend, final_time);
   const auto terminal_gradient =
      [&](const TaylorGreenAdjointState &state, real_t time,
          TaylorGreenAdjointState &result)
   {
      objective.TerminalGradient(state, time, result);
   };
   const auto running_gradient =
      [&](const TaylorGreenAdjointState &state, real_t time,
          TaylorGreenAdjointState &result)
   {
      objective.RunningGradient(state, time, result);
   };
   Array<int> no_essential_dofs;
   auto compute_adjoint_fields = [&](real_t step_dt, int step_count,
                                     TaylorGreenAdjointState &fields)
   {
      TaylorGreenAdjointState gradient(h1_size, l2_size);
      if (ode_solver_type == 4)
      {
         ComputeTaylorGreenRK4Adjoint(
            H1, L2, ess_tdofs_v, rho0_coefficient, rho0, material,
            initial_state, hydro, step_dt, step_count, terminal_gradient,
            running_gradient, gradient);
      }
      else
      {
         ComputeTaylorGreenRK2AvgAdjoint(
            H1, L2, ess_tdofs_v, rho0_coefficient, rho0, material,
            initial_state, hydro, step_dt, step_count, terminal_gradient,
            running_gradient, gradient);
      }
      ApplyLagrangianInverseMass(H1, true, ir, no_essential_dofs,
                                 gradient.x, fields.x);
      ApplyLagrangianInverseMass(H1, true, ir, ess_tdofs_v,
                                 gradient.v, fields.v);
      ApplyLagrangianInverseMass(L2, false, ir, no_essential_dofs,
                                 gradient.e, fields.e);
   };

   TaylorGreenAdjointState adjoint_fields(h1_size, l2_size);
   compute_adjoint_fields(dt, num_steps, adjoint_fields);

   ParGridFunction lambda_x_gf(&H1), lambda_v_gf(&H1), lambda_e_gf(&L2);
   lambda_x_gf.SetFromTrueDofs(adjoint_fields.x);
   lambda_v_gf.SetFromTrueDofs(adjoint_fields.v);
   lambda_e_gf.SetFromTrueDofs(adjoint_fields.e);
   VectorFunctionCoefficient exact_x(2,
                                      ExactInitialLagrangianVectorAdjoint);
   VectorFunctionCoefficient exact_v(2,
                                      ExactInitialLagrangianVectorAdjoint);
   FunctionCoefficient exact_e(ExactInitialLagrangianEnergyAdjoint);
   const real_t x_error = lambda_x_gf.ComputeL2Error(exact_x);
   const real_t v_error = lambda_v_gf.ComputeL2Error(exact_v);
   const real_t e_error = lambda_e_gf.ComputeL2Error(exact_e);
   const real_t combined_error =
      std::sqrt(x_error*x_error + v_error*v_error + e_error*e_error);
   const long long global_elements = H1.GetMesh()->GetGlobalNE();
   const HYPRE_BigInt global_h1_size = H1.GlobalTrueVSize();
   const HYPRE_BigInt global_l2_size = L2.GlobalTrueVSize();

   if (Mpi::Root())
   {
      out << std::scientific << std::setprecision(16)
          << "\nTaylor--Green direct Lagrangian manufactured-adjoint sample\n"
          << "  elements                         = " << global_elements << '\n'
          << "  H1 true dofs                     = " << global_h1_size << '\n'
          << "  L2 true dofs                     = " << global_l2_size << '\n'
          << "  time integrator                  = "
          << (ode_solver_type == 4 ? "RK4" : "RK2Avg") << '\n'
          << "  dt                               = " << dt << '\n'
          << "  final time                       = " << final_time << '\n'
          << "  initial coordinate-adjoint error = " << x_error << '\n'
          << "  initial velocity-adjoint error   = " << v_error << '\n'
          << "  initial energy-adjoint error     = " << e_error << '\n'
          << "  combined initial-adjoint error   = " << combined_error << '\n';
   }

   real_t previous_difference = 0.0;
   TaylorGreenAdjointState previous_fields(h1_size, l2_size);
   previous_fields.x = adjoint_fields.x;
   previous_fields.v = adjoint_fields.v;
   previous_fields.e = adjoint_fields.e;
   for (int level = 1; level <= temporal_refinements; level++)
   {
      const int refined_steps = num_steps * (1 << level);
      const real_t refined_dt = dt / (1 << level);
      TaylorGreenAdjointState refined_fields(h1_size, l2_size);
      compute_adjoint_fields(refined_dt, refined_steps, refined_fields);
      const real_t x_difference = LagrangianFieldDifferenceL2(
                                     H1, true, ir, previous_fields.x,
                                     refined_fields.x);
      const real_t v_difference = LagrangianFieldDifferenceL2(
                                     H1, true, ir, previous_fields.v,
                                     refined_fields.v);
      const real_t e_difference = LagrangianFieldDifferenceL2(
                                     L2, false, ir, previous_fields.e,
                                     refined_fields.e);
      const real_t combined_difference =
         std::sqrt(x_difference*x_difference + v_difference*v_difference +
                   e_difference*e_difference);
      if (Mpi::Root())
      {
         out << "  temporal refinement " << level
             << ": steps = " << refined_steps
             << ", dt = " << refined_dt
             << ", successive difference = " << combined_difference;
         if (level > 1)
         {
            out << ", observed order = "
                << std::log(previous_difference / combined_difference) /
                   std::log(2.0);
         }
         out << '\n';
      }
      previous_difference = combined_difference;
      previous_fields.x = refined_fields.x;
      previous_fields.v = refined_fields.v;
      previous_fields.e = refined_fields.e;
   }
   return std::isfinite(combined_error);
}

} // namespace mfem::hydrodynamics
