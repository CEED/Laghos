// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.

#include "laghos_tg_manufactured.hpp"

#include "laghos_qf.hpp"
#include "laghos_solver.hpp"
#include "laghos_tg_adjoint.hpp"

#include <cmath>
#include <iomanip>
#include <memory>

namespace mfem::hydrodynamics
{

constexpr int MANUFACTURED_TIME = 20;
constexpr int MANUFACTURED_VALUE = 21;

template <int DIM>
struct TaylorGreenManufacturedObjectiveQFunction
{
   static_assert(DIM == 2, "the manufactured adjoint is two-dimensional");
   using vec = vecd_t<DIM, dscalar_t>;
   using mat = matd_t<DIM, dscalar_t>;
   using real_mat = matd_t<DIM>;

   real_t final_time;
   bool terminal;

   MFEM_HOST_DEVICE inline
   void operator()(const vec &x, const mat &J, const vec &v,
                   const dscalar_t &energy, const real_t &rho0,
                   const real_mat &J0, const real_t &time,
                   const real_t &weight, dscalar_t &value) const
   {
      using std::cos;
      using std::sin;

      constexpr real_t gamma_minus_one = 2.0 / 3.0;
      const dscalar_t pix = M_PI * x(0);
      const dscalar_t piy = M_PI * x(1);
      const dscalar_t sx = sin(pix), cx = cos(pix);
      const dscalar_t sy = sin(piy), cy = cos(piy);
      const dscalar_t s2x = sin(2.0 * pix), c2x = cos(2.0 * pix);
      const dscalar_t s2y = sin(2.0 * piy), c2y = cos(2.0 * piy);

      vec velocity;
      velocity(0) = sx * cy;
      velocity(1) = -cx * sy;

      mat grad_velocity;
      grad_velocity(0, 0) = M_PI * cx * cy;
      grad_velocity(0, 1) = -M_PI * sx * sy;
      grad_velocity(1, 0) = M_PI * sx * sy;
      grad_velocity(1, 1) = -M_PI * cx * cy;

      const dscalar_t pressure = 1.0 + 0.25 * (c2x + c2y);
      const dscalar_t exact_energy = pressure / gamma_minus_one;
      vec grad_energy;
      grad_energy(0) = -0.5 * M_PI / gamma_minus_one * s2x;
      grad_energy(1) = -0.5 * M_PI / gamma_minus_one * s2y;
      const dscalar_t source = velocity(0) * grad_energy(0) +
                               velocity(1) * grad_energy(1);

      const dscalar_t phi = sx * sx * sy * sy;
      vec grad_phi;
      grad_phi(0) = M_PI * s2x * sy * sy;
      grad_phi(1) = M_PI * s2y * sx * sx;

      const real_t tau = time / final_time;
      const real_t alpha_factor = 1.0 + tau;
      const real_t beta_factor = 1.0 + 2.0 * tau;
      const real_t chi_factor = 1.0 + 3.0 * tau;

      dscalar_t alpha;
      vec beta;
      dscalar_t chi;
      if (terminal)
      {
         alpha = alpha_factor * phi;
         beta(0) = beta_factor * phi;
         beta(1) = -beta_factor * phi;
         chi = chi_factor * phi;
      }
      else
      {
         const dscalar_t div_beta =
            beta_factor * (grad_phi(0) - grad_phi(1));
         const dscalar_t velocity_grad_phi =
            velocity(0) * grad_phi(0) + velocity(1) * grad_phi(1);
         const dscalar_t div_energy_beta =
            beta_factor * phi * (grad_energy(0) - grad_energy(1)) +
            exact_energy * div_beta;

         const dscalar_t astar_alpha =
            alpha_factor * velocity_grad_phi +
            gamma_minus_one * div_energy_beta - source * chi_factor * phi;

         vec astar_beta;
         for (int j = 0; j < DIM; j++)
         {
            const real_t sign = (j == 0) ? 1.0 : -1.0;
            const dscalar_t grad_velocity_t_beta =
               grad_velocity(0, j) * beta_factor * phi -
               grad_velocity(1, j) * beta_factor * phi;
            const dscalar_t grad_energy_chi =
               chi_factor * phi * grad_energy(j) +
               exact_energy * chi_factor * grad_phi(j);
            astar_beta(j) =
               alpha_factor * grad_phi(j) +
               sign * beta_factor * velocity_grad_phi -
               grad_velocity_t_beta - chi_factor * phi * grad_energy(j) +
               gamma_minus_one * grad_energy_chi;
         }

         const dscalar_t astar_chi =
            gamma_minus_one * div_beta +
            chi_factor * velocity_grad_phi;

         alpha = -phi / final_time - astar_alpha;
         beta(0) = -2.0 * phi / final_time - astar_beta(0);
         beta(1) = 2.0 * phi / final_time - astar_beta(1);
         chi = -3.0 * phi / final_time - astar_chi;
      }

      const dscalar_t det_j = det(J);
      const real_t det_j0 = det(J0);
      const dscalar_t density_part =
         alpha * (rho0 * det_j0 - det_j);
      const dscalar_t velocity_part =
         beta(0) * (v(0) - velocity(0)) +
         beta(1) * (v(1) - velocity(1));
      const dscalar_t energy_part = chi * (energy - exact_energy);
      value = (density_part + det_j * (velocity_part + energy_part)) * weight;
   }
};

std::shared_ptr<DifferentiableOperator> CreateManufacturedObjective(
   ParFiniteElementSpace &H1, ParFiniteElementSpace &L2,
   const IntegrationRule &ir, int qfunction_backend, real_t final_time,
   bool terminal)
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
         FieldDescriptor{DENSITY0, &L2},
         FieldDescriptor{COORDINATES0, &H1},
         FieldDescriptor{MANUFACTURED_TIME, &L2}
      };
      std::vector outputs{FieldDescriptor{MANUFACTURED_VALUE, &L2}};
      auto objective = std::make_shared<DifferentiableOperator>(
                          inputs, outputs, mesh);
      objective->SetMultLevel(DifferentiableOperator::LVECTOR);
      future::tuple kernel_inputs =
      {
         Value<COORDINATES>{}, Gradient<COORDINATES>{}, Value<VELOCITY>{},
         Value<INTERNAL_ENERGY>{}, Value<DENSITY0>{},
         Gradient<COORDINATES0>{}, Value<MANUFACTURED_TIME>{}, Weight{}
      };
      future::tuple kernel_outputs = {Value<MANUFACTURED_VALUE>{}};
      auto derivatives = std::integer_sequence<size_t, COORDINATES, VELOCITY,
           INTERNAL_ENERGY> {};
      TaylorGreenManufacturedObjectiveQFunction<2> qf
      {
         final_time, terminal
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

class TaylorGreenManufacturedObjective
{
private:
   Vector rho0, initial_coordinates, time_values, output_ones;
   std::shared_ptr<DifferentiableOperator> terminal, running;

   MultiVector Inputs(const TaylorGreenAdjointState &state, real_t time)
   {
      time_values = time;
      return MultiVector{state.x, state.v, state.e, rho0,
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
   TaylorGreenManufacturedObjective(
      ParFiniteElementSpace &H1, ParFiniteElementSpace &L2,
      const ParGridFunction &rho0_gf, const Vector &initial_x,
      const IntegrationRule &ir, int qfunction_backend, real_t final_time) :
      rho0(L2.GetTrueVSize()), initial_coordinates(initial_x),
      time_values(L2.GetTrueVSize()), output_ones(L2.GetTrueVSize()),
      terminal(CreateManufacturedObjective(H1, L2, ir, qfunction_backend,
                                           final_time, true)),
      running(CreateManufacturedObjective(H1, L2, ir, qfunction_backend,
                                          final_time, false))
   {
      rho0_gf.GetTrueDofs(rho0);
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

void ApplyInverseMass(ParFiniteElementSpace &space, bool vector_space,
                      const IntegrationRule &ir,
                      const Array<int> &essential_dofs,
                      const Vector &covector,
                      Vector &field)
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

void ExactInitialCoordinateAdjoint(const Vector &x, Vector &lambda)
{
   const real_t pix = M_PI * x(0), piy = M_PI * x(1);
   const real_t sx = std::sin(pix), cx = std::cos(pix);
   const real_t sy = std::sin(piy), cy = std::cos(piy);
   const real_t phi = sx * sx * sy * sy;
   const real_t phix = M_PI * std::sin(2.0 * pix) * sy * sy;
   const real_t phiy = M_PI * std::sin(2.0 * piy) * sx * sx;
   const real_t ex = -0.75 * M_PI * std::sin(2.0 * pix);
   const real_t ey = -0.75 * M_PI * std::sin(2.0 * piy);
   const real_t du00 = M_PI * cx * cy;
   const real_t du01 = -M_PI * sx * sy;
   const real_t du10 = M_PI * sx * sy;
   const real_t du11 = -M_PI * cx * cy;
   lambda.SetSize(2);
   lambda(0) = phix - phi * (du00 - du10) - phi * ex;
   lambda(1) = phiy - phi * (du01 - du11) - phi * ey;
}

void ExactInitialVelocityAdjoint(const Vector &x, Vector &lambda)
{
   const real_t phi = std::pow(std::sin(M_PI * x(0)), 2) *
                      std::pow(std::sin(M_PI * x(1)), 2);
   lambda.SetSize(2);
   lambda(0) = phi;
   lambda(1) = -phi;
}

real_t ExactInitialEnergyAdjoint(const Vector &x)
{
   return std::pow(std::sin(M_PI * x(0)), 2) *
          std::pow(std::sin(M_PI * x(1)), 2);
}

bool VerifyTaylorGreenManufacturedAdjoint(
   ParFiniteElementSpace &H1,
   ParFiniteElementSpace &L2,
   const Array<int> &ess_tdofs_v,
   Coefficient &rho0_coefficient,
   const ParGridFunction &rho0,
   const ParGridFunction &material,
   const BlockVector &initial_state,
   const IntegrationRule &ir,
   int qfunction_backend,
   LagrangianHydroOperator &hydro,
   real_t dt,
   int num_steps)
{
   MFEM_VERIFY(dt > 0.0 && num_steps > 0,
               "manufactured-adjoint time step and step count must be positive");
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

   TaylorGreenManufacturedObjective objective(
      H1, L2, rho0, initial_x, ir, qfunction_backend, final_time);
   TaylorGreenAdjointState gradient(h1_size, l2_size);
   ComputeTaylorGreenRK2AvgAdjoint(
      H1, L2, ess_tdofs_v, rho0_coefficient, rho0, material, initial_state,
      hydro, dt, num_steps,
      [&](const TaylorGreenAdjointState &state, real_t time,
          TaylorGreenAdjointState &result)
   {
      objective.TerminalGradient(state, time, result);
   },
   [&](const TaylorGreenAdjointState &state, real_t time,
       TaylorGreenAdjointState &result)
   {
      objective.RunningGradient(state, time, result);
   },
   gradient);

   Vector lambda_x, lambda_v, lambda_e;
   Array<int> no_essential_dofs;
   ApplyInverseMass(H1, true, ir, no_essential_dofs, gradient.x, lambda_x);
   ApplyInverseMass(H1, true, ir, ess_tdofs_v, gradient.v, lambda_v);
   ApplyInverseMass(L2, false, ir, no_essential_dofs, gradient.e, lambda_e);

   ParGridFunction lambda_x_gf(&H1), lambda_v_gf(&H1), lambda_e_gf(&L2);
   lambda_x_gf.SetFromTrueDofs(lambda_x);
   lambda_v_gf.SetFromTrueDofs(lambda_v);
   lambda_e_gf.SetFromTrueDofs(lambda_e);
   VectorFunctionCoefficient exact_x(2, ExactInitialCoordinateAdjoint);
   VectorFunctionCoefficient exact_v(2, ExactInitialVelocityAdjoint);
   FunctionCoefficient exact_e(ExactInitialEnergyAdjoint);
   const real_t x_error = lambda_x_gf.ComputeL2Error(exact_x);
   const real_t v_error = lambda_v_gf.ComputeL2Error(exact_v);
   const real_t e_error = lambda_e_gf.ComputeL2Error(exact_e);
   const real_t combined_error =
      std::sqrt(x_error*x_error + v_error*v_error + e_error*e_error);

   if (Mpi::Root())
   {
      out << std::scientific << std::setprecision(16)
          << "\nTaylor--Green manufactured-adjoint convergence sample\n"
          << "  elements                         = " << H1.GetMesh()->GetNE() << '\n'
          << "  H1 true dofs                     = " << h1_size << '\n'
          << "  L2 true dofs                     = " << l2_size << '\n'
          << "  dt                               = " << dt << '\n'
          << "  final time                       = " << final_time << '\n'
          << "  initial coordinate-adjoint error = " << x_error << '\n'
          << "  initial velocity-adjoint error   = " << v_error << '\n'
          << "  initial energy-adjoint error     = " << e_error << '\n'
          << "  combined initial-adjoint error   = " << combined_error << '\n';
   }
   return std::isfinite(combined_error);
}

} // namespace mfem::hydrodynamics
