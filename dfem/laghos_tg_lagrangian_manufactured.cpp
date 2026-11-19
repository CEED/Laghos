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

struct ReferenceJet2
{
   real_t value;
   real_t gradient[2];
   real_t hessian[2][2];

   MFEM_HOST_DEVICE ReferenceJet2(real_t value_ = 0.0) : value(value_)
   {
      for (int i = 0; i < 2; i++)
      {
         gradient[i] = 0.0;
         for (int j = 0; j < 2; j++) { hessian[i][j] = 0.0; }
      }
   }

   MFEM_HOST_DEVICE static ReferenceJet2 Variable(real_t value, int direction)
   {
      ReferenceJet2 result(value);
      result.gradient[direction] = 1.0;
      return result;
   }
};

MFEM_HOST_DEVICE ReferenceJet2 operator+(const ReferenceJet2 &a,
                                          const ReferenceJet2 &b)
{
   ReferenceJet2 result(a.value + b.value);
   for (int i = 0; i < 2; i++)
   {
      result.gradient[i] = a.gradient[i] + b.gradient[i];
      for (int j = 0; j < 2; j++)
      {
         result.hessian[i][j] = a.hessian[i][j] + b.hessian[i][j];
      }
   }
   return result;
}

MFEM_HOST_DEVICE ReferenceJet2 operator-(const ReferenceJet2 &a)
{
   ReferenceJet2 result(-a.value);
   for (int i = 0; i < 2; i++)
   {
      result.gradient[i] = -a.gradient[i];
      for (int j = 0; j < 2; j++)
      {
         result.hessian[i][j] = -a.hessian[i][j];
      }
   }
   return result;
}

MFEM_HOST_DEVICE ReferenceJet2 operator-(const ReferenceJet2 &a,
                                          const ReferenceJet2 &b)
{
   return a + (-b);
}

MFEM_HOST_DEVICE ReferenceJet2 operator*(const ReferenceJet2 &a,
                                          const ReferenceJet2 &b)
{
   ReferenceJet2 result(a.value * b.value);
   for (int i = 0; i < 2; i++)
   {
      result.gradient[i] = a.gradient[i] * b.value +
                           a.value * b.gradient[i];
      for (int j = 0; j < 2; j++)
      {
         result.hessian[i][j] =
            a.hessian[i][j] * b.value + a.value * b.hessian[i][j] +
            a.gradient[i] * b.gradient[j] +
            a.gradient[j] * b.gradient[i];
      }
   }
   return result;
}

MFEM_HOST_DEVICE ReferenceJet2 UnaryReferenceJet(
   const ReferenceJet2 &a, real_t value, real_t first, real_t second)
{
   ReferenceJet2 result(value);
   for (int i = 0; i < 2; i++)
   {
      result.gradient[i] = first * a.gradient[i];
      for (int j = 0; j < 2; j++)
      {
         result.hessian[i][j] = first * a.hessian[i][j] +
                                second * a.gradient[i] * a.gradient[j];
      }
   }
   return result;
}

MFEM_HOST_DEVICE ReferenceJet2 Inverse(const ReferenceJet2 &a)
{
   const real_t inverse = 1.0 / a.value;
   return UnaryReferenceJet(a, inverse, -inverse * inverse,
                            2.0 * inverse * inverse * inverse);
}

MFEM_HOST_DEVICE ReferenceJet2 operator/(const ReferenceJet2 &a,
                                          const ReferenceJet2 &b)
{
   return a * Inverse(b);
}

MFEM_HOST_DEVICE ReferenceJet2 Sin(const ReferenceJet2 &a)
{
   using std::cos;
   using std::sin;
   return UnaryReferenceJet(a, sin(a.value), cos(a.value), -sin(a.value));
}

MFEM_HOST_DEVICE ReferenceJet2 Cos(const ReferenceJet2 &a)
{
   using std::cos;
   using std::sin;
   return UnaryReferenceJet(a, cos(a.value), -sin(a.value), -cos(a.value));
}

MFEM_HOST_DEVICE ReferenceJet2 Sqrt(const ReferenceJet2 &a)
{
   using std::sqrt;
   const real_t root = sqrt(a.value);
   return UnaryReferenceJet(a, root, 0.5 / root,
                            -0.25 / (root * root * root));
}

MFEM_HOST_DEVICE ReferenceJet2 Acos(const ReferenceJet2 &a)
{
   using std::acos;
   using std::sqrt;
   const real_t denominator = sqrt(1.0 - a.value * a.value);
   return UnaryReferenceJet(a, acos(a.value), -1.0 / denominator,
                            -a.value / (denominator * denominator * denominator));
}

template <int ORDER = 24>
MFEM_HOST_DEVICE void JacobiEllipticSmallArgument(
   const ReferenceJet2 &parameter, real_t argument,
   ReferenceJet2 &sn, ReferenceJet2 &cn, ReferenceJet2 &dn)
{
   // Power-series solution of sn'=cn*dn, cn'=-sn*dn,
   // dn'=-m*sn*cn.  In this test |argument| <= pi/20, so ORDER=24
   // evaluates the Jacobi functions to roundoff while retaining analytic
   // first and second derivatives with respect to the parameter.
   ReferenceJet2 sn_coefficient[ORDER + 1];
   ReferenceJet2 cn_coefficient[ORDER + 1];
   ReferenceJet2 dn_coefficient[ORDER + 1];
   cn_coefficient[0] = 1.0;
   dn_coefficient[0] = 1.0;
   for (int n = 0; n < ORDER; n++)
   {
      ReferenceJet2 cn_dn, sn_dn, sn_cn;
      for (int k = 0; k <= n; k++)
      {
         cn_dn = cn_dn + cn_coefficient[k] * dn_coefficient[n-k];
         sn_dn = sn_dn + sn_coefficient[k] * dn_coefficient[n-k];
         sn_cn = sn_cn + sn_coefficient[k] * cn_coefficient[n-k];
      }
      const real_t scale = 1.0 / (n + 1.0);
      sn_coefficient[n+1] = cn_dn * scale;
      cn_coefficient[n+1] = -sn_dn * scale;
      dn_coefficient[n+1] = -(parameter * sn_cn) * scale;
   }
   sn = sn_coefficient[ORDER];
   cn = cn_coefficient[ORDER];
   dn = dn_coefficient[ORDER];
   for (int n = ORDER - 1; n >= 0; n--)
   {
      sn = sn * argument + sn_coefficient[n];
      cn = cn * argument + cn_coefficient[n];
      dn = dn * argument + dn_coefficient[n];
   }
}

MFEM_HOST_DEVICE void ExactTaylorGreenTrajectory(
   const ReferenceJet2 initial_x[2], real_t time, ReferenceJet2 x[2])
{
   using std::cos;
   using std::sin;
   const ReferenceJet2 sin_x = Sin(initial_x[0] * M_PI);
   const ReferenceJet2 cos_x = Cos(initial_x[0] * M_PI);
   const ReferenceJet2 sin_y = Sin(initial_x[1] * M_PI);
   const ReferenceJet2 cos_y = Cos(initial_x[1] * M_PI);
   const ReferenceJet2 streamline = sin_x * sin_y;
   const ReferenceJet2 parameter = 1.0 - streamline * streamline;

   if (parameter.value <= 1e-28)
   {
      // The vortex center is stationary.  Its flow-map gradient is the
      // rotation generated by the linearized Taylor--Green velocity, and
      // symmetry makes the second derivative zero at the center.
      const real_t cosine = cos(M_PI * time);
      const real_t sine = sin(M_PI * time);
      x[0] = ReferenceJet2(0.5);
      x[1] = ReferenceJet2(0.5);
      x[0].gradient[0] = cosine;
      x[0].gradient[1] = -sine;
      x[1].gradient[0] = sine;
      x[1].gradient[1] = cosine;
      return;
   }

   const ReferenceJet2 modulus = Sqrt(parameter);
   const ReferenceJet2 sn_initial = cos_x / modulus;
   const ReferenceJet2 cn_initial = sin_x * cos_y / modulus;
   const ReferenceJet2 dn_initial = sin_x;
   ReferenceJet2 sn_time, cn_time, dn_time;
   JacobiEllipticSmallArgument(parameter, M_PI * time,
                               sn_time, cn_time, dn_time);
   const ReferenceJet2 denominator =
      1.0 - parameter * sn_initial * sn_initial * sn_time * sn_time;
   const ReferenceJet2 sn =
      (sn_initial * cn_time * dn_time -
       sn_time * cn_initial * dn_initial) / denominator;
   const ReferenceJet2 cn =
      (cn_initial * cn_time +
       sn_initial * sn_time * dn_initial * dn_time) / denominator;
   const ReferenceJet2 dn =
      (dn_initial * dn_time + parameter * sn_initial * sn_time *
       cn_initial * cn_time) / denominator;
   x[0] = Acos(modulus * sn) / M_PI;
   x[1] = Acos(modulus * cn / dn) / M_PI;
}

struct ReferenceJet1
{
   real_t value;
   real_t gradient[2];

   MFEM_HOST_DEVICE ReferenceJet1(real_t value_ = 0.0) : value(value_)
   {
      gradient[0] = gradient[1] = 0.0;
   }
};

MFEM_HOST_DEVICE ReferenceJet1 FirstJet(const ReferenceJet2 &a)
{
   ReferenceJet1 result(a.value);
   result.gradient[0] = a.gradient[0];
   result.gradient[1] = a.gradient[1];
   return result;
}

MFEM_HOST_DEVICE ReferenceJet1 DerivativeJet(const ReferenceJet2 &a,
                                              int direction)
{
   ReferenceJet1 result(a.gradient[direction]);
   result.gradient[0] = a.hessian[direction][0];
   result.gradient[1] = a.hessian[direction][1];
   return result;
}

MFEM_HOST_DEVICE ReferenceJet1 operator+(const ReferenceJet1 &a,
                                          const ReferenceJet1 &b)
{
   ReferenceJet1 result(a.value + b.value);
   for (int i = 0; i < 2; i++)
   {
      result.gradient[i] = a.gradient[i] + b.gradient[i];
   }
   return result;
}

MFEM_HOST_DEVICE ReferenceJet1 operator-(const ReferenceJet1 &a)
{
   ReferenceJet1 result(-a.value);
   for (int i = 0; i < 2; i++) { result.gradient[i] = -a.gradient[i]; }
   return result;
}

MFEM_HOST_DEVICE ReferenceJet1 operator-(const ReferenceJet1 &a,
                                          const ReferenceJet1 &b)
{
   return a + (-b);
}

MFEM_HOST_DEVICE ReferenceJet1 operator*(const ReferenceJet1 &a,
                                          const ReferenceJet1 &b)
{
   ReferenceJet1 result(a.value * b.value);
   for (int i = 0; i < 2; i++)
   {
      result.gradient[i] = a.gradient[i] * b.value +
                           a.value * b.gradient[i];
   }
   return result;
}

MFEM_HOST_DEVICE ReferenceJet1 Inverse(const ReferenceJet1 &a)
{
   ReferenceJet1 result(1.0 / a.value);
   for (int i = 0; i < 2; i++)
   {
      result.gradient[i] = -a.gradient[i] / (a.value * a.value);
   }
   return result;
}

MFEM_HOST_DEVICE ReferenceJet1 operator/(const ReferenceJet1 &a,
                                          const ReferenceJet1 &b)
{
   return a * Inverse(b);
}

template <int DIM>
MFEM_HOST_DEVICE void TaylorGreenLagrangianManufacturedCoefficients(
   const vecd_t<DIM> &initial_position, real_t time, real_t final_time,
   vecd_t<DIM> &lambda_x, vecd_t<DIM> &lambda_v, real_t &lambda_e,
   vecd_t<DIM> &running_x, vecd_t<DIM> &running_v, real_t &running_e)
{
   static_assert(DIM == 2,
                 "the manufactured Lagrangian adjoint is two-dimensional");
   ReferenceJet2 initial_x[DIM] =
   {
      ReferenceJet2::Variable(initial_position(0), 0),
      ReferenceJet2::Variable(initial_position(1), 1)
   };
   ReferenceJet2 exact_x[DIM];
   ExactTaylorGreenTrajectory(initial_x, time, exact_x);

   const ReferenceJet2 sin_x = Sin(exact_x[0] * M_PI);
   const ReferenceJet2 cos_x = Cos(exact_x[0] * M_PI);
   const ReferenceJet2 sin_y = Sin(exact_x[1] * M_PI);
   const ReferenceJet2 cos_y = Cos(exact_x[1] * M_PI);
   ReferenceJet2 exact_v[DIM];
   exact_v[0] = sin_x * cos_y;
   exact_v[1] = -cos_x * sin_y;
   constexpr real_t gamma_minus_one = 2.0 / 3.0;
   const ReferenceJet2 exact_pressure =
      1.0 + 0.25 * (Cos(exact_x[0] * (2.0*M_PI)) +
                    Cos(exact_x[1] * (2.0*M_PI)));
   const ReferenceJet2 exact_energy = exact_pressure / gamma_minus_one;
   const ReferenceJet2 exact_source = 3.0 / 8.0 * M_PI *
      (Cos(exact_x[0] * (3.0*M_PI)) * Cos(exact_x[1] * M_PI) -
       Cos(exact_x[0] * M_PI) * Cos(exact_x[1] * (3.0*M_PI)));

   const ReferenceJet2 sin_initial_x = Sin(initial_x[0] * M_PI);
   const ReferenceJet2 sin_initial_y = Sin(initial_x[1] * M_PI);
   const ReferenceJet2 sin_initial_x2 = sin_initial_x * sin_initial_x;
   const ReferenceJet2 sin_initial_y2 = sin_initial_y * sin_initial_y;
   const ReferenceJet2 phi = sin_initial_x2 * sin_initial_x2 *
                             sin_initial_x2 * sin_initial_y2 *
                             sin_initial_y2 * sin_initial_y2;
   const real_t tau = time / final_time;
   ReferenceJet2 lambda_x_jet[DIM], lambda_v_jet[DIM];
   ReferenceJet2 lambda_e_jet = phi * (1.0 + 3.0*tau);
   for (int i = 0; i < DIM; i++)
   {
      const real_t direction = i == 0 ? 1.0 : -1.0;
      lambda_x_jet[i] = phi * ((1.0 + tau) * direction);
      lambda_v_jet[i] = phi * ((1.0 + 2.0*tau) * direction);
      lambda_x(i) = lambda_x_jet[i].value;
      lambda_v(i) = lambda_v_jet[i].value;
   }
   lambda_e = lambda_e_jet.value;

   ReferenceJet1 deformation[DIM][DIM];
   ReferenceJet1 velocity_gradient[DIM][DIM];
   ReferenceJet1 lambda_v_gradient[DIM][DIM];
   for (int i = 0; i < DIM; i++)
   {
      for (int j = 0; j < DIM; j++)
      {
         deformation[i][j] = DerivativeJet(exact_x[i], j);
         velocity_gradient[i][j] = DerivativeJet(exact_v[i], j);
         lambda_v_gradient[i][j] = DerivativeJet(lambda_v_jet[i], j);
      }
   }
   const ReferenceJet1 determinant =
      deformation[0][0] * deformation[1][1] -
      deformation[0][1] * deformation[1][0];
   ReferenceJet1 inverse_transpose[DIM][DIM];
   inverse_transpose[0][0] = deformation[1][1] / determinant;
   inverse_transpose[0][1] = -deformation[1][0] / determinant;
   inverse_transpose[1][0] = -deformation[0][1] / determinant;
   inverse_transpose[1][1] = deformation[0][0] / determinant;

   const ReferenceJet1 lambda_e_first = FirstJet(lambda_e_jet);
   const ReferenceJet1 energy_first = FirstJet(exact_energy);
   const ReferenceJet1 source_first = FirstJet(exact_source);
   ReferenceJet1 B[DIM][DIM], pressure_stress[DIM][DIM];
   for (int i = 0; i < DIM; i++)
   {
      for (int j = 0; j < DIM; j++)
      {
         B[i][j] = -lambda_v_gradient[i][j] +
                   lambda_e_first * velocity_gradient[i][j];
         pressure_stress[i][j] =
            -gamma_minus_one * energy_first * inverse_transpose[i][j];
      }
   }

   ReferenceJet1 coordinate_flux[DIM][DIM];
   for (int i = 0; i < DIM; i++)
   {
      for (int j = 0; j < DIM; j++)
      {
         ReferenceJet1 geometric_flux;
         for (int k = 0; k < DIM; k++)
         {
            for (int l = 0; l < DIM; l++)
            {
               geometric_flux = geometric_flux +
                  inverse_transpose[i][k] * B[l][k] *
                  inverse_transpose[l][j];
            }
         }
         coordinate_flux[i][j] =
            gamma_minus_one * energy_first * geometric_flux +
            lambda_e_first * source_first * inverse_transpose[i][j];
      }
   }

   real_t source_spatial_gradient[DIM];
   for (int i = 0; i < DIM; i++)
   {
      source_spatial_gradient[i] = 0.0;
      for (int j = 0; j < DIM; j++)
      {
         source_spatial_gradient[i] += inverse_transpose[i][j].value *
                                       exact_source.gradient[j];
      }
   }
   for (int i = 0; i < DIM; i++)
   {
      real_t coordinate_divergence = 0.0;
      real_t velocity_divergence = 0.0;
      for (int j = 0; j < DIM; j++)
      {
         coordinate_divergence += coordinate_flux[i][j].gradient[j];
         const ReferenceJet1 velocity_flux =
            lambda_e_first * pressure_stress[i][j];
         velocity_divergence += velocity_flux.gradient[j];
      }
      const real_t adjoint_x = -coordinate_divergence +
                               lambda_e * source_spatial_gradient[i];
      const real_t adjoint_v = lambda_x(i) - velocity_divergence;
      const real_t direction = i == 0 ? 1.0 : -1.0;
      running_x(i) = -phi.value / final_time * direction - adjoint_x;
      running_v(i) = -2.0 * phi.value / final_time * direction - adjoint_v;
   }
   real_t contraction = 0.0;
   for (int i = 0; i < DIM; i++)
   {
      for (int j = 0; j < DIM; j++)
      {
         contraction += B[i][j].value * inverse_transpose[i][j].value;
      }
   }
   const real_t adjoint_e = -gamma_minus_one * contraction;
   running_e = -3.0 * phi.value / final_time - adjoint_e;
}

template <int DIM>
struct TaylorGreenLagrangianObjectiveQFunction
{
   static_assert(DIM == 2,
                 "the manufactured Lagrangian adjoint is two-dimensional");
   using dvec = vecd_t<DIM, dscalar_t>;
   using vec = vecd_t<DIM>;
   using mat = matd_t<DIM>;

   real_t final_time;
   bool terminal;

   MFEM_HOST_DEVICE inline
   void operator()(const dvec &x, const dvec &v, const dscalar_t &energy,
                   const vec &initial_x, const mat &J0, const real_t &time,
                   const real_t &weight, dscalar_t &value) const
   {
      vec lambda_x, lambda_v, running_x, running_v;
      real_t lambda_e, running_e;
      TaylorGreenLagrangianManufacturedCoefficients<DIM>(
         initial_x, time, final_time, lambda_x, lambda_v, lambda_e,
         running_x, running_v, running_e);
      const real_t material_measure = det(J0) * weight;
      value = (terminal ? lambda_e : running_e) * energy;
      for (int i = 0; i < DIM; i++)
      {
         value += (terminal ? lambda_x(i) : running_x(i)) * x(i) +
                  (terminal ? lambda_v(i) : running_v(i)) * v(i);
      }
      value *= material_measure;
   }
};

std::shared_ptr<DifferentiableOperator> CreateLagrangianObjective(
   ParFiniteElementSpace &H1, ParFiniteElementSpace &L2,
   const IntegrationRule &ir, int qfunction_backend,
   real_t final_time, bool terminal)
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
         Value<COORDINATES>{}, Value<VELOCITY>{},
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
      int qfunction_backend, real_t final_time) :
      initial_coordinates(initial_x), time_values(L2.GetTrueVSize()),
      output_ones(L2.GetTrueVSize()),
      terminal(CreateLagrangianObjective(H1, L2, ir, qfunction_backend,
                                         final_time, true)),
      running(CreateLagrangianObjective(H1, L2, ir, qfunction_backend,
                                        final_time, false))
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
      H1, L2, initial_x, ir, qfunction_backend, final_time);
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
