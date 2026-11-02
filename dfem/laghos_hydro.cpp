// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include "laghos_qfunctions.hpp"

namespace mfem::hydrodynamics
{

// ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
template <int DIM>
LagrangianHydroOperator *CreateLagrangianHydroOperator(
   ParFiniteElementSpace &H1,
   ParFiniteElementSpace &L2,
   Array<int> &ess_tdof,
   FunctionCoefficient &rho0_coeff,
   ParGridFunction &x0_gf,
   ParGridFunction &rho0_gf,
   ParGridFunction &material_gf,
   Vector &external_data,
   const IntegrationRule &ir,
   const bool &fd_gradient,
   const int &dump_jacobians,
   const int &nonlinear_maximum_iterations,
   const real_t &nonlinear_relative_tolerance,
   const int &krylov_maximum_iterations,
   const int &preconditioner_lag,
   const LagrangianHydroOperator::PRECONDITIONER_TYPE &preconditioner_type)
{
   using matd = matd_t<DIM>;
   ParMesh &mesh = *H1.GetParMesh();

   auto qdata = std::make_shared<QuadratureData>(mesh, ir);

   int ne_loc = mesh.GetNE(), ne_global = 0;
   real_t vol_loc = 0.0, vol_global = 0.0;
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      vol_loc += mesh.GetElementVolume(e);
   }
   MPI_Allreduce(&vol_loc, &vol_global, 1, MPI_DOUBLE, MPI_SUM, mesh.GetComm());
   MPI_Allreduce(&ne_loc, &ne_global, 1, MPI_INT, MPI_SUM, mesh.GetComm());

   real_t &h0 = external_data[EXT_DATA_IDX::H0];
   switch (mesh.GetElementBaseGeometry(0))
   {
      case Geometry::SEGMENT: h0 = vol_global / ne_global; break;
      case Geometry::SQUARE: h0 = sqrt(vol_global / ne_global); break;
      case Geometry::TRIANGLE: h0 = sqrt(2.0 * vol_global / ne_global); break;
      case Geometry::CUBE: h0 = pow(vol_global / ne_global, 1./3.); break;
      case Geometry::TETRAHEDRON: h0 = pow(6.0 * vol_global / ne_global,
                                              1./3.); break;
      default: MFEM_ABORT("Unknown zone type!");
   }
   h0 /= (double) H1.GetOrder(0);

   qdata->dt_est = std::numeric_limits<real_t>::infinity();

   auto d_external_data = external_data.ReadWrite();

   Array<int> all_domain_attr(mesh.attributes.Max());
   all_domain_attr = 1;

   // Output on a quadrature space
   std::shared_ptr<DifferentiableOperator> dt_est_mf;
   {
      future::tuple dt_est_kernel_ao =
      {
         Gradient<VELOCITY>{},
         Value<DENSITY0>{},
         Gradient<COORDINATES0>{},
         Gradient<COORDINATES>{},
         Value<MATERIAL>{},
         Value<SPECIFIC_INTERNAL_ENERGY>{},
         Weight{}
      };
      future::tuple dt_est_kernel_oo = {Identity<DT_EST>{}};
      std::vector dt_est_inputs =
      {
         FieldDescriptor{VELOCITY, &H1},
         FieldDescriptor{DENSITY0, &L2},
         FieldDescriptor{COORDINATES0, &H1},
         FieldDescriptor{COORDINATES, &H1},
         FieldDescriptor{MATERIAL, material_gf.ParFESpace()},
         FieldDescriptor{SPECIFIC_INTERNAL_ENERGY, &L2},
      };
      std::vector dt_est_outputs =
      {
         FieldDescriptor{DT_EST, &qdata->R}
      };
      dt_est_mf = std::make_shared<DifferentiableOperator>(
                     dt_est_inputs, dt_est_outputs, mesh);
      dt_est_mf->SetMultLevel(DifferentiableOperator::LVECTOR);
      struct TimeStepEstimateQFunction
      {
         real_t *external_data;

         MFEM_HOST_DEVICE inline
         void operator()(const matd &dvdxi, const real_t &rho0, const matd &J0,
                         const matd &J, const real_t &gamma, const real_t &E,
                         const real_t &w, real_t &dt_est) const
         {
            dt_est = std::numeric_limits<real_t>::infinity();
            // Reuse qdata_setup() for the dt estimate logic; ignore its stress output.
            (void)qdata_setup<DIM>(
               dvdxi, rho0, J0, J, gamma, E, w,
               external_data[EXT_DATA_IDX::H0],
               external_data[EXT_DATA_IDX::ORDER_VEL],
               external_data[EXT_DATA_IDX::CFL],
               static_cast<bool>(external_data[EXT_DATA_IDX::VISCOSITY_FLAG]),
               static_cast<int>(external_data[EXT_DATA_IDX::VISCOSITY_TYPE]),
               external_data[EXT_DATA_IDX::VISC_Q1],
               external_data[EXT_DATA_IDX::VISC_Q2],
               dt_est);
         }

      } dt_est_qf({d_external_data});
      dt_est_mf->AddDomainIntegrator(dt_est_qf, dt_est_kernel_ao,
                                     dt_est_kernel_oo,
                                     ir,
                                     all_domain_attr);
   }

   // Output on a quadrature space
   std::shared_ptr<DifferentiableOperator> update_qdata;
   {
      future::tuple update_qdata_kernel_ao =
      {
         Gradient<VELOCITY>{},
         Value<DENSITY0>{},
         Gradient<COORDINATES0>{},
         Gradient<COORDINATES>{},
         Value<MATERIAL>{},
         Value<SPECIFIC_INTERNAL_ENERGY>{},
         Weight{}
      };
      future::tuple update_qdata_kernel_oo = {Identity<STRESS_TENSOR>{}};
      std::vector<FieldDescriptor> update_qdata_inputs =
      {
         {VELOCITY, &H1},
         {DENSITY0, &L2},
         {COORDINATES0, &H1},
         {COORDINATES, &H1},
         {MATERIAL, material_gf.ParFESpace()},
         {SPECIFIC_INTERNAL_ENERGY, &L2},
      };
      std::vector<FieldDescriptor> update_qdata_outputs =
      {
         {STRESS_TENSOR, &qdata->StressSpace}
      };
      update_qdata = std::make_shared<DifferentiableOperator>(
                        update_qdata_inputs, update_qdata_outputs, mesh);
      update_qdata->SetMultLevel(DifferentiableOperator::LVECTOR);
      struct UpdateQuadratureDataQFunction
      {
         real_t *external_data;
         MFEM_HOST_DEVICE inline
         void operator()(const matd &dvdxi, const real_t &rho0, const matd &J0,
                         const matd &J, const real_t &gamma, const real_t &E,
                         const real_t &w, matd &stressJiT) const
         {
            // Do not update the global dt estimate here; that will be computed
            // through a proper reduction over quadrature points.
            real_t dt_est_dummy = std::numeric_limits<real_t>::infinity();
            stressJiT =
               qdata_setup<DIM>(
                  dvdxi, rho0, J0, J, gamma, E, w,
                  external_data[EXT_DATA_IDX::H0],
                  external_data[EXT_DATA_IDX::ORDER_VEL],
                  external_data[EXT_DATA_IDX::CFL],
                  static_cast<bool>(external_data[EXT_DATA_IDX::VISCOSITY_FLAG]),
                  static_cast<int>(external_data[EXT_DATA_IDX::VISCOSITY_TYPE]),
                  external_data[EXT_DATA_IDX::VISC_Q1],
                  external_data[EXT_DATA_IDX::VISC_Q2],
                  dt_est_dummy);
         }
      } update_qdata_qf({d_external_data});
      update_qdata->AddDomainIntegrator(update_qdata_qf, update_qdata_kernel_ao,
                                        update_qdata_kernel_oo,
                                        ir,
                                        all_domain_attr);
   }

   // Create momentum operator
   auto momentum_mf = CreateMomentumOperator<DIM>(H1, L2, material_gf, ir,
                                                  all_domain_attr,
                                                  d_external_data);

   std::shared_ptr<DifferentiableOperator> momentum_pa;
   {
      future::tuple momentum_pa_kernel_ao = {Identity<STRESS_TENSOR>{}};
      future::tuple momentum_pa_kernel_oo = {Gradient<VELOCITY>{}};
      std::vector<FieldDescriptor> momentum_pa_inputs = {{STRESS_TENSOR, &qdata->StressSpace}};
      std::vector<FieldDescriptor> momentum_pa_outputs = {{VELOCITY, &H1}};
      momentum_pa = std::make_shared<DifferentiableOperator>(momentum_pa_inputs,
                                                             momentum_pa_outputs, mesh);
      struct MomentumPAQFunction
      {
         MFEM_HOST_DEVICE inline
         void operator()(const matd &stressJiT, matd &f) const { f = stressJiT; }
      } momentum_pa_qf;
      momentum_pa->AddDomainIntegrator(momentum_pa_qf, momentum_pa_kernel_ao,
                                       momentum_pa_kernel_oo, ir, all_domain_attr);
   }

   // Create energy conservation operator
   auto energy_conservation_mf =
      CreateEnergyConservationOperator<DIM>(H1, L2, material_gf, ir,
                                            all_domain_attr, d_external_data);

   std::shared_ptr<DifferentiableOperator> energy_conservation_pa;
   {
      future::tuple energy_conservation_pa_kernel_ao = {Gradient<VELOCITY>{}, Identity<STRESS_TENSOR>{}};
      future::tuple energy_conservation_pa_kernel_oo = {Value<SPECIFIC_INTERNAL_ENERGY>{}};
      std::vector<FieldDescriptor> energy_conservation_pa_inputs =
      {
         {VELOCITY, &H1},
         {STRESS_TENSOR, &qdata->StressSpace}
      };
      std::vector<FieldDescriptor> energy_conservation_pa_outputs =
      {
         {SPECIFIC_INTERNAL_ENERGY, &L2}
      };
      energy_conservation_pa =
         std::make_shared<DifferentiableOperator>(energy_conservation_pa_inputs,
                                                  energy_conservation_pa_outputs, mesh);
      energy_conservation_pa->SetMultLevel(DifferentiableOperator::LVECTOR);
      struct EnergyConservationPAQFunction
      {
         MFEM_HOST_DEVICE inline
         void operator()(const matd &dvdxi, const matd &stressJiT, real_t &f) const
         {
            f = ddot(stressJiT, dvdxi);
         }
      } energy_conservation_pa_qf;
      energy_conservation_pa->AddDomainIntegrator(
         energy_conservation_pa_qf, energy_conservation_pa_kernel_ao,
         energy_conservation_pa_kernel_oo, ir, all_domain_attr);
   }

   // Create total internal energy operator
   std::shared_ptr<DifferentiableOperator> total_internal_energy_mf;
   {
      future::tuple total_internal_energy_kernel_ao =
      {
         Value<SPECIFIC_INTERNAL_ENERGY>{},
         Value<DENSITY0>{},
         Gradient<COORDINATES0>{},
         Weight{}
      };
      future::tuple total_internal_energy_kernel_oo = {Value<SPECIFIC_INTERNAL_ENERGY>{}};
      std::vector total_internal_energy_inputs =
      {
         FieldDescriptor{SPECIFIC_INTERNAL_ENERGY, &L2},
         FieldDescriptor{DENSITY0, &L2},
         FieldDescriptor{COORDINATES0, &H1}
      };
      std::vector total_internal_energy_outputs =
      {
         FieldDescriptor{SPECIFIC_INTERNAL_ENERGY, &L2}
      };
      total_internal_energy_mf =
         std::make_shared<DifferentiableOperator>(
            total_internal_energy_inputs,
            total_internal_energy_outputs,
            mesh);
      total_internal_energy_mf->SetMultLevel(DifferentiableOperator::LVECTOR);
      struct TotalInternalEnergyQFunction
      {
         MFEM_HOST_DEVICE inline
         void operator()(const real_t &E,
                         const real_t &rho0,
                         const matd &J0,
                         const real_t &w,
                         real_t &ie) const
         {
            ie = rho0 * E * det(J0) * w;
         }
      } total_internal_energy_qf;
      total_internal_energy_mf->AddDomainIntegrator(
         total_internal_energy_qf, total_internal_energy_kernel_ao,
         total_internal_energy_kernel_oo, ir, all_domain_attr);
   }

   // Create total kinetic energy operator
   std::shared_ptr<DifferentiableOperator> total_kinetic_energy_mf;
   {
      future::tuple total_kinetic_energy_kernel_ao =
      {
         Value<VELOCITY>{},
         Value<DENSITY0>{},
         Gradient<COORDINATES0>{},
         Weight{}
      };
      future::tuple total_kinetic_energy_kernel_oo = {Value<DENSITY0>{}};
      std::vector total_kinetic_energy_inputs =
      {
         FieldDescriptor{VELOCITY, &H1},
         FieldDescriptor{DENSITY0, &L2},
         FieldDescriptor{COORDINATES0, &H1}
      };
      // The kinetic energy density is an L2 field
      std::vector total_kinetic_energy_outputs =
      {
         FieldDescriptor{DENSITY0, &L2}
      };
      total_kinetic_energy_mf =
         std::make_shared<DifferentiableOperator>(
            total_kinetic_energy_inputs,
            total_kinetic_energy_outputs, mesh);
      total_kinetic_energy_mf->SetMultLevel(DifferentiableOperator::LVECTOR);
      struct TotalKineticEnergyQFunction
      {
         MFEM_HOST_DEVICE inline
         void operator()(const vecd_t<DIM> &v,
                         const real_t &rho0,
                         const matd &J0,
                         const real_t &w,
                         real_t &ke) const
         {
            ke = rho0 * 0.5 * v * v * det(J0) * w;
         }
      } total_kinetic_energy_qf;
      total_kinetic_energy_mf->AddDomainIntegrator(
         total_kinetic_energy_qf, total_kinetic_energy_kernel_ao,
         total_kinetic_energy_kernel_oo, ir, all_domain_attr);
   }

   // Create density operator
   std::shared_ptr<DifferentiableOperator> density_mf;
   {
      future::tuple density_kernel_ao =
      {
         Value<DENSITY0>{},
         Gradient<COORDINATES0>{},
         Weight{}
      };
      future::tuple density_kernel_oo = {Value<DENSITY0>{}};
      std::vector density_inputs =
      {
         FieldDescriptor{DENSITY0, &L2},
         FieldDescriptor{COORDINATES0, &H1}
      };
      std::vector density_outputs =
      {
         FieldDescriptor{DENSITY0, &L2}
      };
      density_mf = std::make_shared<DifferentiableOperator>(
                      density_inputs, density_outputs, mesh);
      density_mf->SetMultLevel(DifferentiableOperator::LVECTOR);
      struct DensityQFunction
      {
         MFEM_HOST_DEVICE inline
         void operator()(const real_t &rho0,
                         const matd &J0,
                         const real_t &w,
                         real_t &m) const
         {
            m = rho0 * det(J0) * w;
         }
      } density_qf;
      density_mf->AddDomainIntegrator(density_qf, density_kernel_ao,
                                      density_kernel_oo, ir, all_domain_attr);
   }

   // Create taylor source operator
   std::shared_ptr<DifferentiableOperator> taylor_source_mf;
   {
      future::tuple taylor_source_kernel_ao =
      {
         Value<COORDINATES>{},
         Gradient<COORDINATES>{},
         Weight{}
      };
      future::tuple taylor_source_kernel_oo = {Value<SPECIFIC_INTERNAL_ENERGY>{}};
      std::vector taylor_source_inputs = { FieldDescriptor{COORDINATES, &H1} };
      std::vector taylor_source_outputs =
      {
         FieldDescriptor{SPECIFIC_INTERNAL_ENERGY, &L2}
      };
      taylor_source_mf = std::make_shared<DifferentiableOperator>(
                            taylor_source_inputs, taylor_source_outputs, mesh);
      struct TaylorSourceQFunction
      {
         using vecd = vecd_t<DIM, dscalar_t>;
         using matd = matd_t<DIM, dscalar_t>;

         MFEM_HOST_DEVICE inline
         void operator()(const vecd &x, const matd &J, const real_t &w,
                         dscalar_t &s) const
         {
            const auto f = 3.0 / 8.0 * M_PI *
                           (cos(3.0*M_PI*x(0)) * cos(M_PI*x(1)) -
                            cos(M_PI*x(0)) * cos(3.0*M_PI*x(1)));

            s = f * det(J) * w;
         }
      } taylor_source_qf;
      auto derivatives = std::integer_sequence<size_t, COORDINATES> {};
      // Only the matrix-free action of dTaylorSourcedx is used
      taylor_source_mf->AddDomainIntegrator<future::LocalQFBackend,
                                            future::DerivativeKernels::Action>(
         taylor_source_qf, taylor_source_kernel_ao, taylor_source_kernel_oo,
         ir, all_domain_attr, derivatives);
   }

   return new LagrangianHydroOperator(
             H1,
             L2,
             ess_tdof,
             ir,
             rho0_coeff,
             x0_gf,
             rho0_gf,
             material_gf,
             update_qdata,
             dt_est_mf,
             momentum_mf,
             momentum_pa,
             energy_conservation_mf,
             energy_conservation_pa,
             total_internal_energy_mf,
             total_kinetic_energy_mf,
             density_mf,
             taylor_source_mf,
             qdata,
             fd_gradient,
             dump_jacobians,
             nonlinear_maximum_iterations,
             nonlinear_relative_tolerance,
             krylov_maximum_iterations,
             preconditioner_lag,
             preconditioner_type,
             external_data);
}

// Explicit instantiations for the 2D and 3D meshes

template LagrangianHydroOperator *CreateLagrangianHydroOperator<2>(
   ParFiniteElementSpace &H1,
   ParFiniteElementSpace &L2,
   Array<int> &ess_tdof,
   FunctionCoefficient &rho0_coeff,
   ParGridFunction &x0_gf,
   ParGridFunction &rho0_gf,
   ParGridFunction &material_gf,
   Vector &external_data,
   const IntegrationRule &ir,
   const bool &fd_gradient,
   const int &dump_jacobians,
   const int &nonlinear_maximum_iterations,
   const real_t &nonlinear_relative_tolerance,
   const int &krylov_maximum_iterations,
   const int &preconditioner_lag,
   const LagrangianHydroOperator::PRECONDITIONER_TYPE &preconditioner_type);

template LagrangianHydroOperator *CreateLagrangianHydroOperator<3>(
   ParFiniteElementSpace &H1,
   ParFiniteElementSpace &L2,
   Array<int> &ess_tdof,
   FunctionCoefficient &rho0_coeff,
   ParGridFunction &x0_gf,
   ParGridFunction &rho0_gf,
   ParGridFunction &material_gf,
   Vector &external_data,
   const IntegrationRule &ir,
   const bool &fd_gradient,
   const int &dump_jacobians,
   const int &nonlinear_maximum_iterations,
   const real_t &nonlinear_relative_tolerance,
   const int &krylov_maximum_iterations,
   const int &preconditioner_lag,
   const LagrangianHydroOperator::PRECONDITIONER_TYPE &preconditioner_type);

} // namespace mfem::hydrodynamics
