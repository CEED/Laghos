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

#include "laghos_qf.hpp"
#include "laghos_solver.hpp"

namespace mfem::hydrodynamics
{

template <int DIM>
struct HydroQFunctions
{
   using matd = matd_t<DIM>;

   struct UpdateQuadratureDataQFunction
   {
      QConstants c;
      MFEM_HOST_DEVICE inline
      void operator()(const matd &dvdxi, const real_t &rho0, const matd &J0,
                      const matd &J, const real_t &gamma, const real_t &E,
                      const real_t &w, matd &stressJiT,
                      real_t &dt_est) const
      {
         const auto [stress, visc_coeff, cs, rho] =
            StressSetup<DIM>(c, dvdxi, rho0, J0, J, gamma, E, w);
         stressJiT = stress;
         dt_est = TimeStepEstimate<DIM>(c, J, rho, cs, visc_coeff);
      }
   };

   struct MomentumPAQFunction
   {
      MFEM_HOST_DEVICE inline
      void operator()(const matd &stressJiT, matd &f) const { f = stressJiT; }
   };

   struct EnergyConservationPAQFunction
   {
      MFEM_HOST_DEVICE inline
      void operator()(const matd &dvdxi, const matd &stressJiT, real_t &f) const
      {
         f = ddot(stressJiT, dvdxi);
      }
   };

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
   };

   struct TotalKineticEnergyQFunction
   {
      MFEM_HOST_DEVICE inline
      void operator()(const vecd_t<DIM, dscalar_t> &v,
                      const real_t &rho0,
                      const matd &J0,
                      const real_t &w,
                      dscalar_t &ke) const
      {
         ke = rho0 * 0.5 * v * v * det(J0) * w;
      }
   };

   struct DensityQFunction
   {
      MFEM_HOST_DEVICE inline
      void operator()(const real_t &rho0,
                      const matd &J0,
                      const real_t &w,
                      real_t &m) const { m = rho0 * det(J0) * w; }
   };

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
   };
};

// Initial zone size of the mesh, over the order of the velocity space
real_t InitialMeshSize(ParMesh &mesh, const int order)
{
   int ne_loc = mesh.GetNE(), ne_global = 0;
   real_t vol_loc = 0.0, vol_global = 0.0;
   for (int e = 0; e < ne_loc; e++)
   {
      vol_loc += mesh.GetElementVolume(e);
   }
   MPI_Allreduce(&vol_loc, &vol_global, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_SUM, mesh.GetComm());
   MPI_Allreduce(&ne_loc, &ne_global, 1, MPI_INT, MPI_SUM, mesh.GetComm());

   real_t h = 0.0;
   switch (mesh.GetTypicalElementGeometry())
   {
      case Geometry::SQUARE: h = sqrt(vol_global / ne_global); break;
      case Geometry::TRIANGLE: h = sqrt(2.0 * vol_global / ne_global); break;
      case Geometry::CUBE: h = pow(vol_global / ne_global, 1./3.); break;
      case Geometry::TETRAHEDRON:
         h = pow(6.0 * vol_global / ne_global, 1./3.); break;
      default: MFEM_ABORT("Unknown zone type!");
   }
   return h / order;
}

template <int DIM, typename backend_t>
static LagrangianHydroOperator *CreateLagrangianHydroOperator(
   ParFiniteElementSpace &H1,
   ParFiniteElementSpace &L2,
   Array<int> &ess_tdof,
   FunctionCoefficient &rho0_coeff,
   ParGridFunction &x0_gf,
   ParGridFunction &rho0_gf,
   ParGridFunction &material_gf,
   const QConstants &qc,
   const IntegrationRule &ir,
   const LagrangianHydroOperator::Options &options)
{
   ParMesh &mesh = *H1.GetParMesh();

   auto qdata = std::make_shared<QuadratureData>(mesh, ir);
   qdata->dt_est = std::numeric_limits<real_t>::infinity();

   Array<int> all_domain_attr(mesh.attributes.Max());
   all_domain_attr = 1;

   // Stress and time step estimate, outputs on quadrature spaces
   std::shared_ptr<DifferentiableOperator> update_qdata;
   {
      future::tuple update_qdata_kernel_oo = {Identity<STRESS_TENSOR>{},
                                              Identity<DT_EST>{}
                                             };
      std::vector<FieldDescriptor> update_qdata_outputs =
      {
         {STRESS_TENSOR, &qdata->StressSpace}, {DT_EST, &qdata->R}
      };
      update_qdata = std::make_shared<DifferentiableOperator>(
                        QDataInputs(H1, L2, material_gf), update_qdata_outputs,
                        mesh);
      update_qdata->SetMultLevel(DifferentiableOperator::LVECTOR);
      typename HydroQFunctions<DIM>::UpdateQuadratureDataQFunction
      update_qdata_qf{qc};
      update_qdata->AddDomainIntegrator<backend_t>(
         update_qdata_qf, QDataFields(), update_qdata_kernel_oo, ir,
         all_domain_attr);
   }

   // Create momentum operator
   auto momentum_mf =
      CreateMomentumOperator(DIM, options.qfunction_backend, H1, L2,
                             material_gf, ir, all_domain_attr, qc);

   std::shared_ptr<DifferentiableOperator> momentum_pa;
   {
      future::tuple momentum_pa_kernel_ao = {Identity<STRESS_TENSOR>{}};
      future::tuple momentum_pa_kernel_oo = {Gradient<VELOCITY>{}};
      std::vector<FieldDescriptor> momentum_pa_inputs =
      {
         {STRESS_TENSOR, &qdata->StressSpace}
      };
      std::vector<FieldDescriptor> momentum_pa_outputs = {{VELOCITY, &H1}};
      momentum_pa = std::make_shared<DifferentiableOperator>(
                       momentum_pa_inputs, momentum_pa_outputs, mesh);
      typename HydroQFunctions<DIM>::MomentumPAQFunction
      momentum_pa_qf;
      momentum_pa->AddDomainIntegrator<backend_t>(
         momentum_pa_qf, momentum_pa_kernel_ao, momentum_pa_kernel_oo, ir,
         all_domain_attr);
   }

   // Create energy conservation operator
   auto energy_conservation_mf =
      CreateEnergyOperator(DIM, options.qfunction_backend, H1, L2,
                           material_gf, ir, all_domain_attr, qc);

   std::shared_ptr<DifferentiableOperator> energy_conservation_pa;
   {
      future::tuple energy_conservation_pa_kernel_ao =
      {
         Gradient<VELOCITY>{}, Identity<STRESS_TENSOR>{}
      };
      future::tuple energy_conservation_pa_kernel_oo =
      {
         Value<INTERNAL_ENERGY>{}
      };
      std::vector<FieldDescriptor> energy_conservation_pa_inputs =
      {
         {VELOCITY, &H1},
         {STRESS_TENSOR, &qdata->StressSpace}
      };
      std::vector<FieldDescriptor> energy_conservation_pa_outputs =
      {
         {INTERNAL_ENERGY, &L2}
      };
      energy_conservation_pa =
         std::make_shared<DifferentiableOperator>(
            energy_conservation_pa_inputs, energy_conservation_pa_outputs,
            mesh);
      // LVECTOR as Mult passes the v L-vector, momentum_pa stays at TVECTOR
      energy_conservation_pa->SetMultLevel(DifferentiableOperator::LVECTOR);
      typename HydroQFunctions<DIM>::EnergyConservationPAQFunction
      energy_conservation_pa_qf;
      energy_conservation_pa->AddDomainIntegrator<backend_t>(
         energy_conservation_pa_qf, energy_conservation_pa_kernel_ao,
         energy_conservation_pa_kernel_oo, ir, all_domain_attr);
   }

   // Create total internal energy operator
   std::shared_ptr<DifferentiableOperator> total_internal_energy_mf;
   {
      future::tuple total_internal_energy_kernel_ao =
      {
         Value<INTERNAL_ENERGY>{},
         Value<DENSITY0>{},
         Gradient<COORDINATES0>{},
         Weight{}
      };
      future::tuple total_internal_energy_kernel_oo =
      {
         Value<INTERNAL_ENERGY>{}
      };
      std::vector total_internal_energy_inputs =
      {
         FieldDescriptor{INTERNAL_ENERGY, &L2},
         FieldDescriptor{DENSITY0, &L2},
         FieldDescriptor{COORDINATES0, &H1}
      };
      std::vector total_internal_energy_outputs =
      {
         FieldDescriptor{INTERNAL_ENERGY, &L2}
      };
      total_internal_energy_mf =
         std::make_shared<DifferentiableOperator>(
            total_internal_energy_inputs,
            total_internal_energy_outputs,
            mesh);
      total_internal_energy_mf->SetMultLevel(DifferentiableOperator::LVECTOR);
      typename HydroQFunctions<DIM>::TotalInternalEnergyQFunction
      total_internal_energy_qf;
      total_internal_energy_mf->AddDomainIntegrator<backend_t>(
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
      typename HydroQFunctions<DIM>::TotalKineticEnergyQFunction
      total_kinetic_energy_qf;
      auto derivatives = std::integer_sequence<size_t, VELOCITY> {};
      total_kinetic_energy_mf->AddDomainIntegrator<
      backend_t, future::DerivativeKernels::ApplyTranspose>(
         total_kinetic_energy_qf, total_kinetic_energy_kernel_ao,
         total_kinetic_energy_kernel_oo, ir, all_domain_attr, derivatives);
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
      typename HydroQFunctions<DIM>::DensityQFunction
      density_qf;
      density_mf->AddDomainIntegrator<backend_t>(
         density_qf, density_kernel_ao, density_kernel_oo, ir, all_domain_attr);
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
      future::tuple taylor_source_kernel_oo = {Value<INTERNAL_ENERGY>{}};
      std::vector taylor_source_inputs = { FieldDescriptor{COORDINATES, &H1} };
      std::vector taylor_source_outputs =
      {
         FieldDescriptor{INTERNAL_ENERGY, &L2}
      };
      taylor_source_mf = std::make_shared<DifferentiableOperator>(
                            taylor_source_inputs, taylor_source_outputs, mesh);
      typename HydroQFunctions<DIM>::TaylorSourceQFunction
      taylor_source_qf;
      auto derivatives = std::integer_sequence<size_t, COORDINATES> {};
      // The implicit solver uses the action; the explicit adjoint uses its
      // transpose.
      taylor_source_mf->AddDomainIntegrator<backend_t,
                       future::DerivativeKernels::Action |
                       future::DerivativeKernels::ApplyTranspose>(
                          taylor_source_qf, taylor_source_kernel_ao,
                          taylor_source_kernel_oo, ir, all_domain_attr,
                          derivatives);
   }

   return new LagrangianHydroOperator(H1, L2, ess_tdof, ir,
                                      rho0_coeff, x0_gf, rho0_gf, material_gf,
                                      update_qdata, momentum_mf, momentum_pa,
                                      energy_conservation_mf,
                                      energy_conservation_pa,
                                      total_internal_energy_mf,
                                      total_kinetic_energy_mf,
                                      density_mf, taylor_source_mf,
                                      qdata, options);
}

LagrangianHydroOperator *CreateLagrangianHydroOperator(
   ParFiniteElementSpace &H1,
   ParFiniteElementSpace &L2,
   Array<int> &ess_tdof,
   FunctionCoefficient &rho0_coeff,
   ParGridFunction &x0_gf,
   ParGridFunction &rho0_gf,
   ParGridFunction &material_gf,
   const QConstants &qc,
   const IntegrationRule &ir,
   const LagrangianHydroOperator::Options &options)
{
   const int dim = H1.GetMesh()->Dimension();
   return Dispatch(dim, options.qfunction_backend, [&](auto DIM, auto backend)
   {
      return CreateLagrangianHydroOperator<DIM, decltype(backend)>(
                H1, L2, ess_tdof, rho0_coeff, x0_gf, rho0_gf, material_gf,
                qc, ir, options);
   });
}

} // namespace mfem::hydrodynamics
