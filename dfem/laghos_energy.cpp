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
std::shared_ptr<DifferentiableOperator> CreateEnergyConservationOperator(
   ParFiniteElementSpace &H1,
   ParFiniteElementSpace &L2,
   ParGridFunction &material_gf,
   const IntegrationRule &ir,
   const Array<int> &all_domain_attr,
   real_t *d_external_data)
{
   ParMesh &mesh = *H1.GetParMesh();

   future::tuple energy_conservation_mf_kernel_oo = {Value<SPECIFIC_INTERNAL_ENERGY>{}};
   // <sigma, grad(v) * inv(J) * phi> * det(J) * w
   // <sigma(J^-T det(J) w), grad(v) * inv(J)>
   std::vector energy_conservation_mf_outputs =
   {
      FieldDescriptor{SPECIFIC_INTERNAL_ENERGY, &L2}
   };
   auto energy_conservation_mf =
      std::make_shared<DifferentiableOperator>(
         QDataInputs(H1, L2, material_gf), energy_conservation_mf_outputs,
         mesh);
   struct EnergyConservationQFunction
   {
      using matd = matd_t<DIM>;
      using dmatd = matd_t<DIM, dscalar_t>;
      const real_t *external_data;
      MFEM_HOST_DEVICE inline
      void operator()(
         const dmatd &dvdxi,
         const real_t &rho0,
         const matd &J0,
         const dmatd &J,
         const real_t &gamma,
         const dscalar_t &E,
         const real_t &w,
         dscalar_t &de) const
      {
         real_t dt_est = std::numeric_limits<real_t>::infinity();
         const auto stressJiT =
            qdata_setup<DIM>(dvdxi, rho0, J0, J, gamma, E, w,
                             external_data, dt_est);
         de = ddot(stressJiT, dvdxi);
      }
   } energy_conservation_qf{d_external_data};
   auto derivatives =
      std::integer_sequence<size_t, VELOCITY, COORDINATES, SPECIFIC_INTERNAL_ENERGY> {};
   energy_conservation_mf->AddDomainIntegrator<future::LocalQFBackend,
                                               JACOBIAN_KERNELS>(
      energy_conservation_qf, QDataFields(),
      energy_conservation_mf_kernel_oo, ir, all_domain_attr, derivatives);
   return energy_conservation_mf;
}

// Explicit instantiations for the 2D and 3D meshes

template std::shared_ptr<DifferentiableOperator>
CreateEnergyConservationOperator<2>(
   ParFiniteElementSpace &H1,
   ParFiniteElementSpace &L2,
   ParGridFunction &material_gf,
   const IntegrationRule &ir,
   const Array<int> &all_domain_attr,
   real_t *d_external_data);

template std::shared_ptr<DifferentiableOperator>
CreateEnergyConservationOperator<3>(
   ParFiniteElementSpace &H1,
   ParFiniteElementSpace &L2,
   ParGridFunction &material_gf,
   const IntegrationRule &ir,
   const Array<int> &all_domain_attr,
   real_t *d_external_data);

} // namespace mfem::hydrodynamics
