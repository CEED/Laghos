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

namespace mfem::hydrodynamics
{

template <int DIM>
std::shared_ptr<DifferentiableOperator> CreateMomentumOperator(
   ParFiniteElementSpace &H1,
   ParFiniteElementSpace &L2,
   ParGridFunction &material_gf,
   const IntegrationRule &ir,
   const Array<int> &all_domain_attr,
   real_t *d_external_data)
{
   using matd = matd_t<DIM>;
   ParMesh &mesh = *H1.GetParMesh();

   future::tuple momentum_mf_kernel_oo = {Gradient<VELOCITY>{}};
   // <sigma, grad(w) * J^-T> * det(J) * weights
   // <sigma(J^-T det(J) weights), grad(w)>
   std::vector momentum_mf_outputs =
   {
      FieldDescriptor{VELOCITY, &H1}
   };
   auto momentum_mf = std::make_shared<DifferentiableOperator>(
                         QDataInputs(H1, L2, material_gf), momentum_mf_outputs,
                         mesh);
   struct MomentumQFunction
   {
      using dmatd = matd_t<DIM, dscalar_t>;
      real_t *external_data;
      MFEM_HOST_DEVICE inline
      void operator()(const dmatd &dvdxi, const real_t &rho0, const matd &J0,
                      const dmatd &J, const real_t &gamma, const dscalar_t &E,
                      const real_t &w, dmatd &stressJiT) const
      {
         real_t dt_est = std::numeric_limits<real_t>::infinity();
         stressJiT =
            qdata_setup<DIM>(dvdxi, rho0, J0, J, gamma, E, w,
                             external_data, dt_est);
      }
   } momentum_qf{d_external_data};
   auto derivatives =
      std::integer_sequence<size_t, VELOCITY, COORDINATES, SPECIFIC_INTERNAL_ENERGY> {};
   momentum_mf->AddDomainIntegrator<future::LocalQFBackend, JACOBIAN_KERNELS>(
      momentum_qf, QDataFields(), momentum_mf_kernel_oo, ir,
      all_domain_attr, derivatives);
   return momentum_mf;
}

template std::shared_ptr<DifferentiableOperator> CreateMomentumOperator<2>(
   ParFiniteElementSpace &H1,
   ParFiniteElementSpace &L2,
   ParGridFunction &material_gf,
   const IntegrationRule &ir,
   const Array<int> &all_domain_attr,
   real_t *d_external_data);

template std::shared_ptr<DifferentiableOperator> CreateMomentumOperator<3>(
   ParFiniteElementSpace &H1,
   ParFiniteElementSpace &L2,
   ParGridFunction &material_gf,
   const IntegrationRule &ir,
   const Array<int> &all_domain_attr,
   real_t *d_external_data);

} // namespace mfem::hydrodynamics
