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
struct MomentumQFunction
{
   using matd = matd_t<DIM>;
   using dmatd = matd_t<DIM, dscalar_t>;

   QConstants c;
   MFEM_HOST_DEVICE inline
   void operator()(const dmatd &dvdxi, const real_t &rho0,
                   const matd &J0, const dmatd &J, const real_t &gamma,
                   const dscalar_t &E, const real_t &w,
                   dmatd &stressJiT) const
   {
      auto setup = StressSetup<DIM>(c, dvdxi, rho0, J0, J, gamma, E, w);
      stressJiT = future::get<0>(setup);
   }
};

std::shared_ptr<DifferentiableOperator> CreateMomentumOperator(
   int dim,
   int qf_backend,
   ParFiniteElementSpace &H1,
   ParFiniteElementSpace &L2,
   ParGridFunction &material_gf,
   const IntegrationRule &ir,
   const Array<int> &all_domain_attr,
   const QConstants &qc)
{
   return Dispatch(dim, qf_backend, [&](auto DIM, auto backend)
   {
      constexpr int dim_v = decltype(DIM)::value;
      using backend_t = decltype(backend);
      // <sigma, grad(w) * J^-T> * det(J) * weights
      // <sigma(J^-T det(J) weights), grad(w)>
      future::tuple kernel_oo = {Gradient<VELOCITY>{}};
      FieldDescriptor output{VELOCITY, &H1};
      MomentumQFunction<dim_v> momentum_qf{qc};
      return CreateQDataOperator<dim_v, backend_t>(
                H1, L2, material_gf, ir, all_domain_attr, output, kernel_oo,
                momentum_qf);
   });
}

} // namespace mfem::hydrodynamics
