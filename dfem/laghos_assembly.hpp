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

#pragma once

#include "mfem.hpp"

// fem/dfem/doperator.hpp has non-inline definitions, see laghos_solver.cpp
#include "fem/dfem/parametricspace.hpp"
#include "fem/dfem/tuple.hpp"
#include "linalg/tensor.hpp"

namespace mfem
{

namespace hydrodynamics
{

using mfem::internal::tensor;

constexpr int VELOCITY = 0;
constexpr int DENSITY0 = 1;
constexpr int COORDINATES0 = 2;
constexpr int COORDINATES = 3;
constexpr int MATERIAL = 4;
constexpr int SPECIFIC_INTERNAL_ENERGY = 5;
constexpr int DT_EST = 8;
constexpr int STRESS_TENSOR = 9;

template <int DIM>
using vecd_t = tensor<real_t, DIM>;

template <int DIM>
using matd_t = tensor<real_t, DIM, DIM>;

template <int DIM>
struct TaylorSourceQFunction
{
   using vecd = vecd_t<DIM>;
   using matd = matd_t<DIM>;

   TaylorSourceQFunction() = default;

   MFEM_HOST_DEVICE inline
   auto operator()(const vecd &x, const matd &J, const real_t &w) const
   {
      auto f = 3.0 / 8.0 * M_PI * ( cos(3.0*M_PI*x(0)) * cos(M_PI*x(1)) -
                                    cos(M_PI*x(0))     * cos(3.0*M_PI*x(1)) );

      return mfem::tuple{f * det(J) * w};
   }
};

template <int DIM>
class MomentumPAQFunction
{
   using matd = matd_t<DIM>;

public:
   MomentumPAQFunction() = default;

   MFEM_HOST_DEVICE inline
   auto operator()(const matd &stressJiT) const
   {
      return mfem::tuple{stressJiT};
   }
};

template <int DIM>
class EnergyConservationPAQFunction
{
   using matd = matd_t<DIM>;

public:
   EnergyConservationPAQFunction() = default;

   MFEM_HOST_DEVICE inline
   auto operator()(const matd &dvdxi, const matd &stressJiT) const
   {
      return mfem::tuple{ddot(stressJiT, dvdxi)};
   }
};

template <int DIM>
class TotalInternalEnergyQFunction
{
   using matd = matd_t<DIM>;

public:
   TotalInternalEnergyQFunction() = default;

   MFEM_HOST_DEVICE inline
   auto operator() (
      const real_t &E,
      const real_t &rho0,
      const matd &J0,
      const real_t &w) const
   {
      return mfem::tuple{rho0 * E * det(J0) * w};
   }
};

template <int DIM>
class TotalKineticEnergyQFunction
{
   using vecd = vecd_t<DIM>;
   using matd = matd_t<DIM>;

public:
   TotalKineticEnergyQFunction() = default;

   MFEM_HOST_DEVICE inline
   auto operator() (
      const vecd &v,
      const real_t &rho0,
      const matd &J0,
      const real_t &w) const
   {
      return mfem::tuple{rho0 * 0.5 * v * v * det(J0) * w};
   }
};

template <int DIM>
class DensityQFunction
{
   using matd = matd_t<DIM>;

public:
   DensityQFunction() = default;

   MFEM_HOST_DEVICE inline
   auto operator() (
      const real_t &rho0,
      const matd &J0,
      const real_t &w) const
   {
      return mfem::tuple{rho0 * det(J0) * w};
   }
};

struct QuadratureData
{
   static constexpr int aux_dim = 1;
   QuadratureData(const ParMesh &mesh, const IntegrationRule &ir) :
      StressSpace(mesh.Dimension(), mesh.Dimension()*mesh.Dimension(),
                  ir.GetNPoints(),
                  mesh.Dimension()*mesh.Dimension()*ir.GetNPoints()*mesh.GetNE()),
      stressp(StressSpace),
      R(mesh.Dimension(),
        aux_dim,
        ir.GetNPoints(),
        aux_dim*ir.GetNPoints()*mesh.GetNE()),
      dt_est(R)
   {
      dt_est.UseDevice(true);
      stressp.UseDevice(true);
   }

   ParametricSpace StressSpace;
   ParametricFunction stressp;

   ParametricSpace R;
   ParametricFunction dt_est;
};

class MassPAOperator : public Operator
{
public:
   MassPAOperator(ParFiniteElementSpace &pfes,
                  const IntegrationRule &ir,
                  Coefficient &Q);

   void SetEssentialTrueDofs(Array<int> &dofs);

   void EliminateRHS(Vector &b) const;

   void Mult(const Vector &x, Vector &y) const override;

   void FullAddMult(const Vector &x, Vector &y) const;

   const ParBilinearForm &GetBF() const { return pabf; }

   const MPI_Comm comm;
   const int dim, NE, vsize;
   ParBilinearForm pabf;
   int ess_tdofs_count;
   Array<int> ess_tdofs;
   OperatorPtr mass;
};

} // namespace hydrodynamics

} // namespace mfem
