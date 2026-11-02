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
#include "linalg/tensor.hpp"
#include <fem/dfem/doperator.hpp>

#include <type_traits>

namespace mfem::hydrodynamics
{

using mfem::future::DifferentiableOperator;
using mfem::future::DerivativeOperator;
using mfem::future::FDJacobian;
using mfem::future::FieldDescriptor;
using mfem::future::Gradient;
using mfem::future::Identity;
using mfem::future::Value;
using mfem::future::Weight;
using mfem::future::pretty_print;

enum EXT_DATA_IDX
{
   CFL = 0,
   ORDER_VEL,
   VISCOSITY_FLAG,
   VISCOSITY_TYPE,
   VISC_Q1,
   VISC_Q2,
   H0,
   DT_ESTIMATE,
   COUNT
};

constexpr int VELOCITY = 0;
constexpr int DENSITY0 = 1;
constexpr int COORDINATES0 = 2;
constexpr int COORDINATES = 3;
constexpr int MATERIAL = 4;
constexpr int SPECIFIC_INTERNAL_ENERGY = 5;
constexpr int DT_EST = 8;
constexpr int STRESS_TENSOR = 9;

using mfem::future::tensor;

using dscalar_t =
   std::conditional_t<mfem::UseEnzyme, real_t,
   mfem::future::dual<real_t, real_t>>;

template <int DIM, typename T = real_t>
using vecd_t = tensor<T, DIM>;

template <int DIM, typename T = real_t>
using matd_t = tensor<T, DIM, DIM>;

template <int DIM, typename T>
struct TaylorSourceQFunction
{
   using vecd = vecd_t<DIM, T>;
   using matd = matd_t<DIM, T>;

   TaylorSourceQFunction() = default;

   MFEM_HOST_DEVICE inline
   void operator()(const vecd &x, const matd &J, const real_t &w,
                   T &s) const
   {
      const auto f = 3.0 / 8.0 * M_PI *
                     (cos(3.0*M_PI*x(0)) * cos(M_PI*x(1)) -
                      cos(M_PI*x(0)) * cos(3.0*M_PI*x(1)));

      s = f * det(J) * w;
   }
};

template <int DIM>
class MomentumPAQFunction
{
   using matd = matd_t<DIM>;

public:
   MomentumPAQFunction() = default;

   MFEM_HOST_DEVICE inline
   void operator()(const matd &stressJiT, matd &f) const
   {
      f = stressJiT;
   }
};

template <int DIM>
class EnergyConservationPAQFunction
{
   using matd = matd_t<DIM>;

public:
   EnergyConservationPAQFunction() = default;

   MFEM_HOST_DEVICE inline
   void operator()(const matd &dvdxi, const matd &stressJiT, real_t &f) const
   {
      f = ddot(stressJiT, dvdxi);
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
      const real_t &w,
      real_t &ie) const
   {
      ie = rho0 * E * det(J0) * w;
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
      const real_t &w,
      real_t &ke) const
   {
      ke = rho0 * 0.5 * v * v * det(J0) * w;
   }
};

template <int DIM>
class DensityQFunction
{
   using matd = matd_t<DIM>;

public:
   DensityQFunction() = default;

   MFEM_HOST_DEVICE inline
   void operator() (
      const real_t &rho0,
      const matd &J0,
      const real_t &w,
      real_t &m) const
   {
      m = rho0 * det(J0) * w;
   }
};

struct QuadratureData
{
   static constexpr int aux_dim = 1;
   QuadratureData(ParMesh &mesh, const IntegrationRule &ir) :
      qspace(mesh, ir),
      StressSpace(qspace, mesh.Dimension()*mesh.Dimension()),
      stressp(StressSpace),
      R(qspace, aux_dim),
      dt_est(R)
   {
      dt_est.UseDevice(true);
      stressp.UseDevice(true);
   }

   QuadratureSpace qspace;

   VectorQuadratureSpace StressSpace;
   QuadratureFunction stressp;

   VectorQuadratureSpace R;
   QuadratureFunction dt_est;
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
   const int dim;
   ParBilinearForm pabf;
   int ess_tdofs_count;
   Array<int> ess_tdofs;
   OperatorPtr mass;
};

} // namespace mfem::hydrodynamics
