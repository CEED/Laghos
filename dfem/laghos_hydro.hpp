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
#include "laghos_solver.hpp"

namespace mfem::hydrodynamics
{

enum EXT_DATA_IDX
{
   CFL = 0,
   ORDER_VEL,
   VISCOSITY_FLAG,
   VISCOSITY_TYPE,
   VISC_Q1,
   VISC_Q2,
   H0,
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

// Builds the dFEM operators of the mesh dimension and returns a new
// LagrangianHydroOperator
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
   const LagrangianHydroOperator::Options &options);

} // namespace mfem::hydrodynamics
