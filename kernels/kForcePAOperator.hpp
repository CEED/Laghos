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

#ifndef MFEM_LAGHOS_KERNEL_FORCE_PA_OPERATOR_HPP
#define MFEM_LAGHOS_KERNEL_FORCE_PA_OPERATOR_HPP

#include "mfem.hpp"

#ifdef MFEM_USE_MPI

#include <memory>
#include <iostream>

namespace mfem
{

namespace hydrodynamics
{

class kForcePAOperator : public AbcForcePAOperator
{
private:
   const int dim, nzones;
   QuadratureData *quad_data;
   const ParFiniteElementSpace &h1fes, &l2fes;
   const kernels::kFiniteElementSpace &h1k, &l2k;
   const IntegrationRule &integ_rule;
   const IntegrationRule &ir1D;
   const int NUM_DOFS_1D;
   const int NUM_QUAD_1D;
   const int L2_DOFS_1D;
   const int H1_DOFS_1D;
   const int h1sz;
   const int l2sz;
   const kernels::kDofQuadMaps *l2D2Q, *h1D2Q;
   mutable mfem::Vector gVecL2, gVecH1;
public:
   kForcePAOperator(QuadratureData*,
                    ParFiniteElementSpace&,
                    ParFiniteElementSpace&,
                    const IntegrationRule&,
                    const bool);
   void Mult(const Vector&, Vector&) const;
   void MultTranspose(const Vector&, Vector&) const;
};

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_LAGHOS_KERNEL_FORCE_PA_OPERATOR_HPP
