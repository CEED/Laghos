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

#ifndef MFEM_LAGHOS_ASSEMBLY
#define MFEM_LAGHOS_ASSEMBLY

#include "mfem.hpp"

namespace mfem
{

namespace hydrodynamics
{

// Container for all data needed at quadrature points.
struct QuadratureData
{
   QuadratureSpace qspace;

   VectorQuadratureSpace StressSpace;
   QuadratureFunction stressp;

   VectorQuadratureSpace R;
   QuadratureFunction dt_est;

public:
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
};

class MassPAOperator : public Operator
{
   const MPI_Comm comm;
   const int dim;
   ParBilinearForm pabf;
   int ess_tdofs_count;
   Array<int> ess_tdofs;
   OperatorPtr mass;

public:
   MassPAOperator(ParFiniteElementSpace &pfes,
                  const IntegrationRule &ir,
                  Coefficient &Q);

   void Mult(const Vector &x, Vector &y) const override;

   void FullAddMult(const Vector &x, Vector &y) const;

   void SetEssentialTrueDofs(Array<int> &dofs);

   void EliminateRHS(Vector &b) const;

   const ParBilinearForm &GetBF() const { return pabf; }
};

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_LAGHOS_ASSEMBLY
