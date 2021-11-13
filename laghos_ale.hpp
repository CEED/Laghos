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

#ifndef MFEM_LAGHOS_ALE
#define MFEM_LAGHOS_ALE

#include "mfem.hpp"

namespace mfem
{

namespace hydrodynamics
{

class SolutionMover;

// Performs the full remap advection loop.
class RemapAdvector
{
private:
   ParMesh pmesh;
   int dim;
   L2_FECollection fec_L2;
   H1_FECollection fec_H1;
   ParFiniteElementSpace pfes_L2, pfes_H1;

   // Remap state variables.
   Array<int> offsets;
   BlockVector S;
   ParGridFunction d, v, rho, e;

   RK3SSPSolver ode_solver;
   Vector x0;

public:
   RemapAdvector(const ParMesh &m, int order_v, int order_e);

   void InitFromLagr(const Vector &nodes0,
                     const ParGridFunction &dist, const ParGridFunction &v,
                     const IntegrationRule &rho_ir, const Vector &rhoDetJw);

   virtual void ComputeAtNewPosition(const Vector &new_nodes);

   void TransferToLagr(ParGridFunction &dist, ParGridFunction &vel,
                       const IntegrationRule &rho_ir, Vector &rhoDetJw);
};

// Performs a single remap advection step.
class AdvectorOper : public TimeDependentOperator
{
protected:
   const Vector &x0;
   Vector &x_now;
   GridFunction &u;
   VectorGridFunctionCoefficient u_coeff;
   mutable ParBilinearForm M, K;

public:
   /** Here @a pfes is the ParFESpace of the function that will be moved. Note
       that Mult() moves the nodes of the mesh corresponding to @a pfes. */
   AdvectorOper(int size, const Vector &x_start, GridFunction &velocity,
                ParFiniteElementSpace &pfes);

   virtual void Mult(const Vector &U, Vector &dU) const;
};

// Transfers of data between Lagrange and remap phases.
class SolutionMover
{
   // Integration points for the density.
   const IntegrationRule &ir_rho;

public:
   SolutionMover(const IntegrationRule &ir) : ir_rho(ir) { }

   // Density Lagrange -> Remap.
   void MoveDensityLR(const Vector &quad_rho, ParGridFunction &rho);
};

class LocalInverseHOSolver
{
protected:
   ParBilinearForm &M, &K;

public:
   LocalInverseHOSolver(ParBilinearForm &Mbf, ParBilinearForm &Kbf)
      : M(Mbf), K(Kbf) { }

   void CalcHOSolution(const Vector &u, Vector &du) const;
};

class DiscreteUpwindLOSolver
{
protected:
   ParFiniteElementSpace &pfes;
   const SparseMatrix &K;
   mutable SparseMatrix D;
   const Array<int> &K_smap;
   const Vector &M_lumped;

   void ComputeDiscreteUpwindMatrix() const;
   void ApplyDiscreteUpwindMatrix(ParGridFunction &u, Vector &du) const;

public:
   DiscreteUpwindLOSolver(ParFiniteElementSpace &space, const SparseMatrix &adv,
                          const Array<int> &adv_smap, const Vector &Mlump)
      : pfes(space), K(adv), D(), K_smap(adv_smap), M_lumped(Mlump) { }

   virtual void CalcLOSolution(const Vector &u, Vector &du) const;
};

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_LAGHOS_ALE
