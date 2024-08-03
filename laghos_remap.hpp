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
struct MaterialData;

class InterpolationRemap
{
   public:
   InterpolationRemap() { }

   void Remap(const ParGridFunction &source, const ParGridFunction &x_new,
              ParGridFunction &interpolated);
};

// Performs the full remap advection loop.
class RemapAdvector
{
private:
   ParMesh pmesh;
   int dim;
   L2_FECollection fec_L2;
   H1_FECollection fec_H1;
   ParFiniteElementSpace pfes_L2, pfes_H1;
   bool remap_v;

   const double cfl_factor;

   // Remap state variables.
   Array<int> offsets;
   BlockVector S;
   ParGridFunction v, rho, e;

   double e_max;

   RK3SSPSolver ode_solver;
   Vector x0;

public:
   RemapAdvector(const ParMesh &m, int order_v, int order_e,
                 double cfl, bool remap_vel);

   void InitFromLagr(const Vector &nodes0,
                     const ParGridFunction &vel,
                     const IntegrationRule &rho_ir,
                     const Vector &rhoDetJw,
                     const ParGridFunction &energy);

   virtual void ComputeAtNewPosition(const Vector &new_nodes,
                                     const Array<int> &ess_tdofs);

   void TransferToLagr(ParGridFunction &rho0_gf, ParGridFunction &vel,
                       const IntegrationRule &ir_rho, Vector &rhoDetJw,
                       const IntegrationRule &ir_rho_b, Vector &rhoDetJ_be,
                       ParGridFunction &energy);
};

// Performs a single remap advection step.
class AdvectorOper : public TimeDependentOperator
{
protected:
   bool remap_v = true;

   const Vector &x0;
   Vector &x_now;
   const Array<int> &v_ess_tdofs;
   ParGridFunction &u;
   VectorGridFunctionCoefficient u_coeff;
   GridFunctionCoefficient rho_coeff;
   ScalarVectorProductCoefficient rho_u_coeff;
   mutable ParBilinearForm Mr_H1, Kr_H1;
   mutable ParBilinearForm M_L2, M_L2_Lump, K_L2;
   mutable ParBilinearForm Mr_L2, Mr_L2_Lump, Kr_L2;
   double dt = 0.0;

   // Piecewise min and max of gf over all elements.
   void ComputeElementsMinMax(const ParGridFunction &gf,
                              Vector &el_min, Vector &el_max) const;
   // Bounds at dofs taking the current element and its face-neighbors.
   void ComputeSparsityBounds(const ParFiniteElementSpace &pfes,
                              const Vector &el_min, const Vector &el_max,
                              Vector &dof_min, Vector &dof_max) const;

public:
   // Here pfes is the ParFESpace of the function that will be moved.
   // Mult() moves the nodes of the mesh corresponding to pfes.
   AdvectorOper(int size, const Vector &x_start, const Array<int> &v_ess_td,
                ParGridFunction &mesh_vel,
                ParGridFunction &rho,
                ParFiniteElementSpace &pfes_H1,
                ParFiniteElementSpace &pfes_H1_s,
                ParFiniteElementSpace &pfes_L2);

   void SetVelocityRemap(bool flag) { remap_v = flag; }

   // Single RK stage solve for all fields contained in U.
   virtual void Mult(const Vector &U, Vector &dU) const;

   void SetDt(double delta_t) { dt = delta_t; }

   double Momentum(ParGridFunction &v, double t);
   double Interface(ParGridFunction &xi, double t);
   double Energy(ParGridFunction &e, double t);
};

// Transfer of data between the Lagrange and the remap phases.
class SolutionMover
{
   // Integration points for the density.
   const IntegrationRule &ir_rho;

public:
   SolutionMover(const IntegrationRule &ir) : ir_rho(ir) { }

   // Density transfer: Lagrange -> Remap.
   // Projects the quad points data to a GridFunction, while preserving the
   // bounds for rho taken from the current element and its face-neighbors.
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

   Array<int> K_smap;
   const Vector &M_lumped;

   void ComputeDiscreteUpwindMatrix() const;
   void ApplyDiscreteUpwindMatrix(ParGridFunction &u, Vector &du) const;

public:
   DiscreteUpwindLOSolver(ParFiniteElementSpace &space, const SparseMatrix &adv,
                          const Vector &Mlump);

   virtual void CalcLOSolution(const Vector &u, Vector &du) const;
   Array<int> &GetKmap() { return K_smap; }
};

class FluxBasedFCT
{
protected:
   ParFiniteElementSpace &pfes;
   double dt;

   const SparseMatrix &K, &M;
   const Array<int> &K_smap;

   // Temporary computation objects.
   mutable SparseMatrix flux_ij;
   mutable ParGridFunction gp, gm;

   void ComputeFluxMatrix(const ParGridFunction &u, const Vector &du_ho,
                          SparseMatrix &flux_mat) const;
   void AddFluxesAtDofs(const SparseMatrix &flux_mat,
                        Vector &flux_pos, Vector &flux_neg) const;
   void ComputeFluxCoefficients(const Vector &u, const Vector &du_lo,
      const Vector &m, const Vector &u_min, const Vector &u_max,
      Vector &coeff_pos, Vector &coeff_neg) const;
   void UpdateSolutionAndFlux(const Vector &du_lo, const Vector &m,
      ParGridFunction &coeff_pos, ParGridFunction &coeff_neg,
      SparseMatrix &flux_mat, Vector &du) const;

public:
   FluxBasedFCT(ParFiniteElementSpace &space, double delta_t,
                const SparseMatrix &adv_mat, const Array<int> &adv_smap,
                const SparseMatrix &mass_mat)
      : pfes(space), dt(delta_t),
        K(adv_mat), M(mass_mat), K_smap(adv_smap), flux_ij(adv_mat),
        gp(&pfes), gm(&pfes) { }

   virtual void CalcFCTSolution(const ParGridFunction &u, const Vector &m,
                                const Vector &du_ho, const Vector &du_lo,
                                const Vector &u_min, const Vector &u_max,
                                Vector &du) const;
};

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_LAGHOS_ALE