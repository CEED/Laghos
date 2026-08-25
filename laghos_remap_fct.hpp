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

#ifndef MFEM_LAGHOS_REMAP_FCT
#define MFEM_LAGHOS_REMAP_FCT

#include "mfem.hpp"

namespace mfem
{
namespace ale
{

// Monotone, High-order, Conservative Solver.
class FCTSolver
{
protected:
   ParFiniteElementSpace &pfes;
   real_t dt;

   // Computes a compatible slope (piecewise constan = mass_us / mass_u).
   // It could also update s_min and s_max, if required.
   void CalcCompatibleLOProduct(const ParGridFunction &us,
                                const Vector &m, const Vector &d_us_HO,
                                Vector &s_min, Vector &s_max,
                                const Vector &u_new,
                                const Array<bool> &active_el,
                                const Array<bool> &active_dofs,
                                Vector &d_us_LO_new);
   void ScaleProductBounds(const Vector &s_min, const Vector &s_max,
                           const Vector &u_new, const Array<bool> &active_el,
                           const Array<bool> &active_dofs,
                           Vector &us_min, Vector &us_max);

public:
   FCTSolver(ParFiniteElementSpace &space,
             real_t dt_)
      : pfes(space), dt(dt_) { }

   virtual ~FCTSolver() { }

   virtual void UpdateTimeStep(real_t dt_new) { dt = dt_new; }

   // Calculate du that satisfies the following:
   // bounds preservation: u_min_i <= u_i + dt du_i <= u_max_i,
   // conservation:        sum m_i (u_i + dt du_ho_i) = sum m_i (u_i + dt du_i).
   // Some methods utilize du_lo as a backup choice, as it satisfies the above.
   virtual void CalcFCTSolution(const ParGridFunction &u, const Vector &m,
                                const Vector &du_ho, const Vector &du_lo,
                                const Vector &u_min, const Vector &u_max,
                                Vector &du) const = 0;

   // Used in the case of product remap.
   // Given the input, calculates d_us, so that:
   // bounds preservation: s_min_i <= (us_i + dt d_us_i) / u_new_i <= s_max_i,
   // conservation: sum m_i (us_i + dt d_us_HO_i) = sum m_i (us_i + dt d_us_i).
   virtual void CalcFCTProduct(const ParGridFunction &us, const Vector &m,
                               const Vector &d_us_HO, const Vector &d_us_LO,
                               Vector &s_min, Vector &s_max,
                               const Vector &u_new,
                               const Array<bool> &active_el,
                               const Array<bool> &active_dofs, Vector &d_us)
   {
      MFEM_ABORT("Product remap is not implemented for the chosen solver");
   }
};

class FluxBasedFCT : public FCTSolver
{
protected:
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
      : FCTSolver(space, delta_t),
        K(adv_mat), M(mass_mat), K_smap(adv_smap), flux_ij(adv_mat),
        gp(&pfes), gm(&pfes) { }

   virtual void CalcFCTSolution(const ParGridFunction &u, const Vector &m,
                                const Vector &du_ho, const Vector &du_lo,
                                const Vector &u_min, const Vector &u_max,
                                Vector &du) const;

   virtual void CalcFCTProduct(const ParGridFunction &us, const Vector &m,
                               const Vector &d_us_HO, const Vector &d_us_LO,
                               Vector &s_min, Vector &s_max,
                               const Vector &u_new,
                               const Array<bool> &active_el,
                               const Array<bool> &active_dofs, Vector &d_us);
};

} // namespace ale
} // namespace mfem

#endif // MFEM_LAGHOS_REMAP_FCT
