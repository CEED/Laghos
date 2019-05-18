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

#ifndef MFEM_LAGHOS_SOLVER
#define MFEM_LAGHOS_SOLVER

#include "mfem.hpp"
#include "laghos_assembly.hpp"
#include "laghos_qupdate.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{

namespace hydrodynamics
{

/// Visualize the given parallel grid function, using a GLVis server on the
/// specified host and port. Set the visualization window title, and optionally,
/// its geometry.
void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    ParGridFunction &gf, const char *title,
                    int x = 0, int y = 0, int w = 400, int h = 400,
                    bool vec = false);

struct TimingData
{
   // Total times for all major computations:
   // CG solves (H1 and L2) / force RHS assemblies / quadrature computations.
   StopWatch sw_cgH1, sw_cgL2, sw_force, sw_qdata;

   // Store the number of dofs of the the coresponding local CG
   const HYPRE_Int L2dof;

   // These accumulate the total processed dofs or quad points:
   // #(CG iterations) for the L2 CG solve.
   // #quads * #(RK sub steps) for the quadrature data computations.
   HYPRE_Int H1iter, L2iter;
   HYPRE_Int quad_tstep;

   TimingData(const HYPRE_Int l2d) :
      L2dof(l2d), H1iter(0), L2iter(0), quad_tstep(0) { }
};

// Given a solutions state (x, v, e), this class performs all necessary
// computations to evaluate the new slopes (dx_dt, dv_dt, de_dt).
class LagrangianHydroOperator : public TimeDependentOperator
{
protected:
   ParFiniteElementSpace &H1FESpace, &L2FESpace;
   mutable ParFiniteElementSpace H1compFESpace;

   // FE spaces local and global sizes
   const int H1Vsize;
   const int H1TVSize;
   const HYPRE_Int H1GTVSize;
   const int H1compTVSize;
   const int L2Vsize;
   const int L2TVSize;
   const HYPRE_Int L2GTVSize;
   Array<int> block_offsets;

   // Reference to the current mesh configuration.
   mutable ParGridFunction x_gf;

   const Array<int> &ess_tdofs;

   const int dim, nzones, l2dofs_cnt, h1dofs_cnt, source_type;
   const double cfl;
   const bool use_viscosity, p_assembly, okina;
   const double cg_rel_tol;
   const int cg_max_iter;
   Coefficient *material_pcf;

   // Velocity mass matrix and local inverses of the energy mass matrices. These
   // are constant in time, due to the pointwise mass conservation property.
   mutable ParBilinearForm Mv;
   SparseMatrix Mv_spmat_copy;
   DenseTensor Me, Me_inv;

   // Integration rule for all assemblies.
   const IntegrationRule &integ_rule;

   // Data associated with each quadrature point in the mesh. These values are
   // recomputed at each time step.
   mutable QuadratureData quad_data;
   mutable bool quad_data_is_current, forcemat_is_assembled;

   // Structures used to perform partial assembly.
   Tensors1D tensors1D;
   FastEvaluator evaluator;

   // Force matrix that combines the kinematic and thermodynamic spaces. It is
   // assembled in each time step and then it is used to compute the final
   // right-hand sides for momentum and specific internal energy.
   mutable MixedBilinearForm Force;

   // Same as above, but done through partial assembly.
   AbcForcePAOperator *ForcePA;

   // Mass matrices done through partial assembly:
   // velocity (coupled H1 assembly) and energy (local L2 assemblies).
   AbcMassPAOperator *VMassPA, *EMassPA;
   mutable DiagonalSolver VMassPA_prec;
   mutable LocalMassPAOperator locEMassPA;

   // Linear solver for energy.
   CGSolver CG_VMass, CG_EMass, locCG;

   mutable TimingData timer;

   const bool qupdate;
   const double gamma;
   mutable QUpdate Q;

   mutable Vector X, B, one, rhs, e_rhs;
   mutable ParGridFunction rhs_c_gf, dvc_gf;

   virtual void ComputeMaterialProperties(int nvalues, const double gamma[],
                                          const double rho[], const double e[],
                                          double p[], double cs[]) const
   {
      for (int v = 0; v < nvalues; v++)
      {
         p[v]  = (gamma[v] - 1.0) * rho[v] * e[v];
         cs[v] = sqrt(gamma[v] * (gamma[v]-1.0) * e[v]);
      }
   }

   void UpdateQuadratureData(const Vector &S) const;
   void AssembleForceMatrix() const;

public:
   LagrangianHydroOperator(Coefficient &q,
                           const int size,
                           ParFiniteElementSpace &h1_fes,
                           ParFiniteElementSpace &l2_fes,
                           const Array<int> &essential_tdofs, ParGridFunction &rho0,
                           const int source_type_, const double cfl_,
                           Coefficient *material_, const bool visc, const bool pa,
                           const double cgt, const int cgiter,
                           const int order_q, const bool qupdate,
                           const double gamma, const bool okina,
                           int h1_basis_type);

   // Solve for dx_dt, dv_dt and de_dt.
   virtual void Mult(const Vector &S, Vector &dS_dt) const;

   virtual MemoryClass GetMemoryClass() const
   { return Device::GetMemoryClass(); }

   void SolveVelocity(const Vector &S, Vector &dS_dt) const;
   void SolveEnergy(const Vector &S, const Vector &v, Vector &dS_dt) const;
   void UpdateMesh(const Vector &S) const;

   // Calls UpdateQuadratureData to compute the new quad_data.dt_estimate.
   double GetTimeStepEstimate(const Vector &S) const;
   void ResetTimeStepEstimate() const;
   void ResetQuadratureData() const { quad_data_is_current = false; }

   // The density values, which are stored only at some quadrature points, are
   // projected as a ParGridFunction.
   void ComputeDensity(ParGridFunction &rho) const;

   double InternalEnergy(const ParGridFunction &e) const;
   double KineticEnergy(const ParGridFunction &v) const;

   void PrintTimingData(bool IamRoot, int steps, const bool fom) const;

   int GetH1VSize() const { return H1FESpace.GetVSize(); }

   const Array<int> &GetBlockOffsets() const { return block_offsets; }
};

class TaylorCoefficient : public Coefficient
{
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      Vector x(2);
      T.Transform(ip, x);
      return 3.0 / 8.0 * M_PI * ( cos(3.0*M_PI*x(0)) * cos(M_PI*x(1)) -
                                  cos(M_PI*x(0))     * cos(3.0*M_PI*x(1)) );
   }
};

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_LAGHOS
