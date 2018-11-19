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
#include "laghos_utils.hpp"

#ifdef MFEM_USE_MPI

#include <memory>
#include <iostream>
#include <fstream>

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

// These are defined in laghos.cpp
double rho0(const Vector &);
void v0(const Vector &, Vector &);
double e0(const Vector &);
double gamma(const Vector &);

struct TimingData
{
   // Total times for all major computations:
   // CG solves (H1 and L2) / force RHS assemblies / quadrature computations.
   StopWatch sw_cgH1, sw_cgL2, sw_force, sw_qdata;

   // These accumulate the total processed dofs or quad points:
   // #(CG iterations) for the H1 CG solve.
   // #dofs  * #(CG iterations) for the L2 CG solve.
   // #quads * #(RK sub steps) for the quadrature data computations.
   int H1cg_iter, L2dof_iter, quad_tstep;

   TimingData()
      : H1cg_iter(0), L2dof_iter(0), quad_tstep(0) { }
};

// Given a solutions state (x, v, e), this class performs all necessary
// computations to evaluate the new slopes (dx_dt, dv_dt, de_dt).
class LagrangianHydroOperator : public TimeDependentOperator
{
protected:
   const ProblemOption problem;

   occa::device device;
   OccaFiniteElementSpace &o_H1FESpace;
   OccaFiniteElementSpace &o_L2FESpace;
   mutable OccaFiniteElementSpace o_H1compFESpace;

   ParFiniteElementSpace &H1FESpace;
   ParFiniteElementSpace &L2FESpace;

   Array<int> &essential_tdofs;

   int dim, elements, l2dofs_cnt, h1dofs_cnt;
   double cfl;
   bool use_viscosity;

   // Velocity mass matrix and local inverses of the energy mass matrices. These
   // are constant in time, due to the pointwise mass conservation property.
   mutable ParBilinearForm Mv;
   DenseTensor Me_inv;

   // Integration rule for all assemblies.
   const IntegrationRule &integ_rule;

   int cg_print_level, cg_max_iter;
   double cg_rel_tol, cg_abs_tol;

   // Data associated with each quadrature point in the mesh. These values are
   // recomputed at each time step.
   mutable QuadratureData quad_data;
   mutable bool quad_data_is_current;

   // Force matrix that combines the kinematic and thermodynamic spaces. It is
   // assembled in each time step and then it's used to compute the final
   // right-hand sides for momentum and specific internal energy.
   mutable OccaMassOperator VMass, EMass;
   mutable OccaForceOperator Force;

   occa::kernel updateKernel;

   // Linear solver for energy.
   OccaCGSolver locCG;

   mutable TimingData timer;

   void UpdateQuadratureData(const OccaVector &S) const;

public:
   LagrangianHydroOperator(ProblemOption problem_,
                           OccaFiniteElementSpace &o_H1FESpace_,
                           OccaFiniteElementSpace &o_L1FESpace_,
                           Array<int> &essential_tdofs,
                           OccaGridFunction &rho0,
                           double cfl_,
                           Coefficient *material_, bool use_viscosity_,
                           double cg_tol_, int cg_max_iter_,
                           occa::properties &props);

   // Solve for dx_dt, dv_dt and de_dt.
   virtual void Mult(const OccaVector &S, OccaVector &dS_dt) const;

   // Calls UpdateQuadratureData to compute the new quad_data.dt_est.
   double GetTimeStepEstimate(const OccaVector &S) const;
   void ResetTimeStepEstimate() const;
   void ResetQuadratureData() const
   {
      quad_data_is_current = false;
   }

   // The density values, which are stored only at some quadrature points, are
   // projected as a ParGridFunction.
   void ComputeDensity(ParGridFunction &rho);

   void PrintTimingData(bool IamRoot, int steps);

   ~LagrangianHydroOperator();
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
