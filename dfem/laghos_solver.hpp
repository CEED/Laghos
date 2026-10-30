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
#ifdef MFEM_USE_PETSC
#include "linalg/petsc.hpp"
#include "petscmat.h"
#endif

#include <limits>
#include <memory>

#include "laghos_assembly.hpp"

// Set from the command line in laghos.cpp
extern int problem;
extern int petsc_lag_steps;

namespace mfem
{

// From fem/dfem, whose doperator.hpp is only included by laghos_solver.cpp
namespace future
{
class DifferentiableOperator;
class DerivativeOperator;
class FDJacobian;
}

namespace hydrodynamics
{

using future::DifferentiableOperator;
using future::DerivativeOperator;
using future::FDJacobian;

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

enum PRECONDITIONER_TYPE
{
   SUPERLU,
   BLOCK_DIAGONAL_AMG,
};

// Visualize the given parallel grid function, using a GLVis server on the
// specified host and port. Set the visualization window title, and optionally,
// its geometry
void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    ParGridFunction &gf, const char *title,
                    int x, int y, int w, int h, bool vec = false);

class LagrangianHydroOperator : public TimeDependentOperator
{
public:

   class LineSearchNewtonSolver : public NewtonSolver
   {
   public:
      LineSearchNewtonSolver(MPI_Comm comm, const LagrangianHydroOperator &hydro)
         : NewtonSolver(comm),
           beta(0.5),    // Backtracking reduction factor
           alpha(1e-2),  // Sufficient decrease constant (Armijo condition)
           max_line_iter(5),
           hydro(hydro) {}

   protected:
      void ProcessNewState(const Vector &x) const override;

      // Override the scaling factor computation to implement line search
      real_t ComputeScalingFactor(const Vector &x, const Vector &b) const override;

   private:
      real_t beta;        // Backtracking factor (how much to reduce step)
      real_t alpha;       // Sufficient decrease parameter
      int max_line_iter;  // Maximum line search iterations
      mutable Vector x_new, r_new, s_new;
      const LagrangianHydroOperator &hydro;
   };

   class LagrangianHydroJacobianOperator : public Operator
   {
   public:
      LagrangianHydroJacobianOperator(
         LagrangianHydroOperator &hydro,
         std::shared_ptr<DerivativeOperator> dRvdx,
         std::shared_ptr<DerivativeOperator> dRvdv,
         std::shared_ptr<DerivativeOperator> dRvde,
         std::shared_ptr<DerivativeOperator> dRedx,
         std::shared_ptr<DerivativeOperator> dRedv,
         std::shared_ptr<DerivativeOperator> dRede,
         std::shared_ptr<DerivativeOperator> dTaylorSourcedx,
         real_t h) :
         Operator(2*hydro.H1.GetTrueVSize() + hydro.L2.GetTrueVSize()),
         h(h),
         H1tsize(hydro.H1.GetTrueVSize()), L2tsize(hydro.L2.GetTrueVSize()),
         w(height),
         z(height),
         hydro(hydro),
         dRvdx(dRvdx), dRvdv(dRvdv), dRvde(dRvde),
         dRedx(dRedx), dRedv(dRedv), dRede(dRede),
         dTaylorSourcedx(dTaylorSourcedx) { }

      void Mult(const Vector &u, Vector &y) const override;

      virtual MemoryClass GetMemoryClass() const override;

      real_t h;
      const int H1tsize, L2tsize;
      mutable Vector w, z;

      LagrangianHydroOperator &hydro;
      std::shared_ptr<DerivativeOperator> dRvdx;
      std::shared_ptr<DerivativeOperator> dRvdv;
      std::shared_ptr<DerivativeOperator> dRvde;
      std::shared_ptr<DerivativeOperator> dRedx;
      std::shared_ptr<DerivativeOperator> dRedv;
      std::shared_ptr<DerivativeOperator> dRede;
      std::shared_ptr<DerivativeOperator> dTaylorSourcedx;
   };

   class Preconditioner : public Solver
   {
   public:
      Preconditioner(LagrangianHydroOperator &hydro) :
         Solver(hydro.Height()),
         hydro(hydro)
      {};

      void SetRebuildFlag(bool flag);

      void SetOperator(const Operator &op) override;

      void Mult(const Vector &x, Vector &y) const override;

      mutable Vector w;
      LagrangianHydroOperator &hydro;
      bool rebuild = true;
      const LagrangianHydroJacobianOperator *jacobian = nullptr;
      std::shared_ptr<HypreParMatrix> block_hypre;
      std::shared_ptr<HypreParMatrix> vv_mat;
      std::shared_ptr<HypreBoomerAMG> amg_v;
      std::shared_ptr<HypreParMatrix> ee_mat;
      std::shared_ptr<HypreBoomerAMG> amg_e;
#ifdef MFEM_USE_SUPERLU
      std::shared_ptr<SuperLURowLocMatrix> superlu_mat;
      std::shared_ptr<SuperLUSolver> superlu_solver;
#endif
   };

   class LagrangianHydroResidualOperator : public Operator
   {
   public:
      LagrangianHydroResidualOperator(LagrangianHydroOperator &hydro,
                                      const Vector &x, bool fd_gradient, int dump_jacobians) :
         Operator(2*hydro.H1.GetTrueVSize()+hydro.L2.GetTrueVSize()),
         hydro(hydro),
         H1tsize(hydro.H1.GetTrueVSize()),
         H1vsize(hydro.H1.GetVSize()),
         L2tsize(hydro.L2.GetTrueVSize()),
         L2vsize(hydro.L2.GetVSize()),
         x(x),
         u(x.Size()),
         u_l(2*H1vsize + L2vsize),
         e_source_t(hydro.L2.GetTrueVSize()),
         fd_gradient(fd_gradient),
         dump_jacobians(dump_jacobians) {}

      void SetTimeStep(const real_t &time_step) { this->dt = time_step; }

      void Mult(const Vector &k, Vector &R) const override;

      Operator& GetGradient(const Vector &k) const override;

      LagrangianHydroOperator &hydro;
      real_t dt;
      const int H1tsize;
      const int H1vsize;
      const int L2tsize;
      const int L2vsize;
      const Vector &x;
      mutable Vector u, u_l, e_source_t;
      mutable std::shared_ptr<FDJacobian> fd_jacobian;
      mutable std::shared_ptr<LagrangianHydroJacobianOperator> jacobian;
#ifdef MFEM_USE_PETSC
      mutable std::unique_ptr<PetscParMatrix> petsc_jacobian;
#endif
      bool fd_gradient;
      int dump_jacobians;
   };

   LagrangianHydroOperator(
      ParFiniteElementSpace &H1,
      ParFiniteElementSpace &L2,
      Array<int> &ess_tdof,
      const IntegrationRule &ir,
      FunctionCoefficient &rho0_coeff,
      ParGridFunction &x0_gf,
      ParGridFunction &rho0_gf,
      ParGridFunction &material_gf,
      std::shared_ptr<DifferentiableOperator> update_qdata,
      std::shared_ptr<DifferentiableOperator> dtest_mf,
      std::shared_ptr<DifferentiableOperator> momentum_mf,
      std::shared_ptr<DifferentiableOperator> momentum_pa,
      std::shared_ptr<DifferentiableOperator> energy_conservation_mf,
      std::shared_ptr<DifferentiableOperator> energy_conservation_pa,
      std::shared_ptr<DifferentiableOperator> total_internal_energy_mf,
      std::shared_ptr<DifferentiableOperator> total_kinetic_energy_mf,
      std::shared_ptr<DifferentiableOperator> density_mf,
      std::shared_ptr<DifferentiableOperator> taylor_source_mf,
      std::shared_ptr<QuadratureData> qdata,
      const bool &fd_gradient,
      const int &dump_jacobians,
      const int &nonlinear_maximum_iterations,
      const real_t &nonlinear_relative_tolerance,
      const int &krylov_maximum_iterations,
      const int &preconditioner_lag,
      const PRECONDITIONER_TYPE &preconditioner_type,
      Vector& external_data,
      const bool use_petsc);

   void Mult(const Vector &S, Vector &dSdt) const override;

   void ImplicitSolve(const real_t dt, const Vector &x, Vector &k) override;

   void UpdateMesh(const Vector &S) const;

   void ResetTimeStepEstimate();

   void ResetQuadratureData() const;

   void ResetNewtonInfo();

   real_t GetTimeStepEstimate(const Vector &S);

   real_t InternalEnergy(ParGridFunction &e);

   real_t KineticEnergy(ParGridFunction &v);

   void ComputeDensity(ParGridFunction &rho);

   void DensityScatter();

   real_t ComputeMinDet(const Vector &S) const;

   void UpdateQuadratureData(const Vector &S) const;

   // Min of the time step estimate over the local quadrature points at S
   real_t LocalTimeStepEstimate(const Vector &S) const;

   virtual MemoryClass GetMemoryClass() const override;

   ~LagrangianHydroOperator();

   ParFiniteElementSpace &H1;
   ParFiniteElementSpace &L2;
   mutable ParFiniteElementSpace H1c;
   Array<int> &ess_tdof;
   mutable Array<int> c_tdofs[3];
   const IntegrationRule &ir;
   ParGridFunction &x0;
   ParGridFunction &rho0;
   ParGridFunction &material;
   std::shared_ptr<DifferentiableOperator> update_qdata;
   std::shared_ptr<DifferentiableOperator> dtest_mf;
   std::shared_ptr<DifferentiableOperator> momentum_mf;
   std::shared_ptr<DifferentiableOperator> momentum_pa;
   std::shared_ptr<DifferentiableOperator> energy_conservation_mf;
   std::shared_ptr<DifferentiableOperator> energy_conservation_pa;
   std::shared_ptr<DifferentiableOperator> total_internal_energy_mf;
   std::shared_ptr<DifferentiableOperator> total_kinetic_energy_mf;
   std::shared_ptr<DifferentiableOperator> density_mf;
   std::shared_ptr<DifferentiableOperator> taylor_source_mf;
   std::shared_ptr<QuadratureData> qdata;
   mutable ParGridFunction mesh_nodes, rhsvc, dvc;
   mutable MassPAOperator *Mv = nullptr, *Me = nullptr;
   ParBilinearForm Mv_blf, Me_blf;
   HypreParMatrix Mv_mat, Me_mat;

   std::shared_ptr<LagrangianHydroResidualOperator> residual;
   std::shared_ptr<Preconditioner> preconditioner;
   std::shared_ptr<Solver> krylov;
   std::shared_ptr<NewtonSolver> newton;
#ifdef MFEM_USE_PETSC
   PetscNonlinearSolver *snes = nullptr;
#endif
   real_t current_dt = 0.0;
   int petsc_lag = 0;
   int lag = 0;

   mutable FunctionCoefficient rho0_coeff;
   OperatorJacobiSmoother *Mv_Jprec = nullptr;
   mutable Vector RHSv, rhsv, X, Xx, Xv, Xvc, Xe, K, Kx, Kv, Ke, B, RHSe, rhse;
   bool fd_gradient;
   int dump_jacobians;
   const int nonlinear_maximum_iterations;
   const real_t nonlinear_relative_tolerance;
   const int krylov_maximum_iterations;
   const int preconditioner_lag;
   const PRECONDITIONER_TYPE preconditioner_type;
   bool newton_converged = true;
   int  newton_max_iter = -1;
   Vector &external_data;
   const bool use_petsc;
   mutable bool qdata_is_current = false;
   // Stock Laghos time step estimate, min over all the RK stages (explicit)
   bool stage_dt_estimate = false;
   mutable real_t dt_est_stages = std::numeric_limits<real_t>::infinity();
};

// Builds the dFEM operators and returns a new LagrangianHydroOperator
template <int DIM>
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
   const bool &fd_gradient,
   const int &dump_jacobians,
   const int &nonlinear_maximum_iterations,
   const real_t &nonlinear_relative_tolerance,
   const int &krylov_maximum_iterations,
   const int &preconditioner_lag,
   const PRECONDITIONER_TYPE &preconditioner_type,
   const bool use_petsc);

} // namespace hydrodynamics

} // namespace mfem
