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
#include "laghos_assembly.hpp"
#include <fem/dfem/doperator.hpp>

using mfem::future::DifferentiableOperator;
using mfem::future::DerivativeOperator;
using mfem::future::FieldDescriptor;
using mfem::future::FDJacobian;
using mfem::future::Gradient;
using mfem::future::Identity;
using mfem::future::Weight;
using mfem::future::Value;

using mfem::future::tensor;

namespace mfem
{

namespace hydrodynamics
{

// Visualize the given parallel grid function, using a GLVis server on the
// specified host and port. Set the visualization window title, and optionally,
// its geometry
void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    ParGridFunction &gf, const char *title,
                    int x, int y, int w, int h, bool vec = false);

// Given a solutions state (x, v, e), this class performs all necessary
// computations to evaluate the new slopes (dx_dt, dv_dt, de_dt).
class LagrangianHydroOperator : public TimeDependentOperator
{
public:

   enum PRECONDITIONER_TYPE { BLOCK_DIAGONAL_AMG = 0 };

   // Solver options, with the laghos command line defaults
   struct Options
   {
      real_t cg_relative_tolerance = 1e-8;
      bool fd_gradient = false;
      int dump_jacobians = 0;
      int nonlinear_maximum_iterations = 10;
      real_t nonlinear_relative_tolerance = 1e-5;
      int krylov_maximum_iterations = 10;
      int preconditioner_lag = 0;
      int preconditioner_type = BLOCK_DIAGONAL_AMG;
   };

   class LineSearchNewtonSolver : public NewtonSolver
   {
      real_t beta;        // Backtracking factor (how much to reduce step)
      real_t alpha;       // Sufficient decrease parameter
      int max_line_iter;  // Maximum line search iterations
      mutable Vector x_new, r_new, s_new;
      const LagrangianHydroOperator &hydro;
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
   };

   class LagrangianHydroJacobianOperator : public Operator
   {
      const real_t h;
      const int H1tsize, L2tsize;
      mutable Vector w, z;
      LagrangianHydroOperator &hydro;
      std::shared_ptr<DerivativeOperator>
      dRvdx, dRvdv, dRvde, dRedx, dRedv, dRede, dTaylorSourcedx;
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
      const real_t GetH() const { return h; }
      auto GetdRvdv() const { return dRvdv; }
      auto GetdRede() const { return dRede; }
   };

   class Preconditioner : public Solver
   {
      mutable Vector w;
      LagrangianHydroOperator &hydro;
      bool rebuild = true;
      const LagrangianHydroJacobianOperator *jacobian = nullptr;
      std::shared_ptr<HypreParMatrix> vv_mat, ee_mat;
      std::shared_ptr<HypreBoomerAMG> amg_v, amg_e;
   public:
      Preconditioner(LagrangianHydroOperator &hydro) :
         Solver(hydro.Height()), hydro(hydro) {};
      void SetRebuildFlag(bool flag) { rebuild = flag; }
      void SetOperator(const Operator &op) override;
      void Mult(const Vector &x, Vector &y) const override;
   };

   class LagrangianHydroResidualOperator : public Operator
   {
   public:
      LagrangianHydroOperator &hydro;
      real_t dt;
      const int H1tsize, H1vsize, L2tsize, L2vsize;
      const Vector &x;
      mutable Vector u, u_l, e_source_t;
      mutable std::shared_ptr<FDJacobian> fd_jacobian;
      mutable std::shared_ptr<LagrangianHydroJacobianOperator> jacobian;
      bool fd_gradient;
      int dump_jacobians;
   public:
      LagrangianHydroResidualOperator(LagrangianHydroOperator &hydro, const Vector &x,
                                      const bool fd_gradient, const int dump_jacobians) :
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
   };

   LagrangianHydroOperator(ParFiniteElementSpace &H1,
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
                           const Options &options,
                           Vector& external_data);
   ~LagrangianHydroOperator();

   // Solve for dx_dt, dv_dt and de_dt.
   void Mult(const Vector &S, Vector &dSdt) const override;

   virtual MemoryClass GetMemoryClass() const override
   { return Device::GetDeviceMemoryClass(); }

   void UpdateMesh(const Vector &S) const;

   // Calls UpdateQuadratureData to compute the new qdata.dt_estimate.
   real_t GetTimeStepEstimate(const Vector &S);
   void ResetTimeStepEstimate() const;
   void ResetQuadratureData() const;

   // The density values, which are stored only at some quadrature points,
   // are projected as a ParGridFunction.
   void ComputeDensity(ParGridFunction &rho) const;
   real_t InternalEnergy(ParGridFunction &e) const;
   real_t KineticEnergy(ParGridFunction &v) const;

   void ImplicitSolve(const real_t dt, const Vector &x, Vector &k) override;

   void ResetNewtonInfo();

   void DensityScatter();

   real_t ComputeMinDet(const Vector &S) const;

   void UpdateQuadratureData(const Vector &S) const;

   // Min of the time step estimate over the local quadrature points at S
   real_t LocalTimeStepEstimate(const Vector &S) const;

   ParFiniteElementSpace &H1, &L2;
   mutable ParFiniteElementSpace H1c;
   Array<int> &ess_tdof;
   mutable Array<int> c_tdofs[3];
   const IntegrationRule &ir;
   ParGridFunction &x0, &rho0, &material;
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
   // Taylor-Green energy source, 2D only as in stock Laghos
   const bool taylor_source;
   std::shared_ptr<QuadratureData> qdata;
   mutable ParGridFunction mesh_nodes, rhsvc, dvc;
   mutable MassPAOperator *Mv = nullptr, *Me = nullptr;
   ParBilinearForm Mv_blf, Me_blf;
   HypreParMatrix Mv_mat, Me_mat;

   std::shared_ptr<LagrangianHydroResidualOperator> residual;
   std::shared_ptr<Preconditioner> preconditioner;
   std::shared_ptr<Solver> krylov;
   std::shared_ptr<NewtonSolver> newton;
   real_t current_dt = 0.0;
   int lag = 0;

   mutable FunctionCoefficient rho0_coeff;
   OperatorJacobiSmoother *Mv_Jprec = nullptr;
   mutable Vector RHSv, rhsv, X, Xvc, K, B, RHSe, rhse;
   const Options options;
   bool newton_converged = true;
   int  newton_max_iter = -1;
   Vector &external_data;
   mutable bool qdata_is_current = false;
   // Laghos time step estimate, min over all the RK stages
   bool stage_dt_estimate = false;
   mutable real_t dt_est_stages = std::numeric_limits<real_t>::infinity();
};

} // namespace hydrodynamics

} // namespace mfem
