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

#include <array>
#include <mfem.hpp>
#include "laghos_assembly.hpp"
#include "laghos_qf.hpp"

namespace mfem::hydrodynamics
{

// Visualize the given parallel grid function, using a GLVis server on the
// specified host and port. Set the visualization window title, and optionally,
// its geometry
void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    ParGridFunction &gf, const char *title,
                    int x, int y, int w, int h, bool vec = false);

struct HydroTimings
{
   enum Section
   {
      CG_H1, CG_L2, FORCE, QDATA, JACOBIAN_SETUP, JACOBIAN_ACTION,
      AMG_SETUP, AMG_APPLY, NEWTON, SOLVE, COMMUNICATION, COUNT
   };
   struct Entry
   {
      StopWatch watch;
      HYPRE_BigInt calls = 0, iterations = 0;
   };
   const bool enabled;
   std::array<Entry, COUNT> entries;
   HYPRE_BigInt h1_work = 0, l2_work = 0, force_work = 0, quad_work = 0;

   explicit HydroTimings(bool enabled) : enabled(enabled) {}

   class Scope
   {
      Entry *entry;
      static void Synchronize()
      {
         if (Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK))
         {
            MFEM_DEVICE_SYNC;
         }
      }
   public:
      Scope(HydroTimings &timing, Section section, bool count = true) :
         entry(timing.enabled ? &timing.entries[section] : nullptr)
      {
         if (!entry) { return; }
         Synchronize();
         entry->watch.Start();
         entry->calls += count;
      }
      Scope(const Scope &) = delete;
      Scope &operator=(const Scope &) = delete;
      void Stop()
      {
         if (!entry) { return; }
         Synchronize();
         entry->watch.Stop();
         entry = nullptr;
      }
      ~Scope() { Stop(); }
   };
};

struct OperatorVectors
{
   MultiVector input, output;
   OperatorVectors(int inputs, int outputs) : input(inputs), output(outputs) {}
};

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
      bool taylor_green = false;
      bool timings = false;
      int dump_jacobians = 0;
      int nonlinear_maximum_iterations = 10;
      real_t nonlinear_relative_tolerance = 1e-5;
      int krylov_maximum_iterations = 10;
      int preconditioner_lag = 0;
      int preconditioner_type = BLOCK_DIAGONAL_AMG;
      int qfunction_backend = LOCAL_QF;
   };

   class LineSearch : public NewtonSolver
   {
      real_t beta;        // Backtracking factor (how much to reduce step)
      real_t alpha;       // Sufficient decrease parameter (Armijo condition)
      int max_line_iter;  // Maximum line search iterations
      mutable Vector x_new, r_new, s_new;
      const LagrangianHydroOperator &hydro;
   public:
      LineSearch(MPI_Comm comm, const LagrangianHydroOperator &hydro)
         : NewtonSolver(comm), beta(0.5), alpha(1e-2), max_line_iter(5),
           hydro(hydro) {}
   protected:
      void ProcessNewState(const Vector &x) const override;
      // Override the scaling factor computation to implement line search
      real_t ComputeScalingFactor(const Vector &x, const Vector &b) const override;
   };

   class Jacobian : public Operator
   {
      const real_t h;
      mutable Vector w, z, source;
      mutable MultiVector zv_output{1}, ze_output{1}, source_output{1};
      LagrangianHydroOperator &hydro;
      std::shared_ptr<DerivativeOperator> dRvdx, dRvdv, dRvde, dRedx, dRedv, dRede,
          dTaylorSourcedx;
   public:
      Jacobian(LagrangianHydroOperator &hydro,
               std::shared_ptr<DerivativeOperator> dRvdx,
               std::shared_ptr<DerivativeOperator> dRvdv,
               std::shared_ptr<DerivativeOperator> dRvde,
               std::shared_ptr<DerivativeOperator> dRedx,
               std::shared_ptr<DerivativeOperator> dRedv,
               std::shared_ptr<DerivativeOperator> dRede,
               std::shared_ptr<DerivativeOperator> dTaylorSourcedx,
               real_t h) :
         Operator(2*hydro.H1tsize + hydro.L2tsize),
         h(h),
         w(height), z(height),
         source(hydro.taylor_source ? hydro.L2tsize : 0),
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
      mutable bool setup_pending = false;
      const Jacobian *jacobian = nullptr;
      std::shared_ptr<HypreParMatrix> vv_mat, ee_mat;
      std::shared_ptr<HypreBoomerAMG> amg_v, amg_e;
   public:
      Preconditioner(LagrangianHydroOperator &hydro) :
         Solver(hydro.Height()), hydro(hydro) {};
      void SetRebuildFlag(bool flag) { rebuild = flag; }
      void SetOperator(const Operator &op) override;
      void Mult(const Vector &x, Vector &y) const override;
   };

   struct Residual : public Operator
   {
      LagrangianHydroOperator &hydro;
      real_t dt;
      const Vector &x;
      mutable Vector u, u_l, e_source_t, rho0_t, x0_t, material_t;
      mutable MultiVector state{6};
      mutable std::shared_ptr<FDJacobian> fd_jacobian;
      mutable std::shared_ptr<Jacobian> jacobian;
      bool fd_gradient;
      int dump_jacobians;

      Residual(LagrangianHydroOperator &hydro, const Vector &x,
               const bool fd_gradient, const int dump_jacobians) :
         Operator(2*hydro.H1tsize+hydro.L2tsize),
         hydro(hydro),
         x(x),
         u(x.Size()),
         u_l(2*hydro.H1vsize + hydro.L2vsize),
         e_source_t(hydro.taylor_source ? hydro.L2tsize : 0),
         rho0_t(hydro.L2tsize), x0_t(hydro.H1tsize),
         material_t(hydro.material.ParFESpace()->GetTrueVSize()),
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
                           std::shared_ptr<DifferentiableOperator> momentum_mf,
                           std::shared_ptr<DifferentiableOperator> momentum_pa,
                           std::shared_ptr<DifferentiableOperator> energy_conservation_mf,
                           std::shared_ptr<DifferentiableOperator> rk2avg_energy_mf,
                           std::shared_ptr<DifferentiableOperator> energy_conservation_pa,
                           std::shared_ptr<DifferentiableOperator> total_internal_energy_mf,
                           std::shared_ptr<DifferentiableOperator> total_kinetic_energy_mf,
                           std::shared_ptr<DifferentiableOperator> density_mf,
                           std::shared_ptr<DifferentiableOperator> taylor_source_mf,
                           std::shared_ptr<QuadratureData> qdata,
                           const Options &options);
   ~LagrangianHydroOperator();

   // Solve for dx_dt, dv_dt and de_dt.
   void Mult(const Vector &S, Vector &dSdt) const override;

   virtual MemoryClass GetMemoryClass() const override
   { return Device::GetDeviceMemoryClass(); }

   void UpdateMesh(const Vector &S) const;
   void SolveVelocity(const Vector &S, Vector &dSdt) const;
   void SolveEnergy(const Vector &S, const Vector &velocity,
                    Vector &dSdt) const;
   void ApplyMomentum(Vector &force) const;
   void ApplyEnergy(const Vector &velocity, Vector &force) const;

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
   void PrintTimingData(bool fom) const;

   void DensityScatter();

   real_t ComputeMinDet(const Vector &S) const;

   void UpdateQuadratureData(const Vector &S) const;

   // Min of the time step estimate over the local quadrature points at S
   real_t LocalTimeStepEstimate(const Vector &S) const;

   ParFiniteElementSpace &H1, &L2;
   mutable ParFiniteElementSpace H1c;
   // Spaces stay fixed; UpdateMesh only moves nodes
   const int H1vsize, L2vsize, H1tsize, L2tsize, H1ctsize;
   const int L2dofs, dim, ne;
   Array<int> &ess_tdof;
   mutable Array<int> c_tdofs[3];
   const IntegrationRule &ir;
   const int nqp;
   ParGridFunction &x0, &rho0, &material;
   std::shared_ptr<DifferentiableOperator> update_qdata;
   std::shared_ptr<DifferentiableOperator> momentum_mf;
   std::shared_ptr<DifferentiableOperator> momentum_pa;
   std::shared_ptr<DifferentiableOperator> energy_mf;
   std::shared_ptr<DifferentiableOperator> rk2avg_energy_mf;
   std::shared_ptr<DifferentiableOperator> energy_pa;
   std::shared_ptr<DifferentiableOperator> internal_energy_mf;
   std::shared_ptr<DifferentiableOperator> kinetic_energy_mf;
   std::shared_ptr<DifferentiableOperator> density_mf;
   std::shared_ptr<DifferentiableOperator> taylor_source_mf;
   // Taylor-Green energy source, 2D only as in stock Laghos
   const bool taylor_source;
   std::shared_ptr<QuadratureData> qdata;
   mutable ParGridFunction mesh_nodes, rhsvc, dvc;
   mutable MassPAOperator *Mv = nullptr, *Me = nullptr;
   ParBilinearForm Mv_blf, Me_blf;
   HypreParMatrix Mv_mat, Me_mat;

   std::shared_ptr<Residual> residual;
   std::shared_ptr<Preconditioner> preconditioner;
   std::shared_ptr<Solver> krylov;
   std::shared_ptr<NewtonSolver> newton;
   real_t current_dt = 0.0;
   int lag = 0;

   mutable FunctionCoefficient rho0_coeff;
   OperatorJacobiSmoother *Mv_Jprec = nullptr;
   mutable Vector RHSv, rhsv, X, Xvc, K, B, RHSe, rhse;
   mutable CGSolver cg_v, cg_e;
   mutable Vector source_x, source_e, energy_state, energy_values;
   mutable ParGridFunction density_rhs;
   mutable Vector density_rho0, density_rhs_t, density_rhs_e, density_values;
   mutable DenseMatrix density_mass;
   mutable DenseMatrixInverse density_inverse;
   mutable Array<int> density_dofs;
   mutable MassIntegrator density_integrator;
   mutable OperatorVectors momentum_vectors{1, 1}, energy_vectors{2, 1};
   mutable OperatorVectors qdata_vectors{6, 2}, source_vectors{1, 1};
   mutable OperatorVectors diagnostic_vectors{3, 1};
   const Options options;
   mutable HydroTimings timing;
   bool newton_converged = true;
   int  newton_max_iter = -1;
   mutable bool qdata_is_current = false;
   // Laghos time step estimate, min over all the RK stages
   bool stage_dt_estimate = false;
   mutable real_t dt_est_stages = std::numeric_limits<real_t>::infinity();
};

// Initial zone size of the mesh, over the order of the velocity space
real_t InitialMeshSize(ParMesh &mesh, const int order);

// Builds the dFEM operators of the mesh dimension
// and returns a new LagrangianHydroOperator
LagrangianHydroOperator *CreateLagrangianHydroOperator(
   ParFiniteElementSpace &H1,
   ParFiniteElementSpace &L2,
   Array<int> &ess_tdof,
   FunctionCoefficient &rho0_coeff,
   ParGridFunction &x0_gf,
   ParGridFunction &rho0_gf,
   ParGridFunction &material_gf,
   const QConstants &qc,
   const IntegrationRule &ir,
   const LagrangianHydroOperator::Options &options);

} // namespace mfem::hydrodynamics

namespace mfem
{

class HydroODESolver : public ODESolver
{
protected:
   hydrodynamics::LagrangianHydroOperator *hydro_oper;
public:
   HydroODESolver() : hydro_oper(NULL) { }
   void Init(TimeDependentOperator &tdop) override;
   void Step(Vector&, double&, double&) override
   { MFEM_ABORT("Time stepping is undefined."); }
};

class RK2AvgSolver : public HydroODESolver
{
protected:
   Vector V, dS_dt, S0;
public:
   RK2AvgSolver() { }
   void Init(TimeDependentOperator &tdop) override;
   void Step(Vector &S, double &t, double &dt) override;
};

} // namespace mfem
