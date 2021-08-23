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
#include "laghos_shift.hpp"

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

class PressureFunction
{
public:
   enum PressureSpace {L2, H1};

private:
   const int p_order     = 1;
   const int basis_type  = BasisType::GaussLobatto;
   PressureSpace p_space;

   L2_FECollection p_fec_L2;
   H1_FECollection p_fec_H1;
   ParFiniteElementSpace p_fes_L2, p_fes_H1;
   ParGridFunction p_L2, p_H1;
   // Stores rho0 * det(J0)  at the pressure GF's nodes.
   Vector rho0DetJ0;
   ParGridFunction &gamma_gf;
   int problem = -1;

public:
   PressureFunction(ParMesh &pmesh, PressureSpace space,
                    ParGridFunction &rho0, int e_order, ParGridFunction &gamma);

   void UpdatePressure(const ParGridFunction &e);

   void SetProblem(int prob) { problem = prob; }

   ParGridFunction &GetPressure() { return (p_space == L2) ? p_L2 : p_H1; }
};

// Given a solutions state (x, v, e), this class performs all necessary
// computations to evaluate the new slopes (dx_dt, dv_dt, de_dt).
class LagrangianHydroOperator : public TimeDependentOperator
{
protected:
   ParFiniteElementSpace &H1, &H1cut, &L2;
   ParMesh *pmesh;
   // FE spaces local and global sizes
   const int H1Vsize, H1cutVsize;
   const int L2Vsize;
   Array<int> block_offsets;
   // Reference to the current mesh configuration.
   mutable ParGridFunction x_gf;
   PressureFunction &p_func;
   const Array<int> &ess_tdofs;
   const int dim, NE, l2dofs_cnt, source_type;
   const double cfl;
   const bool use_viscosity, use_vorticity;
   const double cg_rel_tol;
   const int cg_max_iter;
   const ParGridFunction &gamma_gf;
   // Velocity mass matrix and local inverses of the energy mass matrices. These
   // are constant in time, due to the pointwise mass conservation property.
   mutable BilinearForm Mv;
   SparseMatrix Mv_spmat_copy;
   DenseTensor Me, Me_inv;
   // Integration rule for all assemblies.
   const IntegrationRule &ir;
   const IntegrationRule *cfir;
   // Data associated with each quadrature point in the mesh.
   // These values are recomputed at each time step.
   const int Q1D;
   mutable QuadratureData qdata;
   mutable CutFaceQuadratureData cfqdata;
   mutable bool qdata_is_current, forcemat_is_assembled;
   // Force matrix that combines the kinematic and thermodynamic spaces. It is
   // assembled in each time step and then it is used to compute the final
   // right-hand sides for momentum and specific internal energy.
   mutable MixedBilinearForm Force, FaceForce;
   mutable Vector one, v_rhs, e_rhs;
   mutable LinearForm FaceForce_e, FaceForce_v, FaceForce_es;

   int v_shift_type = 0, e_shift_type = 0;

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
   LagrangianHydroOperator(const int size,
                           ParFiniteElementSpace &h1_fes,
                           ParFiniteElementSpace &h1cut_fes,
                           ParFiniteElementSpace &l2_fes,
                           const Array<int> &ess_tdofs,
                           Coefficient &rho0_coeff,
                           ParGridFunction &rho0_gf,
                           ParGridFunction &v_gf,
                           ParGridFunction &gamma_gf,
                           PressureFunction &pressure,
                           const int source,
                           const double cfl,
                           const bool visc, const bool vort,
                           const double cgt, const int cgiter,
                           const int order_q,
                           double *dt);
   ~LagrangianHydroOperator() { }

   // Solve for dx_dt, dv_dt and de_dt.
   virtual void Mult(const Vector &S, Vector &dS_dt) const;

   void SolveVelocity(const Vector &S, Vector &dS_dt) const;
   void SolveEnergy(const Vector &S, const Vector &v, Vector &dS_dt) const;
   void UpdateMesh(const Vector &S) const;

   // Calls UpdateQuadratureData to compute the new qdata.dt_estimate.
   double GetTimeStepEstimate(const Vector &S) const;
   void ResetTimeStepEstimate() const;
   void ResetQuadratureData() const { qdata_is_current = false; }

   void SetShiftingOptions(int problem, int v_shift, int e_shift,
                           double v_scale, double e_scale,
                           double v_stability_scale, double e_stability_scale)
   {
      p_func.SetProblem(problem);
      v_shift_type = v_shift;
      e_shift_type = e_shift;

      auto tfi_v = FaceForce.GetFBFI();
      auto v_integ = dynamic_cast<FaceForceIntegrator *>((*tfi_v)[0]);
      v_integ->SetScale(v_scale);

      auto tfi = FaceForce_e.GetTLFI();
      auto en_integ = dynamic_cast<EnergyInterfaceIntegrator *>((*tfi)[0]);
      en_integ->SetShiftType(e_shift_type);
      en_integ->SetScale(e_scale);

      auto tfi_vs = FaceForce_v.GetTLFI();
      auto vs_integ = dynamic_cast<VelocityStabilizerLFI *>((*tfi_vs)[0]);
      vs_integ->SetScale(v_stability_scale);

      auto tfi_es = FaceForce_es.GetTLFI();
      auto es_integ = dynamic_cast<EnergyStabilizerLFI *>((*tfi_es)[0]);
      es_integ->SetScale(e_stability_scale);
   }

   // The density values, which are stored only at some quadrature points,
   // are projected as a ParGridFunction.
   void ComputeDensity(ParGridFunction &rho) const;
   ParGridFunction &GetPressure(const ParGridFunction &e)
   {
      p_func.UpdatePressure(e);
      return p_func.GetPressure();
   }
   double InternalEnergy(const ParGridFunction &e) const;
   double KineticEnergy(const ParGridFunction &v) const;

   const Array<int> &GetBlockOffsets() const { return block_offsets; }
};

// TaylorCoefficient used in the 2D Taylor-Green problem.
class TaylorCoefficient : public Coefficient
{
public:
   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      Vector x(2);
      T.Transform(ip, x);
      return 3.0 / 8.0 * M_PI * ( cos(3.0*M_PI*x(0)) * cos(M_PI*x(1)) -
                                  cos(M_PI*x(0))     * cos(3.0*M_PI*x(1)) );
   }
};

// Acceleration source coefficient used in the 2D Rayleigh-Taylor problem.
class RTCoefficient : public VectorCoefficient
{
public:
   RTCoefficient(int dim) : VectorCoefficient(dim) { }
   using VectorCoefficient::Eval;
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      V = 0.0; V(1) = -1.0;
   }
};

} // namespace hydrodynamics

class HydroODESolver : public ODESolver
{
protected:
   hydrodynamics::LagrangianHydroOperator *hydro_oper;
public:
   HydroODESolver() : hydro_oper(NULL) { }
   virtual void Init(TimeDependentOperator&);
   virtual void Step(Vector&, double&, double&)
   { MFEM_ABORT("Time stepping is undefined."); }
};

class RK2AvgSolver : public HydroODESolver
{
protected:
   Vector V;
   BlockVector dS_dt, S0;
public:
   RK2AvgSolver() { }
   virtual void Init(TimeDependentOperator &_f);
   virtual void Step(Vector &S, double &t, double &dt);
};

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_LAGHOS
