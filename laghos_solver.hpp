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
#include "dist_solver.hpp"
#include "marking.hpp"
#include "ghost_penalty.hpp"
#include "volume_fractions.hpp" 

#ifdef MFEM_USE_MPI
class Dist_Level_Set_Coefficient;
class Combo_Level_Set_Coefficient;

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

    // Given a solutions state (x, v, e), this class performs all necessary
    // computations to evaluate the new slopes (dx_dt, dv_dt, de_dt).
    class LagrangianHydroOperator : public TimeDependentOperator
    {
    protected:
      ParFiniteElementSpace &H1, &L2, &P_L2, &PFace_L2;
      mutable ParFiniteElementSpace H1c;
      mfem::ParFiniteElementSpace* alpha_fes;
      mfem::L2_FECollection* alpha_fec;     
      ParMesh *pmesh;
      // FE spaces local and global sizes
      const int H1Vsize;
      const int L2Vsize;
      Array<int> block_offsets;
      // Reference to the current mesh configuration.
      mutable ParGridFunction x_gf;
      Array<int> ess_tdofs;
      const int dim, NE, l2dofs_cnt, h1dofs_cnt, source_type;
      const double cfl;
      const bool use_viscosity, use_vorticity;
      const double cg_rel_tol;
      const int cg_max_iter;
      const double ftz_tol;
      const ParGridFunction &rho0_gf;
      const ParGridFunction &gamma_gf;
      ParGridFunction &v_gf;
      ParGridFunction &p_gf;
      ParGridFunction &e_gf;
      ParGridFunction &cs_gf;
      ParGridFunction &rho_gf;
      // Grid Functions for face terms
      ParGridFunction &pface_gf;
      ParGridFunction &csface_gf;
      ParGridFunction &rhoface_gf;
      ParGridFunction &viscousface_gf;
      ParGridFunction &rho0DetJ0face_gf;
      ParGridFunction &Jac0invface_gf;
			    
      IntegrationRules GLIntRules;
      double &globalmax_rho;
      double &globalmax_cs;
      double &globalmax_viscous_coef;
      
      // Velocity mass matrix and local inverses of the energy mass matrices. These
      // are constant in time, due to the pointwise mass conservation property.
      mutable ParBilinearForm Mv;
      SparseMatrix Mv_spmat_copy;
      DenseTensor Me, Me_inv;
      // Integration rule for all assemblies.
      const IntegrationRule &ir;
      const IntegrationRule &b_ir;
      // Data associated with each quadrature point in the mesh.
      // These values are recomputed at each time step.
      const int Q1D;
      mutable QuadratureData qdata;
      mutable QuadratureDataGL gl_qdata; 
      mutable bool qdata_is_current, forcemat_is_assembled, energyforcemat_is_assembled, bv_qdata_is_current, be_qdata_is_current, bv_forcemat_is_assembled, be_forcemat_is_assembled, bvemb_forcemat_is_assembled, bvemb_qdata_is_current, beemb_forcemat_is_assembled, beemb_qdata_is_current;
      // Force matrix that combines the kinematic and thermodynamic spaces. It is
      // assembled in each time step and then it is used to compute the final
      // right-hand sides for momentum and specific internal energy.
      mutable ParLinearForm Force;
      mutable ParLinearForm EnergyForce;
      mutable ParLinearForm VelocityBoundaryForce;
      mutable ParLinearForm EnergyBoundaryForce;
      mutable ParLinearForm ShiftedVelocityBoundaryForce;
      mutable ParLinearForm ShiftedEnergyBoundaryForce;
      mutable Vector X, B, one, rhs, e_rhs, b_rhs, be_rhs;
      const double penaltyParameter;
      const double nitscheVersion;
      const bool useEmbedded;
      const int geometricShape;
      Array<int> ess_elem;

      ForceIntegrator *fi;
      EnergyForceIntegrator *efi;
      VelocityBoundaryForceIntegrator *v_bfi;
      EnergyBoundaryForceIntegrator *e_bfi;
      NormalVelocityMassIntegrator *nvmi;
      
      ShiftedVelocityBoundaryForceIntegrator *shifted_v_bfi;
      ShiftedEnergyBoundaryForceIntegrator *shifted_e_bfi;
      ShiftedNormalVelocityMassIntegrator *shifted_nvmi;
      
      Dist_Level_Set_Coefficient *wall_dist_coef;
      // in case we are using level set to get distance and normal vectors
      ParFiniteElementSpace *distance_vec_space;
      ParGridFunction *distance;
      ParFiniteElementSpace *normal_vec_space;
      ParGridFunction *normal;
      ParGridFunction *ls_func;
      ParGridFunction *level_set_gf;
      ParGridFunction *alphaCut;
      //  
      ShiftedFaceMarker *analyticalSurface;
      VectorCoefficient *dist_vec;
      VectorCoefficient *normal_vec;
      int nTerms;
      bool fullPenalty;
      double C_I_E;
      double C_I_V;
  
      void UpdateQuadratureData(const Vector &S) const;
      void AssembleForceMatrix() const;
      void AssembleEnergyForceMatrix() const;
      void AssembleVelocityBoundaryForceMatrix() const;
      void AssembleEnergyBoundaryForceMatrix() const;
      void AssembleShiftedEnergyBoundaryForceMatrix() const;

    public:
      LagrangianHydroOperator(const int size, const int order_e, const int order_v,
			      double &globalmax_rho,
			      double &globalmax_cs, double &globalmax_viscous_coef,
			      ParFiniteElementSpace &h1_fes,
			      ParFiniteElementSpace &l2_fes,
			      ParFiniteElementSpace &p_l2_fes,
			      ParFiniteElementSpace &pface_l2_fes,
			      Coefficient &rho0_coeff,
			      ParGridFunction &rho0_gf,
			      ParGridFunction &rho_gf,
			      ParGridFunction &rhoface_gf,
			      ParGridFunction &gamma_gf,
			      ParGridFunction &p_gf,
			      ParGridFunction &pface_gf,
			      ParGridFunction &v_gf,
			      ParGridFunction &e_gf,
			      ParGridFunction &cs_gf,
			      ParGridFunction &csface_gf,
			      ParGridFunction &viscousface_gf,
			      ParGridFunction &rho0DetJ0face_gf,
			      ParGridFunction &Jac0invface_gf,
			      const int source,
			      const double cfl,
			      const bool visc, const bool vort,
			      const double cgt, const int cgiter, double ftz_tol,
			      const int order_q, const double penaltyParameter,
			      const double nitscheVersion, const bool useEmb, const int gS, int nT, bool fP);
      ~LagrangianHydroOperator();

      // Solve for dx_dt, dv_dt and de_dt.
      virtual void Mult(const Vector &S, Vector &dS_dt, const Vector &S_init) const;

      void SolveVelocity(const Vector &S, Vector &dS_dt, const Vector &S_init, const double dt) const;
      void SolveEnergy(const Vector &S, const Vector &v, Vector &dS_dt) const;
      void UpdateMesh(const Vector &S) const;

      // Calls UpdateQuadratureData to compute the new qdata.dt_estimate.
      double GetTimeStepEstimate(const Vector &S) const;
      void ResetTimeStepEstimate() const;
      void ResetQuadratureData() const {
	qdata_is_current = false;
	bv_qdata_is_current = false;
	be_qdata_is_current = false;
	bvemb_qdata_is_current = false;
	beemb_qdata_is_current = false;}

      // The density values, which are stored only at some quadrature points,
      // are projected as a ParGridFunction.
      void ComputeDensity(ParGridFunction &rho) const;
      double InternalEnergy(const ParGridFunction &e) const;
      double KineticEnergy(const ParGridFunction &v) const;

      int GetH1VSize() const { return H1.GetVSize(); }
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
    // S_init is needed to store the initial solution state (x, v, e)
    BlockVector dS_dt, S0, S_init;
    int counter;
  public:
    RK2AvgSolver():counter(0) { }
    virtual void Init(TimeDependentOperator &_f);
    virtual void Step(Vector &S, double &t, double &dt);
  };

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_LAGHOS
