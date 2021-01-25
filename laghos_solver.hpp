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
#include "laghos_assembly.hpp" // QuadratureData

#ifdef MFEM_USE_MPI

namespace mfem
{

// Choice for the problem setup, statically used in functions below.
   static int problem, dim;

static double gamma_func(const Vector &x)
{
   switch (problem)
   {
      case 0: return 5.0 / 3.0;
      case 1: return 1.4;
      case 2: return 1.4;
      case 3: return (x(0) > 1.0 && x(1) <= 1.5) ? 1.4 : 1.5;
      case 4: return 5.0 / 3.0;
      case 5: return 1.4;
      case 6: return 1.4;
      case 7: return 5.0 / 3.0;
      default: MFEM_ABORT("Bad number given for problem id!"); return 0.0;
   }
}

static double rho0(const Vector &x)
{
   switch (problem)
   {
      case 0: return 1.0;
      case 1: return 1.0;
      case 2: return (x(0) < 0.5) ? 1.0 : 0.1;
   case 3: return (dim == 2) ? (x(0) > 1.0 && x(1) > 1.5) ? 0.125 : 1.0
         : x(0) > 1.0 && ((x(1) < 1.5 && x(2) < 1.5) ||
                          (x(1) > 1.5 && x(2) > 1.5)) ? 0.125 : 1.0;

      case 4: return 1.0;
      case 5:
      {
         if (x(0) >= 0.5 && x(1) >= 0.5) { return 0.5313; }
         if (x(0) <  0.5 && x(1) <  0.5) { return 0.8; }
         return 1.0;
      }
      case 6:
      {
         if (x(0) <  0.5 && x(1) >= 0.5) { return 2.0; }
         if (x(0) >= 0.5 && x(1) <  0.5) { return 3.0; }
         return 1.0;
      }
      case 7: return x(1) >= 0.0 ? 2.0 : 1.0;
      default: MFEM_ABORT("Bad number given for problem id!"); return 0.0;
   }
}

static double radius(double x, double y) { return sqrt(x*x + y*y); }

static double e0(const Vector &x)
{
   switch (problem)
   {
      case 0:
      {
         const double denom = 2.0 / 3.0;  // (5/3 - 1) * density.
         double val;
         if (x.Size() == 2)
         {
            val = 1.0 + (cos(2*M_PI*x(0)) + cos(2*M_PI*x(1))) / 4.0;
         }
         else
         {
            val = 100.0 + ((cos(2*M_PI*x(2)) + 2) *
                           (cos(2*M_PI*x(0)) + cos(2*M_PI*x(1))) - 2) / 16.0;
         }
         return val/denom;
      }
      case 1: return 0.0; // This case in initialized in main().
      case 2: return (x(0) < 0.5) ? 1.0 / rho0(x) / (gamma_func(x) - 1.0)
                        : 0.1 / rho0(x) / (gamma_func(x) - 1.0);
      case 3: return (x(0) > 1.0) ? 0.1 / rho0(x) / (gamma_func(x) - 1.0)
                        : 1.0 / rho0(x) / (gamma_func(x) - 1.0);
      case 4:
      {
         const double r = radius(x(0), x(1)), rsq = x(0) * x(0) + x(1) * x(1);
         const double gamma = 5.0 / 3.0;
         if (r < 0.2)
         {
            return (5.0 + 25.0 / 2.0 * rsq) / (gamma - 1.0);
         }
         else if (r < 0.4)
         {
            const double t1 = 9.0 - 4.0 * log(0.2) + 25.0 / 2.0 * rsq;
            const double t2 = 20.0 * r - 4.0 * log(r);
            return (t1 - t2) / (gamma - 1.0);
         }
         else { return (3.0 + 4.0 * log(2.0)) / (gamma - 1.0); }
      }
      case 5:
      {
         const double irg = 1.0 / rho0(x) / (gamma_func(x) - 1.0);
         if (x(0) >= 0.5 && x(1) >= 0.5) { return 0.4 * irg; }
         if (x(0) <  0.5 && x(1) >= 0.5) { return 1.0 * irg; }
         if (x(0) <  0.5 && x(1) <  0.5) { return 1.0 * irg; }
         if (x(0) >= 0.5 && x(1) <  0.5) { return 1.0 * irg; }
         MFEM_ABORT("Error in problem 5!");
         return 0.0;
      }
      case 6:
      {
         const double irg = 1.0 / rho0(x) / (gamma_func(x) - 1.0);
         if (x(0) >= 0.5 && x(1) >= 0.5) { return 1.0 * irg; }
         if (x(0) <  0.5 && x(1) >= 0.5) { return 1.0 * irg; }
         if (x(0) <  0.5 && x(1) <  0.5) { return 1.0 * irg; }
         if (x(0) >= 0.5 && x(1) <  0.5) { return 1.0 * irg; }
         MFEM_ABORT("Error in problem 5!");
         return 0.0;
      }
      case 7:
      {
         const double rho = rho0(x), gamma = gamma_func(x);
         return (6.0 - rho * x(1)) / (gamma - 1.0) / rho;
      }
      default: MFEM_ABORT("Bad number given for problem id!"); return 0.0;
   }
}

static void v0(const Vector &x, Vector &v)
{
   const double atn = pow((x(0)*(1.0-x(0))*4*x(1)*(1.0-x(1))*4.0),0.4);
   switch (problem)
   {
      case 0:
         v(0) =  sin(M_PI*x(0)) * cos(M_PI*x(1));
         v(1) = -cos(M_PI*x(0)) * sin(M_PI*x(1));
         if (x.Size() == 3)
         {
            v(0) *= cos(M_PI*x(2));
            v(1) *= cos(M_PI*x(2));
            v(2) = 0.0;
         }
         break;
      case 1: v = 0.0; break;
      case 2: v = 0.0; break;
      case 3: v = 0.0; break;
      case 4:
      {
         v = 0.0;
         const double r = radius(x(0), x(1));
         if (r < 0.2)
         {
            v(0) =  5.0 * x(1);
            v(1) = -5.0 * x(0);
         }
         else if (r < 0.4)
         {
            v(0) =  2.0 * x(1) / r - 5.0 * x(1);
            v(1) = -2.0 * x(0) / r + 5.0 * x(0);
         }
         else { }
         break;
      }
      case 5:
      {
         v = 0.0;
         if (x(0) >= 0.5 && x(1) >= 0.5) { v(0)=0.0*atn, v(1)=0.0*atn; return;}
         if (x(0) <  0.5 && x(1) >= 0.5) { v(0)=0.7276*atn, v(1)=0.0*atn; return;}
         if (x(0) <  0.5 && x(1) <  0.5) { v(0)=0.0*atn, v(1)=0.0*atn; return;}
         if (x(0) >= 0.5 && x(1) <  0.5) { v(0)=0.0*atn, v(1)=0.7276*atn; return; }
         MFEM_ABORT("Error in problem 5!");
         return;
      }
      case 6:
      {
         v = 0.0;
         if (x(0) >= 0.5 && x(1) >= 0.5) { v(0)=+0.75*atn, v(1)=-0.5*atn; return;}
         if (x(0) <  0.5 && x(1) >= 0.5) { v(0)=+0.75*atn, v(1)=+0.5*atn; return;}
         if (x(0) <  0.5 && x(1) <  0.5) { v(0)=-0.75*atn, v(1)=+0.5*atn; return;}
         if (x(0) >= 0.5 && x(1) <  0.5) { v(0)=-0.75*atn, v(1)=-0.5*atn; return;}
         MFEM_ABORT("Error in problem 6!");
         return;
      }
      case 7:
      {
         v = 0.0;
         v(1) = 0.02 * exp(-2*M_PI*x(1)*x(1)) * cos(2*M_PI*x(0));
         break;
      }
      default: MFEM_ABORT("Bad number given for problem id!");
   }
}

static void GetZeroBCDofs(ParMesh *pmesh, ParFiniteElementSpace &H1,
                          const int bdr_attr_max,
                          Array<int> &ess_tdofs,
                          Array<int> &ess_vdofs)
{
   ess_tdofs.SetSize(0);
   ess_vdofs.SetSize(0);
   Array<int> ess_bdr(bdr_attr_max), dofs_marker, dofs_list;
   for (int d = 0; d < pmesh->Dimension(); d++)
   {
      // Attributes 1/2/3 correspond to fixed-x/y/z boundaries,
      // i.e., we must enforce v_x/y/z = 0 for the velocity components.
      ess_bdr = 0; ess_bdr[d] = 1;
      H1.GetEssentialTrueDofs(ess_bdr, dofs_list, d);
      ess_tdofs.Append(dofs_list);
      H1.GetEssentialVDofs(ess_bdr, dofs_marker, d);
      FiniteElementSpace::MarkerToList(dofs_marker, dofs_list);
      ess_vdofs.Append(dofs_list);
   }
}

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

   // Store the number of dofs of the corresponding local CG
   const HYPRE_Int L2dof;

   // These accumulate the total processed dofs or quad points:
   // #(CG iterations) for the L2 CG solve.
   // #quads * #(RK sub steps) for the quadrature data computations.
   HYPRE_Int H1iter, L2iter;
   HYPRE_Int quad_tstep;

   TimingData(const HYPRE_Int l2d) :
      L2dof(l2d), H1iter(0), L2iter(0), quad_tstep(0) { }
};

class QUpdate;

// Given a solutions state (x, v, e), this class performs all necessary
// computations to evaluate the new slopes (dx_dt, dv_dt, de_dt).
class LagrangianHydroOperator : public TimeDependentOperator
{
protected:
   ParFiniteElementSpace &H1, &L2;
   mutable ParFiniteElementSpace H1c;
   ParMesh *pmesh;
   // FE spaces local and global sizes
   int H1Vsize;
   int H1TVSize;
   HYPRE_Int H1GTVSize;
   int L2Vsize;
   int L2TVSize;
   HYPRE_Int L2GTVSize;
   Array<int> block_offsets;
   // Reference to the current mesh configuration.
   mutable ParGridFunction x_gf;
   const Array<int> &ess_tdofs;
   int NE;
   const int dim;
   int l2dofs_cnt, h1dofs_cnt;
   const int source_type;
   const double cfl;
   const bool use_viscosity, use_vorticity, p_assembly, amr;
   const double cg_rel_tol;
   const int cg_max_iter;
   const double ftz_tol;
   ParGridFunction &gamma_gf;

   ParGridFunction rho0_gf;
   ParGridFunction x0_gf; // copy of initial mesh position
   Coefficient &rho0_coeff;
   GridFunctionCoefficient rho0_gf_coeff; // TODO: remove when Mv update improved

   // Velocity mass matrix and local inverses of the energy mass matrices. These
   // are constant in time, due to the pointwise mass conservation property.
   mutable ParBilinearForm Mv;
   SparseMatrix Mv_spmat_copy;
   DenseTensor Me, Me_inv;
   // Integration rule for all assemblies.
   const IntegrationRule &ir;
   // Data associated with each quadrature point in the mesh.
   // These values are recomputed at each time step.
   const int Q1D;
   mutable QuadratureData qdata;
   mutable bool qdata_is_current, forcemat_is_assembled;
   // Force matrix that combines the kinematic and thermodynamic spaces. It is
   // assembled in each time step and then it is used to compute the final
   // right-hand sides for momentum and specific internal energy.
   mutable MixedBilinearForm Force;
   // Same as above, but done through partial assembly.
   ForcePAOperator *ForcePA;
   // Mass matrices done through partial assembly:
   // velocity (coupled H1 assembly) and energy (local L2 assemblies).
   MassPAOperator *VMassPA, *EMassPA;
   OperatorJacobiSmoother *VMassPA_Jprec;
   // Linear solver for energy.
   CGSolver CG_VMass, CG_EMass;
   mutable TimingData timer;
   mutable QUpdate *qupdate;
   mutable Vector X, B, one, rhs, e_rhs;
   mutable ParGridFunction rhs_c_gf, dvc_gf;
   mutable Array<int> c_tdofs[3];
   mutable Vector zone_max_visc, zone_vgrad;

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
                           ParFiniteElementSpace &l2_fes,
                           const Array<int> &ess_tdofs,
                           Coefficient &rho0_coeff,
                           ParGridFunction &rho0_gf,
                           ParGridFunction &gamma_gf,
                           const int source,
                           const double cfl,
                           const bool visc, const bool vort,
                           const bool p_assembly, const bool amr,
                           const double cgt, const int cgiter, double ftz_tol,
                           const int order_q);
   ~LagrangianHydroOperator();

   // Solve for dx_dt, dv_dt and de_dt.
   virtual void Mult(const Vector &S, Vector &dS_dt) const;

   virtual MemoryClass GetMemoryClass() const
   { return Device::GetMemoryClass(); }

   void SolveVelocity(const Vector &S, Vector &dS_dt) const;
   void SolveEnergy(const Vector &S, const Vector &v, Vector &dS_dt) const;
   void UpdateMesh(const Vector &S) const;

   // Calls UpdateQuadratureData to compute the new qdata.dt_estimate.
   double GetTimeStepEstimate(const Vector &S) const;
   void ResetTimeStepEstimate() const;
   void ResetQuadratureData() const { qdata_is_current = false; }

   // The density values, which are stored only at some quadrature points,
   // are projected as a ParGridFunction.
   void ComputeDensity(ParGridFunction &rho) const;
   double InternalEnergy(const ParGridFunction &e) const;
   double KineticEnergy(const ParGridFunction &v) const;

   int GetH1VSize() const { return H1.GetVSize(); }
   const Array<int> &GetBlockOffsets() const { return block_offsets; }

   void PrintTimingData(bool IamRoot, int steps, const bool fom) const;

   void SetH0(double h0) { qdata.h0 = h0; }
   double GetH0() const { return qdata.h0; }

   Vector& GetZoneMaxVisc() { return zone_max_visc; }
   Vector& GetZoneVGrad() { return zone_vgrad; }

   void AMRUpdate(const Vector&, const bool quick);
};

class QUpdate
{
private:
   const int dim, vdim, NQ, NE, Q1D;
   const bool use_viscosity, use_vorticity;
   const double cfl;
   TimingData *timer;
   const IntegrationRule &ir;
   ParFiniteElementSpace &H1, &L2;
   const Operator *H1R;
   Vector q_dt_est, q_e, e_vec, q_dx, q_dv;
   const QuadratureInterpolator *q1,*q2;
   const ParGridFunction &gamma_gf;
public:
   QUpdate(const int d, const int ne, const int q1d,
           const bool visc, const bool vort,
           const double cfl, TimingData *t,
           const ParGridFunction &gamma_gf,
           const IntegrationRule &ir,
           ParFiniteElementSpace &h1, ParFiniteElementSpace &l2):
      dim(d), vdim(h1.GetVDim()),
      NQ(ir.GetNPoints()), NE(ne), Q1D(q1d),
      use_viscosity(visc), use_vorticity(vort), cfl(cfl),
      timer(t), ir(ir), H1(h1), L2(l2),
      H1R(H1.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC)),
      q_dt_est(NE*NQ),
      q_e(NE*NQ),
      e_vec(NQ*NE*vdim),
      q_dx(NQ*NE*vdim*vdim),
      q_dv(NQ*NE*vdim*vdim),
      q1(H1.GetQuadratureInterpolator(ir)),
      q2(L2.GetQuadratureInterpolator(ir)),
      gamma_gf(gamma_gf) { }

   void UpdateQuadratureData(const Vector&, QuadratureData&, Vector&, Vector&);
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
