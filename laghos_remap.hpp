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

#ifndef MFEM_LAGHOS_REMAP
#define MFEM_LAGHOS_REMAP

#include "mfem.hpp"
#include <functional>

namespace mfem
{

namespace ale
{

class SolutionTransfer;
struct MaterialData;

#ifdef MFEM_USE_GSLIB
class InterpolationRemap
{
   public:
   InterpolationRemap() { }

   void Remap(const ParGridFunction &source, const ParGridFunction &x_new,
              ParGridFunction &interpolated);
};
#endif

// Performs the full remap advection loop.
class RemapAdvector
{
public:
   enum StateVars 
   {
      Velocity,
      Density,
      Energy,
      //------
      NVars
   };

   enum class RemapVelocity
   {
      None = -1,
      LowOrder,
      HighOrderTarget,
      MCL,
      ClipAndScale,
   };

   enum class RemapThermo
   {
      Nonconservative,
      GeomConsistent,
   };

private:
   ParMesh pmesh;
   int dim;
   L2_FECollection fec_L2;
   H1_FECollection fec_H1, fec_H1Lag;
   ParFiniteElementSpace pfes_L2, pfes_H1, pfes_H1Lag;
   const Array<int> &v_ess_tdofs;
   const IntegrationRule *ir_rho{};

   RemapVelocity remap_v;
   bool remap_v_stable;
   RemapThermo remap_th;

   const double cfl_factor;

   // Remap state variables.
   Array<int> offsets;
   BlockVector S;
   ParGridFunction v, rho, e;
   ParGridFunction detJ;

   double e_max;

   std::unique_ptr<ODESolver> ode_solver;
   Vector x0;

   socketstream vis_rho, vis_v, vis_e;

public:
   RemapAdvector(const ParMesh &m, int order_v, int order_e, double cfl,
                 RemapVelocity remap_v_, bool remap_v_stable_,
                 RemapThermo remap_th_,
                 const Array<int> &ess_tdofs);

   void InitFromLagr(const Vector &nodes0,
                     const ParGridFunction &vel,
                     const IntegrationRule &rho_ir,
                     const Vector &rhoDetJw,
                     const ParGridFunction &energy);

   virtual void ComputeAtNewPosition(const Vector &new_nodes,
                                     const Array<int> &ess_tdofs,
                                     const Array<int> &ess_vdofs);

   void TransferToLagr(ParGridFunction &rho0_gf, ParGridFunction &vel,
                       const IntegrationRule &ir_rho, Vector &rhoDetJw,
                       const IntegrationRule &ir_rho_b, Vector &rhoDetJ_be,
                       ParGridFunction &energy);
};

class TimeDependentGeomConsOperator : public TimeDependentOperator
{
public:
   virtual void ImplicitSolveFlux(real_t dt, ParGridFunction &flux) { }
   virtual void MultConserv(const ParGridFunction &flux, const Vector &U, Vector &dU) const
   { this->Mult(U, dU); }
   virtual void LimitUpdate(real_t dt, const Vector &U, Vector &dU) { }
};

class AdvectorVelocityOper;
class AdvectorThermoOper;

// Performs a single remap advection step.
class AdvectorOper : public TimeDependentGeomConsOperator
{
protected:
   std::unique_ptr<AdvectorVelocityOper> op_v;
   std::unique_ptr<AdvectorThermoOper> op_th;

   Array<int> offsets;
   const Vector &x0;
   Vector &x_now;
   ParGridFunction &u;
   VectorGridFunctionCoefficient u_coeff;
   GridFunctionCoefficient rho_coeff;

public:
   // Here pfes is the ParFESpace of the function that will be moved.
   // Mult() moves the nodes of the mesh corresponding to pfes.
   AdvectorOper(const Vector &x_start, const Array<int> &v_ess_td,
                const Array<int> &v_ess_vd,
                ParGridFunction &mesh_vel,
                ParGridFunction &rho,
                const IntegrationRule &ir_rho,
                ParFiniteElementSpace &pfes_H1,
                ParFiniteElementSpace &pfes_H1_s,
                ParFiniteElementSpace &pfes_L2,
                RemapAdvector::RemapVelocity scheme_v,
                bool remap_v_s,
                RemapAdvector::RemapThermo scheme_th);

   // Single RK stage solve for all fields contained in U.
   void Mult(const Vector &U, Vector &dU) const override;

   void ImplicitSolveFlux(real_t dt, ParGridFunction &flux) override;
   void MultConserv(const ParGridFunction &flux, const Vector &U, Vector &dU) const override;
   void LimitUpdate(real_t dt, const Vector &U, Vector &dU) override;

   void SetDt(real_t delta_t);
   void SetTime(real_t t) override;

   real_t Momentum(ParGridFunction &v, real_t t);
   //real_t Interface(ParGridFunction &xi, real_t t);
   real_t InternalEnergy(ParGridFunction &e, real_t t);
};

// Performs a single velocity remap advection step.
class AdvectorVelocityOper : public TimeDependentOperator
{
protected:
   RemapAdvector::RemapVelocity remap_v = RemapAdvector::RemapVelocity::ClipAndScale;
   bool remap_v_stable = false;

   const Array<int> &v_ess_tdofs, &v_ess_vdofs;
   Coefficient &rho_coeff;
   VectorCoefficient &u_coeff;
   mutable ScalarVectorProductCoefficient rho_u_coeff;
   mutable ParBilinearForm Mr_H1, Mr_H1_s, Kr_H1, KrT_H1, lummpedMr_H1;
   mutable Vector lumpedMr_H1_vec;

   void LowOrderVel(const SparseMatrix &K_glb, const SparseMatrix &KT_glb,
                    const Vector &v, Vector &dv) const;

   void HighOrderTargetSchemeVel(const SparseMatrix &K_glb, const SparseMatrix &KT_glb,
                                 const SparseMatrix &M_glb, const Vector &v,
                                 Vector &d_v) const;

   void MCLVel(const SparseMatrix &K_glb, const SparseMatrix &KT_glb,
               const SparseMatrix &M_glb, const Vector &v,
               Vector &d_v) const;

   void ClipAndScale(const ParFiniteElementSpace &pfesV_H1_s, const Vector &v, Vector &d_v) const;
   void ComputeVelocityMinMax(const Vector &v, Array<double> &v_min, Array<double> &v_max) const;
   void ComputeTimeDerivatives(const Vector &v, ConvectionIntegrator* conv_int, const ParFiniteElementSpace &pfes, Vector &vdot) const;

public:
   // Here pfes is the ParFESpace of the function that will be transferred.
   AdvectorVelocityOper(const Array<int> &v_ess_td,
                        const Array<int> &v_ess_vd,
                        Coefficient &rho_coeff,
                        VectorCoefficient &u_coeff,
                        ParFiniteElementSpace &pfes_H1,
                        ParFiniteElementSpace &pfes_H1_s,
                        RemapAdvector::RemapVelocity scheme,
                        bool remap_v_s);

   // Single RK stage solve for all fields contained in U.
   void Mult(const Vector &U, Vector &dU) const override;

   real_t Momentum(ParGridFunction &v);
};

// Performs a single thermodynamic remap advection step.
class AdvectorThermoOper : public TimeDependentGeomConsOperator
{
public:
   enum StateVars 
   {
      Density,
      Energy,
      //------
      NVars
   };

protected:
   ParFiniteElementSpace &pfes_L2;
   
   Array<int> offsets;
   real_t dt = 0.0;

public:
   AdvectorThermoOper(ParFiniteElementSpace &pfes_L2);

   void SetDt(double delta_t) { dt = delta_t; }

   virtual real_t InternalEnergy(ParGridFunction &e) = 0;
};

// Performs a single thermodynamic remap advection step - nonsconservative scheme
class AdvectorThermoNonconservativeOper : public AdvectorThermoOper
{
protected:
   Coefficient &rho_coeff;
   VectorCoefficient &u_coeff;
   mutable ScalarVectorProductCoefficient rho_u_coeff;
   mutable ParBilinearForm M_L2, M_L2_Lump, K_L2;
   mutable ParBilinearForm Mr_L2, Mr_L2_Lump, Kr_L2;

   // Piecewise min and max of gf over all elements.
   void ComputeElementsMinMax(const ParGridFunction &gf,
                              Vector &el_min, Vector &el_max) const;
   // Bounds at dofs taking the current element and its face-neighbors.
   void ComputeSparsityBounds(const ParFiniteElementSpace &pfes,
                              const Vector &el_min, const Vector &el_max,
                              Vector &dof_min, Vector &dof_max) const;

public:
   // Here pfes is the ParFESpace of the function that will be transferred.
   AdvectorThermoNonconservativeOper(Coefficient &rho_coeff,
                                     VectorCoefficient &u_coeff,
                                     ParFiniteElementSpace &pfes_L2);

   // Single RK stage solve for all fields contained in U.
   void Mult(const Vector &U, Vector &dU) const override;

   real_t InternalEnergy(ParGridFunction &e) override;
};

// Performs a single thermodynamic remap advection step - geometrically consistent scheme
class AdvectorThermoGeomConsistentOper : public AdvectorThermoOper
{
protected:
   const Vector &x0;
   const ParGridFunction &u;
   const IntegrationRule &ir_rho;

   RT_FECollection fec_RT;
   ParFiniteElementSpace pfes_RT;
   Vector &x_now;

   ParBilinearForm DD;
   Array<int> ess_tdofs_f;

   std::unique_ptr<SolutionTransfer> trans;
   mutable ParGridFunction detJ;

   DenseMatrix MJi[Geometry::NUM_GEOMETRIES];
   Array<int> MJi_piv[Geometry::NUM_GEOMETRIES];

   class DivRDivRIntegrator : public BilinearFormIntegrator
   {
#ifndef MFEM_THREAD_SAFE
      Vector divshape;
#endif

   public:
      void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat) override;
   };

   class RefConvectionIntegrator : public BilinearFormIntegrator
   {
      const ParGridFunction &f;
      DenseMatrix dshape;
      Vector shape, v, vxt, vdshape;

   public:
      RefConvectionIntegrator(const ParGridFunction &flux, const IntegrationRule *ir = NULL)
      : BilinearFormIntegrator(ir), f(flux) { }

      void AssembleElementMatrix(const FiniteElement &fe,
                                 ElementTransformation &Tr,
                                 DenseMatrix &elmat) override;
   };

   class RefFaceConvectionIntegrator : public BilinearFormIntegrator
   {
      const ParGridFunction &f;
      Vector shape1, shape2, shape_face, f_f;
      Array<int> vdofs_face;

   public:
      RefFaceConvectionIntegrator(const ParGridFunction &flux, const IntegrationRule *ir = NULL)
      : BilinearFormIntegrator(ir), f(flux) { }

      void AssembleFaceMatrix(const FiniteElement &fe1, 
                              const FiniteElement &fe2,
                              FaceElementTransformations &Trans,
                              DenseMatrix &elmat) override;
   };

   void ImplicitSolveFluxRHS(Vector &rhs) const;

public:
   // Here pfes is the ParFESpace of the function that will be transferred.
   AdvectorThermoGeomConsistentOper(const Vector &x_0, const ParGridFunction &u,
                                    const IntegrationRule &ir_rho,
                                    ParFiniteElementSpace &pfes_L2);

   // Single RK stage solve for all fields contained in U.
   void Mult(const Vector &U, Vector &dU) const override
   { MFEM_ABORT("Geometrically conservative operator cannot be integrated classically!"); }

   void ImplicitSolveFlux(real_t dt, ParGridFunction &flux) override;
   void MultConserv(const ParGridFunction &flux, const Vector &U, Vector &dU) const override;
   void LimitUpdate(real_t dt, const Vector &U, Vector &dU) override;

   real_t InternalEnergy(ParGridFunction &e) override;
};

// Transfer of data between the Lagrange and the remap phases.
class SolutionTransfer
{
protected:
   L2_FECollection fec0;
   ParFiniteElementSpace pfes0;

   // Integration points for the density.
   const IntegrationRule &ir_rho;

   friend class AdvectorThermoGeomConsistentOper;
   class RefMassIntegrator : public BilinearFormIntegrator
   {
   public:
      RefMassIntegrator(const IntegrationRule *ir)
      : BilinearFormIntegrator(ir) { }

      void AssembleElementMatrix(const FiniteElement &el,
                                 ElementTransformation &Trans,
                                 DenseMatrix &elmat) override;
   };

   void ComputeMinMax(const Vector &lmins, const Vector &lmaxs, Vector &mins, Vector &maxs);
   void LimitFluxes(real_t y_avg, real_t y_min, real_t y_max, std::function<real_t(int)> &&w_z, DenseMatrix &F);
   void TransferL2Monotonous(std::function<void(int, DenseMatrix &)> &&M, const Vector &mins, const Vector &maxs,
                             std::function<void(int, Vector&)> &&b, ParGridFunction &y);
   void TransferXYL2Monotonous(std::function<void(int, DenseMatrix &)> &&M, const Vector &mins, const Vector &maxs,
                               const ParGridFunction &x, std::function<void(int, Vector&)> &&b, ParGridFunction &y);

public:
   SolutionTransfer(const ParMesh &pmesh, const IntegrationRule &ir);

   // Nonconservative

   // Density transfer: Lagrange -> Remap.
   // Projects the quad points data to a GridFunction, while preserving the
   // bounds for rho taken from the current element and its face-neighbors.
   void TransferDensity_Lagr2Remap(const Vector &rhoDetJw, ParGridFunction &rho);

   // Geometrically consistent

   void TransferJac_Larg2Remap(ParGridFunction &detJ);
   void TransferDensityJac_Lagr2Remap(const Vector &rhoDetJw, const ParGridFunction &detJ, ParGridFunction &rhoJ);
   void TransferEnergyJac_Lagr2Remap(const Vector &rhoDetJw, const ParGridFunction &rhoJ, const ParGridFunction &eps, ParGridFunction &rhoeJ);
   
   void TransferDensityJac_Remap2Lagr(const ParGridFunction &detJ, const ParGridFunction &rhoJ, ParGridFunction &rho);
   void TransferEnergyJac_Remap2Lagr(const Vector &rhoDetJw, const ParGridFunction &rhoJ, const ParGridFunction &rhoeJ, ParGridFunction &eps);
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

} // namespace ale

} // namespace mfem

#endif // MFEM_LAGHOS_REMAP
