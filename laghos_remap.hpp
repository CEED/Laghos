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
#include "laghos_remap_solvers.hpp"
#include <functional>

namespace mfem
{

namespace ale
{

class SolutionTransfer_L2;
class SolutionTransfer_H1;
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
      HighOrder,
      LowOrder,
      HighOrderTarget,
      MCL,
      ClipAndScale,
   };

   enum class RemapScheme
   {
      Nonconservative,
      GeomConsistent,
   };

private:
   ParMesh pmesh;
   int dim;
   L2_FECollection fec_L2;
   H1_FECollection fec_H1, fec_H1Lag;
   ParFiniteElementSpace pfes_L2, pfes_H1, pfes_H1_s, pfes_H1Lag;
   const Array<int> &v_ess_tdofs, v_ess_vdofs;
   const IntegrationRule *ir_rho{};

   RemapScheme remap_scheme;
   RemapVelocity remap_v;

   const double cfl_factor;
   const bool visualize;

   // Remap state variables.
   Array<int> offsets;
   BlockVector S;
   ParGridFunction Sgf[NVars];
   ParGridFunction detJ_H1, detJ_L2, rhoJ_H1;

   double e_max;

   std::unique_ptr<ODESolver> ode_solver;
   std::unique_ptr<geom_consistent_solvers::GeomConsODESolver> ode_solver_gc;
   Vector x0;

   socketstream vis_rho, vis_v, vis_e;

public:
   RemapAdvector(const ParMesh &m, int order_v, int order_e, double cfl,
                 RemapScheme remap, RemapVelocity remap_v,
                 const Array<int> &ess_tdofs, const Array<int> &ess_vdofs,
                 bool visualize);

   void InitFromLagr(const Vector &nodes0,
                     const ParGridFunction &vel,
                     const IntegrationRule &rho_ir,
                     const Vector &rhoDetJw,
                     const ParGridFunction &energy);

   virtual void ComputeAtNewPosition(const Vector &new_nodes);

   void TransferToLagr(ParGridFunction &rho0_gf, ParGridFunction &vel,
                       const IntegrationRule &ir_rho, Vector &rhoDetJw,
                       const IntegrationRule &ir_rho_b, Vector &rhoDetJ_be,
                       ParGridFunction &energy);
};

class TimeDependentGeomConsOperator : virtual public TimeDependentOperator
{
public:
   virtual void ImplicitSolveFlux(real_t dt, ParGridFunction &flux) { }
   virtual void MultConserv(const ParGridFunction &flux, const Vector &U, Vector &dU) const
   { this->Mult(U, dU); }
   virtual void LimitUpdate(real_t dt, const Vector &U, Vector &dU) { }
};

class AdvectorVelocityOper;
class AdvectorThermoOper;

// Performs a single remap advection step
class AdvectorOper : virtual public TimeDependentOperator
{
protected:
   std::unique_ptr<AdvectorVelocityOper> op_v;
   std::unique_ptr<AdvectorThermoOper> op_th;

   Array<int> offsets;
   const Vector &x0;
   Vector &x_now;
   ParGridFunction &u;

public:
   AdvectorOper(
      const Vector &x_start,
      ParGridFunction &mesh_vel,
      ParFiniteElementSpace &pfes_H1,
      ParFiniteElementSpace &pfes_L2);

   void SetDt(real_t delta_t);
   void SetTime(real_t t) override;

   real_t Momentum(const ParGridFunction &v, real_t t);
   //real_t Interface(ParGridFunction &xi, real_t t);
   real_t Mass(const ParGridFunction &rho, real_t t);
   real_t InternalEnergy(const ParGridFunction &e, real_t t);
};

// Performs a single remap advection step - nonconservative scheme
class AdvectorNonconservativeOper : public AdvectorOper
{
   VectorGridFunctionCoefficient u_coeff;
   GridFunctionCoefficient rho_coeff;

public:
   // Here pfes is the ParFESpace of the function that will be moved.
   // Mult() moves the nodes of the mesh corresponding to pfes.
   AdvectorNonconservativeOper(
      const Vector &x_start, const Array<int> &v_ess_td,
      const Array<int> &v_ess_vd,
      ParGridFunction &mesh_vel,
      ParGridFunction &rho,
      const IntegrationRule &ir_rho,
      ParFiniteElementSpace &pfes_H1,
      ParFiniteElementSpace &pfes_H1_s,
      ParFiniteElementSpace &pfes_L2,
      RemapAdvector::RemapVelocity scheme_v);

   // Single RK stage solve for all fields contained in U.
   void Mult(const Vector &U, Vector &dU) const override;
};

// Performs a single remap advection step - geometrically consistent scheme
class AdvectorGeomConsOper : public AdvectorOper, public TimeDependentGeomConsOperator
{
   VectorGridFunctionCoefficient u_coeff;
   GridFunctionCoefficient rho_coeff;

   const IntegrationRule &ir_rho;

   RT_FECollection fec_f;
   H1_FECollection fec_a;
   ParFiniteElementSpace pfes_f, pfes_a;

   ParBilinearForm DD, CC;
   Array<int> ess_bdr;
   Array<int> ess_tdofs_f;

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

   void ImplicitSolveFluxRHS(Vector &rhs) const;
   void ImplicitSolveSolenoidalRHS(const Vector &f, Vector &rhs) const;

public:
   // Here pfes is the ParFESpace of the function that will be moved.
   // Mult() moves the nodes of the mesh corresponding to pfes.
   AdvectorGeomConsOper(
      const Vector &x_start, const Array<int> &v_ess_td,
      const Array<int> &v_ess_vd,
      ParGridFunction &mesh_vel,
      const IntegrationRule &ir_rho,
      ParFiniteElementSpace &pfes_H1,
      ParFiniteElementSpace &pfes_H1_s,
      ParFiniteElementSpace &pfes_L2,
      RemapAdvector::RemapVelocity scheme_v);

   // Single RK stage solve for all fields contained in U.
   void Mult(const Vector &U, Vector &dU) const override
   { MFEM_ABORT("Cannot be integrated with a plain ODESolver!"); }

   void ImplicitSolveFlux(real_t dt, ParGridFunction &flux) override;
   void MultConserv(const ParGridFunction &flux, const Vector &U, Vector &dU) const override;
   void LimitUpdate(real_t dt, const Vector &U, Vector &dU) override;
};

// Performs a single velocity remap advection step.
class AdvectorVelocityOper : virtual public TimeDependentOperator
{
protected:
   RemapAdvector::RemapVelocity remap_v = RemapAdvector::RemapVelocity::ClipAndScale;
   bool remap_v_stable = false;

   const Array<int> &v_ess_tdofs, &v_ess_vdofs;
   ParFiniteElementSpace &pfes_H1, &pfes_H1_s;
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

   void ComputeVelocityMinMax(const Vector &v, Array<double> &v_min, Array<double> &v_max) const;
   void ComputeTimeDerivatives(const Vector &v, ConvectionIntegrator* conv_int, const ParFiniteElementSpace &pfes, Vector &vdot) const;

public:
   // Here pfes is the ParFESpace of the function that will be transferred.
   AdvectorVelocityOper(
      const Array<int> &v_ess_td,
      const Array<int> &v_ess_vd,
      ParFiniteElementSpace &pfes_H1,
      ParFiniteElementSpace &pfes_H1_s,
      RemapAdvector::RemapVelocity scheme);

   virtual real_t Momentum(const ParGridFunction &v) const = 0;
};

// Performs a single velocity remap advection step - nonconservative scheme
class AdvectorVelocityNonconservativeOper : public AdvectorVelocityOper
{
protected:
   Coefficient &rho_coeff;
   VectorCoefficient &u_coeff;
   mutable ScalarVectorProductCoefficient rho_u_coeff;

   void ClipAndScale(const ParFiniteElementSpace &pfesV_H1_s, const Vector &v, Vector &d_v) const;

public:
   // Here pfes is the ParFESpace of the function that will be transferred.
   AdvectorVelocityNonconservativeOper(
      const Array<int> &v_ess_td,
      const Array<int> &v_ess_vd,
      Coefficient &rho_coeff,
      VectorCoefficient &u_coeff,
      ParFiniteElementSpace &pfes_H1,
      ParFiniteElementSpace &pfes_H1_s,
      RemapAdvector::RemapVelocity scheme);

   // Single RK stage solve for all fields contained in U.
   void Mult(const Vector &U, Vector &dU) const override;

   real_t Momentum(const ParGridFunction &v) const override;
};

// Performs a single velocity remap advection step - geometrically consistent scheme
class AdvectorVelocityGeomConsOper : public AdvectorVelocityOper, public TimeDependentGeomConsOperator
{
protected:
   const IntegrationRule &ir_rho;
   std::unique_ptr<SolutionTransfer_H1> trans;
   mutable ParGridFunction detJ;

   class RefConvectionIntegrator : public BilinearFormIntegrator
   {
      const ParGridFunction &f;
      int vdim = 1;
      DenseMatrix dshape, elmat_d;
      Vector shape, v, vxt, vdshape;

   public:
      RefConvectionIntegrator(const ParGridFunction &flux, const IntegrationRule *ir = NULL)
      : BilinearFormIntegrator(ir), f(flux) { }

      void SetVDim(int vdim_) { vdim = vdim_; }

      void AssembleElementMatrix(const FiniteElement &el,
                                 ElementTransformation &Trans,
                                 DenseMatrix &elvec) override;

      void AssembleElementVector(const FiniteElement &el,
                                 ElementTransformation &Trans,
                                 const Vector &elfun,
                                 Vector &elvec) override;
   };

public:
   // Here pfes is the ParFESpace of the function that will be transferred.
   AdvectorVelocityGeomConsOper(
      const IntegrationRule &ir_rho,
      const Array<int> &v_ess_td,
      const Array<int> &v_ess_vd,
      ParFiniteElementSpace &pfes_H1,
      ParFiniteElementSpace &pfes_H1_s,
      RemapAdvector::RemapVelocity scheme);

   // Single RK stage solve for all fields contained in U.
   void Mult(const Vector &U, Vector &dU) const override
   { MFEM_ABORT("Geometrically conservative operator cannot be integrated classically!"); }

   void MultConserv(const ParGridFunction &flux, const Vector &U, Vector &dU) const override;
   void LimitUpdate(real_t dt, const Vector &U, Vector &dU) override;

   real_t Momentum(const ParGridFunction &v) const override;
};

// Performs a single thermodynamic remap advection step.
class AdvectorThermoOper : virtual public TimeDependentOperator
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

   // Piecewise min and max of gf over all elements.
   void ComputeElementsMinMax(const Vector &u, Vector &el_min, Vector &el_max,
                              const Array<bool> *active_el = nullptr,
                              const Array<bool> *active_dof = nullptr) const;
   // Bounds at dofs taking the current element and its face-neighbors.
   void ComputeSparsityBounds(const ParFiniteElementSpace &pfes,
                              const Vector &el_min, const Vector &el_max,
                              Vector &dof_min, Vector &dof_max) const;

public:
   AdvectorThermoOper(ParFiniteElementSpace &pfes_L2);

   void SetDt(double delta_t) { dt = delta_t; }

   virtual real_t Mass(const ParGridFunction &rho) const = 0;
   virtual real_t InternalEnergy(const ParGridFunction &e) const = 0;
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

public:
   // Here pfes is the ParFESpace of the function that will be transferred.
   AdvectorThermoNonconservativeOper(Coefficient &rho_coeff,
                                     VectorCoefficient &u_coeff,
                                     ParFiniteElementSpace &pfes_L2);

   // Single RK stage solve for all fields contained in U.
   void Mult(const Vector &U, Vector &dU) const override;

   real_t Mass(const ParGridFunction &rho) const override;
   real_t InternalEnergy(const ParGridFunction &e) const override;
};

// Performs a single thermodynamic remap advection step - geometrically consistent scheme
class AdvectorThermoGeomConsOper : public AdvectorThermoOper, public TimeDependentGeomConsOperator
{
protected:
   const IntegrationRule &ir_rho;

   ParFiniteElementSpace pfes_vL2;
   std::unique_ptr<SolutionTransfer_L2> trans;
   mutable ParGridFunction detJ;
   Vector MJ_lumped;
   mutable SparseMatrix MJ, KJ;

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

public:
   // Here pfes is the ParFESpace of the function that will be transferred.
   AdvectorThermoGeomConsOper(const IntegrationRule &ir_rho,
                              ParFiniteElementSpace &pfes_L2);

   // Single RK stage solve for all fields contained in U.
   void Mult(const Vector &U, Vector &dU) const override
   { MFEM_ABORT("Geometrically conservative operator cannot be integrated classically!"); }

   void MultConserv(const ParGridFunction &flux, const Vector &U, Vector &dU) const override;
   void LimitUpdate(real_t dt, const Vector &U, Vector &dU) override;

   real_t Mass(const ParGridFunction &rhoJ) const override;
   real_t InternalEnergy(const ParGridFunction &rhoeJ) const override;
};

// Transfer of data between the Lagrange and the remap phases.
class SolutionTransfer_L2
{
protected:
   const ParMesh &pmesh;
   L2_FECollection fec0;
   ParFiniteElementSpace pfes0;

   // Integration points for the density.
   const IntegrationRule &ir_rho;

   DenseMatrix MJ[Geometry::NUM_GEOMETRIES];
   DenseMatrix MJi[Geometry::NUM_GEOMETRIES];
   Array<int> MJi_piv[Geometry::NUM_GEOMETRIES];

   friend class AdvectorThermoGeomConsOper;
   friend class SolutionTransfer_H1;
   class RefMassIntegrator : public BilinearFormIntegrator
   {
      int vdim = 1;
   public:
      RefMassIntegrator(const IntegrationRule *ir = NULL)
      : BilinearFormIntegrator(ir) { }

      void SetVDim(int vdim_) { vdim = vdim_; }

      void AssembleElementMatrix(const FiniteElement &el,
                                 ElementTransformation &Trans,
                                 DenseMatrix &elmat) override;

      void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                  const FiniteElement &test_fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &elmat) override;
   };

   void ComputeMinMax(const Vector &lmins, const Vector &lmaxs, Vector &mins, Vector &maxs);
   void LimitFluxes(real_t y_avg, real_t y_min, real_t y_max, std::function<real_t(int)> &&w_z, DenseMatrix &F);
   void TransferL2Monotonous(std::function<void(int, DenseMatrix &, LUFactors &)> &&M, const Vector &mins, const Vector &maxs,
                             std::function<void(int, Vector&)> &&b, ParGridFunction &y);
   void TransferXYL2Monotonous(std::function<void(int, DenseMatrix &, LUFactors &)> &&M, const Vector &mins, const Vector &maxs,
                               const ParGridFunction &x, std::function<void(int, Vector&)> &&b, ParGridFunction &y);

public:
   SolutionTransfer_L2(const ParFiniteElementSpace &pfes_L2, const IntegrationRule &ir);

   // Nonconservative

   // Density transfer: Lagrange -> Remap.
   // Projects the quad points data to a GridFunction, while preserving the
   // bounds for rho taken from the current element and its face-neighbors.
   void TransferDensity_Lagr2Remap(const Vector &rhoDetJw, ParGridFunction &rho);

   // Geometrically consistent

   inline const DenseMatrix& GetRefMassMatrix(Geometry::Type g) const { return MJ[g]; }
   inline const LUFactors GetRefMassInverse(Geometry::Type g) const
   { return LUFactors(MJi[g].GetData(), const_cast<int*>(MJi_piv[g].GetData())); }

   void TransferJac_Larg2Remap(ParGridFunction &detJ);
   void TransferDensityJac_Lagr2Remap(const Vector &rhoDetJw, const ParGridFunction &detJ, ParGridFunction &rhoJ);
   void TransferEnergyJac_Lagr2Remap(const Vector &rhoDetJw, const ParGridFunction &rhoJ, const ParGridFunction &eps, ParGridFunction &rhoeJ);
   
   void TransferDensityJac_Remap2Lagr(const ParGridFunction &detJ, const ParGridFunction &rhoJ, ParGridFunction &rho);
   void TransferEnergyJac_Remap2Lagr(const Vector &rhoDetJw, const ParGridFunction &rhoJ, const ParGridFunction &rhoeJ, ParGridFunction &eps);
};

// Transfer of data between the Lagrange and the remap phases.
class SolutionTransfer_H1
{
protected:
   const Array<int> &v_ess_tdofs, v_ess_vdofs;
   // Integration points for the density.
   const IntegrationRule &ir_rho;
   const ParFiniteElementSpace pfes, pfes_s;

   Array<int> v_ess_vdofs_marker;
   std::vector<Array<int>> v_ess_vdofs_v, v_ess_tdofs_v;
   std::vector<OperatorHandle> MJ;
   OperatorHandle MJ_s;
   std::vector<SparseMatrix> MJ_mloc;
   SparseMatrix MJ_smloc;
   std::vector<Vector> mJ;
   Vector mJ_s;

   friend class AdvectorVelocityGeomConsOper;
   using RefMassIntegrator = SolutionTransfer_L2::RefMassIntegrator;

   void TransferH1Monotonous(const Array<int> &ess_vdofs, const HypreParMatrix &M, const SparseMatrix &M_loc, const Vector &m,
                             const Vector &b, const Vector &dof_min, const Vector &dof_max,
                             ParGridFunction &y) const;

   void TransferH1Monotonous(const Array<int> &ess_vdofs, const Vector &b, const Vector &dof_min,
                             const Vector &dof_max, ParGridFunction &y) const
   { TransferH1Monotonous(ess_vdofs, *MJ_s.As<HypreParMatrix>(), MJ_smloc, mJ_s, b, dof_min, dof_max, y); }

   void TransferXYH1Monotonous(const Array<int> &ess_vdofs, const HypreParMatrix &M, const SparseMatrix &M_loc, const Vector &m,
                               const Vector &b, const Vector &dof_min_y, const Vector &dof_max_y,
                               const ParGridFunction &x, ParGridFunction &y) const;

   void TransferXYH1Monotonous(const Array<int> &ess_vdofs, const Vector &b, const Vector &dof_min_y,
                                const Vector &dof_max_y, const ParGridFunction &x, ParGridFunction &y) const
   { TransferXYH1Monotonous(ess_vdofs, *MJ_s.As<HypreParMatrix>(), MJ_smloc, mJ_s, b, dof_min_y, dof_max_y, x, y); }

   void ComputeH1SparsityBounds(const Vector &el_min, const Vector &el_max,
                                Vector &dof_min, Vector &dof_max) const;

public:
   SolutionTransfer_H1(const Array<int> &v_ess_tdofs, const Array<int> &v_ess_vdofs,
      const ParFiniteElementSpace &pfes_H1, const ParFiniteElementSpace &pfes_H1_s,
      const IntegrationRule &ir);

   // Nonconservative

   void TransferVelocity_Lagr2Remap(const ParGridFunction &vel_Lag, ParGridFunction &vel);
   void TransferVelocity_Remap2Lagr(const ParGridFunction &vel, ParGridFunction &vel_Lag);

   // Geometrically consistent

   void TransferJac_Larg2Remap(ParGridFunction &detJ);
   void TransferDensityJac_Lagr2Remap(const Vector &rhoDetJw, const ParGridFunction &detJ, ParGridFunction &rhoJ);
   void TransferMomentumJac_Lagr2Remap(const Vector &rhoDetJw, const ParGridFunction &rhoJ, const ParGridFunction &vel, ParGridFunction &rhouJ);
   void TransferMomentumJac_Remap2Lagr(const Vector &rhoDetJw, const ParGridFunction &rhoJ, const ParGridFunction &rhouJ, ParGridFunction &vel);

   HypreParMatrix &GetInterpolationMatrix(int v) const { return *MJ[v].As<HypreParMatrix>(); }
   const Vector &GetLumpedInterpolationMatrix(int v) const { return mJ[v]; }
   HypreParMatrix &GetInterpolationMatrix_s() const { return *MJ_s.As<HypreParMatrix>(); }
   const Vector &GetLumpedInterpolationMatrix_s() const { return mJ_s; }
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

} // namespace ale

} // namespace mfem

#endif // MFEM_LAGHOS_REMAP
