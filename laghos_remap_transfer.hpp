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

#ifndef MFEM_LAGHOS_REMAP_TRANSFER
#define MFEM_LAGHOS_REMAP_TRANSFER

#include "mfem.hpp"
#include <functional>

namespace mfem
{
namespace ale
{

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
   void TransferDensityJac_L22H1(const ParGridFunction &detJ_L2, const ParGridFunction &rhoJ_L2, const ParGridFunction &detJ, ParGridFunction &rhoJ);
   void TransferMomentumJac_Remap2Lagr(const Vector &rhoDetJw, const ParGridFunction &rhoJ, const ParGridFunction &rhouJ, ParGridFunction &vel);

   HypreParMatrix &GetInterpolationMatrix(int v) const { return *MJ[v].As<HypreParMatrix>(); }
   const Vector &GetLumpedInterpolationMatrix(int v) const { return mJ[v]; }
   HypreParMatrix &GetInterpolationMatrix_s() const { return *MJ_s.As<HypreParMatrix>(); }
   const Vector &GetLumpedInterpolationMatrix_s() const { return mJ_s; }
};

} // namespace ale
} // namespace mfem
#endif // MFEM_LAGHOS_REMAP_TRANSFER
