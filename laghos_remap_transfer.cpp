// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project (17-SC-20-SC)
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include "laghos_remap_transfer.hpp"
#include "laghos_solver.hpp"

namespace mfem
{
namespace ale
{

void SolutionTransfer_L2::RefMassIntegrator::AssembleElementMatrix(
   const FiniteElement &fe, ElementTransformation &Trans, DenseMatrix &elmat)
{
   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      const int order = 2 * fe.GetOrder();
      ir = &IntRules.Get(fe.GetGeomType(), order);
   }
   const int nqp = ir->GetNPoints();

   const int ndof = fe.GetDof();
   Vector shape(ndof);
   elmat.SetSize(ndof * vdim);
   elmat = 0.;

   DenseMatrix elmat_d;
   if (vdim > 1)
   {
      elmat_d.SetSize(ndof);
      elmat_d = 0.;
   }
   else
   {
      elmat_d.MakeRef(elmat.GetMemory(), 0, ndof, ndof);
   }

   for (int q = 0; q < nqp; q++)
   {
      const IntegrationPoint &ip = ir->IntPoint(q);
      fe.CalcShape(ip, shape);
      AddMult_a_VVt(ip.weight, shape, elmat_d);
   }

   if (vdim > 1)
   {
      for (int v = 0; v < vdim; v++)
         elmat.SetSubMatrix(v*ndof, v*ndof, elmat_d);
   }
}

void SolutionTransfer_L2::RefMassIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      const int order = trial_fe.GetOrder() + test_fe.GetOrder();
      ir = &IntRules.Get(trial_fe.GetGeomType(), order);
   }
   const int nqp = ir->GetNPoints();

   const int tr_ndof = trial_fe.GetDof();
   const int te_ndof = test_fe.GetDof();
   Vector tr_shape(tr_ndof), te_shape(te_ndof);
   elmat.SetSize(te_ndof * vdim, tr_ndof * vdim);
   elmat = 0.;

   DenseMatrix elmat_d;
   if (vdim > 1)
   {
      elmat_d.SetSize(te_ndof, tr_ndof);
      elmat_d = 0.;
   }
   else
   {
      elmat_d.MakeRef(elmat.GetMemory(), 0, te_ndof, tr_ndof);
   }

   for (int q = 0; q < nqp; q++)
   {
      const IntegrationPoint &ip = ir->IntPoint(q);
      trial_fe.CalcShape(ip, tr_shape);
      test_fe.CalcShape(ip, te_shape);
      AddMult_a_VWt(ip.weight, te_shape, tr_shape, elmat_d);
   }

   if (vdim > 1)
   {
      for (int v = 0; v < vdim; v++)
         elmat.SetSubMatrix(v*te_ndof, v*tr_ndof, elmat_d);
   }
}

SolutionTransfer_L2::SolutionTransfer_L2(const ParFiniteElementSpace &pfes_L2, const IntegrationRule &ir)
: pmesh(*pfes_L2.GetParMesh()), fec0(0, pmesh.Dimension()), pfes0(const_cast<ParMesh*>(&pmesh), &fec0), ir_rho(ir)
{
   // Interpolation matrix (inverse)
   RefMassIntegrator mi(&ir_rho);
   Array<Geometry::Type> geoms;
   pmesh.GetGeometries(pmesh.Dimension(), geoms);
   const FiniteElementCollection *fec_L2 = pfes_L2.FEColl();
   IsoparametricTransformation Tr; // dummy
   for (Geometry::Type g : geoms)
   {
      const FiniteElement *fe = fec_L2->GetFE(g, fec_L2->GetOrder());
      const int ndof = fe->GetDof();
      mi.AssembleElementMatrix(*fe, Tr, MJ[g]);

      MJi[g] = MJ[g];
      MJi_piv[g].SetSize(ndof);
      LUFactors lu(MJi[g].GetData(), MJi_piv[g].GetData());
      lu.Factor(ndof);
   }
}

void SolutionTransfer_L2::ComputeMinMax(const Vector &lmins, const Vector &lmaxs, Vector &mins, Vector &maxs)
{
   ParMesh &pmesh = *pfes0.GetParMesh();
   const int NE = pmesh.GetNE();
   
   mins = lmins;
   maxs = lmaxs;

   // One-level face neighbors max / min.
   ParGridFunction min_pgf(&pfes0, const_cast<Vector&>(lmins));
   ParGridFunction max_pgf(&pfes0, const_cast<Vector&>(lmaxs));
   min_pgf.ExchangeFaceNbrData();
   max_pgf.ExchangeFaceNbrData();
   const Vector &gmins = min_pgf.FaceNbrData();
   const Vector &gmaxs = max_pgf.FaceNbrData();
   const Table &el_to_el = pmesh.ElementToElementTable();
   Array<int> face_nbr_el;
   for (int k = 0; k < NE; k++)
   {
      el_to_el.GetRow(k, face_nbr_el);
      for (int n = 0; n < face_nbr_el.Size(); n++)
      {
         if (face_nbr_el[n] < NE)
         {
            // Local neighbor.
            mins(k) = std::min(mins(k), lmins(face_nbr_el[n]));
            maxs(k) = std::max(maxs(k), lmaxs(face_nbr_el[n]));
         }
         else
         {
            // MPI face neighbor.
            mins(k) = std::min(mins(k), gmins(face_nbr_el[n] - NE));
            maxs(k) = std::max(maxs(k), gmaxs(face_nbr_el[n] - NE));
         }
      }
   }
}

void SolutionTransfer_L2::LimitFluxes(real_t y_avg, real_t y_min, real_t y_max, std::function<real_t(int)> &&w, DenseMatrix &F)
{
   const int dof_cnt = F.Width();
   Vector gp(dof_cnt), gm(dof_cnt);

   // Calculate incoming/outgoing fluxes
   gp = 0.0;
   gm = 0.0;
   for (int i = 1; i < dof_cnt; i++)
   {
      for (int j = 0; j < i; j++)
      {
         real_t fij = F(i, j);
         if (fij >= 0.0)
         {
            gp(i) += fij;
            gm(j) -= fij;
         }
         else
         {
            gm(i) += fij;
            gp(j) -= fij;
         }
      }
   }

   // Calculate Zalesak limiter
   for (int i = 0; i < dof_cnt; i++)
   {
      real_t rp = std::max(w(i) * (y_max - y_avg), 0.0);
      real_t rm = std::min(w(i) * (y_min - y_avg), 0.0);
      real_t sp = gp(i), sm = gm(i);

      gp(i) = (rp < sp) ? rp / sp : 1.0;
      gm(i) = (rm > sm) ? rm / sm : 1.0;
   }

   // Calculate local increments
   for (int i = 1; i < dof_cnt; i++)
   {
      for (int j = 0; j < i; j++)
      {
         real_t &fij = F(i, j), aij;

         if (fij >= 0.0)
         {
            aij = std::min(gp(i), gm(j));
         }
         else
         {
            aij = std::min(gm(i), gp(j));
         }

         fij *= aij;
      }
   }
}

void SolutionTransfer_L2::TransferL2Monotonous(
    std::function<void(int, DenseMatrix &, LUFactors &)> &&M, const Vector &lmins, const Vector &lmaxs,
    std::function<void(int, Vector &)> &&b, ParGridFunction &y)
{
   ParMesh &pmesh = *y.ParFESpace()->GetParMesh();
   const int NE = pmesh.GetNE();

   Vector mins, maxs;
   ComputeMinMax(lmins, lmaxs, mins, maxs);

   // HO solution - FCT_Project.
   const int dof_cnt = y.Size() / NE;
   DenseMatrix M_k(dof_cnt), F(dof_cnt);
   DenseMatrixInverse M_ki(&M_k);
   LUFactors M_klu(nullptr, nullptr);
   Vector rhs(dof_cnt), y_HO(dof_cnt), y_k(dof_cnt), m_k(dof_cnt),
          beta(dof_cnt), z(dof_cnt);
   Array<int> dofs(dof_cnt);
   
   for (int k = 0; k < NE; k++)
   {
      // Get local rhs
      b(k, rhs);

      // Get local mass matrix
      M(k, M_k, M_klu);

      // Construct contracted mass matrix
      M_k.GetRowSums(m_k);

      // Calculate high-order solution
      if (M_klu.data)
      {
         y_HO = rhs;
         M_klu.Solve(dof_cnt, 1, y_HO.GetData());
      }
      else
      {
         M_ki.Factor();
         M_ki.Mult(rhs, y_HO);
      }

      // Calculate the average
      const real_t y_avg = rhs.Sum() / m_k.Sum();

      beta = m_k;
      beta /= beta.Sum();

      // Calculate antisymmetric fluxes
      for (int i = 0; i < dof_cnt; i++) { z(i) = rhs(i) - m_k(i) * y_avg; }

      for (int i = 1; i < dof_cnt; i++)
      {
         for (int j = 0; j < i; j++)
         {
            F(i, j) = M_k(i, j) * (y_HO(i) - y_HO(j)) +
                      (beta(j) * z(i) - beta(i) * z(j));
         }
      }

      // Limit the fluxes
      LimitFluxes(y_avg, mins(k), maxs(k), [&](int i) { return m_k(i); }, F);

      // Calculate local increments
      y_k = y_avg;
      for (int i = 1; i < dof_cnt; i++)
      {
         for (int j = 0; j < i; j++)
         {
            real_t fij = F(i, j);
            y_k(i) += fij / m_k(i);
            y_k(j) -= fij / m_k(j);
         }
      }

      y.ParFESpace()->GetElementDofs(k, dofs);
      y.SetSubVector(dofs, y_k);
   }
}

void SolutionTransfer_L2::TransferXYL2Monotonous(
   std::function<void(int, DenseMatrix &, LUFactors &)> &&M, const Vector &lmins, const Vector &lmaxs,
   const ParGridFunction &x, std::function<void(int, Vector &)> &&b, ParGridFunction &y)
{
      ParMesh &pmesh = *y.ParFESpace()->GetParMesh();
   const int NE = pmesh.GetNE();

   Vector mins, maxs;
   ComputeMinMax(lmins, lmaxs, mins, maxs);

   // HO solution - FCT_Project.
   const int dof_cnt = y.Size() / NE;
   DenseMatrix M_k(dof_cnt), F(dof_cnt);
   DenseMatrixInverse M_ki(&M_k);
   LUFactors M_klu(nullptr, nullptr);
   Vector x_k(dof_cnt), rhs(dof_cnt), xy_HO(dof_cnt), y_k(dof_cnt), m_k(dof_cnt),
          beta(dof_cnt), z(dof_cnt);
   Array<int> dofs(dof_cnt);

   for (int k = 0; k < NE; k++)
   {
      // Get local x
      y.ParFESpace()->GetElementDofs(k, dofs);
      x.GetSubVector(dofs, x_k);

      // Get local rhs
      b(k, rhs);

      // Get local mass matrix
      M(k, M_k, M_klu);

      // Construct contracted mass matrix
      M_k.GetRowSums(m_k);

      // Calculate high-order solution
      if (M_klu.data)
      {
         xy_HO = rhs;
         M_klu.Solve(dof_cnt, 1, xy_HO.GetData());
      }
      else
      {
         M_ki.Factor();
         M_ki.Mult(rhs, xy_HO);
      }

      // Calculate the average
      const real_t mx_sum = m_k * x_k;
      const real_t y_avg = (mx_sum != 0.) ? (rhs.Sum() / mx_sum):(0.);

      beta = m_k;
      beta /= beta.Sum();

      // Calculate antisymmetric fluxes
      for (int i = 0; i < dof_cnt; i++) { z(i) = rhs(i) - m_k(i) * x_k(i) * y_avg; }

      for (int i = 1; i < dof_cnt; i++)
      {
         for (int j = 0; j < i; j++)
         {
            F(i, j) = M_k(i, j) * (xy_HO(i) - xy_HO(j)) +
                      (beta(j) * z(i) - beta(i) * z(j));
         }
      }

      // Limit the fluxes
      LimitFluxes(y_avg, mins(k), maxs(k), [&](int i) { return m_k(i) * x_k(i); }, F);

      // Calculate local increments
      y_k = y_avg;
      for (int i = 1; i < dof_cnt; i++)
      {
         for (int j = 0; j < i; j++)
         {
            real_t fij = F(i, j);
            y_k(i) += (x_k(i) != 0.) ? (fij / (m_k(i) * x_k(i))) : (0.);
            y_k(j) -= (x_k(j) != 0.) ? (fij / (m_k(j) * x_k(j))) : (0.);
         }
      }

      y.SetSubVector(dofs, y_k);
   }
}

void SolutionTransfer_L2::TransferDensity_Lagr2Remap(const Vector &rhoDetJw,
                                                  ParGridFunction &rho)
{
   const ParFiniteElementSpace &pfes = *rho.ParFESpace();
   const int NE = pfes.GetNE(), nqp = ir_rho.GetNPoints();
   Vector rho_min_loc(NE), rho_max_loc(NE);

   // Local max / min.
   for (int k = 0; k < NE; k++)
   {
      ElementTransformation &T = *pfes.GetElementTransformation(k);
      rho_min_loc(k) = +infinity();
      rho_max_loc(k) = -infinity();

      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir_rho.IntPoint(q);
         T.SetIntPoint(&ip);
         const real_t detJ = T.Jacobian().Det();
         const real_t rho = rhoDetJw(k * nqp + q) / detJ / ip.weight;

         rho_min_loc(k) = std::min(rho_min_loc(k), rho);
         rho_max_loc(k) = std::max(rho_max_loc(k), rho);
      }
   }

   // Mass matrix
   MassIntegrator mi(&ir_rho);
   auto M = [&pfes,&mi](int k, DenseMatrix &M_k, LUFactors &) {
      const FiniteElement &fe = *pfes.GetFE(k);
      ElementTransformation &T = *pfes.GetElementTransformation(k);
      mi.AssembleElementMatrix(fe, T, M_k);
   };

   // Righ hand side
   hydrodynamics::DensityIntegrator di(rhoDetJw);
   di.SetIntRule(&ir_rho);

   auto brho = [&pfes,&di](int k, Vector &rhs) {
      const FiniteElement &fe = *pfes.GetFE(k);
      ElementTransformation &T = *pfes.GetElementTransformation(k);
      di.AssembleRHSElementVect(fe, T, rhs);
   };

   TransferL2Monotonous(M, rho_min_loc, rho_max_loc, brho, rho);
}

void SolutionTransfer_L2::TransferJac_Larg2Remap(ParGridFunction &detJ)
{
   const ParFiniteElementSpace &pfes = *detJ.ParFESpace();
   const int NE = pfes.GetNE(), nqp = ir_rho.GetNPoints();
   Vector detJ_min_loc(NE), detJ_max_loc(NE);

   // Local max / min.
   for (int k = 0; k < NE; k++)
   {
      ElementTransformation &T = *pfes.GetElementTransformation(k);
      detJ_min_loc(k) = +infinity();
      detJ_max_loc(k) = 0.;

      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir_rho.IntPoint(q);
         T.SetIntPoint(&ip);
         const real_t detJ = T.Jacobian().Det();
         MFEM_ASSERT(detJ > 0., "Non-positive Jacobian!");
         detJ_min_loc(k) = std::min(detJ_min_loc(k), detJ);
         detJ_max_loc(k) = std::max(detJ_max_loc(k), detJ);
      }
   }

   // Interpolation matrix
   auto M = [&pfes, this]
   (int k, DenseMatrix &M_k, LUFactors &M_klu) {
      const FiniteElement &fe = *pfes.GetFE(k);
      const Geometry::Type g = fe.GetGeomType();
      M_k = MJ[g];
      M_klu.data = MJi[g].GetData();
      M_klu.ipiv = MJi_piv[g].GetData();
   };

   // Right hand side
   ConstantCoefficient one;
   DomainLFIntegrator dlfi(one, &ir_rho);

   auto bdetJ = [&pfes,&dlfi](int k, Vector &rhs) {
      const FiniteElement &fe = *pfes.GetFE(k);
      ElementTransformation &T = *pfes.GetElementTransformation(k);
      dlfi.AssembleRHSElementVect(fe, T, rhs);
   };

   TransferL2Monotonous(M, detJ_min_loc, detJ_max_loc, bdetJ, detJ);
}

void SolutionTransfer_L2::TransferDensityJac_Lagr2Remap(
   const Vector &rhoDetJw, const ParGridFunction &detJ, ParGridFunction &rhoJ)
{
   const ParFiniteElementSpace &pfes = *rhoJ.ParFESpace();
   const int NE = pfes.GetNE(), nqp = ir_rho.GetNPoints();
   Vector rho_min_loc(NE), rho_max_loc(NE);

   // Local max / min.
   for (int k = 0; k < NE; k++)
   {
      ElementTransformation &T = *pfes.GetElementTransformation(k);
      rho_min_loc(k) = +infinity();
      rho_max_loc(k) = -infinity();

      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir_rho.IntPoint(q);
         T.SetIntPoint(&ip);
         const real_t detJ = T.Jacobian().Det();
         const real_t rho = rhoDetJw(k * nqp + q) / detJ / ip.weight;

         rho_min_loc(k) = std::min(rho_min_loc(k), rho);
         rho_max_loc(k) = std::max(rho_max_loc(k), rho);
      }
   }

   // Interpolation matrix
   auto M = [&pfes, this](int k, DenseMatrix &M_k, LUFactors &M_klu) {
      const FiniteElement &fe = *pfes.GetFE(k);
      const Geometry::Type g = fe.GetGeomType();
      M_k = MJ[g];
      M_klu.data = MJi[g].GetData();
      M_klu.ipiv = MJi_piv[g].GetData();
   };

   // Righ hand side
   hydrodynamics::DensityIntegrator di(rhoDetJw);
   di.SetIntRule(&ir_rho);

   auto brho = [&pfes,&di](int k, Vector &rhs) {
      const FiniteElement &fe = *pfes.GetFE(k);
      ElementTransformation &T = *pfes.GetElementTransformation(k);
      di.AssembleRHSElementVect(fe, T, rhs);
   };

   TransferXYL2Monotonous(M, rho_min_loc, rho_max_loc, detJ, brho, rhoJ);

   // Jacobian product
   for (int i = 0; i < rhoJ.Size(); i++)
      rhoJ(i) *= detJ(i);
}

void SolutionTransfer_L2::TransferEnergyJac_Lagr2Remap(
   const Vector &rhoDetJw, const ParGridFunction &rhoJ, const ParGridFunction &eps,
   ParGridFunction &rhoeJ)
{
   const ParFiniteElementSpace &pfes = *rhoeJ.ParFESpace();
   const int NE = pfes.GetNE();
   Vector eps_min_loc(NE), eps_max_loc(NE);
   Vector eps_k;

   // Local max / min.
   for (int k = 0; k < NE; k++)
   {
      eps.GetElementDofValues(k, eps_k);
      eps_min_loc(k) = eps_k.Min();
      eps_max_loc(k) = eps_k.Max();
   }

   // Interpolation matrix
   auto M = [&pfes, this](int k, DenseMatrix &M_k, LUFactors &M_klu) {
      const FiniteElement &fe = *pfes.GetFE(k);
      const Geometry::Type g = fe.GetGeomType();
      M_k = MJ[g];
      M_klu.data = MJi[g].GetData();
      M_klu.ipiv = MJi_piv[g].GetData();
   };

   // Righ hand side
   hydrodynamics::InternalEnergyIntegrator iei(rhoDetJw, eps);
   iei.SetIntRule(&ir_rho);

   auto beps = [&pfes,&iei](int k, Vector &rhs) {
      const FiniteElement &fe = *pfes.GetFE(k);
      ElementTransformation &T = *pfes.GetElementTransformation(k);
      iei.AssembleRHSElementVect(fe, T, rhs);
   };

   TransferXYL2Monotonous(M, eps_min_loc, eps_max_loc, rhoJ, beps, rhoeJ);

   // density and Jacobian product
   for (int i = 0; i < rhoJ.Size(); i++)
      rhoeJ(i) *= rhoJ(i);
}

void SolutionTransfer_L2::TransferDensityJac_Remap2Lagr(
   const ParGridFunction &detJ, const ParGridFunction &rhoJ, ParGridFunction &rho)
{
   const ParFiniteElementSpace &pfes = *rhoJ.ParFESpace();
   const int NE = pfes.GetNE();
   Vector rho_min_loc(NE), rho_max_loc(NE);
   Vector detJ_k, rhoJ_k;

   // Local max / min.
   for (int k = 0; k < NE; k++)
   {
      detJ.GetElementDofValues(k, detJ_k);
      rhoJ.GetElementDofValues(k, rhoJ_k);
      rho_min_loc(k) = +infinity();
      rho_max_loc(k) = -infinity();
      const int ndof = detJ_k.Size();

      for (int i = 0; i < ndof; i++)
      {
         const real_t rho = rhoJ_k(i) / detJ_k(i);

         rho_min_loc(k) = std::min(rho_min_loc(k), rho);
         rho_max_loc(k) = std::max(rho_max_loc(k), rho);
      }
   }

   // Mass matrix
   MassIntegrator mi(&ir_rho);
   auto M = [&pfes,&mi](int k, DenseMatrix &M_k, LUFactors &) {
      const FiniteElement &fe = *pfes.GetFE(k);
      ElementTransformation &T = *pfes.GetElementTransformation(k);
      mi.AssembleElementMatrix(fe, T, M_k);
   };

   // Right hand side
   auto brho = [&pfes,&rhoJ,this](int k, Vector &rhs) {
      const FiniteElement &fe = *pfes.GetFE(k);
      ElementTransformation &T = *pfes.GetElementTransformation(k);
      Vector shape(fe.GetDof()), rhoJ_k;
      rhoJ.GetElementDofValues(T.ElementNo, rhoJ_k);
      const int nqp = ir_rho.GetNPoints();
      rhs.SetSize(fe.GetDof());
      rhs = 0.;
      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir_rho.IntPoint(q);
         T.SetIntPoint(&ip);
         fe.CalcShape(ip, shape);
         const real_t rhoJ = rhoJ_k * shape;
         rhs.Add(ip.weight * rhoJ, shape);
      }
   };

   TransferL2Monotonous(M, rho_min_loc, rho_max_loc, brho, rho);
}

void SolutionTransfer_L2::TransferEnergyJac_Remap2Lagr(
   const Vector &rhoDetJw, const ParGridFunction &rhoJ, const ParGridFunction &rhoeJ, ParGridFunction &eps)
{
   const ParFiniteElementSpace &pfes = *rhoeJ.ParFESpace();
   const int NE = pfes.GetNE();
   Vector eps_min_loc(NE), eps_max_loc(NE);
   Vector rhoJ_k, rhoeJ_k;

   // Local max / min.
   for (int k = 0; k < NE; k++)
   {
      rhoJ.GetElementDofValues(k, rhoJ_k);
      rhoeJ.GetElementDofValues(k, rhoeJ_k);
      eps_min_loc(k) = +infinity();
      eps_max_loc(k) = -infinity();
      const int ndof = rhoJ_k.Size();

      for (int i = 0; i < ndof; i++)
      {
         const real_t eps = (rhoJ_k(i) != 0.) ? (rhoeJ_k(i) / rhoJ_k(i)) : (0.);

         eps_min_loc(k) = std::min(eps_min_loc(k), eps);
         eps_max_loc(k) = std::max(eps_max_loc(k), eps);
      }
   }

   // Energy mass matrix
   auto Me = [&pfes,&rhoDetJw,this](int k, DenseMatrix &M_k, LUFactors &) {
      const FiniteElement &fe = *pfes.GetFE(k);
      const int nqp = ir_rho.GetNPoints();
      Vector shape(fe.GetDof());
      M_k.SetSize(fe.GetDof());
      M_k = 0.;
      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir_rho.IntPoint(q);
         fe.CalcShape(ip, shape);
         AddMult_a_VVt(rhoDetJw(k*nqp + q), shape, M_k);
      }
   };

   // Right hand side
   auto beps = [&pfes,&rhoeJ,this](int k, Vector &rhs) {
      const FiniteElement &fe = *pfes.GetFE(k);
      ElementTransformation &T = *pfes.GetElementTransformation(k);
      Vector shape(fe.GetDof()), rhoeJ_k;
      rhoeJ.GetElementDofValues(T.ElementNo, rhoeJ_k);
      const int nqp = ir_rho.GetNPoints();
      rhs.SetSize(fe.GetDof());
      rhs = 0.;
      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir_rho.IntPoint(q);
         T.SetIntPoint(&ip);
         fe.CalcShape(ip, shape);
         const real_t rhoJ = rhoeJ_k * shape;
         rhs.Add(ip.weight * rhoJ, shape);
      }
   };

   TransferL2Monotonous(Me, eps_min_loc, eps_max_loc, beps, eps);
}

// Bounds-preserving H1 projection inspired by FCT and MCL
void SolutionTransfer_H1::TransferH1Monotonous(
   const Array<int> &ess_vdofs, const HypreParMatrix &M, const SparseMatrix &M_loc, const Vector &m,
   const  Vector &b, const Vector &dof_min, const Vector &dof_max,
   ParGridFunction &y) const
{
   ParFiniteElementSpace &pfes_H1_s = *y.ParFESpace();
   const int dofs_h1 = pfes_H1_s.GetVSize();

   // Build lumped mass matrix
   Vector m_loc(dofs_h1);
   pfes_H1_s.GetProlongationMatrix()->Mult(m, m_loc);

   // Step 1: Compute high-order solution M y_HO = b
   HypreSmoother prec;
   prec.SetType(HypreSmoother::l1GS, 1);

   CGSolver lin_solver(pfes_H1_s.GetComm());
   lin_solver.SetRelTol(1e-10);
   lin_solver.SetAbsTol(0.0);
   lin_solver.SetMaxIter(200);
   lin_solver.SetPrintLevel(0);
   lin_solver.SetPreconditioner(prec);
   lin_solver.SetOperator(M);

   Vector B(pfes_H1_s.GetTrueVSize());
   pfes_H1_s.GetProlongationMatrix()->MultTranspose(b, B);

   Vector Y_HO(pfes_H1_s.GetTrueVSize());
   Y_HO = 0.;
   lin_solver.Mult(B, Y_HO);

   Vector y_HO(dofs_h1);
   pfes_H1_s.GetProlongationMatrix()->Mult(Y_HO, y_HO);

   // Step 2: Compute low-order solution y_LO = b_LO / m_loc
   Vector y_LO(dofs_h1), b_LO(dofs_h1);
   pfes_H1_s.GetProlongationMatrix()->Mult(B, b_LO);
   for (int i = 0; i < dofs_h1; i++)
   {
      if (ess_vdofs.Size() && ess_vdofs[i])
      {
         y_LO(i) = y_HO(i) = 0.;
         continue;
      }
      y_LO(i) = (m_loc(i) != 0.)?(b_LO(i) / m_loc(i)):(0.5 * (dof_max(i) + dof_min(i)));
   }

   // Step 3: FCT-style flux limiting
   // Get diagonal block of the mass matrix (local connectivity)
   GroupCommunicator &gcomm = pfes_H1_s.GroupComm();

   // Compute sum of incoming and outgoing antidiffusive fluxes
   Vector P_plus(dofs_h1), P_minus(dofs_h1);
   P_plus = 0.0;
   P_minus = 0.0;

   const int *I = M_loc.GetI();
   const int *J_local = M_loc.GetJ();
   const real_t *M_data = M_loc.GetData();

   for (int i = 0; i < dofs_h1; i++)
   {
      if (ess_vdofs.Size() && ess_vdofs[i]) { continue; }
      for (int k = I[i]; k < I[i+1]; k++)
      {
         int j = J_local[k];
         if (i == j) continue;

         // Antidiffusive flux: f_ij = M_ij * (y_HO_i - y_HO_j)
         real_t f_ij = M_data[k] * (y_HO(i) - y_HO(j));

         if (f_ij > 0.0)
         {
            P_plus(i) += f_ij;
         }
         else
         {
            P_minus(i) += f_ij;
         }
      }
   }

   // Share flux sums across processors
   Array<real_t> P_plus_array(P_plus.GetData(), P_plus.Size());
   Array<real_t> P_minus_array(P_minus.GetData(), P_minus.Size());
   gcomm.Reduce<real_t>(P_plus_array, GroupCommunicator::Sum);
   gcomm.Bcast(P_plus_array);
   gcomm.Reduce<real_t>(P_minus_array, GroupCommunicator::Sum);
   gcomm.Bcast(P_minus_array);

   // Compute allowable flux bounds from min/max constraints
   Vector Q_plus(dofs_h1), Q_minus(dofs_h1);
   for (int i = 0; i < dofs_h1; i++)
   {
      Q_plus(i) = m_loc(i) * (dof_max(i) - y_LO(i));
      Q_minus(i) = m_loc(i) * (dof_min(i) - y_LO(i));
   }

   // Compute flux limiters
   Vector alpha_plus(dofs_h1), alpha_minus(dofs_h1);
   for (int i = 0; i < dofs_h1; i++)
   {
      alpha_plus(i) = (P_plus(i) != 0.) ?
         std::min(1.0, Q_plus(i) / P_plus(i)) : 1.0;

      alpha_minus(i) = (P_minus(i) != 0.) ?
         std::min(1.0, Q_minus(i) / P_minus(i)) : 1.0;
   }

   // Step 4: Apply limited fluxes
   Vector flux_limited(dofs_h1);
   flux_limited = 0.0;

   for (int i = 0; i < dofs_h1; i++)
   {
      for (int k = I[i]; k < I[i+1]; k++)
      {
         int j = J_local[k];
         if (i == j) continue;

         // Antidiffusive flux
         real_t f_ij = M_data[k] * (y_HO(i) - y_HO(j));

         // Apply limiter: min of sender and receiver alphas
         real_t alpha_ij;
         if (f_ij > 0.0)
         {
            alpha_ij = std::min(alpha_plus(i), alpha_minus(j));
         }
         else
         {
            alpha_ij = std::min(alpha_minus(i), alpha_plus(j));
         }

         flux_limited(i) += alpha_ij * f_ij;
      }
   }

   // Share limited fluxes across processors
   Array<real_t> flux_array(flux_limited.GetData(), flux_limited.Size());
   gcomm.Reduce<real_t>(flux_array, GroupCommunicator::Sum);
   gcomm.Bcast(flux_array);

   // Final solution: y = y_LO + flux_limited / m_lumped
   y = y_LO;
   for (int i = 0; i < dofs_h1; i++)
   {
      if (ess_vdofs.Size() && ess_vdofs[i]) { continue; }
      y(i) += (m_loc(i) != 0.)?(flux_limited(i) / m_loc(i)):(0.);
   }
}

// Product version: Transfer x*y with bounds on y, given RHS b = integral(x*y)
void SolutionTransfer_H1::TransferXYH1Monotonous(
   const Array<int> &ess_vdofs, const HypreParMatrix &M, const SparseMatrix &M_loc, const Vector &m,
   const Vector &b, const Vector &dof_min_y, const Vector &dof_max_y,
   const ParGridFunction &x, ParGridFunction &y) const
{
   ParFiniteElementSpace &pfes_H1_s = *y.ParFESpace();
   const int dofs_h1 = pfes_H1_s.GetVSize();
   GroupCommunicator &gcomm = pfes_H1_s.GroupComm();

   // Build lumped mass matrix
   Vector m_loc(dofs_h1);
   pfes_H1_s.GetProlongationMatrix()->Mult(m, m_loc);

   // Step 1: Compute high-order solution M (x*y)_HO = b
   HypreSmoother prec;
   prec.SetType(HypreSmoother::l1GS, 1);

   CGSolver lin_solver(pfes_H1_s.GetComm());
   lin_solver.SetRelTol(1e-10);
   lin_solver.SetAbsTol(0.0);
   lin_solver.SetMaxIter(200);
   lin_solver.SetPrintLevel(0);
   lin_solver.SetPreconditioner(prec);
   lin_solver.SetOperator(M);

   Vector B(pfes_H1_s.GetTrueVSize());
   pfes_H1_s.GetProlongationMatrix()->MultTranspose(b, B);

   Vector XY_HO(pfes_H1_s.GetTrueVSize());
   XY_HO = 0.;
   lin_solver.Mult(B, XY_HO);

   Vector xy_HO(dofs_h1);
   pfes_H1_s.GetProlongationMatrix()->Mult(XY_HO, xy_HO);

   // Step 2: Compute low-order solution per DOF: y_LO = b_LO / (m_loc * x)
   Vector y_LO(dofs_h1), b_LO(dofs_h1);
   pfes_H1_s.GetProlongationMatrix()->Mult(B, b_LO);
   for (int i = 0; i < dofs_h1; i++)
   {
      if (ess_vdofs.Size() && ess_vdofs[i])
      {
         y_LO(i) = 0.;
         continue;
      }
      real_t mx_i = m_loc(i) * x(i);
      if (mx_i != 0.)
      {
         y_LO(i) = b_LO(i) / mx_i;
      }
      else
      {
         // If x is zero, y is undefined; use midpoint of bounds
         y_LO(i) = 0.5 * (dof_min_y(i) + dof_max_y(i));
      }
   }

   // Step 3: FCT-style flux limiting with product formulation
   const int *I = M_loc.GetI();
   const int *J_local = M_loc.GetJ();
   const real_t *M_data = M_loc.GetData();

   // Compute sum of incoming and outgoing antidiffusive fluxes
   // weighted by the product formulation
   Vector P_plus(dofs_h1), P_minus(dofs_h1);
   P_plus = 0.0;
   P_minus = 0.0;

   for (int i = 0; i < dofs_h1; i++)
   {
      if (ess_vdofs.Size() && ess_vdofs[i]) { continue; }
      for (int k = I[i]; k < I[i+1]; k++)
      {
         int j = J_local[k];
         if (i == j) continue;

         // Antidiffusive flux: f_ij = M_ij * (xy_HO_i - xy_HO_j)
         const real_t f_ij = M_data[k] * (xy_HO(i) - xy_HO(j));

         if (f_ij > 0.0)
         {
            P_plus(i) += f_ij;
         }
         else
         {
            P_minus(i) += f_ij;
         }
      }
   }

   // Share flux sums across processors
   Array<real_t> P_plus_array(P_plus.GetData(), P_plus.Size());
   Array<real_t> P_minus_array(P_minus.GetData(), P_minus.Size());
   gcomm.Reduce<real_t>(P_plus_array, GroupCommunicator::Sum);
   gcomm.Bcast(P_plus_array);
   gcomm.Reduce<real_t>(P_minus_array, GroupCommunicator::Sum);
   gcomm.Bcast(P_minus_array);

   // Compute allowable flux bounds from min/max constraints on y
   // Q is based on the product mass: m_i * x_i
   Vector Q_plus(dofs_h1), Q_minus(dofs_h1);
   for (int i = 0; i < dofs_h1; i++)
   {
      const real_t mx_i = m_loc(i) * x(i);
      Q_plus(i) = mx_i * (dof_max_y(i) - y_LO(i));
      Q_minus(i) = mx_i * (dof_min_y(i) - y_LO(i));
   }

   // Compute flux limiters
   Vector alpha_plus(dofs_h1), alpha_minus(dofs_h1);
   for (int i = 0; i < dofs_h1; i++)
   {
      alpha_plus(i) = (P_plus(i) != 0.) ?
         std::min(1.0, Q_plus(i) / P_plus(i)) : 1.0;

      alpha_minus(i) = (P_minus(i) != 0.) ?
         std::min(1.0, Q_minus(i) / P_minus(i)) : 1.0;
   }

   // Share alpha values across processors for shared DOFs
   // Use Min reduction to ensure consistency
   Array<real_t> alpha_plus_array(alpha_plus.GetData(), alpha_plus.Size());
   Array<real_t> alpha_minus_array(alpha_minus.GetData(), alpha_minus.Size());
   gcomm.Reduce<real_t>(alpha_plus_array, GroupCommunicator::Min);
   gcomm.Bcast(alpha_plus_array);
   gcomm.Reduce<real_t>(alpha_minus_array, GroupCommunicator::Min);
   gcomm.Bcast(alpha_minus_array);

   // Step 4: Apply limited fluxes
   Vector flux_limited(dofs_h1);
   flux_limited = 0.0;

   for (int i = 0; i < dofs_h1; i++)
   {
      for (int k = I[i]; k < I[i+1]; k++)
      {
         int j = J_local[k];
         if (i == j) continue;

         // Antidiffusive flux
         const real_t f_ij = M_data[k] * (xy_HO(i) - xy_HO(j));

         // Apply limiter: min of sender and receiver alphas
         real_t alpha_ij;
         if (f_ij > 0.0)
         {
            alpha_ij = std::min(alpha_plus(i), alpha_minus(j));
         }
         else
         {
            alpha_ij = std::min(alpha_minus(i), alpha_plus(j));
         }

         flux_limited(i) += alpha_ij * f_ij;
      }
   }

   // Share limited fluxes across processors
   Array<real_t> flux_array(flux_limited.GetData(), flux_limited.Size());
   gcomm.Reduce<real_t>(flux_array, GroupCommunicator::Sum);
   gcomm.Bcast(flux_array);

   // Final solution: y = y_LO + flux_limited / (m_lumped * x)
   y = y_LO;
   for (int i = 0; i < dofs_h1; i++)
   {
      if (ess_vdofs.Size() && ess_vdofs[i]) { continue; }
      const real_t mx_i = m_loc(i) * x(i);
      y(i) += (mx_i != 0.)?(flux_limited(i) / mx_i):(0.);
   }
}

// Helper function: Compute DOF bounds from element bounds
// For H1 elements, DOFs are at vertices, so we expand to vertex neighbors
void SolutionTransfer_H1::ComputeH1SparsityBounds(
   const Vector &el_min,
   const Vector &el_max,
   Vector &dof_min,
   Vector &dof_max) const
{
   const ParMesh *pmesh = pfes_s.GetParMesh();
   const int NE = pmesh->GetNE();
   const int ndofs = pfes_s.GetVSize();

   dof_min.SetSize(ndofs);
   dof_max.SetSize(ndofs);

   // Initialize with extreme values
   dof_min = +infinity();
   dof_max = -infinity();

   // Collect bounds from all elements touching each vertex/DOF
   Array<int> dofs;
   for (int k = 0; k < NE; k++)
   {
      pfes_s.GetElementDofs(k, dofs);
      for (int i = 0; i < dofs.Size(); i++)
      {
         int dof = dofs[i];
         dof_min(dof) = std::min(dof_min(dof), el_min(k));
         dof_max(dof) = std::max(dof_max(dof), el_max(k));
      }
   }

   // Share across processors
   const GroupCommunicator &gcomm = pfes_s.GroupComm();
   Array<real_t> min_array(dof_min.GetData(), dof_min.Size());
   Array<real_t> max_array(dof_max.GetData(), dof_max.Size());
   gcomm.Reduce<real_t>(min_array, GroupCommunicator::Min);
   gcomm.Bcast(min_array);
   gcomm.Reduce<real_t>(max_array, GroupCommunicator::Max);
   gcomm.Bcast(max_array);
}

SolutionTransfer_H1::SolutionTransfer_H1(
   const Array<int> &v_ess_tdofs_, const Array<int> &v_ess_vdofs_,
   const ParFiniteElementSpace &pfes_H1, const ParFiniteElementSpace &pfes_H1_s,
   const IntegrationRule &ir)
: v_ess_tdofs(v_ess_tdofs_), v_ess_vdofs(v_ess_vdofs_), ir_rho(ir),
  pfes(pfes_H1), pfes_s(pfes_H1_s)
{
   const ParMesh &pmesh = *pfes_H1_s.GetParMesh();
   const int vdim = pfes_H1.GetVDim();

   FiniteElementSpace::ListToMarker(v_ess_vdofs, pfes_H1.GetVSize(), v_ess_vdofs_marker);

   // Interpolation matrix and lumped diagonal
   RefMassIntegrator mi(&ir_rho);
   DenseMatrix MJ_g[Geometry::NUM_GEOMETRIES];
   Vector mJ_g[Geometry::NUM_GEOMETRIES];
   Array<Geometry::Type> geoms;
   pmesh.GetGeometries(pmesh.Dimension(), geoms);
   const FiniteElementCollection *fec_H1 = pfes_H1_s.FEColl();
   IsoparametricTransformation Tr; // dummy
   for (Geometry::Type g : geoms)
   {
      const FiniteElement *fe = fec_H1->GetFE(g, fec_H1->GetOrder());
      mi.AssembleElementMatrix(*fe, Tr, MJ_g[g]);
      MJ_g[g].GetRowSums(mJ_g[g]);
   }

   // Assemble the interpolation (lumped) mass matrix
   ParBilinearForm MJ_sbf(const_cast<ParFiniteElementSpace*>(&pfes_H1_s));
   Vector mJ_sloc(pfes_H1_s.GetVSize());
   Array<int> dofs, vdofs;
   MJ_sbf.AllocateMatrix();
   mJ_sloc = 0.;
   const int NE = pmesh.GetNE();
   for (int k = 0; k < NE; k++)
   {
      pfes_H1_s.GetElementDofs(k, dofs);
      Geometry::Type g = pfes_H1_s.GetFE(k)->GetGeomType();
      mJ_sloc.AddElementVector(dofs, mJ_g[g]);

      MJ_sbf.SpMat().AddSubMatrix(dofs, dofs, MJ_g[g], 0);
   }
   mJ_s.SetSize(pfes_H1_s.GetTrueVSize());
   pfes_H1_s.GetProlongationMatrix()->MultTranspose(mJ_sloc, mJ_s);
   
   MJ_sbf.Finalize(0);
   MJ_smloc = MJ_sbf.SpMat();
   MJ_s.SetType(Operator::Hypre_ParCSR);
   MJ_sbf.ParallelAssemble(MJ_s);

   v_ess_tdofs_v.resize(vdim);
   const int ntdofs = pfes_H1_s.GetTrueVSize();
   for (int i = 0; i < v_ess_tdofs.Size(); i++)
   {
      const int v = v_ess_tdofs[i] / ntdofs;
      const int tdof = v_ess_tdofs[i] % ntdofs;
      v_ess_tdofs_v[v].Append(tdof);
   }

   v_ess_vdofs_v.resize(vdim);
   const int nvdofs = pfes_H1_s.GetVSize();
   for (int i = 0; i < v_ess_vdofs.Size(); i++)
   {
      const int v = v_ess_vdofs[i] / nvdofs;
      const int vdof = v_ess_vdofs[i] % nvdofs;
      v_ess_vdofs_v[v].Append(vdof);
   }

   MJ.resize(vdim);
   MJ_mloc.resize(vdim);
   mJ.resize(vdim);
   Vector mJ_loc(pfes_H1_s.GetVSize());
   for (int v = 0; v < vdim; v++)
   {
      ParBilinearForm MJ_bf(const_cast<ParFiniteElementSpace*>(&pfes_H1_s));
      MJ_bf.AllocateMatrix();
      MJ_bf.SpMat() = MJ_smloc;
      MJ[v].SetType(Operator::Hypre_ParCSR);
      MJ_bf.ParallelAssemble(MJ[v]);
      MJ[v].As<HypreParMatrix>()->EliminateBC(v_ess_tdofs_v[v], Operator::DiagonalPolicy::DIAG_ONE);

      MJ_mloc[v] = MJ_smloc;
      MJ_mloc[v].EliminateBC(v_ess_vdofs_v[v], Operator::DiagonalPolicy::DIAG_ONE);
      MJ_mloc[v].GetRowSums(mJ_loc);
      mJ[v].SetSize(ntdofs);
      pfes_s.GetProlongationMatrix()->MultTranspose(mJ_loc, mJ[v]);
      mJ[v].SetSubVector(v_ess_tdofs_v[v], 0.);
   }
}

void SolutionTransfer_H1::TransferVelocity_Lagr2Remap(const ParGridFunction &vel_Lag, ParGridFunction &vel)
{
   const ParFiniteElementSpace &pfes_H1Lag = *vel_Lag.ParFESpace();
   ParFiniteElementSpace &pfes_H1 = *vel.ParFESpace();

   // project velocity field into Bernstein FE space via lumped L2 projection
   ParMixedBilinearForm M_mixed(const_cast<ParFiniteElementSpace*>(&pfes_H1Lag), &pfes_H1);
   M_mixed.AddDomainIntegrator(new VectorMassIntegrator());
   M_mixed.Assemble(0);
   M_mixed.Finalize(0);

   OperatorHandle M;
   M_mixed.FormRectangularSystemMatrix(v_ess_tdofs, v_ess_tdofs, M);

   ParBilinearForm M_lumped(&pfes_H1);
   M_lumped.AddDomainIntegrator(new LumpedIntegrator(new VectorMassIntegrator()));
   M_lumped.Assemble(0);
   M_lumped.Finalize(0);

   Vector lumped_vec(M_lumped.Height());
   M_lumped.SpMat().GetDiag(lumped_vec);
   GroupCommunicator &gcomm = pfes_H1.GroupComm();
   Array<double> lumpedmassmatrix_array(lumped_vec.GetData(), lumped_vec.Size());
   gcomm.Reduce<double>(lumpedmassmatrix_array, GroupCommunicator::Sum);
   gcomm.Bcast(lumpedmassmatrix_array);

   const Operator *R_v = pfes_H1.GetRestrictionMatrix();
   Vector RHS_V(R_v->Height()), X_V(R_v->Height()), VEL(R_v->Height()), M_L(R_v->Height());
   R_v->Mult(vel_Lag, VEL);
   R_v->Mult(lumped_vec, M_L);
   M->Mult(VEL, RHS_V);
   RHS_V /= M_L;
   vel.Distribute(RHS_V);
}

void SolutionTransfer_H1::TransferVelocity_Remap2Lagr(const ParGridFunction &vel, ParGridFunction &vel_Lag)
{
   VectorGridFunctionCoefficient v_coeff(&vel);
   vel_Lag.ProjectCoefficient(v_coeff);

   // // project velocity field back to Lagrange FE space via lumped L2 projection
   // ParMixedBilinearForm M_mixed(&pfes_H1, &pfes_H1Lag);
   // M_mixed.AddDomainIntegrator(new VectorMassIntegrator());
   // M_mixed.Assemble(0);
   // M_mixed.Finalize(0);

   // OperatorHandle M;
   // M_mixed.FormRectangularSystemMatrix(v_ess_tdofs, v_ess_tdofs, M);

   // ParBilinearForm M_lumped(&pfes_H1Lag);
   // M_lumped.AddDomainIntegrator(new LumpedIntegrator(new VectorMassIntegrator()));
   // M_lumped.Assemble(0);
   // M_lumped.Finalize(0);

   // Vector lumped_vec(M_lumped.Height());
   // M_lumped.SpMat().GetDiag(lumped_vec);
   // GroupCommunicator &gcomm = pfes_H1Lag.GroupComm();
   // Array<double> lumpedmassmatrix_array(lumped_vec.GetData(), lumped_vec.Size());
   // gcomm.Reduce<double>(lumpedmassmatrix_array, GroupCommunicator::Sum);
   // gcomm.Bcast(lumpedmassmatrix_array);

   // const Operator *R_v = pfes_H1Lag.GetRestrictionMatrix();
   // Vector RHS_V(R_v->Height()), X_V(R_v->Height()), V(R_v->Height()), M_L(R_v->Height());
   // R_v->Mult(vel, V);
   // R_v->Mult(lumped_vec, M_L);
   // M->Mult(V, RHS_V);
   // RHS_V /= M_L;
   // vel_Lag.Distribute(RHS_V);
}

void SolutionTransfer_H1::TransferJac_Larg2Remap(ParGridFunction &detJ)
{
   ParFiniteElementSpace &pfes_H1_s = *detJ.ParFESpace();

   ConstantCoefficient one;
   ParLinearForm b(&pfes_H1_s);
   b.AddDomainIntegrator(new DomainLFIntegrator(one, &ir_rho));
   b.Assemble();
#if 0
   Vector B(pfes_H1_s.GetTrueVSize());
   b.ParallelAssemble(B);

#if 0
   // H1 lumped projection
   Vector detJ_tv(pfes_H1_s.GetTrueVSize());
   for (int i = 0; i < detJ_tv.Size(); i++)
      detJ_tv(i) = B(i) / mJ(i);
   detJ.Distribute(detJ_tv);
#else
   // H1 projection
   HypreSmoother prec;
   prec.SetType(HypreSmoother::Jacobi, 1);

   CGSolver lin_solver(pfes_H1_s.GetComm());
   lin_solver.SetRelTol(1e-10);
   lin_solver.SetAbsTol(0.0);
   lin_solver.SetMaxIter(100);
   lin_solver.SetPrintLevel(0);
   lin_solver.SetPreconditioner(prec);
   lin_solver.SetOperator(*MJ_s.ParallelAssembleInternalMatrix());

   Vector X(pfes_H1_s.GetTrueVSize());
   X = 0.;
   lin_solver.Mult(B, X);

   detJ.Distribute(X);
#endif
#else
   const int NE = pfes_H1_s.GetNE(), nqp = ir_rho.GetNPoints();
   Vector detJ_min_loc(NE), detJ_max_loc(NE);

   // Local max / min.
   for (int k = 0; k < NE; k++)
   {
      ElementTransformation &T = *pfes_H1_s.GetElementTransformation(k);
      detJ_min_loc(k) = +infinity();
      detJ_max_loc(k) = 0.;

      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir_rho.IntPoint(q);
         T.SetIntPoint(&ip);
         const real_t detJ = T.Jacobian().Det();
         MFEM_ASSERT(detJ > 0., "Non-positive Jacobian!");
         detJ_min_loc(k) = std::min(detJ_min_loc(k), detJ);
         detJ_max_loc(k) = std::max(detJ_max_loc(k), detJ);
      }
   }

   const int ndof_h1 = pfes_H1_s.GetVSize();
   Vector detJ_min_dof(ndof_h1), detJ_max_dof(ndof_h1);
   ComputeH1SparsityBounds(detJ_min_loc, detJ_max_loc, detJ_min_dof, detJ_max_dof);
   const Array<int> ess_vdofs;
   TransferH1Monotonous(ess_vdofs, b, detJ_min_dof, detJ_max_dof, detJ);
#endif
}

void SolutionTransfer_H1::TransferDensityJac_Lagr2Remap(
   const Vector &rhoDetJw, const ParGridFunction &detJ, ParGridFunction &rhoJ)
{
   ParFiniteElementSpace &pfes_H1_s = *rhoJ.ParFESpace();
   const int ndof_h1 = pfes_H1_s.GetNDofs();
   const int NE = pfes_H1_s.GetNE();
   const int nqp = ir_rho.GetNPoints();
   Vector rho_min_el(NE), rho_max_el(NE);
   rho_min_el = +infinity();
   rho_max_el = -infinity();
   Vector b(ndof_h1); b = 0.;
   Vector b_k;
   Vector shape;
   Array<int> dofs;
   for(int k = 0; k < NE; k++)
   {
      const FiniteElement &fe = *pfes_H1_s.GetFE(k);
      ElementTransformation &Tr = *pfes_H1_s.GetElementTransformation(k);
      const int ndofs = fe.GetDof();
      shape.SetSize(ndofs);
      pfes_H1_s.GetElementDofs(k, dofs);
      b_k.SetSize(ndofs);
      b_k = 0.;
      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir_rho.IntPoint(q);
         Tr.SetIntPoint(&ip);
         fe.CalcShape(ip, shape);
         const real_t w = rhoDetJw(k * nqp + q);
         b_k.Add(w, shape);
         const real_t rho = w / ip.weight / Tr.Weight();
         rho_min_el(k) = std::min(rho_min_el(k), rho);
         rho_max_el(k) = std::max(rho_max_el(k), rho);
      }
      b.AddElementVector(dofs, b_k.GetData());
   }

#if 0
   Vector B(pfes_H1_s.GetTrueVSize());
   pfes_H1_s.GetProlongationMatrix()->MultTranspose(b, B);
#if 0
   // H1 lumped projection
   Vector rhouJ_tv(pfes_H1.GetTrueVSize());
   for (int i = 0; i < mJ.Size(); i++)
      for (int v = 0; v < vdim; v++)
         rhouJ_tv(i + v*mJ.Size()) = B(i + v*mJ.Size()) / mJ(i);
   rhouJ.Distribute(rhouJ_tv);
#else
   // H1 projection
   HypreSmoother prec;
   prec.SetType(HypreSmoother::Jacobi, 1);
   CGSolver lin_solver(pfes_H1_s.GetComm());
   lin_solver.SetRelTol(1e-10);
   lin_solver.SetAbsTol(0.0);
   lin_solver.SetMaxIter(100);
   lin_solver.SetPrintLevel(0);
   lin_solver.SetPreconditioner(prec);
   lin_solver.SetOperator(*MJ_s.ParallelAssembleInternalMatrix());

   Vector X(pfes_H1_s.GetTrueVSize());
   X = 0.;
   lin_solver.Mult(B, X);

   rhoJ.Distribute(X);
#endif
#else
   Vector rho_min(ndof_h1), rho_max(ndof_h1);
   ComputeH1SparsityBounds(rho_min_el, rho_max_el, rho_min, rho_max);
   const Array<int> ess_vdofs_v;
   TransferXYH1Monotonous(ess_vdofs_v, b, rho_min, rho_max, detJ, rhoJ);

   // Jacobian product
   for (int i = 0; i < ndof_h1; i++)
   {
      rhoJ(i) *= detJ(i);
   }
#endif
}

void SolutionTransfer_H1::TransferMomentumJac_Lagr2Remap(
   const Vector &rhoDetJw, const ParGridFunction &rhoJ, const ParGridFunction &vel, ParGridFunction &rhouJ)
{
   ParFiniteElementSpace &pfes_H1_Lag = *vel.ParFESpace();
   ParFiniteElementSpace &pfes_H1 = *rhouJ.ParFESpace();
   const int vdim = pfes_H1.GetVDim();
   const int ndof_h1 = pfes_H1.GetNDofs();
   const int NE = pfes_H1.GetNE();
   const int nqp = ir_rho.GetNPoints();
   DenseMatrix v_min_el(NE, vdim), v_max_el(NE, vdim);
   v_min_el = +infinity();
   v_max_el = -infinity();
   Vector b(ndof_h1*vdim); b = 0.;
   DenseMatrix vel_k, b_k;
   Vector shape_Lag, shape, vel_q(vdim);
   Array<int> vdofs_Lag, vdofs;
   for(int k = 0; k < NE; k++)
   {
      const FiniteElement &fe_Lag = *pfes_H1_Lag.GetFE(k);
      const FiniteElement &fe = *pfes_H1.GetFE(k);
      const int ndofs_Lag = fe_Lag.GetDof();
      const int ndofs = fe.GetDof();
      shape_Lag.SetSize(ndofs_Lag);
      shape.SetSize(ndofs);
      pfes_H1_Lag.GetElementVDofs(k, vdofs_Lag);
      pfes_H1.GetElementVDofs(k, vdofs);
      vel_k.SetSize(ndofs_Lag, vdim);
      vel.GetSubVector(vdofs_Lag, vel_k.GetData());
      b_k.SetSize(ndofs, vdim);
      b_k = 0.;
      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir_rho.IntPoint(q);
         fe_Lag.CalcShape(ip, shape_Lag);
         fe.CalcShape(ip, shape);
         vel_k.MultTranspose(shape_Lag, vel_q);
         const real_t w = rhoDetJw(k * nqp + q);
         for (int v = 0; v < vdim; v++)
         {
            Vector b_kv;
            b_k.GetColumnReference(v, b_kv);
            b_kv.Add(w * vel_q(v), shape);
            v_min_el(k,v) = std::min(v_min_el(k,v), vel_q(v));
            v_max_el(k,v) = std::max(v_max_el(k,v), vel_q(v));
         }
      }
      b.AddElementVector(vdofs, b_k.GetData());
   }

#if 0
   Vector B(pfes_H1.GetTrueVSize());
   pfes_H1.GetProlongationMatrix()->MultTranspose(b, B);
#if 0
   // H1 lumped projection
   Vector rhouJ_tv(pfes_H1.GetTrueVSize());
   for (int i = 0; i < mJ.Size(); i++)
      for (int v = 0; v < vdim; v++)
         rhouJ_tv(i + v*mJ.Size()) = B(i + v*mJ.Size()) / mJ(i);
   rhouJ.Distribute(rhouJ_tv);
#else
   // H1 projection
   HypreSmoother prec;
   prec.SetType(HypreSmoother::Jacobi, 1);

   CGSolver lin_solver(pfes_H1.GetComm());
   lin_solver.SetRelTol(1e-10);
   lin_solver.SetAbsTol(0.0);
   lin_solver.SetMaxIter(100);
   lin_solver.SetPrintLevel(0);
   lin_solver.SetPreconditioner(prec);
   lin_solver.SetOperator(*MJ.ParallelAssembleInternalMatrix());

   Vector X(pfes_H1.GetTrueVSize());
   X = 0.;
   lin_solver.Mult(B, X);

   rhouJ.Distribute(X);
#endif
#else
   Vector v_min(ndof_h1), v_max(ndof_h1);
   for (int v = 0; v < vdim; v++)
   {
      const Vector b_v(const_cast<Vector&>(b), v*ndof_h1, ndof_h1);
      ParGridFunction rhouJ_v(rhoJ.ParFESpace(), rhouJ, v*ndof_h1);
      Vector v_min_el_v, v_max_el_v;
      v_min_el.GetColumnReference(v, v_min_el_v);
      v_max_el.GetColumnReference(v, v_max_el_v);
      ComputeH1SparsityBounds(v_min_el_v, v_max_el_v, v_min, v_max);
      const Array<int> ess_vdofs_v(v_ess_vdofs_marker.GetData() + ndof_h1*v, ndof_h1);
      TransferXYH1Monotonous(
         ess_vdofs_v, *MJ[v].As<HypreParMatrix>(), MJ_mloc[v], mJ[v],
         b_v, v_min, v_max, rhoJ, rhouJ_v);

      // density * Jacobian product
      for (int i = 0; i < ndof_h1; i++)
      {
         rhouJ_v(i) *= rhoJ(i);
      }
   }
#endif
}

void SolutionTransfer_H1::TransferDensityJac_L22H1(
   const ParGridFunction &detJ_L2, const ParGridFunction &rhoJ_L2, const ParGridFunction &detJ, ParGridFunction &rhoJ)
{
   ParFiniteElementSpace &pfes_L2_s = *rhoJ_L2.ParFESpace();
   ParFiniteElementSpace &pfes_H1_s = *rhoJ.ParFESpace();
   const int ndof_h1 = pfes_H1_s.GetNDofs();
   const int NE = pfes_H1_s.GetNE();
   const int nqp = ir_rho.GetNPoints();
   Vector rho_min_el(NE), rho_max_el(NE);
   rho_min_el = +infinity();
   rho_max_el = -infinity();
   Vector b(ndof_h1); b = 0.;
   Vector b_k, rhoJ_L2_k, detJ_L2_k;
   Vector shape, shape_L2;
   Array<int> dofs, dofs_L2;
   for(int k = 0; k < NE; k++)
   {
      const FiniteElement &fe = *pfes_H1_s.GetFE(k);
      const FiniteElement &fe_L2 = *pfes_L2_s.GetFE(k);
      ElementTransformation &Tr = *pfes_H1_s.GetElementTransformation(k);
      const int ndofs = fe.GetDof();
      const int ndofs_L2 = fe_L2.GetDof();
      shape.SetSize(ndofs);
      pfes_H1_s.GetElementDofs(k, dofs);
      shape_L2.SetSize(ndofs_L2);
      pfes_L2_s.GetElementDofs(k, dofs_L2);
      rhoJ_L2.GetSubVector(dofs_L2, rhoJ_L2_k);
      detJ_L2.GetSubVector(dofs_L2, detJ_L2_k);
      b_k.SetSize(ndofs);
      b_k = 0.;
      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir_rho.IntPoint(q);
         Tr.SetIntPoint(&ip);
         fe.CalcShape(ip, shape);
         fe_L2.CalcShape(ip, shape_L2);
         const real_t rhoJ = rhoJ_L2_k * shape_L2;
         b_k.Add(rhoJ * ip.weight, shape);
         const real_t rho = rhoJ / (detJ_L2_k * shape_L2);
         rho_min_el(k) = std::min(rho_min_el(k), rho);
         rho_max_el(k) = std::max(rho_max_el(k), rho);
      }
      b.AddElementVector(dofs, b_k.GetData());
   }

#if 0
   Vector B(pfes_H1_s.GetTrueVSize());
   pfes_H1_s.GetProlongationMatrix()->MultTranspose(b, B);
#if 0
   // H1 lumped projection
   Vector rhouJ_tv(pfes_H1.GetTrueVSize());
   for (int i = 0; i < mJ.Size(); i++)
      for (int v = 0; v < vdim; v++)
         rhouJ_tv(i + v*mJ.Size()) = B(i + v*mJ.Size()) / mJ(i);
   rhouJ.Distribute(rhouJ_tv);
#else
   // H1 projection
   HypreSmoother prec;
   prec.SetType(HypreSmoother::Jacobi, 1);
   CGSolver lin_solver(pfes_H1_s.GetComm());
   lin_solver.SetRelTol(1e-10);
   lin_solver.SetAbsTol(0.0);
   lin_solver.SetMaxIter(100);
   lin_solver.SetPrintLevel(0);
   lin_solver.SetPreconditioner(prec);
   lin_solver.SetOperator(*MJ_s.ParallelAssembleInternalMatrix());

   Vector X(pfes_H1_s.GetTrueVSize());
   X = 0.;
   lin_solver.Mult(B, X);

   rhoJ.Distribute(X);
#endif
#else
   Vector rho_min(ndof_h1), rho_max(ndof_h1);
   ComputeH1SparsityBounds(rho_min_el, rho_max_el, rho_min, rho_max);
   const Array<int> ess_vdofs_v;
   TransferXYH1Monotonous(ess_vdofs_v, b, rho_min, rho_max, detJ, rhoJ);

   // Jacobian product
   for (int i = 0; i < ndof_h1; i++)
   {
      rhoJ(i) *= detJ(i);
   }
#endif
}

void SolutionTransfer_H1::TransferMomentumJac_Remap2Lagr(
   const Vector &rhoDetJw, const ParGridFunction &rhoJ, const ParGridFunction &rhouJ, ParGridFunction &vel)
{
   ParFiniteElementSpace &pfes_H1_Lag = *vel.ParFESpace();
   ParFiniteElementSpace &pfes_H1 = *rhouJ.ParFESpace();
   ParFiniteElementSpace &pfes_H1_s = *rhoJ.ParFESpace();
   const int vdim = pfes_H1.GetVDim();

#if 0
   // H1 lumped projection
   Vector m(pfes_H1_Lag.GetVSize()); m = 0.;
   Vector b(pfes_H1_Lag.GetVSize()); b = 0.;
   DenseMatrix rhouJ_k, m_k, b_k;
   Vector shape_Lag, shape, rhouJ_q(vdim);
   Array<int> vdofs_Lag, vdofs;
   const int NE = pfes_H1.GetNE();
   const int nqp = ir_rho.GetNPoints();
   for(int k = 0; k < NE; k++)
   {
      const FiniteElement &fe_Lag = *pfes_H1_Lag.GetFE(k);
      const FiniteElement &fe = *pfes_H1.GetFE(k);
      const int ndofs_Lag = fe_Lag.GetDof();
      const int ndofs = fe.GetDof();
      shape_Lag.SetSize(ndofs_Lag);
      shape.SetSize(ndofs);
      pfes_H1_Lag.GetElementVDofs(k, vdofs_Lag);
      pfes_H1.GetElementVDofs(k, vdofs);
      rhouJ_k.SetSize(ndofs, vdim);
      rhouJ.GetSubVector(vdofs, rhouJ_k.GetData());
      m_k.SetSize(ndofs_Lag, vdim);
      m_k = 0.;
      b_k.SetSize(ndofs_Lag, vdim);
      b_k = 0.;
      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir_rho.IntPoint(q);
         fe_Lag.CalcShape(ip, shape_Lag);
         fe.CalcShape(ip, shape);
         rhouJ_k.MultTranspose(shape, rhouJ_q);
         const real_t w = rhoDetJw(k * nqp + q);
         for (int v = 0; v < vdim; v++)
         {
            Vector m_kv, b_kv;
            m_k.GetColumnReference(v, m_kv);
            b_k.GetColumnReference(v, b_kv);
            m_kv.Add(w, shape_Lag);
            b_kv.Add(ip.weight * rhouJ_q(v), shape_Lag);
         }
      }
      m.AddElementVector(vdofs_Lag, m_k.GetData());
      b.AddElementVector(vdofs_Lag, b_k.GetData());
   }

   Vector M(pfes_H1_Lag.GetTrueVSize());
   pfes_H1_Lag.GetProlongationMatrix()->MultTranspose(m, M);
   Vector B(pfes_H1_Lag.GetTrueVSize());
   pfes_H1_Lag.GetProlongationMatrix()->MultTranspose(b, B);
   Vector vel_tv(pfes_H1_Lag.GetTrueVSize());
   for (int i = 0; i < M.Size(); i++)
      vel_tv(i) = B(i) / M(i);
   vel.Distribute(vel_tv);
#else
   // H1 projection
   ParFiniteElementSpace pfes_H1_Lag_s(pfes_H1_Lag.GetParMesh(), pfes_H1_Lag.FEColl());
   const int ndofs_Lag = pfes_H1_Lag_s.GetVSize();
   const int ntdofs_Lag = pfes_H1_Lag_s.GetTrueVSize();
   SparseMatrix Mv_s(ndofs_Lag);
   Vector b(ndofs_Lag * vdim); b = 0.;
   DenseMatrix rhouJ_k, b_k, Mv_k;
   Vector shape, shape_Lag, rhouJ_q(vdim);
   Array<int> dofs_Lag, vdofs, vdofs_Lag;
   const int NE = pfes_H1_Lag.GetNE();
   const int nqp = ir_rho.GetNPoints();
   for(int k = 0; k < NE; k++)
   {
      const FiniteElement &fe = *pfes_H1_s.GetFE(k);
      const FiniteElement &fe_Lag = *pfes_H1_Lag_s.GetFE(k);
      const int nd = fe.GetDof();
      const int nd_Lag = fe_Lag.GetDof();
      
      shape.SetSize(nd);
      shape_Lag.SetSize(nd_Lag);
      
      pfes_H1_Lag_s.GetElementDofs(k, dofs_Lag);
      vdofs_Lag = dofs_Lag;
      pfes_H1_Lag.DofsToVDofs(vdofs_Lag);
      
      pfes_H1.GetElementVDofs(k, vdofs);
      rhouJ_k.SetSize(nd, vdim);
      rhouJ.GetSubVector(vdofs, rhouJ_k.GetData());

      b_k.SetSize(nd_Lag, vdim);
      b_k = 0.;
      Mv_k.SetSize(nd_Lag);
      Mv_k = 0.;
      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir_rho.IntPoint(q);
         fe.CalcShape(ip, shape);
         fe_Lag.CalcShape(ip, shape_Lag);
         rhouJ_k.MultTranspose(shape, rhouJ_q);

         for (int v = 0; v < vdim; v++)
         {
            Vector b_kv;
            b_k.GetColumnReference(v, b_kv);
            b_kv.Add(ip.weight * rhouJ_q(v), shape_Lag);
         }

         const real_t w = rhoDetJw(k * nqp + q);
         AddMult_a_VVt(w, shape_Lag, Mv_k);
      }

      b.AddElementVector(vdofs_Lag, b_k.GetData());
      Mv_s.AddSubMatrix(dofs_Lag, dofs_Lag, Mv_k);
   }
   Mv_s.Finalize();
   b.SetSubVector(v_ess_vdofs, 0.);
   
   std::vector<OperatorHandle> Mv(vdim);
   std::vector<SparseMatrix> Mv_loc(vdim);
   std::vector<Vector> mv(vdim);
   Vector mv_loc(ndofs_Lag);
   for (int v = 0; v < vdim; v++)
   {
      ParBilinearForm Mv_bf(&pfes_H1_Lag_s);
      Mv_bf.AllocateMatrix();
      Mv_bf.SpMat() = Mv_s;
      Mv[v].SetType(Operator::Hypre_ParCSR);
      Mv_bf.ParallelAssemble(Mv[v]);
      Mv[v].As<HypreParMatrix>()->EliminateBC(v_ess_tdofs_v[v], Operator::DiagonalPolicy::DIAG_ONE);

      Mv_loc[v] = Mv_s;
      Mv_loc[v].EliminateBC(v_ess_vdofs_v[v], Operator::DiagonalPolicy::DIAG_ONE);
      
      Mv_loc[v].GetRowSums(mv_loc);
      mv[v].SetSize(ntdofs_Lag);
      pfes_H1_Lag_s.GetProlongationMatrix()->MultTranspose(mv_loc, mv[v]);
      mv[v].SetSubVector(v_ess_tdofs_v[v], 0.);
   }

#if 0
   HypreSmoother prec;
   prec.SetType(HypreSmoother::Jacobi, 1);
   GMRESSolver lin_solver(pfes_H1_Lag_s.GetComm());
   lin_solver.SetPreconditioner(prec);
   lin_solver.SetRelTol(1e-10);
   lin_solver.SetAbsTol(0.0);
   lin_solver.SetMaxIter(100);
   lin_solver.SetPrintLevel(3);
   lin_solver.SetOperator(Mv_m);
   
   Vector X(ntdofs_Lag * vdim); X = 0.;
   Vector X_v, B_v;

   for (int v = 0; v < vdim; v++)
   {
      B_v.MakeRef(B, v*ntdofs_Lag, ntdofs_Lag);
      X_v.MakeRef(X, v*ntdofs_Lag, ntdofs_Lag);
      lin_solver.Mult(B_v, X_v);
   }
   vel.Distribute(X);
#else
   // Local min/max
   DenseMatrix v_min_el(NE, vdim), v_max_el(NE, vdim);
   v_min_el = +infinity();
   v_max_el = -infinity();
   Vector rhoJ_k;
   Array<int> dofs;

   for(int k = 0; k < NE; k++)
   {
      pfes_H1.GetElementDofs(k, dofs);
      rhoJ.GetSubVector(dofs, rhoJ_k);
      vdofs = dofs;
      pfes_H1.DofsToVDofs(vdofs);
      rhouJ_k.SetSize(dofs.Size(), vdim);
      rhouJ.GetSubVector(vdofs, rhouJ_k.GetData());
      for (int v = 0; v < vdim; v++)
         for (int i = 0; i < dofs.Size(); i++)
         {
            const real_t vel = rhouJ_k(i, v) / rhoJ_k(i);
            v_min_el(k,v) = std::min(v_min_el(k,v), vel);
            v_max_el(k,v) = std::max(v_max_el(k,v), vel);
         }
   }

   Vector v_min(ndofs_Lag), v_max(ndofs_Lag);
   for (int v = 0; v < vdim; v++)
   {
      const Vector b_v(const_cast<Vector&>(b), v*ndofs_Lag, ndofs_Lag);
      ParGridFunction vel_v(&pfes_H1_Lag_s, vel, v*ndofs_Lag);
      Vector v_min_el_v, v_max_el_v;
      v_min_el.GetColumnReference(v, v_min_el_v);
      v_max_el.GetColumnReference(v, v_max_el_v);
      
      ComputeH1SparsityBounds(v_min_el_v, v_max_el_v, v_min, v_max);
      const Array<int> v_ess_vdofs_marker_v(v_ess_vdofs_marker.GetData() + ndofs_Lag*v, ndofs_Lag);
      TransferH1Monotonous(v_ess_vdofs_marker_v, *Mv[v].As<HypreParMatrix>(), Mv_loc[v], mv[v],
                           b_v, v_min, v_max, vel_v);
   }
#endif
#endif
}

} // namespace ale
} // namespace mfem
