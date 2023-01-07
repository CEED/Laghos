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

#include "laghos_materials.hpp"

using namespace std;

namespace mfem
{

namespace hydrodynamics
{

void UpdateAlpha(const ParGridFunction &level_set,
                 ParGridFunction &alpha_1, ParGridFunction &alpha_2,
                 MaterialData *mat_data, bool pointwise_alpha)
{
   IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);
   auto pfes = *alpha_1.ParFESpace();
   const IntegrationRule &ir = IntRulesLo.Get(pfes.GetFE(0)->GetGeomType(), 20);
   const int NE = alpha_1.ParFESpace()->GetNE(),
             nqp = ir.GetNPoints();
   Vector ls_vals;

   const int ndof_l2 = (mat_data) ? mat_data->vol_1.Size() / NE : 0;

   Array<int> l2_dofs;
   Vector vol_1_loc(ndof_l2), vol_2_loc(ndof_l2);
   const IntegrationRule ir_nodes =
      (mat_data) ? mat_data->vol_1.ParFESpace()->GetFE(0)->GetNodes() : ir;
   Vector bounds_max(ndof_l2), bounds_min(ndof_l2), target_1(ndof_l2);
   bounds_max = 1.0; bounds_min = 0.0;
   Vector ls_vals_nodes;
   Vector vol_moments(ndof_l2), l2_shape(ndof_l2);

   for (int e = 0; e < NE; e++)
   {
      const int attr = pfes.GetParMesh()->GetAttribute(e);
      if (attr == 10)
      {
         alpha_1(e) = 1.0;
         alpha_2(e) = 0.0;
         if (mat_data)
         {
            vol_1_loc = 1.0;
            vol_2_loc = 0.0;
            mat_data->vol_1.ParFESpace()->GetElementDofs(e, l2_dofs);
            mat_data->vol_1.SetSubVector(l2_dofs, vol_1_loc);
            mat_data->vol_2.SetSubVector(l2_dofs, vol_2_loc);
         }
         continue;
      }
      if (attr == 20)
      {
         alpha_1(e) = 0.0;
         alpha_2(e) = 1.0;
         if (mat_data)
         {
            vol_1_loc = 0.0;
            vol_2_loc = 1.0;
            mat_data->vol_1.ParFESpace()->GetElementDofs(e, l2_dofs);
            mat_data->vol_1.SetSubVector(l2_dofs, vol_1_loc);
            mat_data->vol_2.SetSubVector(l2_dofs, vol_2_loc);
         }
         continue;
      }

      ElementTransformation &Tr = *pfes.GetElementTransformation(e);
      level_set.GetValues(Tr, ir, ls_vals);
      double volume_1 = 0.0, volume = 0.0;
      vol_moments = 0.0;
      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         volume   += ip.weight * Tr.Weight();
         volume_1 += ip.weight * Tr.Weight() *
                     ((ls_vals(q) < 0.0) ? 1.0 : 0.0);
         if (mat_data)
         {
            mat_data->vol_1.ParFESpace()->GetFE(e)->CalcShape(ip, l2_shape);
            for (int i = 0; i < ndof_l2; i++)
            {
               vol_moments(i) += ip.weight * Tr.Weight() * l2_shape(i);
            }
         }
      }
      alpha_1(e) = volume_1 / volume;
      alpha_2(e) = 1.0 - alpha_1(e);

      // Tilt the alphas.
      // Target values are 1 or 0 at the dof locations.
      if (mat_data)
      {
         mat_data->vol_1.ParFESpace()->GetElementDofs(e, l2_dofs);
         if (pointwise_alpha == false)
         {
            vol_1_loc = alpha_1(e);
            vol_2_loc = alpha_2(e);
            mat_data->vol_1.SetSubVector(l2_dofs, vol_1_loc);
            mat_data->vol_2.SetSubVector(l2_dofs, vol_2_loc);
            continue;
         }
         level_set.GetValues(Tr, ir_nodes, ls_vals_nodes);
         for (int i = 0; i < ndof_l2; i++)
         {
            target_1(i) = ((ls_vals_nodes(i) < 0.0) ? 1.0 : 1e-12);
         }
         SLBQPOptimizer slbqp;
         slbqp.SetBounds(bounds_min, bounds_max);
         slbqp.SetLinearConstraint(vol_moments, volume_1);
         IterativeSolver::PrintLevel print;
         print.None();
         slbqp.SetPrintLevel(print);
         slbqp.Mult(target_1, vol_1_loc);
         for (int i = 0; i < ndof_l2; i++)
         {
            vol_2_loc(i) = 1.0 - vol_1_loc(i);
         }
         mat_data->vol_1.SetSubVector(l2_dofs, vol_1_loc);
         mat_data->vol_2.SetSubVector(l2_dofs, vol_2_loc);
      }
   }
}

void MaterialData::UpdateInitialMasses()
{
   ParFiniteElementSpace &pfes_L2 = *rhoDetJind0_1.ParFESpace();
   const IntegrationRule &ir_L2_nodes = pfes_L2.GetFE(0)->GetNodes();
   const int nd = ir_L2_nodes.GetNPoints(), NE = pfes_L2.GetNE();
   for (int e = 0; e < NE; e++)
   {
      ElementTransformation &tr_e = *pfes_L2.GetElementTransformation(e);
      for (int i = 0; i < nd; i++)
      {
         const IntegrationPoint &ip = ir_L2_nodes.IntPoint(i);
         tr_e.SetIntPoint(&ip);
         const double detJ = tr_e.Weight();
         rhoDetJind0_1(e * nd + i) = rho0_1(e * nd + i) *
                                     detJ * vol_1(e * nd + i);
         rhoDetJind0_2(e * nd + i) = rho0_2(e * nd + i) *
                                     detJ * vol_2(e * nd + i);
      }
   }
}

void MaterialData::ComputeTotalPressure(const ParGridFunction &p1_gf,
                                        const ParGridFunction &p2_gf)
{
   auto pfes = *p1_gf.ParFESpace();
   const int NE = pfes.GetNE();
   Vector ls_vals;
   for (int e = 0; e < NE; e++)
   {
      const IntegrationRule &ir = pfes.GetFE(e)->GetNodes();
      ElementTransformation &Tr = *pfes.GetElementTransformation(e);
      level_set.GetValues(Tr, ir, ls_vals);
      const int nqp = ir.GetNPoints();
      for (int q = 0; q < nqp; q++)
      {
         p(e * nqp + q) = (ls_vals(q) < 0.0) ? p1_gf(e * nqp + q)
                                             : p2_gf(e * nqp + q);
      }
   }
}

PressureFunction::PressureFunction(int prob, int mid, ParMesh &pmesh,
                                   PressureSpace space,
                                   ParGridFunction &alpha0,
                                   ParGridFunction &rho0, double g)
   : gamma(g), problem(prob), mat_id(mid), p_space(space),
     p_fec_L2(p_order, pmesh.Dimension(), BasisType::GaussLegendre),
     p_fec_H1(p_order, pmesh.Dimension(), BasisType::GaussLobatto),
     p_fes_L2(&pmesh, &p_fec_L2), p_fes_H1(&pmesh, &p_fec_H1),
     p_L2(&p_fes_L2), p_H1(&p_fes_H1),
     rho0DetJ0(p_L2.Size())
{
   p_L2 = 0.0;
   p_H1 = 0.0;
   UpdateRho0Alpha0(alpha0, rho0);
}

void PressureFunction::UpdateRho0Alpha0(const ParGridFunction &alpha0,
                                        const ParGridFunction &rho0)
{
   const int NE = p_fes_L2.GetParMesh()->GetNE();
   const int nqp = rho0DetJ0.Size() / NE;
   Vector rho_vals(nqp);

   for (int e = 0; e < NE; e++)
   {
      // The points (and their numbering) coincide with the nodes of p.
      const IntegrationRule &ir = p_fes_L2.GetFE(e)->GetNodes();
      ElementTransformation &Tr = *p_fes_L2.GetElementTransformation(e);

      rho0.GetValues(Tr, ir, rho_vals);
      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         rho0DetJ0(e * nqp + q) = Tr.Weight() *
                                  alpha0.GetValue(Tr, ip) * rho_vals(q);
      }
   }
}

void PressureFunction::UpdatePressure(const ParGridFunction &alpha,
                                      const ParGridFunction &energy)
{
   const int NE = p_fes_L2.GetParMesh()->GetNE();
   Vector e_vals;

   // Compute L2 pressure element by element.
   for (int e = 0; e < NE; e++)
   {
      const int attr = p_fes_L2.GetParMesh()->GetAttribute(e);

      // The points (and their numbering) coincide with the nodes of p.
      const IntegrationRule &ir = p_fes_L2.GetFE(e)->GetNodes();
      const int nqp = ir.GetNPoints();

      if ((attr == 10 && mat_id == 2) || (attr == 20 && mat_id == 1))
      {
         for (int q = 0; q < nqp; q++) { p_L2(e * nqp + q) = 0.0; }
         continue;
      }

      ElementTransformation &Tr = *p_fes_L2.GetElementTransformation(e);
      energy.GetValues(Tr, ir, e_vals);
      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         const double rho = rho0DetJ0(e * nqp + q) /
                            alpha.GetValue(Tr, ip) / Tr.Weight();
         p_L2(e * nqp + q) = fmax(1e-5, (gamma - 1.0) * rho * e_vals(q));

         if (problem == 9 && mat_id == 1)
         {
            // Water pressure in the water/air test.
            p_L2(e * nqp + q) -= gamma * 6.0e8;
         }
      }
   }

   // If H1 pressure is needed, average on the shared faces.
   if (p_space == H1)
   {
      GridFunctionCoefficient p_coeff(&p_L2);
      p_H1.ProjectDiscCoefficient(p_coeff, GridFunction::ARITHMETIC);
   }
}

} // namespace hydrodynamics

} // namespace mfem
