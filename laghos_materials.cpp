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

void MaterialData::UpdateAlpha()
{
   auto pfes = *alpha_1.ParFESpace();
   const IntegrationRule &ir = IntRules.Get(pfes.GetFE(0)->GetGeomType(), 20);
   const int NE = alpha_1.ParFESpace()->GetNE(),
             nqp = ir.GetNPoints();
   Vector ls_vals;

   for (int e = 0; e < NE; e++)
   {
      ElementTransformation &Tr = *pfes.GetElementTransformation(e);
      level_set.GetValues(Tr, ir, ls_vals);
      double volume_1 = 0.0, volume = 0.0;
      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         volume   += ip.weight * Tr.Weight();
         volume_1 += ip.weight * Tr.Weight() *
                     ((ls_vals(q) < 0.0) ? 1.0 : 0.0);
      }
      alpha_1(e) = volume_1 / volume;
      alpha_2(e) = 1.0 - alpha_1(e);
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
                                   ParGridFunction &rho0,
                                   ParGridFunction &gamma)
   : problem(prob), mat_id(mid), p_space(space),
     p_fec_L2(p_order, pmesh.Dimension(), BasisType::GaussLegendre),
     p_fec_H1(p_order, pmesh.Dimension(), BasisType::GaussLobatto),
     p_fes_L2(&pmesh, &p_fec_L2), p_fes_H1(&pmesh, &p_fec_H1),
     p_L2(&p_fes_L2), p_H1(&p_fes_H1),
     rho0DetJ0(p_L2.Size()), gamma_gf(gamma)
{
   p_L2 = 0.0;
   p_H1 = 0.0;

   const int NE = pmesh.GetNE();
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
         rho0DetJ0(e * nqp + q) = Tr.Weight() * alpha0(e) * rho_vals(q);
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
      // The points (and their numbering) coincide with the nodes of p.
      const IntegrationRule &ir = p_fes_L2.GetFE(e)->GetNodes();
      const int nqp = ir.GetNPoints();
      ElementTransformation &Tr = *p_fes_L2.GetElementTransformation(e);

      if (alpha(e) < 1e-12)
      {
         for (int q = 0; q < nqp; q++) { p_L2(e * nqp + q) = 0.0; }
         continue;
      }

      energy.GetValues(Tr, ir, e_vals);
      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         const double rho = rho0DetJ0(e * nqp + q) / alpha(e) / Tr.Weight();
         p_L2(e * nqp + q) = fmax(1e-5, (gamma_gf(e) - 1.0) * rho * e_vals(q));

         if (problem == 9 && mat_id == 1)
         {
            // Water pressure in the water/air test.
            p_L2(e * nqp + q) -= gamma_gf(e) * 6.0e8;
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
