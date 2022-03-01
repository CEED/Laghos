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

PressureFunction::PressureFunction(int prob, ParMesh &pmesh,
                                   PressureSpace space,
                                   ParGridFunction &rho0,
                                   ParGridFunction &gamma)
   : problem(prob), p_space(space),
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
         rho0DetJ0(e * nqp + q) = Tr.Weight() * rho_vals(q);
      }
   }
}

void PressureFunction::UpdatePressure(const ParGridFunction &energy)
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

      energy.GetValues(Tr, ir, e_vals);

      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         double rho = rho0DetJ0(e * nqp + q) / Tr.Weight();
         p_L2(e * nqp + q) = fmax(1e-5, (gamma_gf(e) - 1.0) * rho * e_vals(q));

         if (problem == 9 && p_fes_L2.GetParMesh()->GetAttribute(e) == 1)
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

double InterfaceRhoCoeff::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   return (level_set.GetValue(T, ip) < 0.0) ? rho_1.GetValue(T, ip)
                                            : rho_2.GetValue(T, ip);
}

} // namespace hydrodynamics

} // namespace mfem
