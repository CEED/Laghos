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

#include "laghos_remesh.hpp"
#include "../mfem/miniapps/common/mfem-common.hpp"
#include "laghos_solver.hpp"

using namespace std;
namespace mfem
{

namespace hydrodynamics
{

IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);

void OptimizeMesh(ParGridFunction &coord_x_in,
                  ParGridFunction &coord_x_out)
{
   const int myid = coord_x_in.ParFESpace()->GetMyRank();

   const int    solver_type  = 0,
                solver_iter  = 1000;
   const double solver_rtol  = 1e-6;
   const int    precond      = 2;
   const int    art_type     = 0;
   const int    max_lin_iter = 100;
   const int    quad_order   = 8;
   const bool   glvis        = true;

   ParFiniteElementSpace *pfes_mesh = coord_x_in.ParFESpace();
   ParMesh *pmesh = pfes_mesh->GetParMesh();
   IntegrationRules *irules = &IntRulesLo;

   ParGridFunction x0(coord_x_in), coord_t(pfes_mesh);

   const int dim = pfes_mesh->GetMesh()->Dimension();

   // Compute the minimum det(J) of the starting mesh.
   double min_detJ = infinity();
   const int NE = pmesh->GetNE();
   for (int e = 0; e < NE; e++)
   {
      const IntegrationRule &ir =
          IntRules.Get(pfes_mesh->GetFE(0)->GetGeomType(), quad_order);
      ElementTransformation *transf = pmesh->GetElementTransformation(e);
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         transf->SetIntPoint(&ir.IntPoint(j));
         min_detJ = min(min_detJ, transf->Jacobian().Det());
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &min_detJ, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   if (myid == 0)
   { cout << "Minimum det(J) of the original mesh is " << min_detJ << endl; }
   MFEM_VERIFY(min_detJ > 0.0, "Inverted initial meshes are not supported.");

   // Mark which nodes to move tangentially.
   Array<bool> fit_marker_top(pfes_mesh->GetNDofs());
   Array<bool> fit_marker_right(pfes_mesh->GetNDofs());
   Array<bool> fit_marker_2(pfes_mesh->GetNDofs());
   ParFiniteElementSpace pfes_scalar(pmesh, pfes_mesh->FEColl(), 1);
   ParGridFunction fit_marker_vis_gf(&pfes_scalar);
   Array<int> vdofs, ess_vdofs;
   fit_marker_top   = false;
   fit_marker_right = false;
   fit_marker_2     = false;
   fit_marker_vis_gf = 0.0;
   for (int e = 0; e < pmesh->GetNBE(); e++)
   {
      const int attr = pmesh->GetBdrElement(e)->GetAttribute();
      const int nd = pfes_mesh->GetBE(e)->GetDof();
      pfes_mesh->GetBdrElementVDofs(e, vdofs);

      // Top boundary.
      if (attr == 1)
      {
         for (int j = 0; j < nd; j++)
         {
            // Eliminate y component.
            ess_vdofs.Append(vdofs[j+nd]);
            fit_marker_top[vdofs[j]] = true;
         }
      }
      // Right boundary.
      else if (attr == 2)
      {
         for (int j = 0; j < nd; j++)
         {
            // Eliminate y component.
            ess_vdofs.Append(vdofs[j+nd]);
            fit_marker_right[vdofs[j]] = true;
         }
      }
      else if (attr == 3)
      {
         // Fix y components.
         for (int j = 0; j < nd; j++)
         {
            fit_marker_2[vdofs[j]] = true;
            ess_vdofs.Append(vdofs[j+nd]);
         }
      }
      else if (attr == 4)
      {
         // Fix x components.
         for (int j = 0; j < nd; j++)
         {
            fit_marker_2[vdofs[j]] = true;
            ess_vdofs.Append(vdofs[j]);
         }
      }
   }

   for (int e = 0; e < pmesh->GetNBE(); e++)
   {
      pfes_mesh->GetBdrElementVDofs(e, vdofs);
      const int nd = pfes_mesh->GetBE(e)->GetDof();

      for (int j = 0; j < nd; j++)
      {
         int cnt = 0;
         if (fit_marker_top[vdofs[j]])   { cnt++; }
         if (fit_marker_right[vdofs[j]]) { cnt++; }
         if (fit_marker_2[vdofs[j]])     { cnt++; }

         fit_marker_vis_gf(vdofs[j]) = cnt;

         if (cnt > 1) { ess_vdofs.Append(vdofs[j]); }
      }
   }

   // Visualize the selected nodes and their target positions.
   if (glvis)
   {
      socketstream vis1, vis2, vis3;
      common::VisualizeField(vis1, "localhost", 19916, fit_marker_vis_gf,
                             "Target positions (DOFS with value 1)",
                             0, 600, 400, 400, (dim == 2) ? "Rjm" : "");
      common::VisualizeMesh(vis2, "localhost", 19916, *pmesh, "Initial mesh",
                            400, 600, 400, 400, "me");
   }

   Array<const AnalyticSurface *> surf_array;
   Line_Top line_top(fit_marker_top);
   Curve_Sine_Top curve_top(fit_marker_top);
   Line_Right line_right(fit_marker_right);
   Curve_Sine_Right curve_right(fit_marker_right);
   surf_array.Append(&line_top);
   //surf_array.Append(&curve_top);
   surf_array.Append(&line_right);
   //surf_array.Append(&curve_right);

   AnalyticCompositeSurface surfaces(surf_array);
   surfaces.ConvertPhysCoordToParam(coord_x_in, coord_t);

   if (glvis)
   {
      surfaces.ConvertParamCoordToPhys(coord_t, coord_x_in);
      socketstream vis1;
      common::VisualizeMesh(vis1, "localhost", 19916, *pmesh, "Mesh x->t->x",
                            400, 600, 400, 400, "me");
      coord_x_in = x0;
   }

   // TMOP setup.
   TMOP_QualityMetric *metric;
   if (dim == 2) { metric = new TMOP_Metric_002; }
   else          { metric = new TMOP_Metric_302; }
   metric->use_old_invariants_code = true;
   TargetConstructor target(TargetConstructor::IDEAL_SHAPE_UNIT_SIZE,
                            pfes_mesh->GetComm());
   auto integ = new TMOP_Integrator(metric, &target, nullptr);
   integ->EnableTangentialMovement(surfaces, *pfes_mesh);

   // Linear solver.
   MINRESSolver minres(pfes_mesh->GetComm());
   minres.SetMaxIter(100);
   minres.SetRelTol(1e-8);
   minres.SetAbsTol(0.0);

   // Nonlinear solver.
   ParNonlinearForm a(pfes_mesh);
   a.SetEssentialVDofs(ess_vdofs);
   a.AddDomainIntegrator(integ);
   const IntegrationRule &ir =
       IntRules.Get(pfes_mesh->GetFE(0)->GetGeomType(), quad_order);
   TMOPNewtonSolver solver(pfes_mesh->GetComm(), ir, 0);
   solver.SetOperator(a);
   solver.SetPreconditioner(minres);
   solver.SetPrintLevel(1);
   solver.SetMaxIter(50);
   solver.SetRelTol(1e-6);
   solver.SetAbsTol(0.0);

   // Solve.
   Vector b(0);
   coord_t.SetTrueVector();
   solver.Mult(b, coord_t.GetTrueVector());
   coord_t.SetFromTrueVector();
   surfaces.ConvertParamCoordToPhys(coord_t, coord_x_out);
   if (glvis)
   {
      coord_x_in = coord_x_out;
      socketstream vis2;
      common::VisualizeMesh(vis2, "localhost", 19916, *pmesh, "Final mesh",
                            800, 600, 400, 400, "me");
      coord_x_in = x0;
   }

   delete metric;
}

} // namespace hydrodynamics

} // namespace mfem
