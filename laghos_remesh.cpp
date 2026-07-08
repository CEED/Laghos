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

socketstream vism;

double EvaluateBoundaryDistance(ParGridFunction &coord_x,
                                const IntegrationRule &ir_bdr,
                                const AnalyticCompositeSurface &surfaces,
                                const Array<int> &be_to_surface,
                                bool visualize)
{
   // Compute the distance to the analytical boundary at the boundary
   // quadrature points and use that same field for the max norm and
   // visualization.
   ParMesh *pmesh = coord_x.ParFESpace()->GetParMesh();
   const int dim = pmesh->Dimension();
   Vector pos(dim);
   FaceQuadratureSpace distance_qs(*pmesh, ir_bdr, FaceType::Boundary);
   QuadratureFunction distance_qf(&distance_qs);
   double local_qp_max = 0.0;

   pmesh->NewNodes(coord_x, false);
   distance_qf = 0.0;

   for (int be = 0; be < pmesh->GetNBE(); be++)
   {
      const int surf_id = be_to_surface[be];
      if (surf_id < 0) { MFEM_ABORT("Boundary element not mapped to surface.") }

      FaceElementTransformations *bdr_face_tr = pmesh->GetBdrFaceTransformations(be);
      if (bdr_face_tr == NULL) { MFEM_ABORT("Null boundary face transformation.") }

      const AnalyticSurface *surface = surfaces.GetSurfaceID(surf_id);
      const int qf_be = distance_qf.GetSpace()->GetEntityIndex(*bdr_face_tr);
      Vector qval;

      for (int q = 0; q < ir_bdr.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir_bdr.IntPoint(q);
         bdr_face_tr->SetAllIntPoints(&ip);
         coord_x.GetVectorValue(*bdr_face_tr, ip, pos);
         distance_qf.GetValues(qf_be, q, qval);
         qval(0) = surface->DistanceToSurface(pos);
         local_qp_max = max(local_qp_max, qval(0));
      }
   }

   double global_qp_max = 0.0;
   MPI_Allreduce(&local_qp_max, &global_qp_max, 1, MPI_DOUBLE, MPI_MAX,
                 pmesh->GetComm());

   if (visualize)
   {
      ParFiniteElementSpace pfes_scalar(pmesh, coord_x.ParFESpace()->FEColl(), 1);
      ParGridFunction distance_sum(&pfes_scalar);
      ParGridFunction distance_count(&pfes_scalar);
      ParGridFunction distance_vis_gf(&pfes_scalar);
      Array<int> dofs;
      Vector vis_vals, count_vals, node_pos(dim), q_pos(dim), qval;
      DofTransformation dof_tr;

      distance_sum = 0.0;
      distance_count = 0.0;

      for (int be = 0; be < pmesh->GetNBE(); be++)
      {
         const FiniteElement *fe = pfes_scalar.GetBE(be);
         const IntegrationRule &nodes = fe->GetNodes();
         ElementTransformation *bdr_tr =
            pfes_scalar.GetBdrElementTransformation(be);
         FaceElementTransformations *bdr_face_tr =
            pmesh->GetBdrFaceTransformations(be);
         const int qf_be = distance_qf.GetSpace()->GetEntityIndex(*bdr_face_tr);

         if (bdr_tr == NULL || bdr_face_tr == NULL || qf_be < 0) { continue; }

         const IntegrationRule &ir_q = distance_qf.GetSpace()->GetIntRule(qf_be);
         pfes_scalar.GetBdrElementDofs(be, dofs, dof_tr);
         vis_vals.SetSize(nodes.GetNPoints());
         count_vals.SetSize(nodes.GetNPoints());
         count_vals = 1.0;

         for (int j = 0; j < nodes.GetNPoints(); j++)
         {
            const IntegrationPoint &node_ip = nodes.IntPoint(j);
            bdr_tr->SetIntPoint(&node_ip);
            coord_x.GetVectorValue(*bdr_tr, node_ip, node_pos);

            double min_dist2 = infinity();
            int best_q = 0;
            for (int q = 0; q < ir_q.GetNPoints(); q++)
            {
               const IntegrationPoint &q_ip = ir_q.IntPoint(q);
               bdr_face_tr->SetAllIntPoints(&q_ip);
               coord_x.GetVectorValue(*bdr_face_tr, q_ip, q_pos);

               double dist2 = 0.0;
               for (int d = 0; d < q_pos.Size(); d++)
               {
                  const double delta = node_pos(d) - q_pos(d);
                  dist2 += delta * delta;
               }
               if (dist2 < min_dist2)
               {
                  min_dist2 = dist2;
                  best_q = q;
               }
            }

            distance_qf.GetValues(qf_be, best_q, qval);
            vis_vals(j) = qval(0);
         }

         dof_tr.TransformPrimal(vis_vals);
         dof_tr.TransformPrimal(count_vals);
         distance_sum.AddElementVector(dofs, vis_vals);
         distance_count.AddElementVector(dofs, count_vals);
      }

      HypreParVector *true_sum = distance_sum.ParallelAssemble();
      HypreParVector *true_count = distance_count.ParallelAssemble();
      for (int i = 0; i < true_sum->Size(); i++)
      {
         const double count = (*true_count)[i];
         (*true_sum)[i] = (count > 0.0) ? ((*true_sum)[i] / count) : 0.0;
      }
      distance_vis_gf.SetFromTrueDofs(*true_sum);

      // Open a fresh GLVis connection so each ALE step gets its own window.
      socketstream vis;
      VisualizeField(vis, "localhost", 19916, distance_vis_gf,
                     "Distance to analytical boundary",
                     800, 0, 400, 400);

      delete true_sum;
      delete true_count;
   }

   return global_qp_max;
}

void OptimizeMesh(ParGridFunction &coord_x_in,
                  AnalyticCompositeSurface &surfaces,
                  const IntegrationRule &ir,
                  const IntegrationRule &ir_bdr,
                  double remesh_dist,
                  ParGridFunction &coord_x_out, bool vis)
{
   const int myid = coord_x_in.ParFESpace()->GetMyRank();

   const int    solver_type  = 0,
                solver_iter  = 1000;
   const double solver_rtol  = 1e-6;
   const int    precond      = 2;
   const int    art_type     = 0;
   const int    max_lin_iter = 100;
   const bool   glvis        = vis;

   ParFiniteElementSpace *pfes_mesh = coord_x_in.ParFESpace();
   ParMesh *pmesh = pfes_mesh->GetParMesh();
   const int dim = pfes_mesh->GetMesh()->Dimension();

   ParGridFunction x0(coord_x_in), coord_t(pfes_mesh);

   // Compute the minimum det(J) of the starting mesh.
   double min_detJ = infinity();
   const int NE = pmesh->GetNE();
   for (int e = 0; e < NE; e++)
   {
      ElementTransformation *transf = pmesh->GetElementTransformation(e);
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         transf->SetIntPoint(&ir.IntPoint(j));
         const double detJ = transf->Weight();
         MFEM_VERIFY(detJ > 0, "Inverted volumetric QP before remesh!");
         min_detJ = min(min_detJ, detJ);
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &min_detJ, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   if (myid == 0)
   { cout << "Minimum det(J) of the original mesh is " << min_detJ << endl; }
   MFEM_VERIFY(min_detJ > 0.0, "Inverted initial meshes are not supported.");

   // Mark which nodes to move tangentially.

   ParFiniteElementSpace pfes_scalar(pmesh, pfes_mesh->FEColl(), 1);
   ParGridFunction fit_marker_vis_gf(&pfes_scalar);
   fit_marker_vis_gf = 0.0;

   Array<int> ess_vdofs_marker(pfes_mesh->GetVSize());
   ess_vdofs_marker = 0;
   for (int dof = 0; dof < pfes_mesh->GetNDofs(); dof++)
   {
      int cnt = 0, params = dim;
      for (int s = 0; s < surfaces.GetNumSurfaces(); s++)
      {
         const AnalyticSurface *surf = surfaces.GetSurfaceID(s);
         const Array<bool> &m = surf->GetMarker();
         if (m[dof])
         {
            cnt++;
            params = min(params, surf->NumParams());
         }
      }

      fit_marker_vis_gf(dof) = cnt;
      if (cnt == 0) { continue; }
      if (cnt > 1) { params = 0; }

      for (int d = params; d < dim; d++)
      {
         ess_vdofs_marker[pfes_mesh->DofToVDof(dof, d)] = 1;
      }
   }

   Array<int> vdofs, ess_vdofs;
   FiniteElementSpace::MarkerToList(ess_vdofs_marker, ess_vdofs);

   for (int e = 0; e < pmesh->GetNBE(); e++)
   {
      pfes_mesh->GetBdrElementVDofs(e, vdofs);
      const int nd = pfes_mesh->GetBE(e)->GetDof();

      for (int j = 0; j < nd; j++)
      {
         int nconstrained = 0;
         for (int d = 0; d < dim; d++)
         {
            nconstrained += ess_vdofs_marker[pfes_mesh->DofToVDof(vdofs[j], d)];
         }
         fit_marker_vis_gf(vdofs[j]) = nconstrained;
      }
   }

   // Visualize the selected nodes and their target positions.
   if (glvis)
   {
      socketstream vis;
      common::VisualizeField(vism, "localhost", 19916, fit_marker_vis_gf,
                             "Marked DOFs",
                             0, 600, 400, 400, (dim == 2) ? "Rjm" : "");
      common::VisualizeMesh(vis, "localhost", 19916, *pmesh, "Initial mesh",
                            400, 600, 400, 400, "me");
   }

   surfaces.ConvertPhysCoordToParam(coord_x_in, coord_t);

   // if (glvis)
   // {
   //    surfaces.ConvertParamCoordToPhys(coord_t, coord_x_in);
   //    socketstream vis1;
   //    common::VisualizeMesh(vis1, "localhost", 19916, *pmesh, "Mesh x->t->x",
   //                          400, 600, 400, 400, "me");
   //    coord_x_in = x0;
   // }

   // TMOP setup.
   TMOP_QualityMetric *metric;
   if (dim == 2) { metric = new TMOP_Metric_002; }
   else          { metric = new TMOP_Metric_302; }
   metric->use_old_invariants_code = true;
   TargetConstructor target(TargetConstructor::IDEAL_SHAPE_UNIT_SIZE,
                            pfes_mesh->GetComm());
   target.SetNodes(coord_x_in);
   auto integ = new TMOP_Integrator(metric, &target, nullptr);
   integ->EnableTangentialMovement(surfaces, *pfes_mesh);

   ParFiniteElementSpace pfes_dist(pmesh, pfes_mesh->FEColl(), 1);
   ParGridFunction dist(&pfes_dist);
   dist = remesh_dist; // smaller is less motion.
   ConstantCoefficient limit_coeff(1.0);
   integ->EnableLimiting(x0, dist, limit_coeff);

   integ->ParEnableNormalization(x0);

   // Linear solver.
   MINRESSolver minres(pfes_mesh->GetComm());
   minres.SetMaxIter(100);
   minres.SetRelTol(1e-8);
   minres.SetAbsTol(0.0);

   // Nonlinear solver.
   ParNonlinearForm nlf(pfes_mesh);
   nlf.SetEssentialVDofs(ess_vdofs);
   nlf.AddDomainIntegrator(integ);
   TMOPNewtonSolver solver(pfes_mesh->GetComm(), ir, 0);
   //solver.SetBdrIntegrationRule(ir_bdr);
   solver.SetOperator(nlf);
   solver.SetPreconditioner(minres);
   solver.SetPrintLevel(1);
   solver.SetMaxIter(100);
   solver.SetRelTol(1e-6);
   solver.SetAbsTol(0.0);

   // Solve.
   Vector zero(0);
   coord_t.SetTrueVector();
   solver.Mult(zero, coord_t.GetTrueVector());
   coord_t.SetFromTrueVector();
   surfaces.ConvertParamCoordToPhys(coord_t, coord_x_out);
   if (glvis)
   {
      coord_x_in = coord_x_out;
      socketstream vis;
      common::VisualizeMesh(vis, "localhost", 19916, *pmesh, "Final mesh",
                            800, 600, 400, 400, "me");
      coord_x_in = x0;
   }

   delete metric;
}

} // namespace hydrodynamics

} // namespace mfem
