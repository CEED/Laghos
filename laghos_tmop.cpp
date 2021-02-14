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

#include "laghos_tmop.hpp"
#include "laghos_solver.hpp"

using namespace std;
namespace mfem
{

namespace hydrodynamics
{

int material_id(int el_id, const GridFunction &g)
{
   const FiniteElementSpace *fes = g.FESpace();
   const FiniteElement *fe = fes->GetFE(el_id);
   Vector g_vals;
   const IntegrationRule &ir =
      IntRules.Get(fe->GetGeomType(), fes->GetOrder(el_id) + 2);

   double integral = 0.0;
   g.GetValues(el_id, ir, g_vals);
   ElementTransformation *Tr = fes->GetMesh()->GetElementTransformation(el_id);
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      Tr->SetIntPoint(&ip);
      integral += ip.weight * g_vals(q) * Tr->Weight();
   }
   return (integral > 0.0) ? 1.0 : 0.0;
}

IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);

void OptimizeMesh(ParGridFunction &x, Array<int> &ess_vdofs,
                  ParGridFunction &interface_ls)
{
   const int myid = x.ParFESpace()->GetMyRank();

   const int    solver_type  = 0,
                solver_iter  = 1000;
   const double solver_rtol  = 1e-6;
   const int    precond      = 2;
   const int    art_type     = 0;
   const int    max_lin_iter = 100;
   const int    quad_order   = 8;
   const double surface_fit_const = 10000.0;
   const bool   fix_interface     = true;

   ParFiniteElementSpace *pfespace = x.ParFESpace();
   ParMesh *pmesh = pfespace->GetParMesh();
   IntegrationRules *irules = &IntRulesLo;

   ParGridFunction x0(pfespace);
   x0 = x;

   const int mesh_poly_deg = pfespace->GetOrder(0);
   const int dim = pfespace->GetMesh()->Dimension();

   // Metric.
   TMOP_QualityMetric *metric  = new TMOP_Metric_080(0.5);
   //TMOP_QualityMetric *metric  = new TMOP_Metric_002;
   //TMOP_QualityMetric *metric  = new TMOP_Metric_009;
   //TMOP_QualityMetric *metric  = new TMOP_Metric_302;
   // 328, 333, 334
   //TMOP_QualityMetric *metric  = new TMOP_Metric_334(0.5);
   // Target.
   auto ttype = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE;
   //auto ttype = TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE;
   //auto ttype = TargetConstructor::IDEAL_SHAPE_UNIT_SIZE;
   TargetConstructor *target_c = new TargetConstructor(ttype, MPI_COMM_WORLD);
   target_c->SetNodes(x0);
   // Integrator.
   TMOP_Integrator *he_nlf_integ= new TMOP_Integrator(metric, target_c);
   he_nlf_integ->SetIntegrationRules(*irules, quad_order);

   // Surface fitting.
   L2_FECollection mat_coll(0, dim);
   H1_FECollection sigma_fec(mesh_poly_deg, dim);
   ParFiniteElementSpace sigma_fes(pmesh, &sigma_fec);
   ParFiniteElementSpace mat_fes(pmesh, &mat_coll);
   ParGridFunction mat(&mat_fes);
   ParGridFunction marker_gf(&sigma_fes);
   ParGridFunction ls_0(&sigma_fes);
   Array<bool> marker(ls_0.Size());
   ConstantCoefficient coef_ls(surface_fit_const);
   AdaptivityEvaluator *adapt_surface = NULL;
   Array<int> extra_vdofs(0);
   if (surface_fit_const > 0.0)
   {
      ls_0 = interface_ls;

      for (int i = 0; i < pmesh->GetNE(); i++)
      {
         mat(i) = material_id(i, ls_0);
         pmesh->SetAttribute(i, mat(i) + 1);
      }
      pmesh->SetAttributes();

      GridFunctionCoefficient coeff_mat(&mat);
      marker_gf.ProjectDiscCoefficient(coeff_mat, GridFunction::ARITHMETIC);
      for (int j = 0; j < marker.Size(); j++)
      {
         //if (marker_gf(j) > 0.1 && marker_gf(j) < 0.9)
         if (interface_ls(j) == 0.0)
         {
            marker[j] = true;
            marker_gf(j) = 1.0;
            extra_vdofs.Append(j);
            extra_vdofs.Append(j + marker.Size());
            if (dim == 3) { extra_vdofs.Append(j + 2 * marker.Size()); }
         }
         else
         {
            marker[j] = false;
            marker_gf(j) = 0.0;
         }
      }

      adapt_surface = new InterpolatorFP;

      he_nlf_integ->EnableSurfaceFitting(ls_0, marker, coef_ls, *adapt_surface);
      socketstream vis1, vis2, vis3;
      hydrodynamics::VisualizeField(vis1, "localhost", 19916, ls_0,
                                    "Level Set 0",
                                    300, 600, 300, 300);
      hydrodynamics::VisualizeField(vis2, "localhost", 19916, mat,
                                    "Materials",
                                    600, 600, 300, 300);
      hydrodynamics::VisualizeField(vis3, "localhost", 19916, marker_gf,
                                    "Surface DOF",
                                    900, 600, 300, 300);
   }

   he_nlf_integ->ParEnableNormalization(x0);

   // Objective.
   ParNonlinearForm a(pfespace);
   a.AddDomainIntegrator(he_nlf_integ);
   if (fix_interface) { ess_vdofs.Append(extra_vdofs); }
   if (myid == 0)
   {
      //ess_vdofs.Append(14);
      //ess_vdofs.Append(14 + marker.Size());
   }
   /*
   if (myid == 2)
   {
      ess_vdofs.Append(15);
      ess_vdofs.Append(15 + marker.Size());
   }
   */
   a.SetEssentialVDofs(ess_vdofs);

   // Initial energy.
   const double init_energy = a.GetParGridFunctionEnergy(x);
   double init_m_energy = init_energy;
   if (surface_fit_const > 0.0)
   {
      coef_ls.constant   = 0.0;
      init_m_energy = a.GetParGridFunctionEnergy(x);
      coef_ls.constant   = surface_fit_const;
   }

   // Print the initial mesh in a file.
   {
      std::ostringstream mesh_name;
      mesh_name << "initial" << "_" << "mesh";
      std::ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->PrintAsOne(mesh_ofs);
      mesh_ofs.close();
   }

   // Linear solver.
   const double linsol_rtol = 1e-12;
   MINRESSolver *minres = new MINRESSolver(MPI_COMM_WORLD);
   minres->SetMaxIter(max_lin_iter);
   minres->SetRelTol(linsol_rtol);
   minres->SetAbsTol(0.0);
   minres->SetPrintLevel(-1);

   // Nonlinear solver.
   const IntegrationRule &ir =
      irules->Get(pfespace->GetFE(0)->GetGeomType(), quad_order);
   TMOPNewtonSolver solver(pfespace->GetComm(), ir, solver_type);
   if (solver_type == 0)
   {
      solver.SetPreconditioner(*minres);
      if (precond > 0)
      {
         HypreSmoother *hs = new HypreSmoother;
         hs->SetType((precond == 1) ? HypreSmoother::Jacobi
                                    : HypreSmoother::l1Jacobi, 1);
         minres->SetPreconditioner(*hs);
      }
   }
   solver.SetMaxIter(solver_iter);
   solver.SetRelTol(solver_rtol);
   solver.SetAbsTol(0.0);
   solver.SetPrintLevel(1);
   if (art_type > 0)
   {
      solver.SetAdaptiveLinRtol(art_type, 0.5, 0.9);
   }
   solver.SetOperator(a);

   // Solve.
   Vector b(0);
   x.SetTrueVector();
   solver.Mult(b, x.GetTrueVector());
   x.SetFromTrueVector();

   if (surface_fit_const > 0.0)
   {
      socketstream vis2, vis3;
      hydrodynamics::VisualizeField(vis2, "localhost", 19916, mat,
                                    "Materials (Optimized)",
                                     600, 900, 300, 300);
      hydrodynamics::VisualizeField(vis3, "localhost", 19916, marker_gf,
                                    "Surface DOF (Optimized)",
                                     900, 900, 300, 300);
   }

   // Final energy
   const double fin_energy = a.GetParGridFunctionEnergy(x);
   double fin_metric_energy = fin_energy;
   if (surface_fit_const > 0.0)
   {
      coef_ls.constant   = 0.0;
      fin_metric_energy  = a.GetParGridFunctionEnergy(x);
      coef_ls.constant   = surface_fit_const;
   }
   if (myid == 0)
   {
      std::cout << std::scientific << std::setprecision(4);
      cout << "Initial strain energy: " << init_energy
           << " = metrics: " << init_m_energy
           << " + extra terms: " << init_energy - init_m_energy << endl;
      cout << "  Final strain energy: " << fin_energy
           << " = metrics: " << fin_metric_energy
           << " + extra terms: " << fin_energy - fin_metric_energy << endl;
      cout << "The strain energy decreased by: "
           << (init_m_energy - fin_metric_energy) * 100.0 / init_m_energy << " %." << endl;
   }
   double err_avg, err_max;
   he_nlf_integ->GetSurfaceFittingErrors(err_avg, err_max);
   if (myid == 0)
   {
      cout << "Fitting error max: " << err_max << endl
           << "Fitting error avg: " << err_avg << endl;
   }

   // Visualize the displacement.
   x0 -= x;
   socketstream sock;
   if (myid == 0)
   {
      sock.open("localhost", 19916);
      sock << "solution\n";
   }
   pmesh->PrintAsOne(sock);
   x0.SaveAsOne(sock);
   if (myid == 0)
   {
      sock << "window_title 'Displacements'\n"
              << "window_geometry "
              << 300 << " " << 900 << " " << 300 << " " << 300 << "\n"
              << "keys jRmclA" << endl;
   }

   // Print the final mesh in a file.
   {
      std::ostringstream mesh_name;
      mesh_name << "result" << "_" << "mesh";
      std::ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->PrintAsOne(mesh_ofs);
      mesh_ofs.close();
   }
}

} // namespace hydrodynamics

} // namespace mfem
