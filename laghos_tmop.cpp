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

void ExtendRefinementListToNeighbors(ParMesh &pmesh, Array<int> &intel);

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
                  ParGridFunction &interface_ls,
                  std::vector<ParGridFunction> ind)
{
   const int myid = x.ParFESpace()->GetMyRank();

   //
   // Setup.
   //
   const int    solver_type    = 0,
                solver_iter    = 1000;
   const double solver_rtol    = 1e-6;
   const int    precond        = 2;
   const int    max_lin_iter   = 100;
   const int    quad_order     = 8;
   const bool   normalization  = false;

   // Adaptive limiting.
   real_t adapt_lim_const = 0.0;

   // Limiting the node movement.
   real_t lim_const         = 0.0;

   ParFiniteElementSpace *pfespace = x.ParFESpace();
   ParMesh *pmesh = pfespace->GetParMesh();
   IntegrationRules *irules = &IntRulesLo;

   ParGridFunction x0(pfespace);
   x0 = x;

   const int mesh_poly_deg = pfespace->GetOrder(0);
   const int dim = pfespace->GetMesh()->Dimension();

   //
   // Metric + target.
   //
   // shape config.
   TMOP_QualityMetric *metric  = new TMOP_Metric_002;
   //TMOP_QualityMetric *metric  = new TMOP_Metric_050;
   auto ttype = TargetConstructor::IDEAL_SHAPE_UNIT_SIZE;
   // shape+size config.
   //TMOP_QualityMetric *metric  = new TMOP_Metric_080(0.5);
   //TMOP_QualityMetric *metric  = new TMOP_Metric_009;
   //TMOP_QualityMetric *metric  = new TMOP_Metric_007;
   //TMOP_QualityMetric *metric  = new TMOP_Metric_302;
   //auto ttype = TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE;
   //auto ttype = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE;
   // 328, 333, 334
   //TMOP_QualityMetric *metric  = new TMOP_Metric_334(0.5);

   TargetConstructor *target_c = new TargetConstructor(ttype, MPI_COMM_WORLD);
   target_c->SetNodes(x0);
   // Integrator.
   TMOP_Integrator *he_nlf_integ= new TMOP_Integrator(metric, target_c);
   he_nlf_integ->SetIntegrationRules(*irules, quad_order);

   // Adaptive limiting.
   ParGridFunction ind_combo(ind[0].ParFESpace());
   for (int i = 0; i < ind_combo.Size(); i++)
   {
      if (ind_combo.Size() == 3) { ind_combo(i) += ind[0](i) + 2.0 * ind[1](i); }
      else { ind_combo(i) = ind[0](i); }
   }
   ParGridFunction adapt_lim_gf0(interface_ls.ParFESpace());
   adapt_lim_gf0.ProjectGridFunction(ind_combo);
   InterpolatorFP adapt_lim_eval;
   ConstantCoefficient adapt_lim_coeff(adapt_lim_const);
   if (adapt_lim_const > 0.0)
   {
      socketstream vis1;
      hydrodynamics::VisualizeField(vis1, "localhost", 19916, adapt_lim_gf0,
                                    "Adaptive Region GF",
                                    0, 400, 300, 300);

      he_nlf_integ->EnableAdaptiveLimiting(adapt_lim_gf0, adapt_lim_coeff,
                                           adapt_lim_eval);
   }

   // Limit the node movement.
   // The limiting distances can be given by a general function of space.
   ParFiniteElementSpace dist_pfespace(pmesh,
                                       interface_ls.ParFESpace()->FEColl());
   ParGridFunction dist(&dist_pfespace);
   dist = 0.1;
   ConstantCoefficient lim_coeff(lim_const);
   if (lim_const != 0.0)
   {
      he_nlf_integ->EnableLimiting(x0, dist, lim_coeff);
   }

   if (normalization)
   {
      he_nlf_integ->ParEnableNormalization(x0);
   }

   // Objective.
   ParNonlinearForm a(pfespace);
   a.AddDomainIntegrator(he_nlf_integ);

   // Fix interface if required.
   a.SetEssentialVDofs(ess_vdofs);

   // Initial energy.
   const double init_energy = a.GetParGridFunctionEnergy(x);
   double init_m_energy = init_energy;

   // Print the initial mesh in a file.
   {
      std::ostringstream mesh_name;
      mesh_name << "initial" << "_" << "mesh.mesh";
      std::ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->PrintAsSerial(mesh_ofs);
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
         hs->SetPositiveDiagonal(true);
         minres->SetPreconditioner(*hs);
      }
   }
   solver.SetMaxIter(solver_iter);
   solver.SetRelTol(solver_rtol);
   solver.SetAbsTol(0.0);
   solver.SetPrintLevel(1);
   solver.SetOperator(a);

   // Solve.
   Vector b(0);
   x.SetTrueVector();
   solver.Mult(b, x.GetTrueVector());
   x.SetFromTrueVector();

   // Final energy
   const double fin_energy = a.GetParGridFunctionEnergy(x);
   double fin_metric_energy = fin_energy;
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

   // Print the final mesh in a file.
   {
      std::ostringstream mesh_name;
      mesh_name << "result" << "_" << "mesh.mesh";
      std::ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->PrintAsSerial(mesh_ofs);
      mesh_ofs.close();
   }
}

void ExtendRefinementListToNeighbors(ParMesh &pmesh, Array<int> &intel)
{
   mfem::L2_FECollection l2fec(0, pmesh.Dimension());
   mfem::ParFiniteElementSpace l2fespace(&pmesh, &l2fec);
   mfem::ParGridFunction el_to_refine(&l2fespace);
   const int quad_order = 4;

   el_to_refine = 0.0;

   for (int i = 0; i < intel.Size(); i++)
   {
      el_to_refine(intel[i]) = 1.0;
   }

   mfem::H1_FECollection lhfec(1, pmesh.Dimension());
   mfem::ParFiniteElementSpace lhfespace(&pmesh, &lhfec);
   mfem::ParGridFunction lhx(&lhfespace);

   el_to_refine.ExchangeFaceNbrData();
   GridFunctionCoefficient field_in_dg(&el_to_refine);
   lhx.ProjectDiscCoefficient(field_in_dg, GridFunction::ARITHMETIC);

   IntegrationRules irRules = IntegrationRules(0, Quadrature1D::GaussLobatto);
   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      Array<int> dofs;
      Vector x_vals;
      lhfespace.GetElementDofs(e, dofs);
      const IntegrationRule &ir =
         irRules.Get(pmesh.GetElementGeometry(e), quad_order);
      lhx.GetValues(e, ir, x_vals);
      double max_val = x_vals.Max();
      if (max_val > 0)
      {
         intel.Append(e);
      }
   }

   intel.Sort();
   intel.Unique();
}

} // namespace hydrodynamics

} // namespace mfem
