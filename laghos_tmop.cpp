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

namespace mfem
{

namespace hydrodynamics
{

IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);

void OptimizeMesh(ParGridFunction &x, Array<int> &ess_vdofs)
{
   const int myid = x.ParFESpace()->GetMyRank();

   const int    solver_type = 0,
                solver_iter = 20;
   const double solver_rtol = 1e-6;
   const int max_lin_iter   = 100;
   const int quad_order = 8;

   ParFiniteElementSpace *pfespace = x.ParFESpace();
   IntegrationRules *irules = &IntRulesLo;

   // Metric.
   TMOP_QualityMetric *metric  = new TMOP_Metric_002;
   // Target.
   auto ttype = TargetConstructor::IDEAL_SHAPE_UNIT_SIZE;
   TargetConstructor *target_c = new TargetConstructor(ttype, MPI_COMM_WORLD);
   // Integrator.
   TMOP_Integrator *he_nlf_integ= new TMOP_Integrator(metric, target_c);
   he_nlf_integ->SetIntegrationRules(*irules, quad_order);

   // Objective.
   ParNonlinearForm a(pfespace);
   a.AddDomainIntegrator(he_nlf_integ);
   a.SetEssentialVDofs(ess_vdofs);

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
   solver.SetPreconditioner(*minres);
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
}

} // namespace hydrodynamics

} // namespace mfem
