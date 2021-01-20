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

#define MFEM_DEBUG_COLOR 226
#include "general/debug.hpp"

#include "laghos_solver.hpp"
#include "laghos_amr.hpp"

namespace mfem
{

namespace amr
{

void EstimatorIntegrator::ComputeElementFlux1(const FiniteElement &el,
                                              ElementTransformation &Trans,
                                              const Vector &u,
                                              const FiniteElement &fluxelem,
                                              Vector &flux)
{
   const int dof = el.GetDof();
   const int dim = el.GetDim();
   const int sdim = Trans.GetSpaceDim();

   DenseMatrix dshape(dof, dim);
   DenseMatrix invdfdx(dim, sdim);
   Vector vec(dim), pointflux(sdim);

   const IntegrationRule &ir = fluxelem.GetNodes();
   const int NQ = ir.GetNPoints();
   flux.SetSize(NQ * sdim);

   for (int q = 0; q < NQ; q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      el.CalcDShape(ip, dshape);
      dshape.MultTranspose(u, vec);

      Trans.SetIntPoint (&ip);
      CalcInverse(Trans.Jacobian(), invdfdx);
      invdfdx.MultTranspose(vec, pointflux);

      for (int d = 0; d < sdim; d++)
      {
         flux(NQ*d+q) = pointflux(d);
      }
   }
}

void EstimatorIntegrator::ComputeElementFlux2(const int e,
                                              const FiniteElement &el,
                                              ElementTransformation &Trans,
                                              const FiniteElement &fluxelem,
                                              Vector &flux)
{
   const int dim = el.GetDim();
   const int sdim = Trans.GetSpaceDim();

   DenseMatrix Jadjt(dim, sdim), Jadj(dim, sdim);

   const IntegrationRule &ir = fluxelem.GetNodes();
   const int NQ = ir.GetNPoints();
   flux.SetSize(NQ * sdim);

   constexpr double NL_DMAX = std::numeric_limits<double>::max();
   double minW = +NL_DMAX;
   double maxW = -NL_DMAX;

   const int depth = pmesh->pncmesh->GetElementDepth(e);

   for (int q = 0; q < NQ; q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      Trans.SetIntPoint(&ip);
      CalcAdjugate(Trans.Jacobian(), Jadj);
      Jadjt = Jadj;
      Jadjt.Transpose();
      const double w = Jadjt.Weight();
      minW = std::fmin(minW, w);
      maxW = std::fmax(maxW, w);
      MFEM_VERIFY(std::fabs(maxW) > 1e-13, "");
      const double rho = minW / maxW;
      MFEM_VERIFY(rho <= 1.0, "");
      for (int d = 0; d < sdim; d++)
      {
         const int iq = NQ*d + q;
         flux(iq) = 1.0 - rho;
         //dbg("%f > %f", rho, jac_threshold);
         if (rho > jac_threshold) { continue; }
         //dbg("rho:%f, depth:%d", rho, depth);
         if (depth > max_level) { continue; }
         flux(iq) = rho;
      }
   }
}

void EstimatorIntegrator::ComputeElementFlux(const FiniteElement &el,
                                             ElementTransformation &Trans,
                                             Vector &u,
                                             const FiniteElement &fluxelem,
                                             Vector &flux,
                                             bool with_coef)
{
   MFEM_VERIFY(NE == pmesh->GetNE(), "");
   //dbg("NE:%d, e:%d", NE, e);
   // ZZ comes with with_coef set to true, not Kelly
   switch (flux_mode)
   {
      case mode::diffusion:
      {
         DiffusionIntegrator::ComputeElementFlux(el, Trans, u,
                                                 fluxelem, flux, with_coef);
         break;
      }
      case mode::one:
      {
         ComputeElementFlux1(el, Trans, u, fluxelem, flux);
         break;
      }
      case mode::two:
      {
         ComputeElementFlux2(e++, el, Trans, fluxelem, flux);
         break;
      }
      default: MFEM_ABORT("Unknown mode!");
   }
}

static void Eval(const ParGridFunction &v_gf, const double threshold,
                 ParMesh &pmesh, Array<Refinement> &refs)
{
   Vector val;
   IntegrationPoint ip;
   const int NE = pmesh.GetNE();
   const int dim  = pmesh.Dimension();
   const int kmax = dim - 2;
   constexpr double w = 1.0;

   //ParFiniteElementSpace *pfes = v_gf.ParFESpace();

   for (int e = 0; e < NE; e++)
   {
      for (int x1 = 0; x1 <= 1; x1++)
      {
         for (int x2 = 0; x2 <= 1; x2++)
         {
            for (int x3 = 0; x3 <= kmax; x3++)
            {
               ip.Set(x1, x2, x3, w);
               v_gf.GetVectorValue(e, ip, val);
               const double l2_norm = val.Norml2();
               if (l2_norm > threshold) { refs.Append(Refinement(e)); }
            }
         }
      }
   }
}

Operator::Operator(ParMesh *pmesh,
                   int estimator,
                   double ref_t,
                   double jac_t,
                   double deref_t,
                   int max_level,
                   int nc_limit,
                   double size_b,
                   double energy_b,
                   double *xyz_b):
   pmesh(pmesh),
   myid(pmesh->GetMyRank()),
   dim(pmesh->Dimension()),
   sdim(pmesh->SpaceDimension()),
   flux_fec(order, dim),
   flux_fes(pmesh, &flux_fec, sdim),
   opt(
{
   estimator, ref_t, jac_t, deref_t, max_level, nc_limit,
              size_b, energy_b, Vertex(xyz_b[0], xyz_b[1], xyz_b[2])
}) { dbg("%s", amr::EstimatorName(opt.estimator)); }

Operator::~Operator() { dbg(); }

void Operator::Setup(ParGridFunction &x_gf)
{
   dbg();
   if (myid == 0)
   {
      std::cout << "AMR setup with "
                << amr::EstimatorName(opt.estimator) << " estimator"
                << std::endl;
   }

   if (opt.estimator == amr::estimator::zz)
   {
      dbg("ZZ estimator init");
      integ = new amr::EstimatorIntegrator(pmesh, opt.max_level,
                                           opt.jac_threshold);
      smooth_flux_fec = new RT_FECollection(order-1, dim);
      auto smooth_flux_fes = new ParFiniteElementSpace(pmesh, smooth_flux_fec);
      estimator = new L2ZienkiewiczZhuEstimator(*integ, x_gf, &flux_fes,
                                                smooth_flux_fes);
   }

   if (opt.estimator == amr::estimator::kelly)
   {
      dbg("Kelly estimator init");
      integ = new amr::EstimatorIntegrator(pmesh, opt.max_level,
                                           opt.jac_threshold);
      estimator = new KellyErrorEstimator(*integ, x_gf, flux_fes);
   }

   if (estimator)
   {
      const double hysteresis = 0.25;
      const double max_elem_error = 1.0e-6;
      refiner = new ThresholdRefiner(*estimator);
      refiner->SetTotalErrorFraction(0.0);
      refiner->SetLocalErrorGoal(max_elem_error);
      refiner->PreferConformingRefinement();
      refiner->SetNCLimit(opt.nc_limit);

      derefiner = new ThresholdDerefiner(*estimator);
      derefiner->SetOp(2); // 0:min, 1:sum, 2:max
      derefiner->SetThreshold(hysteresis * max_elem_error);
      derefiner->SetNCLimit(opt.nc_limit);
   }
}

void Operator::Reset()
{
   if (integ) { integ->Reset(); }
   if (refiner) { refiner->Reset(); }
   if (derefiner) { derefiner->Reset(); }
}

void Operator::Update(hydrodynamics::LagrangianHydroOperator &hydro,
                      ODESolver *ode_solver,
                      BlockVector &S,
                      BlockVector &S_old,
                      ParGridFunction &x,
                      ParGridFunction &v,
                      ParGridFunction &e,
                      ParGridFunction &m,
                      Array<int> &true_offset,
                      const int bdr_attr_max,
                      Array<int> &ess_tdofs,
                      Array<int> &ess_vdofs)
{
   Vector v_max, v_min;
   Array<Refinement> refs;
   bool mesh_refined = false;
   const int NE = pmesh->GetNE();
   ParFiniteElementSpace &H1FESpace = *x.ParFESpace();
   constexpr double NL_DMAX = std::numeric_limits<double>::max();

   GetPerElementMinMax(v, v_min, v_max);

   switch (opt.estimator)
   {
      case amr::estimator::custom:
      {
         Vector &error_est = hydro.GetZoneMaxVisc();
         for (int e = 0; e < NE; e++)
         {
            if (error_est(e) > opt.ref_threshold &&
                pmesh->pncmesh->GetElementDepth(e) < opt.max_level &&
                (v_min(e) < 1e-3)) // only refine the still area
            {
               refs.Append(Refinement(e));
            }
         }
         break;
      }

      case amr::estimator::jjt:
      {
         MFEM_VERIFY(dim == 2, "JJt estimator only available for 2D!");
         const double art = opt.ref_threshold;
         const int order = H1FESpace.GetOrder(0) + 1;
         DenseMatrix Jadjt, Jadj(dim, pmesh->SpaceDimension());
         MFEM_VERIFY(art >= 0.0, "AMR threshold should be positive");
         for (int e = 0; e < NE; e++)
         {
            double minW = +NL_DMAX;
            double maxW = -NL_DMAX;
            const int depth = pmesh->pncmesh->GetElementDepth(e);
            ElementTransformation *eTr = pmesh->GetElementTransformation(e);
            const Geometry::Type &type = pmesh->GetElement(e)->GetGeometryType();
            const IntegrationRule *ir = &IntRules.Get(type, order);
            const int NQ = ir->GetNPoints();
            for (int q = 0; q < NQ; q++)
            {
               eTr->SetIntPoint(&ir->IntPoint(q));
               const DenseMatrix &J = eTr->Jacobian();
               CalcAdjugate(J, Jadj);
               Jadjt = Jadj;
               Jadjt.Transpose();
               const double w = Jadjt.Weight();
               minW = std::fmin(minW, w);
               maxW = std::fmax(maxW, w);
            }
            if (std::fabs(maxW) != 0.0)
            {
               const double rho = minW / maxW;
               MFEM_VERIFY(rho <= 1.0, "");
               //dbg("%f", rho);
               if (rho < opt.jac_threshold && depth < opt.max_level)
               {
                  refs.Append(Refinement(e));
               }
            }
         }
         break;
      }

      case amr::estimator::zz:
      case amr::estimator::kelly:
      {
         //dbg("AMR estimator Apply");
         refiner->Apply(*pmesh);
         if (refiner->Refined()) { dbg("\033[1;32mREFINED!"); mesh_refined = true; }
         MFEM_VERIFY(!refiner->Derefined(),"");
         MFEM_VERIFY(!refiner->Rebalanced(),"");
         break;
      }
      default: MFEM_ABORT("Unknown AMR estimator!");
   }

   // custom and JJt uses refs, ZZ and Kelly will set mesh_refined
   const int nref = pmesh->ReduceInt(refs.Size());
   if (nref && !mesh_refined)
   {
      dbg("GeneralRefinement");
      constexpr int non_conforming = 1;
      pmesh->GeneralRefinement(refs, non_conforming, opt.nc_limit);
      mesh_refined = true;
      if (myid == 0)
      {
         std::cout << "Refined " << nref << " elements." << std::endl;
      }
   }
   else if (opt.estimator == amr::estimator::custom &&
            opt.deref_threshold >= 0.0 && !mesh_refined)
   {
      //dbg("Derefinement");
      ParGridFunction rho_gf;
      Vector rho_max, rho_min;
      hydro.ComputeDensity(rho_gf);
      GetPerElementMinMax(rho_gf, rho_min, rho_max);

      // Derefinement based on zone maximum rho in post-shock region
      const double rho_max_max = rho_max.Size() ? rho_max.Max() : 0.0;
      double threshold, loc_threshold = opt.deref_threshold * rho_max_max;
      MPI_Allreduce(&loc_threshold, &threshold, 1, MPI_DOUBLE, MPI_MAX,
                    pmesh->GetComm());

      // make sure the blast point is never derefined
      Array<int> elements;
      FindElementsWithVertex(pmesh, opt.blast_position,
                             opt.blast_size, elements);
      for (int i = 0; i < elements.Size(); i++)
      {
         int index = elements[i];
         if (index >= 0) { rho_max(index) = NL_DMAX; }
      }

      // only derefine where the mesh is in motion, i.e. after the shock
      for (int i = 0; i < pmesh->GetNE(); i++)
      {
         if (v_min(i) < 0.1) { rho_max(i) = NL_DMAX; }
      }

      const int op = 2; // maximum value of fine elements
      mesh_refined = pmesh->DerefineByError(rho_max, threshold,
                                            opt.nc_limit, op);
      if (mesh_refined && myid == 0)
      {
         std::cout << "Derefined, threshold = " << threshold << std::endl;
      }
   }
   else if ((opt.estimator == amr::estimator::zz ||
             opt.estimator == amr::estimator::kelly) && !mesh_refined)
   {
      //dbg("ZZ/Kelly Derefinement");
      MFEM_VERIFY(derefiner,"");
      if (derefiner->Apply(*pmesh))
      {
         if (myid == 0)
         {
            //std::cout << "\nDerefined elements." << std::endl;
         }
      }
      if (derefiner->Derefined()) {  dbg("\033[1;31mDEREFINED!"); mesh_refined = true; }
   }
   else { /* nothing to do */ }

   if (mesh_refined)
   {
      constexpr bool quick = true;

      amr::Update(S, S_old, true_offset, x, v, e, m);
      hydro.AMRUpdate(S, quick);

      pmesh->Rebalance();

      amr::Update(S, S_old, true_offset, x, v, e, m);
      hydro.AMRUpdate(S, !quick);

      GetZeroBCDofs(pmesh, H1FESpace, bdr_attr_max, ess_tdofs, ess_vdofs);
      ode_solver->Init(hydro);
      //H1FESpace.PrintPartitionStats();
   }
}

} // namespace amr

} // namespace mfem
