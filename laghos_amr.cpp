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

#include "laghos_solver.hpp"
#include "laghos_amr.hpp"

namespace mfem
{

namespace amr
{

static const char *EstimatorName(const int est)
{
   switch (static_cast<amr::estimator>(est))
   {
      case amr::estimator::custom: return "Custom";
      case amr::estimator::jjt: return "JJt";
      case amr::estimator::zz: return "ZZ";
      case amr::estimator::kelly: return "Kelly";
      default: MFEM_ABORT("Unknown estimator!");
   }
   return nullptr;
}

static void FindElementsWithVertex(const Mesh* mesh, const Vertex &vert,
                                   const double size, Array<int> &elements)
{
   Array<int> v;

   for (int i = 0; i < mesh->GetNE(); i++)
   {
      mesh->GetElementVertices(i, v);
      for (int j = 0; j < v.Size(); j++)
      {
         double dist = 0.0;
         for (int l = 0; l < mesh->SpaceDimension(); l++)
         {
            double d = vert(l) - mesh->GetVertex(v[j])[l];
            dist += d*d;
         }
         if (dist <= size*size) { elements.Append(i); break; }
      }
   }
}

static void Pow(Vector &vec, double p)
{
   for (int i = 0; i < vec.Size(); i++)
   {
      vec(i) = std::pow(vec(i), p);
   }
}

static void GetPerElementMinMax(const GridFunction &gf,
                                Vector &elem_min, Vector &elem_max,
                                int int_order = -1)
{
   const FiniteElementSpace *space = gf.FESpace();
   int ne = space->GetNE();

   if (int_order < 0) { int_order = space->GetOrder(0) + 1; }

   elem_min.SetSize(ne);
   elem_max.SetSize(ne);

   Vector vals, tmp;
   for (int i = 0; i < ne; i++)
   {
      int geom = space->GetFE(i)->GetGeomType();
      const IntegrationRule &ir = IntRules.Get(geom, int_order);

      gf.GetValues(i, ir, vals);

      if (space->GetVDim() > 1)
      {
         Pow(vals, 2.0);
         for (int vd = 1; vd < space->GetVDim(); vd++)
         {
            gf.GetValues(i, ir, tmp, vd+1);
            Pow(tmp, 2.0);
            vals += tmp;
         }
         Pow(vals, 0.5);
      }

      elem_min(i) = vals.Min();
      elem_max(i) = vals.Max();
   }
}

void AMREstimatorIntegrator::ComputeElementFlux1(const FiniteElement &el,
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

void AMREstimatorIntegrator::ComputeElementFlux2(const int el,
                                                 const FiniteElement &fe,
                                                 ElementTransformation &Trans,
                                                 const FiniteElement &fluxelem,
                                                 Vector &flux)
{
   const int dim = fe.GetDim();
   const int sdim = Trans.GetSpaceDim();

   DenseMatrix Jadjt(dim, sdim), Jadj(dim, sdim);

   const IntegrationRule &ir = fluxelem.GetNodes();
   const int NQ = ir.GetNPoints();
   flux.SetSize(NQ * sdim);

   constexpr double NL_DMAX = std::numeric_limits<double>::max();
   double minW = +NL_DMAX;
   double maxW = -NL_DMAX;

   const int depth = pmesh->pncmesh->GetElementDepth(el);

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
         if (rho > jac_threshold) { continue; }
         if (depth > max_level) { continue; }
         flux(iq) = rho;
      }
   }
}

void AMREstimatorIntegrator::ComputeElementFlux(const FiniteElement &fe,
                                                ElementTransformation &Tr,
                                                Vector &u,
                                                const FiniteElement &fluxelem,
                                                Vector &flux,
                                                bool with_coef,
                                                const IntegrationRule *ir)
{
   MFEM_VERIFY(ir == nullptr, "ir not supported");
   MFEM_VERIFY(NE == pmesh->GetNE(), "");
   // ZZ comes with with_coef set to true, not Kelly
   switch (flux_mode)
   {
      case mode::diffusion:
      {
         DiffusionIntegrator::ComputeElementFlux(fe, Tr, u,
                                                 fluxelem, flux, with_coef);
         break;
      }
      case mode::one:
      {
         AMREstimatorIntegrator::ComputeElementFlux1(fe, Tr, u, fluxelem, flux);
         break;
      }
      case mode::two:
      {
         AMREstimatorIntegrator::ComputeElementFlux2(e++, fe, Tr, fluxelem, flux);
         break;
      }
      default: MFEM_ABORT("Unknown mode!");
   }
}

AMR::AMR(ParMesh *pmesh, int estimator,
         double ref_t, double jac_t, double deref_t,
         int max_level,  int nc_limit,
         double size_b, double energy_b, double *blast_xyz):
   pmesh(pmesh),
   myid(pmesh->GetMyRank()),
   dim(pmesh->Dimension()),
   sdim(pmesh->SpaceDimension()),
   flux_fec(order, dim),
   flux_fes(pmesh, &flux_fec, sdim),
   opt(
{
   estimator, ref_t, jac_t, deref_t, max_level, nc_limit,
   size_b, energy_b, Vertex(blast_xyz[0], blast_xyz[1], blast_xyz[2])
}) { }

AMR::~AMR() { }

void AMR::Setup(ParGridFunction &x_gf)
{
   if (myid == 0)
   {
      std::cout << "AMR setup with "
                << amr::EstimatorName(opt.estimator) << " estimator"
                << std::endl;
   }

   if (opt.estimator == amr::estimator::zz)
   {
      integ = new AMREstimatorIntegrator(pmesh, opt.max_level, opt.jac_threshold);
      smooth_flux_fec = new RT_FECollection(order-1, dim);
      auto smooth_flux_fes = new ParFiniteElementSpace(pmesh, smooth_flux_fec);
      estimator = new L2ZienkiewiczZhuEstimator(*integ, x_gf, &flux_fes,
                                                smooth_flux_fes);
   }

   if (opt.estimator == amr::estimator::kelly)
   {
      integ = new AMREstimatorIntegrator(pmesh, opt.max_level, opt.jac_threshold);
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

void AMR::Reset()
{
   if (integ) { integ->Reset(); }
   if (refiner) { refiner->Reset(); }
   if (derefiner) { derefiner->Reset(); }
}

void AMR::Update(hydrodynamics::LagrangianHydroOperator &hydro,
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
   Array<Refinement> refined;
   bool mesh_updated = false;
   const int NE = pmesh->GetNE();
   ParFiniteElementSpace &H1FESpace = *x.ParFESpace();
   constexpr double NL_DMAX = std::numeric_limits<double>::max();

   GetPerElementMinMax(v, v_min, v_max);

   switch (opt.estimator)
   {
      case amr::estimator::custom:
      {
         Vector &error_est = hydro.GetZoneMaxVisc();
         for (int el = 0; el < NE; el++)
         {
            if (error_est(el) > opt.ref_threshold &&
                pmesh->pncmesh->GetElementDepth(el) < opt.max_level &&
                (v_min(el) < 1e-3)) // only refine the still area
            {
               refined.Append(Refinement(el));
            }
         }
         break;
      }

      case amr::estimator::jjt:
      {
         const double art = opt.ref_threshold;
         const int h1_order = H1FESpace.GetOrder(0) + 1;
         DenseMatrix Jadjt, Jadj(dim, pmesh->SpaceDimension());
         MFEM_VERIFY(art >= 0.0, "AMR threshold should be positive");
         for (int el = 0; el < NE; el++)
         {
            double minW = +NL_DMAX;
            double maxW = -NL_DMAX;
            const int depth = pmesh->pncmesh->GetElementDepth(el);
            ElementTransformation *eTr = pmesh->GetElementTransformation(el);
            const Geometry::Type &type = pmesh->GetElement(el)->GetGeometryType();
            const IntegrationRule *ir = &IntRules.Get(type, h1_order);
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
               if (rho < opt.jac_threshold && depth < opt.max_level)
               {
                  refined.Append(Refinement(el));
               }
            }
         }
         break;
      }

      case amr::estimator::zz:
      case amr::estimator::kelly:
      {
         refiner->Apply(*pmesh);
         if (refiner->Refined()) { mesh_updated = true; }
         MFEM_VERIFY(!refiner->Derefined() && !refiner->Rebalanced(), "");
         break;
      }
      default: MFEM_ABORT("Unknown AMR estimator!");
   }

   // custom and JJt use 'refined', ZZ and Kelly will set 'mesh_updated'
   const int nref = pmesh->ReduceInt(refined.Size());
   if (nref && !mesh_updated)
   {
      constexpr int non_conforming = 1;
      pmesh->GeneralRefinement(refined, non_conforming, opt.nc_limit);
      mesh_updated = true;
      if (myid == 0)
      {
         std::cout << "Refined " << nref << " elements." << std::endl;
      }
   }
   else if (opt.estimator == amr::estimator::custom &&
            opt.deref_threshold >= 0.0 && !mesh_updated)
   {
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
      FindElementsWithVertex(pmesh, opt.blast_position, opt.blast_size, elements);
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

      const int op = 2; // operatoe on the 'maximum' value of the fine elements
      mesh_updated = pmesh->DerefineByError(rho_max, threshold, opt.nc_limit, op);
      if (mesh_updated && myid == 0)
      {
         std::cout << "Derefined, threshold = " << threshold << std::endl;
      }
   }
   else if ((opt.estimator == amr::estimator::zz ||
             opt.estimator == amr::estimator::kelly) && !mesh_updated)
   {
      MFEM_VERIFY(derefiner,"");
      if (derefiner->Apply(*pmesh))
      {
         //if (myid == 0) { std::cout << "\nDerefined elements." << std::endl; }
      }
      if (derefiner->Derefined()) { mesh_updated = true; }
   }
   else { /* nothing to do */ }

   if (mesh_updated)
   {
      auto Update = [&]()
      {
         ParFiniteElementSpace* H1FESpace = x.ParFESpace();
         ParFiniteElementSpace* L2FESpace = e.ParFESpace();
         ParFiniteElementSpace* MEFESpace = m.ParFESpace();

         H1FESpace->Update();
         L2FESpace->Update();
         MEFESpace->Update();

         const int h1_vsize = H1FESpace->GetVSize();
         const int l2_vsize = L2FESpace->GetVSize();

         true_offset[0] = 0;
         true_offset[1] = true_offset[0] + h1_vsize;
         true_offset[2] = true_offset[1] + h1_vsize;
         true_offset[3] = true_offset[2] + l2_vsize;

         S_old = S;
         S.Update(true_offset);
         const Operator* h1_update = H1FESpace->GetUpdateOperator();
         const Operator* l2_update = L2FESpace->GetUpdateOperator();
         h1_update->Mult(S_old.GetBlock(0), S.GetBlock(0));
         h1_update->Mult(S_old.GetBlock(1), S.GetBlock(1));
         l2_update->Mult(S_old.GetBlock(2), S.GetBlock(2));

         x.MakeRef(H1FESpace, S, true_offset[0]);
         v.MakeRef(H1FESpace, S, true_offset[1]);
         e.MakeRef(L2FESpace, S, true_offset[2]);
         x.SyncAliasMemory(S);
         v.SyncAliasMemory(S);
         e.SyncAliasMemory(S);

         S_old.Update(true_offset);
         m.Update();
      };

      constexpr bool quick = true;

      Update();
      hydro.AMRUpdate(S, quick);

      pmesh->Rebalance();

      Update();
      hydro.AMRUpdate(S, not quick);

      GetZeroBCDofs(pmesh, H1FESpace, bdr_attr_max, ess_tdofs, ess_vdofs);
      ode_solver->Init(hydro);
      //H1FESpace.PrintPartitionStats();
   }
}

} // namespace amr

} // namespace mfem
