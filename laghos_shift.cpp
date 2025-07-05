// Copyright (cA) 2017, Lawrence Livermore National Security, LLC. Produced at
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

#include "laghos_shift.hpp"
#include "laghos_solver.hpp"

using namespace std;
namespace mfem
{

namespace hydrodynamics
{

void shift_shape(const ParFiniteElementSpace &pfes_e_const,
                 const ParFiniteElementSpace &pfes_p,
                 int e_id,
                 const IntegrationPoint &ip, const Vector &dist,
                 int nterms, Vector &shape_shift)
{
   auto pfes_e = const_cast<ParFiniteElementSpace *>(&pfes_e_const);
   const int NE = pfes_e->GetNE();
   const FiniteElement &el_e =
      (e_id < NE) ? *pfes_e->GetFE(e_id) : *pfes_e->GetFaceNbrFE(e_id - NE);
   const FiniteElement &el_p =
      (e_id < NE) ? *pfes_p.GetFE(e_id) : *pfes_p.GetFaceNbrFE(e_id - NE);
   const int dim = pfes_e->GetMesh()->Dimension(),
             dof_e = el_e.GetDof(), dof_p = el_p.GetDof();

   IsoparametricTransformation el_tr;
   if (e_id < NE)
   {
      pfes_e->GetElementTransformation(e_id, &el_tr);
   }
   else
   {
      pfes_e->GetParMesh()->GetFaceNbrElementTransformation(e_id - NE, el_tr);
   }
   DenseMatrix grad_phys;
   DenseMatrix Transfer_pe;
   el_p.Project(el_e, el_tr, Transfer_pe);
   el_p.ProjectGrad(el_p, el_tr, grad_phys);

   Vector s(dim*dof_p), t(dof_p);
   for (int j = 0; j < dof_e; j++)
   {
      // Shape function transformed into the p space.
      Vector u_shape_e(dof_e), u_shape_p(dof_p);
      u_shape_e = 0.0;
      u_shape_e(j) = 1.0;
      Transfer_pe.Mult(u_shape_e, u_shape_p);

      t = u_shape_p;
      int factorial = 1;
      for (int i = 1; i < nterms + 1; i++)
      {
         factorial = factorial*i;
         grad_phys.Mult(t, s);
         for (int j = 0; j < dof_p; j++)
         {
            t(j) = 0.0;
            for(int d = 0; d < dim; d++)
            {
               t(j) = t(j) + s(j + d * dof_p) * dist(d);
            }
         }
         u_shape_p.Add(1.0/double(factorial), t);
      }

      el_tr.SetIntPoint(&ip);
      el_p.CalcPhysShape(el_tr, t);
      shape_shift(j) = t * u_shape_p;
   }
}

void get_shifted_value(const ParGridFunction &g, int e_id,
                       const IntegrationPoint &ip, const Vector &dist,
                       int nterms, Vector &shifted_vec)
{
   auto pfes = const_cast<ParFiniteElementSpace *>(g.ParFESpace());
   const int NE = pfes->GetNE();
   const FiniteElement &el =
      (e_id < NE) ? *pfes->GetFE(e_id) : *pfes->GetFaceNbrFE(e_id - NE);
   const int dim = pfes->GetMesh()->Dimension(), dof = el.GetDof();
   const int vdim = pfes->GetVDim();

   IsoparametricTransformation el_tr;
   if (e_id < NE)
   {
      pfes->GetElementTransformation(e_id, &el_tr);
   }
   else
   {
      pfes->GetParMesh()->GetFaceNbrElementTransformation(e_id - NE, el_tr);
   }
   DenseMatrix grad_phys;
   el.ProjectGrad(el, el_tr, grad_phys);

   Array<int> vdofs;
   Vector u(dof), s(dim*dof), t(dof), g_loc(vdim*dof);
   if (e_id < NE)
   {
      g.FESpace()->GetElementVDofs(e_id, vdofs);
      g.GetSubVector(vdofs, g_loc);
   }
   else
   {
      g.ParFESpace()->GetFaceNbrElementVDofs(e_id - NE, vdofs);
      g.FaceNbrData().GetSubVector(vdofs, g_loc);
   }

   for (int c = 0; c < vdim; c++)
   {
      u.SetDataAndSize(g_loc.GetData() + c*dof, dof);

      t = u;
      int factorial = 1;
      for (int i = 1; i < nterms + 1; i++)
      {
         factorial = factorial*i;
         grad_phys.Mult(t, s);
         for (int j = 0; j < dof; j++)
         {
            t(j) = 0.0;
            for(int d = 0; d < dim; d++)
            {
               t(j) = t(j) + s(j + d * dof) * dist(d);
            }
         }
         u.Add(1.0/double(factorial), t);
      }

      el_tr.SetIntPoint(&ip);
      el.CalcPhysShape(el_tr, t);
      shifted_vec(c) = t * u;
   }
}

  
void SBM_BoundaryVectorMassIntegrator::
AssembleFaceMatrix(const FiniteElement &el1, const FiniteElement &el2,
                   FaceElementTransformations &Tr, DenseMatrix &elmat)
{
   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = 2 * el1.GetOrder();
      ir = &IntRules.Get(Tr.GetGeometryType(), order);
   }

   const int nqp_face = IntRule->GetNPoints();
   const int dof = el1.GetDof();
   elmat.SetSize(dof * vdim);
   elmat = 0.0;

   mcoeff.SetSize(vdim);
   shape.SetSize(dof);
   partelmat.SetSize(dof);
   for (int q = 0; q < nqp_face; q++)
   {
      const IntegrationPoint &ip_f = IntRule->IntPoint(q);
      // Set the integration point in the face and the neighboring elements
      Tr.SetAllIntPoints(&ip_f);

      MQ->Eval(mcoeff, Tr, ip_f);
      const IntegrationPoint &eip1 = Tr.GetElement1IntPoint();

      Vector position;
      Tr.Transform(eip1, position);
      Vector dist;
      Vector true_n;
      geom.ComputeDistanceAndNormal(position, dist, true_n);
      shift_shape(H1, H1, Tr.ElementNo, eip1, dist, 0, shape);

      MultVVt(shape, partelmat);

      for (int i = 0; i < vdim; i++)
      {
         for (int j = 0; j < vdim; j++)
         {
            elmat.AddMatrix(mcoeff(i,j), partelmat, dof*i, dof*j);
         }
      }
   }
}

void SBM_BoundaryMixedForceIntegrator::
AssembleFaceMatrix(const FiniteElement &trial_fe, const FiniteElement &test_fe,
                   FaceElementTransformations &Tr, DenseMatrix &elmat)
{
   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = trial_fe.GetOrder() + test_fe.GetOrder();
      ir = &IntRules.Get(Tr.GetGeometryType(), order);
   }

   const int nqp_face  = IntRule->GetNPoints();
   const int vdim      = Q.GetVDim();
   const int dof_trial = trial_fe.GetDof();
   const int dof_test  = test_fe.GetDof();

   elmat.SetSize(dof_test * vdim, dof_trial);
   elmat = 0.0;
   DenseMatrix loc_force(dof_test, vdim);
   Vector shape_trial(dof_trial), shape_test(dof_test),
          Vloc_force(loc_force.Data(), dof_test * vdim);
   Vector qcoeff(vdim);

   for (int q = 0; q < nqp_face; q++)
   {
      const IntegrationPoint &ip_f = IntRule->IntPoint(q);
      Tr.SetAllIntPoints(&ip_f);
      const IntegrationPoint &ip_e = Tr.GetElement1IntPoint();

      test_fe.CalcShape(ip_e, shape_test);
      trial_fe.CalcShape(ip_e, shape_trial);
      Q.Eval(qcoeff, Tr, ip_f);

      MultVWt(shape_test, qcoeff, loc_force);
      AddMultVWt(Vloc_force, shape_trial, elmat);
   }
}

  
void SBM_BoundaryMixedForceTIntegrator::
AssembleFaceMatrix(const FiniteElement &trial_fe, const FiniteElement &test_fe,
                   FaceElementTransformations &Tr, DenseMatrix &elmat)
{
   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = trial_fe.GetOrder() + test_fe.GetOrder();
      ir = &IntRules.Get(Tr.GetGeometryType(), order);
   }

   const int nqp_face  = IntRule->GetNPoints();
   const int vdim      = Q.GetVDim();
   const int dof_trial = trial_fe.GetDof();
   const int dof_test  = test_fe.GetDof();

   elmat.SetSize(dof_test, dof_trial * vdim);
   elmat = 0.0;
   DenseMatrix loc_force(dof_trial, vdim);
   Vector shape_trial(dof_trial), shape_test(dof_test),
          Vloc_force(loc_force.Data(), dof_trial * vdim);
   Vector qcoeff(vdim);

   for (int q = 0; q < nqp_face; q++)
   {
      const IntegrationPoint &ip_f = IntRule->IntPoint(q);
      Tr.SetAllIntPoints(&ip_f);
      const IntegrationPoint &ip_e = Tr.GetElement1IntPoint();

      test_fe.CalcShape(ip_e, shape_test);
      trial_fe.CalcShape(ip_e, shape_trial);
      Q.Eval(qcoeff, Tr, ip_f);

      MultVWt(shape_trial, qcoeff, loc_force);
      AddMultVWt(shape_test, Vloc_force, elmat);
   }
}

} // namespace hydrodynamics

} // namespace mfem
