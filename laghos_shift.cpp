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
      pfes_e->GetParMesh()->GetFaceNbrElementTransformation(e_id - NE, &el_tr);
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
      pfes->GetParMesh()->GetFaceNbrElementTransformation(e_id - NE, &el_tr);
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

int SIMarker::GetMaterialID(int el_id)
{
   IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);

   const ParFiniteElementSpace &pfes = *ls.ParFESpace();
   const FiniteElement *fe = pfes.GetFE(el_id);
   Vector ls_vals;
   const IntegrationRule &ir = IntRulesLo.Get(fe->GetGeomType(), 20);

   ls.GetValues(el_id, ir, ls_vals);
   ElementTransformation *Tr = pfes.GetMesh()->GetElementTransformation(el_id);
   double volume_1 = 0.0, volume_2 = 0.0, volume = 0.0;
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      Tr->SetIntPoint(&ip);

      volume += ip.weight * Tr->Weight();
      if (ls_vals(q) + 1e-12 < 0.0) { volume_1 += ip.weight * Tr->Weight(); }
      if (ls_vals(q) - 1e-12 > 0.0) { volume_2 += ip.weight * Tr->Weight(); }
   }

   if (volume_1 == 0.0) { return 20; }
   if (volume_2 == 0.0) { return 10; }
   return 15;
}

void SIMarker::MarkFaceAttributes()
{
   auto get_face_attr = [&](int a1, int a2)
   {
      if (a1 == 15 && a2 == 10) { return 10; }
      if (a1 == 10 && a2 == 15) { return 10; }
      if (a1 == 15 && a2 == 20) { return 20; }
      if (a1 == 20 && a2 == 15) { return 20; }
      if (a1 == 15 && a2 == 15) { return 15; }
      return 0;
   };

   ParMesh *pmesh = ls.ParFESpace()->GetParMesh();
   pmesh->ExchangeFaceNbrNodes();

   // Mark faces of mixed elements, when both sides are local zones.
   for (int f = 0; f < pmesh->GetNumFaces(); f++)
   {
      pmesh->SetFaceAttribute(f, 0);

      auto *ft = pmesh->GetFaceElementTransformations(f, 3);
      if (ft->Elem2No < 0) { continue; }

      const int attr1 = mat_attr(ft->Elem1No),
                attr2 = mat_attr(ft->Elem2No);
      pmesh->SetFaceAttribute(f, get_face_attr(attr1, attr2));
   }

   // Mark faces of mixed elements, when the faces are shared.
   mat_attr.ExchangeFaceNbrData();
   for (int f = 0; f < pmesh->GetNSharedFaces(); f++)
   {
      auto *ftr = pmesh->GetSharedFaceTransformations(f, true);
      int attr1 = mat_attr(ftr->Elem1No);
      IntegrationPoint ip; ip.Init(0);
      int attr2 = mat_attr.GetValue(*ftr->Elem2, ip);

      int faceno = pmesh->GetSharedFace(f);
      pmesh->SetFaceAttribute(faceno, get_face_attr(attr1, attr2));
   }
}

void SIMarker::GetFaceAttributeGF(ParGridFunction &fa_gf)
{
   fa_gf.SetSpace(ls.ParFESpace());
   fa_gf = 0.0;

   ParMesh &pmesh = *ls.ParFESpace()->GetParMesh();
   Array<int> face_dofs;
   for (int f = 0; f < pmesh.GetNumFaces(); f++)
   {
      ls.ParFESpace()->GetFaceDofs(f, face_dofs);
      for (int i = 0; i < face_dofs.Size(); i++)
      {
         fa_gf(face_dofs[i]) = fmax(fa_gf(face_dofs[i]),
                                    pmesh.GetFaceAttribute(f));
      }
   }

   GroupCommunicator &gcomm = fa_gf.ParFESpace()->GroupComm();
   Array<double> maxvals(fa_gf.GetData(), fa_gf.Size());
   gcomm.Reduce<double>(maxvals, GroupCommunicator::Max);
   gcomm.Bcast(maxvals);
}

double InterfaceCoeff::Eval(ElementTransformation &T,
                            const IntegrationPoint &ip)
{
   Vector x(3);
   T.Transform(ip, x);

   const int dim = pmesh.Dimension();

   // Modes for Taylor-Green (problem 0):
   // 0 - vertical
   // 1 - diagonal
   // 2 - circle
   const int mode_TG = (pure_test) ? 0 : 2;

   switch (problem)
   {
      case 0:
      case 1:
      {
         if (mode_TG == 0)
         {
            // The domain area for the TG is 1.
            const double dx = sqrt(1.0 / glob_NE);

            // The middle of the element after x = 0.5.
            return (pure_test) ? tanh(x(0) - 0.5)
                               : tanh(x(0) - (0.5 + 0.5 * dx));
         }
         else if (mode_TG == 1)
         {
            MFEM_VERIFY(pure_test == false, "Can't do a pure diagonal.");
            return tanh(x(0) - x(1));
         }
         else if (mode_TG == 2)
         {
            MFEM_VERIFY(pure_test == false, "Can't do a pure circle.");
            double center[3] = {0.5, 0.5, 0.5};
            double rad = 0.0;
            for (int d = 0; d < dim; d++)
            {
               rad += (x(d) - center[d]) * (x(d) - center[d]);
            }
            rad = sqrt(rad + 1e-16);
            return tanh(rad - 0.3);
         }
         else { MFEM_ABORT("wrong TG mode"); return 0.0; }
      }
      case 8:
      {
         // Sod - the 1D domain length is 1.
         const double dx = 1.0 / glob_NE;
         return (pure_test) ? x(0) - 0.5
                            : x(0) - (0.5 + 0.5*dx);
      }
      case 9:
      {
         // Water-air - the 1D domain length is 1.
         const double dx = 1.0 / glob_NE;
         return (pure_test) ? x(0) - 0.7
                            : x(0) - (0.7 + 0.5*dx);
      }
      case 10:
      {
         if (pure_test)
         {
            if (x(0) < 1.0) { return -1.0; }
            if (x(1) > 1.5) { return -1.0; }
            return 1.0;
         }

         // The domain volume for the 3point is 21 in 2D, and 63 in 3D.
         double dx;
         if (dim == 2) { dx = sqrt(21.0 / glob_NE); }
         else          { dx =  pow(63.0 / glob_NE, 1.0/3.0); }

         // The middle of the element before x = 1.
         // The middle of the element above y = 1.5.
         if (x(0) < 1.0 - 0.5 * dx) { return -1.0; }
         if (x(1) > 1.5 + 0.5 * dx) { return -1.0; }
         return 1.0;
      }
      case 12:
      {
         if (x(0) >= 0.5   && x(0) <= 0.625) { return -1.0; }
         if (x(0) >= 0.25  && x(0) <= 0.5 &&
             x(1) >= 0.375 && x(1) <= 0.625) { return -1.0; }
         return 1.0;
      }
      default: MFEM_ABORT("error"); return 0.0;
   }
}

void GradAtLocalDofs(ElementTransformation &T, const ParGridFunction &g,
                     DenseMatrix &grad_g)
{
   ParFiniteElementSpace &pfes = *g.ParFESpace();
   const FiniteElement &el = *pfes.GetFE(T.ElementNo);
   const int dim = el.GetDim(), dof = el.GetDof();
   grad_g.SetSize(dof, dim);
   Array<int> dofs;
   Vector g_e;
   DenseMatrix grad_phys; // This will be (dof_p x dim, dof_p).

   pfes.GetElementDofs(T.ElementNo, dofs);
   g.GetSubVector(dofs, g_e);
   el.ProjectGrad(el, T, grad_phys);
   Vector grad_ptr(grad_g.GetData(), dof*dim);
   grad_phys.Mult(g_e, grad_ptr);
}

void StrainTensorAtLocalDofs(ElementTransformation &T, const ParGridFunction &g,
                             DenseTensor &grad_g)
{
   ParFiniteElementSpace &pfes = *g.ParFESpace();
   const int zone_id = T.ElementNo;
   const FiniteElement &el = *pfes.GetFE(zone_id);
   const int dim = el.GetDim(), dof = el.GetDof();
   MFEM_VERIFY(dim == pfes.GetVDim(), " Strain Tensor can only be obtained for"
                                      " vector GridFunctions.");
   grad_g.SetSize(dof, dim, dim);
   Array<int> dofs;
   Vector g_e;
   DenseMatrix grad_phys; // This will be (dof x dim, dof).
   {
      pfes.GetElementVDofs(zone_id, dofs);
      g.GetSubVector(dofs, g_e);
      el.ProjectGrad(el, T, grad_phys);
      for (int d = 0; d < dim; d++)
      {
         Vector g_e_d(g_e.GetData()+d*dof, dof);
         Vector grad_ptr(grad_g.GetData(0)+d*dof*dim, dof*dim);
         grad_phys.Mult(g_e_d, grad_ptr);
      }
   }
}

void MomentumInterfaceIntegrator::AssembleRHSElementVect(
      const FiniteElement &el_1, const FiniteElement &el_2,
      FaceElementTransformations &Trans, Vector &elvect)
{
   const int h1dofs_cnt = el_1.GetDof();
   const int dim = el_1.GetDim();
   const int NE = mat_data.alpha_1.FESpace()->GetNE();
   const bool local_face = (Trans.Elem2No < NE);

   if (local_face == false)
   {
      // This is a shared face between mpi tasks.
      // Each task assembles only its side, but uses info from the neighbor.
      elvect.SetSize(h1dofs_cnt * dim);
   }
   else { elvect.SetSize(h1dofs_cnt * dim * 2); }
   elvect = 0.0;

   // The early return must be done after elmat.SetSize().
   const int attr_face = Trans.Attribute;
   if (attr_face != 10 && attr_face != 20) { return; }

   ElementTransformation &Trans_e1 = Trans.GetElement1Transformation();
   ElementTransformation &Trans_e2 = Trans.GetElement2Transformation();

   const int attr_e1 = Trans_e1.Attribute;
   const ParGridFunction *p_e1, *p_e2;
   if ( (attr_face == 10 && attr_e1 == 10) ||
        (attr_face == 20 && attr_e1 == 15) )
   {
      p_e1     = &mat_data.p_1->GetPressure();
      p_e2     = &mat_data.p_2->GetPressure();
   }
   else if ( (attr_face == 10 && attr_e1 == 15) ||
             (attr_face == 20 && attr_e1 == 20) )
   {
      p_e1     = &mat_data.p_2->GetPressure();
      p_e2     = &mat_data.p_1->GetPressure();
   }
   else { MFEM_ABORT("Invalid marking configuration."); return; }

   // The alpha scaling is always taken from the mixed element.
   // GetValue() is used so that this can work in parallel.
   double alpha_scale = (attr_e1 == 15)
      ? mat_data.alpha_1.GetValue(Trans_e1,
                                  Geometries.GetCenter(el_1.GetGeomType()))
      : mat_data.alpha_1.GetValue(Trans_e2,
                                  Geometries.GetCenter(el_2.GetGeomType()));
   // For 10-faces we use 1-alpha_1, for 20-faces we use 1-alpha_2 = alpha_1.
   if (attr_face == 10) { alpha_scale = 1.0 - alpha_scale; }
   MFEM_VERIFY(alpha_scale > 1e-12 && alpha_scale < 1.0-1e-12,
               "The mixed zone is a 1-material zone! Check it.");

   Vector h1_shape(h1dofs_cnt);
   Vector nor(dim), d_q(dim);

   MFEM_VERIFY(IntRule != nullptr, "Must have been set in advance");
   const int nqp_face = IntRule->GetNPoints();

   for (int q = 0; q < nqp_face; q++)
   {
      // Set the integration point in the face and the neighboring elements
      const IntegrationPoint &ip_f = IntRule->IntPoint(q);
      Trans.SetAllIntPoints(&ip_f);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &ip_e1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &ip_e2 = Trans.GetElement2IntPoint();

      // The normal includes the Jac scaling.
      // The orientation is taken into account in the processing of element 1.
      if (dim == 1)
      {
         nor(0) = (2*ip_e1.x - 1.0) * Trans.Weight();
      }
      else { CalcOrtho(Trans.Jacobian(), nor); }
      nor *= ip_f.weight;

      // Compute el1 quantities. Allows negative pressure.
      // Note that the distance vector is a continuous function.
      Trans_e1.SetIntPoint(&ip_e1);
      dist.Eval(d_q, Trans_e1, ip_e1);
      Vector shift_p_e1(1); double p_q1;
      if (use_mixed_elem == false)
      {
         p_q1 = p_e1->GetValue(Trans_e1, ip_e1);
         get_shifted_value(*p_e1, Trans_e1.ElementNo, ip_e1, d_q,
                           num_taylor, shift_p_e1);
      }
      else if (attr_e1 == 15)
      {
         p_q1 = p_e1->GetValue(Trans_e1, ip_e1);
         get_shifted_value(*p_e1, Trans_e1.ElementNo, ip_e1, d_q,
                           num_taylor, shift_p_e1);
      }
      else
      {
         p_q1 = p_e1->GetValue(Trans_e2, ip_e2);
         get_shifted_value(*p_e1, Trans_e2.ElementNo, ip_e2, d_q,
                           num_taylor, shift_p_e1);
      }
      double grad_p_d_q1 = shift_p_e1(0) - p_q1;

      // Compute el2 quantities. Allows negative pressure.
      Trans_e2.SetIntPoint(&ip_e2);
      Vector shift_p_e2(1); double p_q2;
      if (use_mixed_elem == false)
      {
         p_q2 = p_e2->GetValue(Trans_e2, ip_e2);
         get_shifted_value(*p_e2, Trans_e2.ElementNo, ip_e2, d_q,
                           num_taylor, shift_p_e2);
      }
      else if (attr_e1 == 15)
      {
         p_q2 = p_e2->GetValue(Trans_e1, ip_e1);
         get_shifted_value(*p_e2, Trans_e1.ElementNo, ip_e1, d_q,
                           num_taylor, shift_p_e2);
      }
      else
      {
         p_q2 = p_e2->GetValue(Trans_e2, ip_e2);
         get_shifted_value(*p_e2, Trans_e2.ElementNo, ip_e2, d_q,
                           num_taylor, shift_p_e2);
      }
      double grad_p_d_q2 = shift_p_e2(0) - p_q2;

      // 1st element.
      {
         // Shape functions in the 1st element.
         el_1.CalcShape(ip_e1, h1_shape);

         for (int j = 0; j < h1dofs_cnt; j++)
         {
            double h1_shape_part = h1_shape(j);
            double p_shift_part = grad_p_d_q1;

            // Scalings: (i) user parameter and (ii) cut volume fraction.
            p_shift_part *= v_shift_scale;
            p_shift_part *= alpha_scale;

            for (int d = 0; d < dim; d++)
            {
               elvect(d*h1dofs_cnt + j)
                  += p_shift_part * h1_shape_part * nor(d);
            }
         }
      }
      // 2nd element if there is such (subtracting from the 1st).
      if (local_face)
      {
         // Shape functions in the 2nd element.
         el_2.CalcShape(ip_e2, h1_shape);

         for (int j = 0; j < h1dofs_cnt; j++)
         {
            double h1_shape_part = h1_shape(j);
            double p_shift_part = grad_p_d_q2;

            // Scalings: (i) user parameter and (ii) cut volume fraction.
            p_shift_part *= v_shift_scale;
            p_shift_part *= alpha_scale;

            for (int d = 0; d < dim; d++)
            {
               elvect(dim*h1dofs_cnt + d*h1dofs_cnt + j)
                  -= p_shift_part * h1_shape_part * nor(d);
            }
         }
      }
   }
}

void EnergyInterfaceIntegrator::AssembleRHSElementVect(
      const FiniteElement &el_L, const FiniteElement &el_R,
      FaceElementTransformations &Trans, Vector &elvect)
{
   MFEM_VERIFY(v != nullptr, "Velocity pointer has not been set!");
   MFEM_VERIFY(e != nullptr, "Energy pointer has not been set!");
   MFEM_VERIFY(e_shift_type == 4, "Implemented only for type 4.");

   const int l2dofs_cnt = el_L.GetDof();
   const int dim = el_L.GetDim();
   const int NE = mat_data.alpha_1.FESpace()->GetNE();
   const bool local_face = (Trans.Elem2No < NE);

   if (local_face == false)
   {
      // This is a shared face between mpi tasks.
      // Each task assembles only its side, but uses info from the neighbor.
      elvect.SetSize(l2dofs_cnt);
   }
   else { elvect.SetSize(l2dofs_cnt * 2); }
   elvect = 0.0;

   // The early return must be done after elvect.SetSize().
   const int attr_face = Trans.Attribute;
   if (attr_face != 10 && attr_face != 20 && attr_face != 15) { return; }

   ElementTransformation &Trans_e_L = Trans.GetElement1Transformation();
   ElementTransformation &Trans_e_R = Trans.GetElement2Transformation();

   const IntegrationRule *ir = IntRule;
   MFEM_VERIFY(ir != NULL, "Set the correct IntRule!");
   const int nqp_face = ir->GetNPoints();

   const int attr_e1 = Trans_e_L.Attribute;
   const int geom    = el_L.GetGeomType();

   const ParGridFunction *p_e1, *p_e2, *rho0DetJ_e1, *rho0DetJ_e2,
                         *ind_e1, *ind_e2;

   const ParGridFunction *p_mat, *alpha_mat;
   if (mat_id == 1)
   {
      p_mat          = &mat_data.p_1->GetPressure();
      alpha_mat      = &mat_data.alpha_1;
   }
   else
   {
      p_mat          = &mat_data.p_2->GetPressure();
      alpha_mat      = &mat_data.alpha_2;
   }

   double gamma_e1, gamma_e2;
   if ( (attr_face == 10 && attr_e1 == 10) ||
        (attr_face == 20 && attr_e1 == 15) || attr_face == 15)
   {
      p_e1        = &mat_data.p_1->GetPressure();
      rho0DetJ_e1 = &mat_data.rho0DetJ_1;
      ind_e1      = &mat_data.ind_1;
      gamma_e1    =  mat_data.gamma_1;
      p_e2        = &mat_data.p_2->GetPressure();
      rho0DetJ_e2 = &mat_data.rho0DetJ_2;
      ind_e2      = &mat_data.ind_2;
      gamma_e2    =  mat_data.gamma_2;
   }
   else if ( (attr_face == 10 && attr_e1 == 15) ||
             (attr_face == 20 && attr_e1 == 20) )
   {
      p_e1        = &mat_data.p_2->GetPressure();
      rho0DetJ_e1 = &mat_data.rho0DetJ_2;
      ind_e1      = &mat_data.ind_2;
      gamma_e1    =  mat_data.gamma_2;
      p_e2        = &mat_data.p_1->GetPressure();
      rho0DetJ_e2 = &mat_data.rho0DetJ_1;
      ind_e2      = &mat_data.ind_1;
      gamma_e1    =  mat_data.gamma_1;
   }
   else { MFEM_ABORT("Invalid marking configuration."); return; }

   // The alpha scaling is always taken from the mixed element.
   // GetValue() is used so that this can work in parallel.
   double alpha_scale = (attr_e1 == 15)
      ? mat_data.alpha_1.GetValue(Trans_e_L, Geometries.GetCenter(geom))
      : mat_data.alpha_1.GetValue(Trans_e_R, Geometries.GetCenter(geom));
   // For 10-faces we use 1-alpha_1, for 20-faces we use 1-alpha_2 = alpha_1.
   if (attr_face == 10) { alpha_scale = 1.0 - alpha_scale; }
   MFEM_VERIFY(alpha_scale > 1e-12 && alpha_scale < 1.0-1e-12,
               "The mixed zone is a 1-material zone! Check it.");

   Vector shape_e(l2dofs_cnt);

   Vector nor(dim), p_grad_q1(dim), p_grad_q2(dim), d_q(dim), v_vals(dim);
   DenseMatrix v_grad_q1(dim), v_grad_q2(dim);

   for (int q = 0; q < nqp_face; q++)
   {
      const IntegrationPoint &ip_f = ir->IntPoint(q);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip_f);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &ip_e_L = Trans.GetElement1IntPoint();
      const IntegrationPoint &ip_e_R = Trans.GetElement2IntPoint();

      // The normal includes the Jac scaling.
      // The orientation is taken into account in the processing of element 1.
      if (dim == 1)
      {
         nor(0) = (2*ip_e_L.x - 1.0 ) * Trans.Weight();
      }
      else { CalcOrtho(Trans.Jacobian(), nor); }

      // The four pressures at the face. Allows negative pressure.
      double p_mat_q_L   = p_mat->GetValue(Trans_e_L, ip_e_L),
             p_mat_q_R   = p_mat->GetValue(Trans_e_R, ip_e_R);

      // Gamma values for averaging.
      const double gamma_mat =
            fabs(p_mat_q_L) / (fabs(p_mat_q_L) + fabs(p_mat_q_R));

      // Compute el1 quantities. Allows negative pressure.
      double p_q1;
      if (use_mixed_elem == false) { p_q1 = p_e1->GetValue(Trans_e_L, ip_e_L); }
      else if (attr_e1 == 15)      { p_q1 = p_e1->GetValue(Trans_e_L, ip_e_L); }
      else                         { p_q1 = p_e1->GetValue(Trans_e_R, ip_e_R); }
      p_e1->GetGradient(Trans_e_L, p_grad_q1);
      v->GetVectorGradient(Trans_e_L, v_grad_q1);

      // Compute el2 quantities. Allows negative pressure.
      double p_q2;
      if (use_mixed_elem == false) { p_q2 = p_e2->GetValue(Trans_e_R, ip_e_R); }
      else if (attr_e1 == 15)      { p_q2 = p_e2->GetValue(Trans_e_L, ip_e_L); }
      else                         { p_q2 = p_e2->GetValue(Trans_e_R, ip_e_R); }
      p_e2->GetGradient(Trans_e_R, p_grad_q2);
      v->GetVectorGradient(Trans_e_R, v_grad_q2);

      const double gamma_avg = fabs(p_q1) / (fabs(p_q1) + fabs(p_q2));

      //
      // The velocity jump.
      //
      dist.Eval(d_q, Trans_e_L, ip_e_L);
      v->GetVectorValue(Trans, ip_f, v_vals);
      Vector true_normal = d_q;
      true_normal /= sqrt(d_q * d_q + 1e-12);

      Vector gradv_d(dim);
      get_shifted_value(*v, Trans_e_L.ElementNo, ip_e_L, d_q,
                        num_taylor, gradv_d);
      gradv_d -= v_vals;
      double gradv_d_n = gradv_d * true_normal;
      Vector gradv_d_n_n_e1(true_normal);
      gradv_d_n_n_e1 *= gradv_d_n;

      get_shifted_value(*v, Trans_e_R.ElementNo, ip_e_R, d_q,
                        num_taylor, gradv_d);
      gradv_d -= v_vals;
      gradv_d_n = gradv_d * true_normal;
      Vector gradv_d_n_n_e2(true_normal);
      gradv_d_n_n_e2 *= gradv_d_n;

      double jump_gradv_d_n_n = gradv_d_n_n_e1 * nor - gradv_d_n_n_e2 * nor;

      //
      // mat 1: - < scale * [((grad_v d).n) n], {alpha1 p1 phi} >
      // mat 2: - < scale * [((grad_v d).n) n], {alpha2 p2 phi} >
      // phi is DG, so {alpha p phi} = g alpha_L p_L phi + (1-g) 0 as phi_R = 0.
      //
      // Left element.
      el_L.CalcShape(ip_e_L, shape_e);
      shape_e *= e_shift_scale * ip_f.weight * jump_gradv_d_n_n *
                 gamma_mat * p_mat_q_L *
                 alpha_mat->GetValue(Trans_e_L, Geometries.GetCenter(geom));
      Vector elvect_e_L(elvect.GetData(), l2dofs_cnt);
      elvect_e_L -= shape_e;
      // Right element.
      if (local_face)
      {
         el_R.CalcShape(ip_e_R, shape_e);
         shape_e *= e_shift_scale * ip_f.weight * jump_gradv_d_n_n *
                    (1.0 - gamma_mat) * p_mat_q_R *
                    alpha_mat->GetValue(Trans_e_R, Geometries.GetCenter(geom));
         Vector elvect_e_R(elvect.GetData() + l2dofs_cnt, l2dofs_cnt);
         // Same sign as above, because it's an average of the shape f-s.
         elvect_e_R -= shape_e;
      }

      if (mat_id == 1 && attr_face != 20) { continue; }
      if (mat_id == 2 && attr_face != 10) { continue; }

      if (problem_visc == false) { continue; }
      if (diffusion == false)    { continue; }

      //
      // mat 1:
      // < scale * {h |grad_v|} [p + grad_p d] [phi + grad_phi d] >
      // mat 2:
      // < scale * {h |grad_v|} [p + grad_p d] [phi + grad_phi d] >
      //
      // Take the material's pressure extension.
      Vector p_ext(1);
      if (use_mixed_elem == false)
      {
         get_shifted_value(*p_e1, Trans_e_L.ElementNo, ip_e_L, d_q,
                           num_taylor, p_ext);
      }
      else if (attr_e1 == 15)
      {
         get_shifted_value(*p_e1, Trans_e_L.ElementNo, ip_e_L, d_q,
                           num_taylor, p_ext);
      }
      else
      {
         get_shifted_value(*p_e1, Trans_e_R.ElementNo, ip_e_R, d_q,
                           num_taylor, p_ext);
      }
      double p_q1_ext = p_ext(0);

      if (use_mixed_elem == false)
      {
         get_shifted_value(*p_e2, Trans_e_R.ElementNo, ip_e_R, d_q,
                           num_taylor, p_ext);
      }
      else if (attr_e1 == 15)
      {
         get_shifted_value(*p_e2, Trans_e_L.ElementNo, ip_e_L, d_q,
                           num_taylor, p_ext);
      }
      else
      {
         get_shifted_value(*p_e2, Trans_e_R.ElementNo, ip_e_R, d_q,
                           num_taylor, p_ext);
      }
      double p_q2_ext = p_ext(0);

      double p_gradp_jump = p_q1_ext - p_q2_ext;

      double h_1, h_2, mu_1, mu_2, visc_q1, visc_q2;

      // Distance with grad_v.
      if (diffusion_type == 0)
      {
         h_1  = d_q.Norml2();
         h_2  = d_q.Norml2();
         mu_1 = v_grad_q1.FNorm();
         mu_2 = v_grad_q2.FNorm();
         visc_q1 = h_1 * fabs(mu_1);
         visc_q2 = h_2 * fabs(mu_2);
      }
      else if (diffusion_type == 1)
      {
         // Mesh size with grad_v.
         h_1  = mat_data.e_1.ParFESpace()->GetMesh()->GetElementSize(&Trans_e_L, 0);
         h_2  = mat_data.e_1.ParFESpace()->GetMesh()->GetElementSize(&Trans_e_R, 0);
         mu_1 = v_grad_q1.FNorm();
         mu_2 = v_grad_q2.FNorm();
         visc_q1 = h_1 * fabs(mu_1);
         visc_q2 = h_2 * fabs(mu_2);
      }
      else
      {
         double rho_q1, rho_q2;
         if (use_mixed_elem == false)
         {
            rho_q1 = rho0DetJ_e1->GetValue(Trans_e_L, ip_e_L)
                     / Trans_e_L.Weight() / ind_e1->GetValue(Trans_e_L, ip_e_L);
            rho_q2 = rho0DetJ_e2->GetValue(Trans_e_R, ip_e_R)
                     / Trans_e_R.Weight() / ind_e2->GetValue(Trans_e_R, ip_e_R);
         }
         else if (attr_e1 == 15)
         {
            rho_q1 = rho0DetJ_e1->GetValue(Trans_e_L, ip_e_L)
                     / Trans_e_L.Weight() / ind_e1->GetValue(Trans_e_L, ip_e_L);
            rho_q2 = rho0DetJ_e2->GetValue(Trans_e_L, ip_e_L)
                     / Trans_e_L.Weight() / ind_e2->GetValue(Trans_e_L, ip_e_L);
         }
         else
         {
            rho_q1 = rho0DetJ_e1->GetValue(Trans_e_R, ip_e_R)
                     / Trans_e_R.Weight() / ind_e1->GetValue(Trans_e_R, ip_e_R);
            rho_q2 = rho0DetJ_e2->GetValue(Trans_e_R, ip_e_R)
                     / Trans_e_R.Weight() / ind_e2->GetValue(Trans_e_R, ip_e_R);
         }
         MFEM_VERIFY(rho_q1 > 0.0 && rho_q2 > 0.0,
                     "Negative density at the face, not good: "
                        << rho_q1 << " " << rho_q2);

         // As in the volumetric viscosity.
         v_grad_q1.Symmetrize();
         v_grad_q2.Symmetrize();
         LengthScaleAndCompression(v_grad_q1, Trans_e_L, quad_data.Jac0inv(0),
                                   quad_data.h0, h_1, mu_1);
         LengthScaleAndCompression(v_grad_q2, Trans_e_R, quad_data.Jac0inv(0),
                                   quad_data.h0, h_2, mu_2);
         visc_q1 = 2.0 * h_1 * fabs(mu_1);
         if (mu_1 < 0.0)
         {
            visc_q1 += 0.5 * sqrt(gamma_e1 * fmax(p_q1, 1e-5) / rho_q1);
         }
         visc_q2 = 2.0 * h_2 * fabs(mu_2);
         if (mu_2 < 0.0)
         {
            visc_q2 += 0.5 * sqrt(gamma_e2 * fmax(p_q2, 1e-5) / rho_q2);
         }
      }

      double grad_v_avg = gamma_avg * visc_q1 + (1.0 - gamma_avg) * visc_q2;

      // Left element.
      shift_shape(*e->ParFESpace(), *p_e1->ParFESpace(),
                  Trans_e_L.ElementNo, ip_e_L, d_q, num_taylor, shape_e);
      shape_e *= Trans.Weight() * ip_f.weight * alpha_scale *
                 diffusion_scale * grad_v_avg * p_gradp_jump;
      Vector elvect_e1(elvect.GetData(), l2dofs_cnt);
      elvect_e1 += shape_e;

      // Right element.
      if (local_face)
      {
         shift_shape(*e->ParFESpace(), *p_e1->ParFESpace(),
                     Trans_e_R.ElementNo, ip_e_R, d_q, num_taylor, shape_e);
         shape_e *= Trans.Weight() * ip_f.weight * alpha_scale *
                    diffusion_scale * grad_v_avg * p_gradp_jump;
         Vector elvect_e2(elvect.GetData() + l2dofs_cnt, l2dofs_cnt);
         // Switches sign, because it's a jump of the shape f-s.
         elvect_e2 -= shape_e;
      }
   }
}

void MomentumCutFaceIntegrator::AssembleRHSElementVect(
      const FiniteElement &el_1, const FiniteElement &el_2,
      FaceElementTransformations &Trans, Vector &elvect)
{
   const int h1dofs_cnt = el_1.GetDof();
   const int dim = el_1.GetDim();
   const int NE = mat_data.alpha_1.FESpace()->GetNE();
   const bool local_face = (Trans.Elem2No < NE);

   if (local_face == false)
   {
      // This is a shared face between mpi tasks.
      // Each task assembles only its side, but uses info from the neighbor.
      elvect.SetSize(h1dofs_cnt * dim);
   }
   else { elvect.SetSize(h1dofs_cnt * dim * 2); }
   elvect = 0.0;

   // The early return must be done after elmat.SetSize().
   const int attr_face = Trans.Attribute;
   if (attr_face != 15) { return; }

   ElementTransformation &Trans_e1 = Trans.GetElement1Transformation();
   ElementTransformation &Trans_e2 = Trans.GetElement2Transformation();

   const ParGridFunction *p1, *p2, *alpha1;
   p1     = &mat_data.p_1->GetPressure();
   p2     = &mat_data.p_2->GetPressure();
   alpha1 = &mat_data.alpha_1;

   Vector h1_shape(h1dofs_cnt);
   Vector nor(dim), d_q(dim);

   MFEM_VERIFY(IntRule != nullptr, "Must have been set in advance");
   const int nqp_face = IntRule->GetNPoints();

   for (int q = 0; q < nqp_face; q++)
   {
      // Set the integration point in the face and the neighboring elements
      const IntegrationPoint &ip_f = IntRule->IntPoint(q);
      Trans.SetAllIntPoints(&ip_f);

      // Access the neighboring elements' integration points
      const IntegrationPoint &ip_e1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &ip_e2 = Trans.GetElement2IntPoint();

      // The normal includes the Jac scaling.
      // The orientation is taken into account in the processing of element 1.
      if (dim == 1)
      {
         nor(0) = (2*ip_e1.x - 1.0) * Trans.Weight();
      }
      else { CalcOrtho(Trans.Jacobian(), nor); }
      nor *= ip_f.weight;

      // Compute el1 quantities. Allows negative pressure.
      // Note that the distance vector is a continuous function.
      Trans_e1.SetIntPoint(&ip_e1);
      dist.Eval(d_q, Trans_e1, ip_e1);
      Vector shift_p1_q1(1), shift_p2_q1(1);
      double p1_q1 = p1->GetValue(Trans_e1, ip_e1),
             p2_q1 = p2->GetValue(Trans_e1, ip_e1);
      get_shifted_value(*p1, Trans_e1.ElementNo, ip_e1, d_q,
                        num_taylor, shift_p1_q1);
      get_shifted_value(*p2, Trans_e1.ElementNo, ip_e1, d_q,
                        num_taylor, shift_p2_q1);
      double grad_p1_d_q1 = shift_p1_q1(0) - p1_q1,
             grad_p2_d_q1 = shift_p2_q1(0) - p2_q1;
      double alpha1_q1    = alpha1->GetValue(Trans_e1, ip_e1);

      // Compute el1 quantities. Allows negative pressure.
      Trans_e2.SetIntPoint(&ip_e2);
      Vector shift_p1_q2(1), shift_p2_q2(1);
      double p1_q2 = p1->GetValue(Trans_e2, ip_e2),
             p2_q2 = p2->GetValue(Trans_e2, ip_e2);
      get_shifted_value(*p1, Trans_e2.ElementNo, ip_e2, d_q,
                        num_taylor, shift_p1_q2);
      get_shifted_value(*p2, Trans_e2.ElementNo, ip_e2, d_q,
                        num_taylor, shift_p2_q2);
      double grad_p1_d_q2 = shift_p1_q2(0) - p1_q2,
             grad_p2_d_q2 = shift_p2_q2(0) - p2_q2;
      double alpha1_q2    = alpha1->GetValue(Trans_e2, ip_e2);

      const double p_shift_jump = (grad_p1_d_q1 + grad_p1_d_q2) -
                                  (grad_p2_d_q1 + grad_p2_d_q2);
      // 1st element.
      {
         // Shape functions in the 1st element.
         el_1.CalcShape(ip_e1, h1_shape);

         for (int j = 0; j < h1dofs_cnt; j++)
         {
            for (int d = 0; d < dim; d++)
            {
               elvect(d*h1dofs_cnt + j) +=
                     v_cut_scale * 0.5 * alpha1_q1 *
                     p_shift_jump * h1_shape(j) * nor(d);
            }
         }
      }
      // 2nd element if there is such (subtracting from the 1st).
      if (local_face)
      {
         // Shape functions in the 2nd element.
         el_2.CalcShape(ip_e2, h1_shape);

         for (int j = 0; j < h1dofs_cnt; j++)
         {
            for (int d = 0; d < dim; d++)
            {
               elvect(dim*h1dofs_cnt + d*h1dofs_cnt + j) -=
                     v_cut_scale * 0.5 * alpha1_q2 *
                     p_shift_jump * h1_shape(j) * nor(d);
            }
         }
      }
   }
}

void PointExtractor::SetPoint(int el_id, const Vector &xyz,
                              const ParGridFunction *gf,
                              const IntegrationRule &ir, std::string filename)
{
   g = gf;
   element_id = el_id;

   // Find the integration point and copy it.
   int q_id = FindIntegrPoint(el_id, xyz, ir);
   ip = ir.IntPoint(q_id);
   MFEM_VERIFY(q_id > -1,
               "Integration point not found " << filename);

   ParFiniteElementSpace &pfes = *g->ParFESpace();
   MFEM_VERIFY(pfes.GetNRanks() == 1, "PointExtractor works only in serial.");
   MFEM_VERIFY(pfes.GetMesh()->Dimension() == 1, "Only implemented in 1D.");

   fstream.open(filename);
   fstream.precision(8);

   cout << filename << ": Element " << el_id << "; Quad: " << q_id << endl;
}

void PointExtractor::SetPoint(int el_id, int quad_id, const ParGridFunction *gf,
                              const IntegrationRule &ir, std::string filename)
{
   g = gf;
   element_id = el_id;
   ip = ir.IntPoint(quad_id);

   ParFiniteElementSpace &pfes = *g->ParFESpace();
   MFEM_VERIFY(pfes.GetNRanks() == 1, "PointExtractor works only in serial.");
   MFEM_VERIFY(pfes.GetMesh()->Dimension() == 1, "Only implemented in 1D.");

   fstream.open(filename);
   fstream.precision(8);

   cout << filename << ": Element " << el_id << "; Quad: " << quad_id << endl;
}

int PointExtractor::FindIntegrPoint(const int z_id, const Vector &xyz,
                                    const IntegrationRule &ir)
{
   const int nqp = ir.GetNPoints();
   ElementTransformation &tr =
         *g->ParFESpace()->GetMesh()->GetElementTransformation(z_id);
   Vector position;
   const double eps = 1e-8;
   for (int q = 0; q < nqp; q++)
   {
      const IntegrationPoint &ip_q = ir.IntPoint(q);
      tr.SetIntPoint(&ip_q);
      tr.Transform(ip_q, position);
      // Assumes 1D.
      if (fabs(position(0) - xyz(0)) < eps) { return q; }
   }
   return -1;
}

double RhoPointExtractor::GetValue() const
{
   ParFiniteElementSpace &pfes = *g->ParFESpace();
   ElementTransformation &Tr = *pfes.GetElementTransformation(element_id);
   Tr.SetIntPoint(&ip);

   return (*rho0DetJ0w)(nqp * element_id + q_id) /
          g->GetValue(Tr, ip) / Tr.Weight() / ip.weight;
}

double PPointExtractor::GetValue() const
{
   const double rho = RhoPointExtractor::GetValue(),
                  e = e_gf->GetValue(element_id, ip);

   return (fabs(gamma - 4.4) > 1e-8) ? (gamma - 1.0) * rho * e
                                     : (gamma - 1.0) * rho * e - gamma * 6.0e8;
}

double ShiftedPointExtractor::GetValue() const
{
   ParFiniteElementSpace &pfes = *g->ParFESpace();
   Vector grad_g(1);

   ElementTransformation &tr = *pfes.GetElementTransformation(element_id);
   tr.SetIntPoint(&ip);
   g->GetGradient(*pfes.GetElementTransformation(element_id), grad_g);

   // Assumes 1D.
   return g->GetValue(element_id, ip) +
          dist->GetValue(element_id, ip) * grad_g(0);
}

// Initially the energies are initialized as the single material version, due
// the special Bernstein projection. This only zeroes them in the empty zones.
void InitTG2Mat(MaterialData &mat_data)
{
   ParFiniteElementSpace &pfes = *mat_data.e_1.ParFESpace();
   const int NE    = pfes.GetNE();
   const int ndofs = mat_data.e_1.Size() / NE;

   mat_data.rho0_1  = 0.0;
   mat_data.rho0_2  = 0.0;
   for (int e = 0; e < NE; e++)
   {
      const int attr = pfes.GetParMesh()->GetAttribute(e);

      if (attr == 10)
      {
         for (int i = 0; i < ndofs; i++) { mat_data.e_2(e*ndofs + i) = 0.0; }
      }
      if (attr == 20)
      {
         for (int i = 0; i < ndofs; i++) { mat_data.e_1(e*ndofs + i) = 0.0; }
      }

      if (attr == 10 || attr == 15)
      {
         mat_data.gamma_1 = 5.0 / 3.0;
         for (int i = 0; i < ndofs; i++)
         {
            mat_data.rho0_1(e*ndofs + i) = 1.0;
         }
      }

      if (attr == 15 || attr == 20)
      {
         mat_data.gamma_2 = 5.0 / 3.0;
         for (int i = 0; i < ndofs; i++)
         {
            mat_data.rho0_2(e*ndofs + i) = 1.0;
         }
      }
   }
}

void InitSod2Mat(MaterialData &mat_data)
{
   ParFiniteElementSpace &pfes = *mat_data.e_1.ParFESpace();
   const int NE    = pfes.GetNE();
   const int ndofs = mat_data.e_1.Size() / NE;
   double r, g, p;

   mat_data.rho0_1  = 0.0;
   mat_data.e_1     = 0.0;
   mat_data.rho0_2  = 0.0;
   mat_data.e_2     = 0.0;
   for (int e = 0; e < NE; e++)
   {
      const int attr = pfes.GetParMesh()->GetAttribute(e);

      if (attr == 10 || attr == 15)
      {
         // Left material (high pressure).
         r = 1.0, g = 2.0, p = 2.0;
         mat_data.gamma_1 = g;
         for (int i = 0; i < ndofs; i++)
         {
            mat_data.rho0_1(e*ndofs + i) = r;
            mat_data.e_1(e*ndofs + i)    = p / r / (g - 1.0);
         }
      }

      if (attr == 15 || attr == 20)
      {
         // Right material (low pressure).
         r = 0.125; g = 1.4; p = 0.1;
         mat_data.gamma_2 = g;
         for (int i = 0; i < ndofs; i++)
         {
            mat_data.rho0_2(e*ndofs + i) = r;
            mat_data.e_2(e*ndofs + i)    = p / r / (g - 1.0);
         }
      }
   }
}

void InitWaterAir(MaterialData &mat_data)
{
   ParFiniteElementSpace &pfes = *mat_data.e_1.ParFESpace();
   const int NE    = pfes.GetNE();
   const int ndofs = mat_data.e_1.Size() / NE;
   double r, g, p;

   mat_data.rho0_1  = 0.0;
   mat_data.e_1     = 0.0;
   mat_data.rho0_2  = 0.0;
   mat_data.e_2     = 0.0;
   for (int e = 0; e < NE; e++)
   {
      const int attr = pfes.GetParMesh()->GetAttribute(e);

      if (attr == 10 || attr == 15)
      {
         // Left material - water (high pressure).
         r = 1000; g = 4.4; p = 1.e9;
         mat_data.gamma_1 = g;
         for (int i = 0; i < ndofs; i++)
         {
            mat_data.rho0_1(e*ndofs + i) = r;
            mat_data.e_1(e*ndofs + i)    = (p + g * 6.0e8) / r / (g - 1.0);
         }
      }

      if (attr == 15 || attr == 20)
      {
         // Right material - air (low pressure).
         r = 50; g = 1.4; p = 1.e5;
         mat_data.gamma_2 = g;
         for (int i = 0; i < ndofs; i++)
         {
            mat_data.rho0_2(e*ndofs + i) = r;
            mat_data.e_2(e*ndofs + i)    = p / r / (g - 1.0);
         }
      }
   }
}

void InitTriPoint2Mat(MaterialData &mat_data)
{
   ParFiniteElementSpace &pfes = *mat_data.e_1.ParFESpace();
   const int NE    = pfes.GetNE();
   const int ndofs = mat_data.e_1.Size() / NE;
   const int dim  = pfes.GetMesh()->Dimension();
   double r, p;

   mat_data.gamma_1 = 1.5;
   mat_data.gamma_2 = 1.4;
   mat_data.rho0_1  = 0.0;
   mat_data.e_1     = 0.0;
   mat_data.rho0_2  = 0.0;
   mat_data.e_2     = 0.0;
   for (int e = 0; e < NE; e++)
   {
      const int attr = pfes.GetParMesh()->GetAttribute(e);
      Vector center(dim);
      pfes.GetParMesh()->GetElementCenter(e, center);
      const double x = center(0),
                   z = (dim == 3) ? center(2) : 0.0;

      if (attr == 10 || attr == 15)
      {
         // Left/Top material.
         r = 1.0; p = 1.0;
         if (x > 1.0)
         {
            p = 0.1;
            if (z < 1.5) { r = 0.125; }
         }
         for (int i = 0; i < ndofs; i++)
         {
            mat_data.rho0_1(e*ndofs + i) = r;
            mat_data.e_1(e*ndofs + i)    = p / r / (mat_data.gamma_1 - 1.0);
         }
      }

      if (attr == 15 || attr == 20)
      {
         // Right/Bottom material.
         r = 1.0; p = 0.1;
         if (z > 1.5) { r = 0.125; }
         for (int i = 0; i < ndofs; i++)
         {
            mat_data.rho0_2(e*ndofs + i) = r;
            mat_data.e_2(e*ndofs + i)    = p / r / (mat_data.gamma_2 - 1.0);
         }
      }
   }
}

void InitImpact(MaterialData &mat_data)
{
   ParFiniteElementSpace &pfes = *mat_data.e_1.ParFESpace();
   const int NE    = pfes.GetNE();
   const int ndofs = mat_data.e_1.Size() / NE;
   double r, p;
   Array<int> vdofs;

   mat_data.gamma_1 = 10.0;
   mat_data.gamma_2 = 1.4;
   mat_data.rho0_1  = 0.0;
   mat_data.e_1     = 0.0;
   mat_data.rho0_2  = 0.0;
   mat_data.e_2     = 0.0;
   for (int e = 0; e < NE; e++)
   {
      const int attr = pfes.GetParMesh()->GetAttribute(e);
      Vector center(2);
      pfes.GetParMesh()->GetElementCenter(e, center);
      const double x = center(0), y = center(1);

      if (attr == 10 || attr == 15)
      {
         // Impactor and Wall.
         r = 10.0; p = 1.0;
         if (x <= 0.5) { r = 20.0; }
         for (int i = 0; i < ndofs; i++)
         {
            mat_data.rho0_1(e*ndofs + i) = r;
            mat_data.e_1(e*ndofs + i)    = p / r / (mat_data.gamma_1 - 1.0);
         }
      }

      if (attr == 15 || attr == 20)
      {
         // Background.
         r = 1.0; p = 1.0;
         for (int i = 0; i < ndofs; i++)
         {
            mat_data.rho0_2(e*ndofs + i) = r;
            mat_data.e_2(e*ndofs + i)    = p / r / (mat_data.gamma_2 - 1.0);
         }
      }
   }
}

} // namespace hydrodynamics

} // namespace mfem
