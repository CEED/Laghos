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
         return (pure_test) ? tanh(x(0) - 0.5)
                            : tanh(x(0) - (0.5 + 0.5*dx));
      }
      case 9:
      {
         // Water-air - the 1D domain length is 1.
         const double dx = 1.0 / glob_NE;
         return (pure_test) ? tanh(x(0) - 0.7)
                            : tanh(x(0) - (0.7 + 0.5*dx));
      }
      case 10:
      {
         // The domain area for the 3point is 21.
         const double dx = sqrt(21.0 / glob_NE);

         // The middle of the element after x = 1.
         return (pure_test) ? tanh(x(0) - 1.0)
                            : tanh(x(0) - (1.0 + 0.5*dx));
      }
      case 11:
      {
         // The domain area for the 3point is 21.
         const double dx = sqrt(21.0 / glob_NE);

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

void FaceForceIntegrator::AssembleFaceMatrix(const FiniteElement &trial_fe,
                                             const FiniteElement &test_fe,
                                             FaceElementTransformations &Trans,
                                             DenseMatrix &elmat)
{
   MFEM_VERIFY(mat_data.e_1.ParFESpace()->GetNRanks() == 1,
               "Implemented only in serial.");
   MFEM_VERIFY(v_shift_type == 1, "Implemented only for type 1.");

   if (diffuse_v == true)
   {
      MFEM_VERIFY(v != nullptr, "Velocity pointer has not been set!");
   }

   const int h1dofs_cnt = trial_fe.GetDof();
   const int l2dofs_cnt = test_fe.GetDof();
   const int dim = test_fe.GetDim();

   if (Trans.Elem2No < 0)
   {
      // This case should take care of shared (MPI) faces. They will get
      // processed by both MPI tasks.
      elmat.SetSize(l2dofs_cnt, h1dofs_cnt * dim);
   }
   else { elmat.SetSize(l2dofs_cnt * 2, h1dofs_cnt * dim * 2); }
   elmat = 0.0;

   // The early return must be done after elmat.SetSize().
   const int attr_face = Trans.Attribute;
   if (attr_face != 10 && attr_face != 20) { return; }

   ElementTransformation &Trans_e1 = Trans.GetElement1Transformation();
   ElementTransformation &Trans_e2 = Trans.GetElement2Transformation();
   const int attr_e1 = Trans_e1.Attribute;
   const ParGridFunction *p_e1, *p_e2;
   double gamma_e1, gamma_e2;
   if ( (attr_face == 10 && attr_e1 == 10) ||
        (attr_face == 20 && attr_e1 == 15) )
   {
      p_e1     = &mat_data.p_1->GetPressure();
      gamma_e1 = mat_data.gamma_1;
      p_e2     = &mat_data.p_2->GetPressure();
      gamma_e2 = mat_data.gamma_2;
   }
   else if ( (attr_face == 10 && attr_e1 == 15) ||
             (attr_face == 20 && attr_e1 == 20) )
   {
      p_e1     = &mat_data.p_2->GetPressure();
      gamma_e1 = mat_data.gamma_2;
      p_e2     = &mat_data.p_1->GetPressure();
      gamma_e2 = mat_data.gamma_1;
   }
   else { MFEM_ABORT("Invalid marking configuration."); }

   // The alpha scaling is always taken from the mixed element.
   double alpha_scale = (attr_e1 == 15) ? mat_data.alpha_1(Trans_e1.ElementNo)
                                        : mat_data.alpha_1(Trans_e2.ElementNo);
   // For 10-faces we use 1-alpha_1, for 20-faces we use 1-alpha_2 = alpha_1.
   if (attr_face == 10) { alpha_scale = 1.0 - alpha_scale; }
   if (alpha_scale < 1e-13 || (alpha_scale > 1.0 - 1e-13))
   {
      cout << "\n--- Bad alpha value: " << alpha_scale << endl;
      MFEM_ABORT("bad alpha value");
   }

   h1_shape.SetSize(h1dofs_cnt);
   l2_shape.SetSize(l2dofs_cnt);

   MFEM_VERIFY(IntRule != nullptr, "Must have been set in advance");
   const int nqp_face = IntRule->GetNPoints();

   Vector nor(dim);

   // The distance vector is a continuous function.
   Vector d_q(dim);
   Vector p_grad_q1(dim), p_grad_q2(dim);
   DenseMatrix h1_grads(h1dofs_cnt, dim), v_grad_q1(dim), v_grad_q2(dim);
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

      // Compute el1 quantities.
      Trans_e1.SetIntPoint(&ip_e1);
      p_e1->GetGradient(Trans_e1, p_grad_q1);
      dist.Eval(d_q, Trans_e1, ip_e1);
      const double grad_p_d_q1 = d_q * p_grad_q1;
      const double p_q1 = fmax(1e-5, p_e1->GetValue(Trans_e1, ip_e1));
      if (diffuse_v) { v->GetVectorGradient(Trans_e1, v_grad_q1); }

      // Compute el2 quantities.
      Trans_e2.SetIntPoint(&ip_e2);
      p_e2->GetGradient(Trans_e2, p_grad_q2);
      const double grad_p_d2 = d_q * p_grad_q2;
      const double p_q2 = fmax(1e-5, p_e2->GetValue(Trans_e2, ip_e2));
      if (diffuse_v) { v->GetVectorGradient(Trans_e2, v_grad_q2); }

      MFEM_VERIFY(p_q1 > 0.0 && p_q2 > 0.0, "negative pressure");

      // generic stuff that we need for forms 2, 3 and 4
      Vector gradv_d(dim);
      v_grad_q1.Mult(d_q, gradv_d);
      Vector true_normal = d_q;
      true_normal /= sqrt(d_q * d_q + 1e-12);
      double gradv_d_n = gradv_d * true_normal;
      Vector gradv_d_n_n_e1(true_normal);
      gradv_d_n_n_e1 *= gradv_d_n;

      v_grad_q2.Mult(d_q, gradv_d);
      gradv_d_n = gradv_d * true_normal;
      Vector gradv_d_n_n_e2(true_normal);
      gradv_d_n_n_e2 *= gradv_d_n;

      const double true_normal_nor = true_normal * nor;
      const double gradv_d_n_n_jump = gradv_d_n_n_e1 * nor - gradv_d_n_n_e2 * nor;

      // 1st element.
      {
         // Shape functions in the 1st element.
         trial_fe.CalcShape(ip_e1, h1_shape);
         test_fe.CalcShape(ip_e1, l2_shape);

         // Compute grad_psi in the first element.
         trial_fe.CalcPhysDShape(Trans_e1, h1_grads);

         // TODO reorder/optimize loops.
         for (int i = 0; i < l2dofs_cnt; i++)
         {
            for (int j = 0; j < h1dofs_cnt; j++)
            {
               double h1_shape_part = h1_shape(j);
               if (v_shift_type == 2 || v_shift_type == 3 ||
                   v_shift_type == 4 || v_shift_type == 5)
               {
                  Vector grad_shape_h1;
                  h1_grads.GetRow(j, grad_shape_h1);
                  h1_shape_part = d_q * grad_shape_h1;
               }

               double p_shift_part = grad_p_d_q1;
               if (v_shift_type == 3)
               {
                   p_shift_part = p_q1 + grad_p_d_q1;
               }
               else if (v_shift_type == 4)
               {
                   p_shift_part = p_q1 + grad_p_d_q1 - p_q2 - grad_p_d2;
                   h1_shape_part += h1_shape(j);
               }
               else if (v_shift_type == 5)
               {
                   p_shift_part = grad_p_d_q1 - grad_p_d2;
                   h1_shape_part += h1_shape(j);
               }

               // Scalings: (i) user parameter and (ii) cut volume fraction.
               p_shift_part *= v_shift_scale;
               p_shift_part *= alpha_scale;

               for (int d = 0; d < dim; d++)
               {
                  elmat(i, d*h1dofs_cnt + j)
                        += p_shift_part * h1_shape_part * l2_shape(i) * nor(d);

                  double diffuse_term = 0.0;
                  if (diffuse_v)
                  {
                     MFEM_ABORT("Fix the rho computation to get data "
                                "from mat_data.rhoDetJind0.");
                     double rho_cs_avg = -1.0; // TODO.

                     Vector grad_shape_h1;
                     h1_grads.GetRow(j, grad_shape_h1);
                     double grad_psi_d = (grad_shape_h1 * d_q) * true_normal(d);
                     diffuse_term = diffuse_v_scale *
                                    rho_cs_avg * grad_psi_d * gradv_d_n_n_jump *
                                    true_normal_nor / Trans.Weight() / ip_f.weight;
                     elmat(i, d*h1dofs_cnt + j)
                           += diffuse_term * l2_shape(i);
                  }
               }
            }
         }
      }
      // 2nd element if there is such (subtracting from the 1st).
      if (Trans.Elem2No >= 0)
      {
         // L2 shape functions on the 2nd element.
         trial_fe.CalcShape(ip_e2, h1_shape);
         test_fe.CalcShape(ip_e2, l2_shape);

         // Compute grad_psi in the second element.
         trial_fe.CalcPhysDShape(Trans_e2, h1_grads);

         // TODO reorder/optimize loops.
         for (int i = 0; i < l2dofs_cnt; i++)
         {
            for (int j = 0; j < h1dofs_cnt; j++)
            {
               double h1_shape_part = h1_shape(j);
               if (v_shift_type == 2 || v_shift_type == 3 ||
                   v_shift_type == 4 || v_shift_type == 5)
               {
                  Vector grad_shape_h1;
                  h1_grads.GetRow(j, grad_shape_h1);
                  h1_shape_part = d_q * grad_shape_h1;
               }

               double p_shift_part = grad_p_d2;
               if (v_shift_type == 3)
               {
                   p_shift_part = p_q2 + grad_p_d2;
               }
               else if (v_shift_type == 4)
               {
                   p_shift_part = p_q1 + grad_p_d_q1 - p_q2 - grad_p_d2;
                   h1_shape_part += h1_shape(j);
               }
               else if (v_shift_type == 5)
               {
                   p_shift_part = grad_p_d_q1 - grad_p_d2;
                   h1_shape_part += h1_shape(j);
               }

               // Scalings: (i) user parameter and (ii) cut volume fraction.
               p_shift_part *= v_shift_scale;
               p_shift_part *= alpha_scale;

               for (int d = 0; d < dim; d++)
               {
                  elmat(l2dofs_cnt + i, dim*h1dofs_cnt + d*h1dofs_cnt + j)
                        -= p_shift_part * h1_shape_part * l2_shape(i) * nor(d);

                  double diffuse_term = 0.0;
                  if (diffuse_v)
                  {
                     MFEM_ABORT("Fix the rho computation to get data "
                                "from mat_data.rhoDetJind0.");
                     double rho_cs_avg = -1.0; // TODO.

                     Vector grad_shape_h1;
                     h1_grads.GetRow(j, grad_shape_h1);
                     double grad_psi_d = (grad_shape_h1 * d_q) * true_normal(d);
                     diffuse_term = diffuse_v_scale *
                                    rho_cs_avg * grad_psi_d * gradv_d_n_n_jump *
                                    true_normal_nor / Trans.Weight() / ip_f.weight;
                     elmat(l2dofs_cnt + i, dim*h1dofs_cnt + d*h1dofs_cnt + j)
                           -= diffuse_term * l2_shape(i);
                  }
               }
            }
         }
      }
   }
}

void FaceForceIntegrator::AssembleFaceMatrix(const FiniteElement &trial_face_fe,
                                             const FiniteElement &test_fe1,
                                             const FiniteElement &test_fe2,
                                             FaceElementTransformations &Trans,
                                             DenseMatrix &elmat)
{
   MFEM_ABORT("not used anymore");

   const int h1dofs_cnt_face = trial_face_fe.GetDof();
   const int l2dofs_cnt = test_fe1.GetDof();
   const int dim = test_fe1.GetDim();

   if (Trans.Elem2No < 0)
   {
      // This case should take care of shared (MPI) faces. They will get
      // processed by both MPI tasks.
      elmat.SetSize(l2dofs_cnt, h1dofs_cnt_face * dim);
   }
   else { elmat.SetSize(l2dofs_cnt * 2, h1dofs_cnt_face * dim); }
   elmat = 0.0;

   // Must be done after elmat.SetSize().
   if (Trans.Attribute != 77) { return; }

   h1_shape_face.SetSize(h1dofs_cnt_face);
   l2_shape.SetSize(l2dofs_cnt);

   const int ir_order =
      test_fe1.GetOrder() + trial_face_fe.GetOrder() + Trans.OrderW();
   const IntegrationRule *ir = &IntRules.Get(Trans.GetGeometryType(), ir_order);
   const int nqp_face = ir->GetNPoints();

   // grad_p at all DOFs of the pressure FE space, on both sides.
   const FiniteElement &el_p =
         *mat_data.p_1->GetPressure().ParFESpace()->GetFE(0);
   const int dof_p = el_p.GetDof();
   DenseMatrix p_grad_e_1(dof_p, dim), p_grad_e_2(dof_p, dim);
   GradAtLocalDofs(Trans.GetElement1Transformation(),
                   mat_data.p_1->GetPressure(), p_grad_e_1);
   if (Trans.Elem2No > 0)
   {
      GradAtLocalDofs(Trans.GetElement2Transformation(),
                      mat_data.p_1->GetPressure(), p_grad_e_2);
   }

   Vector nor(dim);

   Vector p_grad_q(dim), d_q(dim), shape_p(dof_p);
   //DenseMatrix h1_grads(h1_vol_fe->GetDof(), dim), grad_psi(dim);
   for (int q = 0; q  < nqp_face; q++)
   {
      const IntegrationPoint &ip_f = ir->IntPoint(q);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip_f);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &ip_e1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &ip_e2 = Trans.GetElement2IntPoint();

      // The normal includes the Jac scaling.
      // The orientation is taken into account in the processing of element 1.
      if (dim == 1)
      {
         nor(0) = (2*ip_e1.x - 1.0 ) * Trans.Weight();
      }
      else { CalcOrtho(Trans.Jacobian(), nor); }
      nor *= ip_f.weight;

      // Shape functions on the face (H1); same for both elements.
      trial_face_fe.CalcShape(ip_f, h1_shape_face);

      // + <[[grad p . d]], psi>
      // + <gradp1*d1*nor, psi> - <gradp2*d2*nor, psi>
      // We keep the positive sign here because in SolveVelocity(), this term
      // will be multipled with -1 and added to the RHS.

      // 1st element.
      {
         // Compute dist * grad_p in the first element.
         el_p.CalcShape(ip_e1, shape_p);
         p_grad_e_1.MultTranspose(shape_p, p_grad_q);
         dist.Eval(d_q, Trans.GetElement1Transformation(), ip_e1);
         const double grad_p_d = d_q * p_grad_q;

         // Compute dist * grad_psi in the first element
         //h1_vol_fe->CalcPhysDShape(Trans.GetElement1Transformation(), h1_grads);

         // L2 shape functions in the 1st element.
         test_fe1.CalcShape(ip_e1, l2_shape);

         for (int i = 0; i < l2dofs_cnt; i++)
         {
            for (int j = 0; j < h1dofs_cnt_face; j++)
            {
               for (int d = 0; d < dim; d++)
               {
                  elmat(i, d*h1dofs_cnt_face + j)
                         += grad_p_d * l2_shape(i) * h1_shape_face(j) * nor(d);

                  /*
                  const int vol_j = LocVolumeDofID(Trans.ElementNo, j,
                                                   Trans.Elem1No, pfes_h1);
                  h1_grads.GetRow(vol_j, grad_psi);
                  const double grad_p_psi = d_q * grad_psi;
                  elmat(i, d*h1dofs_cnt_face + j)
                         += grad_p_d * grad_p_psi * l2_shape(i)  * nod(d);
                  */
               }
            }
         }
      }
      // 2nd element if there is such (subtracting from the 1st).
      if (Trans.Elem2No >= 0)
      {
         // Compute dist * grad_p in the second element.
         el_p.CalcShape(ip_e2, shape_p);
         p_grad_e_2.MultTranspose(shape_p, p_grad_q);
         dist.Eval(d_q, Trans.GetElement2Transformation(), ip_e2);
         const double grad_p_d = d_q * p_grad_q;

         // L2 shape functions on the 2nd element.
         test_fe2.CalcShape(ip_e2, l2_shape);

         for (int i = 0; i < l2dofs_cnt; i++)
         {
            for (int j = 0; j < h1dofs_cnt_face; j++)
            {
               for (int d = 0; d < dim; d++)
               {
                  elmat(l2dofs_cnt + i, d*h1dofs_cnt_face + j)
                          -= grad_p_d * l2_shape(i) * h1_shape_face(j) * nor(d);
               }
            }
         }
      }
   }

   //delete h1_vol_fe;
}

void MomentumInterfaceIntegrator::AssembleRHSElementVect(
      const FiniteElement &el_1, const FiniteElement &el_2,
      FaceElementTransformations &Trans, Vector &elvect)
{
   MFEM_VERIFY(v_shift_type == 1, "Implemented only for type 1.");

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
   double gamma_e1, gamma_e2;
   if ( (attr_face == 10 && attr_e1 == 10) ||
        (attr_face == 20 && attr_e1 == 15) )
   {
      p_e1     = &mat_data.p_1->GetPressure();
      gamma_e1 = mat_data.gamma_1;
      p_e2     = &mat_data.p_2->GetPressure();
      gamma_e2 = mat_data.gamma_2;
   }
   else if ( (attr_face == 10 && attr_e1 == 15) ||
             (attr_face == 20 && attr_e1 == 20) )
   {
      p_e1     = &mat_data.p_2->GetPressure();
      gamma_e1 = mat_data.gamma_2;
      p_e2     = &mat_data.p_1->GetPressure();
      gamma_e2 = mat_data.gamma_1;
   }
   else { MFEM_ABORT("Invalid marking configuration."); }

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
      const double p_q1 = p_e1->GetValue(Trans_e1, ip_e1);
      Vector shift_p(1);
      get_shifted_value(*p_e1, Trans_e1.ElementNo, ip_e1, d_q,
                        num_taylor, shift_p);
      double grad_p_d_q1 = shift_p(0) - p_q1;

      // Compute el2 quantities. Allows negative pressure.
      Trans_e2.SetIntPoint(&ip_e2);
      const double p_q2 = p_e2->GetValue(Trans_e2, ip_e2);
      get_shifted_value(*p_e2, Trans_e2.ElementNo, ip_e2, d_q,
                        num_taylor, shift_p);
      double grad_p_d_q2 = shift_p(0) - p_q2;

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
         // L2 shape functions on the 2nd element.
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
      const FiniteElement &el_1, const FiniteElement &el_2,
      FaceElementTransformations &Trans, Vector &elvect)
{
   MFEM_VERIFY(v != nullptr, "Velocity pointer has not been set!");
   MFEM_VERIFY(e != nullptr, "Energy pointer has not been set!");
   MFEM_VERIFY(e_shift_type != 3 && e_shift_type != 2, "Not implemented");
   MFEM_VERIFY(e_shift_type == 4, "Implemented only for type 4.");

   const int l2dofs_cnt = el_1.GetDof();
   const int dim = el_1.GetDim();
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
   // Material 1 uses 20-faces, material 2 uses 10-faces.
   const int attr_face = Trans.Attribute;
   if (mat_id == 1 && attr_face != 20) { return; }
   if (mat_id == 2 && attr_face != 10) { return; }

   ElementTransformation &Trans_e1 = Trans.GetElement1Transformation();
   ElementTransformation &Trans_e2 = Trans.GetElement2Transformation();

   const IntegrationRule *ir = IntRule;
   MFEM_VERIFY(ir != NULL, "Set the correct IntRule!");
   const int nqp_face = ir->GetNPoints();

   const int attr_e1 = Trans_e1.Attribute;
   const ParGridFunction *p_e1, *p_e2, *rhoDetJind0_e1, *rhoDetJind0_e2,
                         *ind_e1, *ind_e2;
   double gamma_e1, gamma_e2;
   if ( (attr_face == 10 && attr_e1 == 10) ||
        (attr_face == 20 && attr_e1 == 15) )
   {
      p_e1           = &mat_data.p_1->GetPressure();
      rhoDetJind0_e1 = &mat_data.rhoDetJind0_1;
      ind_e1         = &mat_data.vol_1;
      gamma_e1       =  mat_data.gamma_1;
      p_e2           = &mat_data.p_2->GetPressure();
      rhoDetJind0_e2 = &mat_data.rhoDetJind0_2;
      ind_e2         = &mat_data.vol_2;
      gamma_e2       =  mat_data.gamma_2;
   }
   else if ( (attr_face == 10 && attr_e1 == 15) ||
             (attr_face == 20 && attr_e1 == 20) )
   {
      p_e1           = &mat_data.p_2->GetPressure();
      rhoDetJind0_e1 = &mat_data.rhoDetJind0_2;
      ind_e1         = &mat_data.vol_2;
      gamma_e1       =  mat_data.gamma_2;
      p_e2           = &mat_data.p_1->GetPressure();
      rhoDetJind0_e2 = &mat_data.rhoDetJind0_1;
      ind_e2         = &mat_data.vol_1;
      gamma_e2       =  mat_data.gamma_1;
   }
   else { MFEM_ABORT("Invalid marking configuration."); }

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
      const IntegrationPoint &ip_e1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &ip_e2 = Trans.GetElement2IntPoint();

      // The normal includes the Jac scaling.
      // The orientation is taken into account in the processing of element 1.
      if (dim == 1)
      {
         nor(0) = (2*ip_e1.x - 1.0 ) * Trans.Weight();
      }
      else { CalcOrtho(Trans.Jacobian(), nor); }

      // Compute el1 quantities. Allows negative pressure.
      const double p_q1 = p_e1->GetValue(Trans_e1, ip_e1);
      dist.Eval(d_q, Trans_e1, ip_e1);
      p_e1->GetGradient(Trans_e1, p_grad_q1);
      v->GetVectorGradient(Trans_e1, v_grad_q1);

      // Compute el2 quantities. Allows negative pressure.
      const double p_q2 = p_e2->GetValue(Trans_e2, ip_e2);
      p_e2->GetGradient(Trans_e2, p_grad_q2);
      v->GetVectorGradient(Trans_e2, v_grad_q2);

      const double gamma_avg = fabs(p_q1) / (fabs(p_q1) + fabs(p_q2));

      v->GetVectorValue(Trans, ip_f, v_vals);

      // For each term, we keep the sign as if it is on the left hand side
      // because in SolveEnergy, this will be multiplied with -1 and added.
      // + <v, phi[[\grad p . d]]>
      if (e_shift_type == 1)
      {
          double gradp_d_jump_term = (d_q * p_grad_q1 - d_q * p_grad_q2) *
                  (ip_f.weight) * (nor * v_vals);

          // 1st element.
          {
             // L2 shape functions in the 1st element.
             el_1.CalcShape(ip_e1, shape_e);
             for (int i = 0; i < l2dofs_cnt; i++)
             {
                 elvect(i) += shape_e(i) * gradp_d_jump_term;
             }
          }
          // 2nd element.
          {
             // L2 shape functions in the 2nd element.
             el_2.CalcShape(ip_e2, shape_e);
             for (int i = 0; i < l2dofs_cnt; i++)
             {
                 elvect(i + l2dofs_cnt) += shape_e(i) * gradp_d_jump_term;
             }
          }
      }

      // generic stuff.
      Vector true_normal = d_q;
      true_normal /= sqrt(d_q * d_q + 1e-12);

      Vector gradv_d(dim);
      get_shifted_value(*v, Trans_e1.ElementNo, ip_e1, d_q,
                        num_taylor, gradv_d);
      gradv_d -= v_vals;
      double gradv_d_n = gradv_d * true_normal;
      Vector gradv_d_n_n_e1(true_normal);
      gradv_d_n_n_e1 *= gradv_d_n;

      get_shifted_value(*v, Trans_e2.ElementNo, ip_e2, d_q,
                        num_taylor, gradv_d);
      gradv_d -= v_vals;
      gradv_d_n = gradv_d * true_normal;
      Vector gradv_d_n_n_e2(true_normal);
      gradv_d_n_n_e2 *= gradv_d_n;

      double jump_gradv_d_n_n = gradv_d_n_n_e1 * nor - gradv_d_n_n_e2 * nor;

      // - < [((grad_v d).n) n], {p phi} > (form 4)
      // phi is DG, so {p phi} = p1 phi + p2 0 = p1 phi.
      if (e_shift_type == 4)
      {
         // 1st element.
         el_1.CalcShape(ip_e1, shape_e);
         shape_e *= ip_f.weight * alpha_scale * e_shift_scale *
                    jump_gradv_d_n_n * gamma_avg * p_q1;
         Vector elvect_e1(elvect.GetData(), l2dofs_cnt);
         elvect_e1 -= shape_e;

         // 2nd element.
         if (local_face)
         {
            el_2.CalcShape(ip_e2, shape_e);
            shape_e *= ip_f.weight * alpha_scale * e_shift_scale *
                       jump_gradv_d_n_n * (1.0 - gamma_avg) * p_q2;
            Vector elvect_e2(elvect.GetData() + l2dofs_cnt, l2dofs_cnt);
            // Same sign as above, because it's an average of the shape f-s.
            elvect_e2 -= shape_e;
         }
      }

      // - < [((grad_v d).n) n] ({p phi} - gamma(1-gamma) [p + grad_p.d].[phi]>
      if (e_shift_type == 5)
      {
         double jump_gradp_d = p_q1 + d_q * p_grad_q1 - p_q2 - d_q * p_grad_q2;

         // 1st element.
         el_1.CalcShape(ip_e1, shape_e);
         shape_e *= ip_f.weight * (gradv_d_n_n_e1 * nor) *
                    ( gamma_avg * p_q1 -
                      gamma_avg * (1.0 - gamma_avg) * jump_gradp_d );
         Vector elvect_e1(elvect.GetData(), l2dofs_cnt);
         elvect_e1.Add(-1.0, shape_e);

         // 2nd element.
         el_2.CalcShape(ip_e2, shape_e);
         shape_e *= ip_f.weight * (gradv_d_n_n_e2 * nor) *
                    ( (1.0 - gamma_avg) * p_q2 +
                      gamma_avg * (1.0 - gamma_avg) * jump_gradp_d );
         Vector elvect_e2(elvect.GetData() + l2dofs_cnt, l2dofs_cnt);
         elvect_e2.Add(+1.0, shape_e);
      }

      // - < [p + grad_p.d] {phi} v >
      if (e_shift_type == 5)
      {
          double jump_gradp_d = p_q1 + d_q * p_grad_q1 - p_q2 - d_q * p_grad_q2;

          // 1st element.
          el_1.CalcShape(ip_e1, shape_e);
          shape_e *= ip_f.weight * jump_gradp_d *
                     gamma_avg * (nor * v_vals);
          Vector elvect_e1(elvect.GetData(), l2dofs_cnt);
          elvect_e1.Add(-1.0, shape_e);

          // 2nd element.
          el_2.CalcShape(ip_e2, shape_e);
          shape_e *= ip_f.weight * jump_gradp_d *
                     (1.0 - gamma_avg) * (nor * v_vals);
          Vector elvect_e2(elvect.GetData() + l2dofs_cnt, l2dofs_cnt);
          elvect_e2.Add(-1.0, shape_e);
      }

      // + < |[((grad_v d).n) n]| [p] [phi] >
      if (e_shift_type == 6)
      {
         // 1st element.
         el_1.CalcShape(ip_e1, shape_e);
         shape_e *= ip_f.weight * fabs(jump_gradv_d_n_n) * (p_q1 - p_q2);
         Vector elvect_e1(elvect.GetData(), l2dofs_cnt);
         elvect_e1.Add(+1.0, shape_e);

         // 2nd element.
         el_2.CalcShape(ip_e2, shape_e);
         shape_e *= ip_f.weight * fabs(jump_gradv_d_n_n) * (p_q1 - p_q2);
         Vector elvect_e2(elvect.GetData() + l2dofs_cnt, l2dofs_cnt);
         elvect_e2.Add(-1.0, shape_e);
      }

      // + < scale * {h |grad_v|} * [p + grad_p.d] * [phi + grad_phi.d] >.
      if (problem_visc && diffusion)
      {
         Vector p_ext(1);
         get_shifted_value(*p_e1, Trans_e1.ElementNo, ip_e1, d_q,
                           num_taylor, p_ext);
         double p_q1_ext = p_ext(0);
         get_shifted_value(*p_e2, Trans_e2.ElementNo, ip_e2, d_q,
                           num_taylor, p_ext);
         double p_q2_ext = p_ext(0);

         double p_gradp_jump = p_q1_ext - p_q2_ext;

         double rho_q1  = rhoDetJind0_e1->GetValue(Trans_e1, ip_e1) /
                          Trans_e1.Weight() / ind_e1->GetValue(Trans_e1, ip_e1);
         double rho_q2  = rhoDetJind0_e2->GetValue(Trans_e2, ip_e2) /
                          Trans_e2.Weight() / ind_e2->GetValue(Trans_e2, ip_e2);
         MFEM_VERIFY(rho_q1 > 0.0 && rho_q2 > 0.0,
                     "Negative density at the face, not good: "
                     << rho_q1 << " " << rho_q2);

         double h_1, h_2, mu_1, mu_2, visc_q1, visc_q2;

         // Mesh size with grad_v.
//         h_1  = mat_data.e_1.ParFESpace()->GetMesh()->GetElementSize(&Trans_e1, 0);
//         h_2  = mat_data.e_1.ParFESpace()->GetMesh()->GetElementSize(&Trans_e2, 0);
//         mu_1 = v_grad_q1.FNorm();
//         mu_2 = v_grad_q2.FNorm();
//         visc_q1 = h_1 * fabs(mu_1);
//         visc_q2 = h_2 * fabs(mu_2);

         // Distance with grad_v.
         h_1  = d_q.Norml2();
         h_2  = d_q.Norml2();
         mu_1 = v_grad_q1.FNorm();
         mu_2 = v_grad_q2.FNorm();
         visc_q1 = h_1 * fabs(mu_1);
         visc_q2 = h_2 * fabs(mu_2);

         // As in the volumetric viscosity.
//         v_grad_q1.Symmetrize();
//         v_grad_q2.Symmetrize();
//         LengthScaleAndCompression(v_grad_q1, Trans_e1, quad_data.Jac0inv(0),
//                                   quad_data.h0, h_1, mu_1);
//         LengthScaleAndCompression(v_grad_q2, Trans_e2, quad_data.Jac0inv(0),
//                                   quad_data.h0, h_2, mu_2);
//         visc_q1 = 2.0 * h_1 * fabs(mu_1);
//         if (mu_1 < 0.0)
//         {
//            visc_q1 += 0.5 * sqrt(gamma_e1 * fmax(p_q1, 1e-5) / rho_q1);
//         }
//         visc_q2 = 2.0 * h_2 * fabs(mu_2);
//         if (mu_2 < 0.0)
//         {
//            visc_q2 += 0.5 * sqrt(gamma_e2 * fmax(p_q2, 1e-5) / rho_q2);
//         }

         double grad_v_avg = gamma_avg * visc_q1 + (1.0 - gamma_avg) * visc_q2;

         // 1st element.
         shift_shape(*e->ParFESpace(), *p_e1->ParFESpace(),
                     Trans_e1.ElementNo, ip_e1, d_q, num_taylor, shape_e);
         shape_e *= Trans.Weight() * ip_f.weight * alpha_scale *
                    diffusion_scale * grad_v_avg * p_gradp_jump;
         Vector elvect_e1(elvect.GetData(), l2dofs_cnt);
         elvect_e1 += shape_e;

         // 2nd element.
         if (local_face)
         {
            shift_shape(*e->ParFESpace(), *p_e1->ParFESpace(),
                        Trans_e2.ElementNo, ip_e2, d_q, num_taylor, shape_e);
            shape_e *= Trans.Weight() * ip_f.weight * alpha_scale *
                       diffusion_scale * grad_v_avg * p_gradp_jump;
            Vector elvect_e2(elvect.GetData() + l2dofs_cnt, l2dofs_cnt);
            // Switches sign, because it's a jump of the shape f-s.
            elvect_e2 -= shape_e;
         }
      }
   }
}

void EnergyCutFaceIntegrator::AssembleRHSElementVect(
      const FiniteElement &el_1, const FiniteElement &el_2,
      FaceElementTransformations &Trans, Vector &elvect)
{
   MFEM_VERIFY(mat_data.e_1.ParFESpace()->GetNRanks() == 1,
               "Implemented only in serial, because CutFaceQuadratureData "
               "does not know about MPI neighbors.");

   const int l2dofs_cnt = el_1.GetDof();
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
   if (attr_face != 15) { return; }

   ElementTransformation &Trans_e1 = Trans.GetElement1Transformation();
   ElementTransformation &Trans_e2 = Trans.GetElement2Transformation();

   const ParGridFunction *p = (mat_id == 1) ? &mat_data.p_1->GetPressure()
                                            : &mat_data.p_2->GetPressure();
   const ParGridFunction *rhoDetJind0 = (mat_id == 1) ? &mat_data.rhoDetJind0_1
                                                      : &mat_data.rhoDetJind0_2;
   const ParGridFunction *ind = (mat_id == 1) ? &mat_data.vol_1
                                              : &mat_data.vol_2;
   const double gamma = (mat_id == 1) ? mat_data.gamma_1 : mat_data.gamma_2;

   Vector shape_e(l2dofs_cnt);

   const IntegrationRule *ir = IntRule;
   MFEM_VERIFY(ir != NULL, "Set the correct IntRule!");
   const int nqp_face = ir->GetNPoints();

   for (int q = 0; q < nqp_face; q++)
   {
      const IntegrationPoint &ip_f = ir->IntPoint(q);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip_f);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &ip_e1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &ip_e2 = Trans.GetElement2IntPoint();

      const double p_q1 = fmax(1e-5, p->GetValue(Trans_e1, ip_e1)),
                   p_q2 = fmax(1e-5, p->GetValue(Trans_e2, ip_e2));
      const double jump_p = p_q1 - p_q2;
      const double gamma_avg = p_q1 / (p_q1 + p_q2);

      const double rho_q1  = rhoDetJind0->GetValue(Trans_e1, ip_e1) /
                             ind->GetValue(Trans_e1, ip_e1) / Trans_e1.Weight(),
                   rho_q2  = rhoDetJind0->GetValue(Trans_e2, ip_e2) /
                             ind->GetValue(Trans_e2, ip_e2) / Trans_e2.Weight();
      const double cs_q1   = sqrt(gamma * p_q1 / rho_q1),
                   cs_q2   = sqrt(gamma * p_q2 / rho_q2),
                   cs_avg  = gamma_avg * cs_q1 + (1.0 - gamma_avg) * cs_q2;

      MFEM_VERIFY(rho_q1 > 0.0 && rho_q2 > 0.0,
                  "Negative density at the face, not good.");

      // The term is: + < {cs} [p] [phi] >
      // phi is DG, so [phi] = phi - 0.

      // 1st element.
      el_1.CalcShape(ip_e1, shape_e);
      shape_e *= ip_f.weight * Trans.Weight() * cs_avg * jump_p;
      Vector elvect_e1(elvect.GetData(), l2dofs_cnt);
      elvect_e1.Add(+1.0, shape_e);

      // 2nd element.
      if (local_face)
      {
         el_2.CalcShape(ip_e2, shape_e);
         shape_e *= ip_f.weight * Trans.Weight() * cs_avg * jump_p;
         Vector elvect_e2(elvect.GetData() + l2dofs_cnt, l2dofs_cnt);
         elvect_e2.Add(-1.0, shape_e);
      }
   }
}

void PointExtractor::SetPoint(int zone, const Vector &xyz,
                              const ParGridFunction *gf,
                              const IntegrationRule &ir, std::string filename)
{
   g = gf;
   z_id = zone;
   fstream.open(filename);

   ParFiniteElementSpace &pfes = *g->ParFESpace();
   MFEM_VERIFY(pfes.GetNRanks() == 1, "PointExtractor works only in serial.");
   MFEM_VERIFY(pfes.GetMesh()->Dimension() == 1, "Only implemented in 1D.");

   // Find the integration point and copy it.
   int q_id = FindIntegrPoint(zone, xyz, ir);
   ip = ir.IntPoint(q_id);
   MFEM_VERIFY(q_id > -1,
               "Integration point not found " << filename);
   cout << filename << ": Element " << zone << "; Quad: " << q_id << endl;

   fstream.precision(8);
}

void PointExtractor::WriteValue(double time)
{
   fstream << time << " " << GetValue() << "\n";
   fstream.flush();
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
      const IntegrationPoint &ip = ir.IntPoint(q);
      tr.SetIntPoint(&ip);
      tr.Transform(ip, position);
      // Assumes 1D.
      if (fabs(position(0) - xyz(0)) < eps) { return q; }
   }
   return -1;
}

double ShiftedPointExtractor::GetValue() const
{
   ParFiniteElementSpace &pfes = *g->ParFESpace();
   Vector grad_g(1);

   ElementTransformation &tr = *pfes.GetElementTransformation(z_id);
   tr.SetIntPoint(&ip);
   g->GetGradient(*pfes.GetElementTransformation(z_id), grad_g);

   // Assumes 1D.
   return g->GetValue(z_id, ip) + dist->GetValue(z_id, ip) * grad_g(0);
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

void InitTriPoint2Mat(MaterialData &mat_data, int variant)
{
   ParFiniteElementSpace &pfes = *mat_data.e_1.ParFESpace();
   const int NE    = pfes.GetNE();
   const int ndofs = mat_data.e_1.Size() / NE;
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
      Vector center(2);
      pfes.GetParMesh()->GetElementCenter(e, center);
      const double x = center(0), y = center(1);

      if (attr == 10 || attr == 15)
      {
         // Left/Top material.
         r = 1.0; p = 1.0;
         if (variant == 1 && x > 1.0)
         {
            r = 0.125; p = 0.1;
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
         if (variant == 0 && y > 1.5)
         {
            r = 0.125;
         }
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
