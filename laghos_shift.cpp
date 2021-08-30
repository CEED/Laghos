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

namespace mfem
{

namespace hydrodynamics
{



void DenseMatrixFromVecVecT(Vector &x, Vector &y, DenseMatrix &xyT) {
    for (int i = 0; i < x.Size(); i++) {
        for (int j = 0; j < y.Size(); j++) {
            xyT(i, j) = x(i)*y(j);
        }
    }
}

int material_id(int el_id, const ParGridFunction &g)
{
   const ParFiniteElementSpace &pfes =  *g.ParFESpace();
   const FiniteElement *fe = pfes.GetFE(el_id);
   Vector g_vals;
   const IntegrationRule &ir =
      IntRules.Get(fe->GetGeomType(), pfes.GetOrder(el_id) + 7);

   double integral = 0.0;
   bool is_positive = true;
   g.GetValues(el_id, ir, g_vals);
   ElementTransformation *Tr = pfes.GetMesh()->GetElementTransformation(el_id);
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      Tr->SetIntPoint(&ip);
      integral += ip.weight * g_vals(q) * Tr->Weight();
      if (g_vals(q) + 1e-12 < 0.0) { is_positive = false; }
   }
   return (is_positive) ? 1 : 0;
   //return (integral > 0.0) ? 1 : 0;
}

void MarkFaceAttributes(ParFiniteElementSpace &pfes)
{
   ParMesh *pmesh = pfes.GetParMesh();
   pmesh->ExchangeFaceNbrNodes();
   // Set face_attribute = 77 to faces that are on the material interface.
   for (int f = 0; f < pmesh->GetNumFaces(); f++)
   {
      auto *ftr = pmesh->GetFaceElementTransformations(f, 3);
       if (ftr->Elem2No > 0 &&
          pmesh->GetAttribute(ftr->Elem1No) != pmesh->GetAttribute(ftr->Elem2No))
      {
         pmesh->SetFaceAttribute(f, 77);
      }
   }
   for (int f = 0; f < pmesh->GetNSharedFaces(); f++) {
       auto *ftr = pmesh->GetSharedFaceTransformations(f, 3);
       int faceno = pmesh->GetSharedFace(f);
       int Elem1no = ftr->Elem1No;
       int Elem2NbrNo = ftr->Elem2No - pmesh->GetNE();
       auto *nbrftr = pfes.GetFaceNbrElementTransformation(Elem2NbrNo);
       int attr1 = pmesh->GetAttribute(ftr->Elem1No);
       int attr2 = nbrftr->Attribute;
       if (attr1 != attr2) {
           pmesh->SetFaceAttribute(faceno, 77);
       }
   }
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
  const int mode_TG = 0;

   switch (problem)
   {
      case 0:
      {
         if (mode_TG == 0)
         {
            // The domain area for the TG is 1.
            const double dx = sqrt(1.0 / glob_NE);

            // The middle of the element after x = 0.5.
            return tanh(x(0) - (0.5 + 0*dx/2.0));
         }
         else if (mode_TG == 1)
         {
            return tanh(x(0) - x(1));
         }
         else if (mode_TG == 2)
         {
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
      case 8: return tanh(x(0) - 0.5); // 2 mat Sod.
      case 9: return tanh(x(0) - 0.7); // water-air.
      case 10:
      {
         // The domain area for the 3point is 21.
         const double dx = sqrt(21.0 / glob_NE);

         // The middle of the element after x = 1.
         double waveloc = 1. + 0.75*dx;
         return tanh(x(0) - waveloc);
      }
      default: MFEM_ABORT("error"); return 0.0;
   }
}

void GradAtLocalDofs(ElementTransformation &T,
                     const ParGridFunction &g,
                     DenseMatrix &grad_g){
   ParFiniteElementSpace &pfes = *g.ParFESpace();
   const FiniteElement &el = *pfes.GetFE(T.ElementNo);
   const int dim = el.GetDim(), dof = el.GetDof();
   grad_g.SetSize(dof, dim);
   Array<int> dofs;
   Vector g_e;
   DenseMatrix grad_phys; // This will be (dof_p x dim, dof_p).
   {
      pfes.GetElementDofs(T.ElementNo, dofs);
      g.GetSubVector(dofs, g_e);
      el.ProjectGrad(el, T, grad_phys);
      Vector grad_ptr(grad_g.GetData(), dof*dim);
      grad_phys.Mult(g_e, grad_ptr);
   }
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
   DenseMatrix grad_phys; // This will be (dof_p x dim, dof_p).
   {
      pfes.GetElementVDofs(zone_id, dofs);
      g.GetSubVector(dofs, g_e);
      el.ProjectGrad(el, T, grad_phys);
      for (int d = 0; d < dim; d++) {
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
   MFEM_VERIFY(p.ParFESpace()->GetNRanks() == 1,
               "Implemented only in serial.");

   const int h1dofs_cnt = trial_fe.GetDof();
   const int l2dofs_cnt = test_fe.GetDof();
   const int dim = test_fe.GetDim();
   //int nor_dir_mask = 1;

   if (Trans.Elem2No < 0)
   {
      // This case should take care of shared (MPI) faces. They will get
      // processed by both MPI tasks.
      elmat.SetSize(l2dofs_cnt, h1dofs_cnt * dim);
   }
   else { elmat.SetSize(l2dofs_cnt * 2, h1dofs_cnt * dim * 2); }
   elmat = 0.0;

   // Must be done after elmat.SetSize().
   if (Trans.Attribute != 77) { return; }

   h1_shape.SetSize(h1dofs_cnt);
   l2_shape.SetSize(l2dofs_cnt);

   const int ir_order =
      2*trial_fe.GetOrder() + 2*test_fe.GetOrder() + Trans.OrderW();
   const IntegrationRule *ir = &IntRules.Get(Trans.GetGeometryType(), ir_order);
   const int nqp_face = ir->GetNPoints();

   // grad_p at all DOFs of the pressure FE space, on both sides.
   const FiniteElement &el_p = *p.ParFESpace()->GetFE(0);
   const int dof_p = el_p.GetDof();
   DenseMatrix p_grad_e_1(dof_p, dim), p_grad_e_2(dof_p, dim);
   GradAtLocalDofs(Trans.GetElement1Transformation(), p, p_grad_e_1);
   if (Trans.Elem2No > 0)
   {
      GradAtLocalDofs(Trans.GetElement2Transformation(), p, p_grad_e_2);
   }

   Vector nor(dim);

   Vector p_grad_q1(dim), d_q1(dim), shape_p1(dof_p), grad_shape_h1(dim);
   Vector p_grad_q2(dim), d_q2(dim), shape_p2(dof_p);
   DenseMatrix h1_grads(h1dofs_cnt, dim);
   for (int q = 0; q  < nqp_face; q++)
   {
      // Set the integration point in the face and the neighboring elements
      const IntegrationPoint &ip_f = ir->IntPoint(q);
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

      el_p.CalcShape(ip_e1, shape_p1);
      const double p1 = p.GetValue(Trans.GetElement1Transformation(), ip_e1);

      el_p.CalcShape(ip_e2, shape_p2);
      const double p2 = p.GetValue(Trans.GetElement2Transformation(), ip_e2);

      double p_term = 0.5*(p1+p2);
      // 1st element.
      {
         // Shape functions in the 1st element.
         trial_fe.CalcShape(ip_e1, h1_shape);
         test_fe.CalcShape(ip_e1, l2_shape);

         // TODO reorder/optimize loops.
         for (int i = 0; i < l2dofs_cnt; i++)
         {
            for (int j = 0; j < h1dofs_cnt; j++)
            {
               double h1_shape_part = h1_shape(j);
               double p_shift_part = p_term;
               p_shift_part *= scale;

               for (int d = 0; d < dim; d++)
               {
                  elmat(i, d*h1dofs_cnt + j)
                        += ip_f.weight * p_shift_part *
                          h1_shape_part * l2_shape(i) * nor(d);
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

         // TODO reorder/optimize loops.
         for (int i = 0; i < l2dofs_cnt; i++)
         {
            for (int j = 0; j < h1dofs_cnt; j++)
            {
               double h1_shape_part = h1_shape(j);
               double p_shift_part = p_term;
               p_shift_part *= scale;

               for (int d = 0; d < dim; d++)
               {
                  elmat(l2dofs_cnt + i, dim*h1dofs_cnt + d*h1dofs_cnt + j)
                        -= ip_f.weight * p_shift_part *
                          h1_shape_part * l2_shape(i) * nor(d);
               }
            }
         }
      }
   }
}

void VelocityStabilizerLFI::AssembleRHSElementVect(const FiniteElement &el_1,
                                                   const FiniteElement &el_2,
                                                   FaceElementTransformations &Trans,
                                                   Vector &elvect)
{
   const int h1dofs_cnt = el_1.GetDof();
   const int dim = el_1.GetDim();

   elvect.SetSize(h1dofs_cnt * dim * 2);
   if (Trans.Elem2No < 0)
   {
      elvect.SetSize(h1dofs_cnt * dim);
   }
   elvect = 0.0;

   // Must be done after elvect.SetSize().
   if (Trans.Attribute != 77) { return; }

   Vector h1_shape(h1dofs_cnt);

   const int ir_order =
      2*el_1.GetOrder() + Trans.OrderW();
   const IntegrationRule *ir = &IntRules.Get(Trans.GetGeometryType(), ir_order);
   const int nqp_face = ir->GetNPoints();

   // grad_p at all quad points, on both sides.
   const FiniteElement &el_v = *v.ParFESpace()->GetFE(0);
   const int dof_v = el_v.GetDof();
   Vector nor(dim), d_q1(dim), d_q2(dim),
          v1jump(dim), v2jump(dim), shape_v(dof_v),
          psi1jump(dim), psi2jump(dim);
   Vector vx1jump(dim), vx2jump(dim), vy1jump(dim), vy2jump(dim);

   DenseTensor v_strain_e_1(dof_v, dim, dim), v_strain_e_2(dof_v, dim, dim);
   if (Trans.Elem2No > 0)
   {
       StrainTensorAtLocalDofs(Trans.GetElement2Transformation(), v, v_strain_e_2);
   }
   StrainTensorAtLocalDofs(Trans.GetElement1Transformation(), v, v_strain_e_1);

   DenseMatrix v_strain_q1(dim), v_strain_q2(dim);
   DenseMatrix Iden(dim);
   Iden = 0.0;
   for (int d = 0; d < dim; d++) { Iden(d, d) = 1.0; }
   DenseMatrix NN(dim), VN1(dim), VN2(dim), work(dim);
   DenseMatrix IVN1(dim), IVN2(dim);
   DenseMatrix PsiN1(dim), PsiN2(dim);
   DenseMatrix IPsiN1(dim), IPsiN2(dim);

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

      Vector true_normal = nor;
      true_normal *= true_normal.Norml2() == 0. ? 0 : 1./true_normal.Norml2();

      DenseMatrixFromVecVecT(true_normal, true_normal, NN);
      Add(Iden, NN, -1.0, work); //work_ij = I - n_i n_j


      // 1st element stuff.
      Vector v1(dim), v2(dim);
      v.GetVectorValue(Trans.GetElement1Transformation(), ip_e1, v1);
      el_v.CalcShape(ip_e1, shape_v);
      for (int d = 0; d < dim; d++) {
          DenseMatrix v_grad_e_1(v_strain_e_1.GetData(d), dof_v, dim);
          Vector v_grad_q1(v_strain_q1.GetData()+d*dim, dim);
          v_grad_e_1.MultTranspose(shape_v, v_grad_q1);
      }
      v_strain_q1.Transpose(); //du_i/dx_j

      // 2nd element stuff.
      v.GetVectorValue(Trans.GetElement2Transformation(), ip_e2, v2);
      el_v.CalcShape(ip_e2, shape_v);
      for (int d = 0; d < dim; d++) {
          DenseMatrix v_grad_e_2(v_strain_e_2.GetData(d), dof_v, dim);
          Vector v_grad_q2(v_strain_q2.GetData()+d*dim, dim);
          v_grad_e_2.MultTranspose(shape_v, v_grad_q2);
      }
      v_strain_q2.Transpose();

      //when penalize normal is true, we penalize both the tangential and
      //normal components. We get rid of (I-n_in_j) terms from the
      //formulation i.e.e just set work = I instead of I-n_in_j
      bool penalize_normal = false;
      if (penalize_normal) { work = Iden; }
      double alpha = scale*work.FNorm2();
      if (penalize_normal) { alpha = scale; }


      //VN1(i, j) = v1(i)*true_normal(j)
      Vector vx1(dim), vy1(dim), vx2(dim), vy2(dim);
      vx1 = 0.;
      vx2 = 0.;
      vy1 = 0.;
      vy2 = 0.;
      vx1(0) = v1(0);
      vy1(1) = v1(1);
      vx2(0) = v2(0);
      vy2(1) = v2(1);
      //vy1.Print();

      DenseMatrixFromVecVecT(vx1, true_normal, VN1);
      Mult(work, VN1, IVN1);
      IVN1.Mult(nor, vx1jump);
      DenseMatrixFromVecVecT(vy1, true_normal, VN1);
      Mult(work, VN1, IVN1);
      IVN1.Mult(nor, vy1jump);

      DenseMatrixFromVecVecT(vx2, true_normal, VN2);
      Mult(work, VN2, IVN2);
      IVN2.Mult(nor, vx2jump);
      DenseMatrixFromVecVecT(vy2, true_normal, VN2);
      Mult(work, VN2, IVN2);
      IVN2.Mult(nor, vy2jump);

      //VN1(i, j) = v1(i)*true_normal(j)
      DenseMatrixFromVecVecT(v1, true_normal, VN1);
      Mult(work, VN1, IVN1);
      IVN1.Mult(nor, v1jump);

      //VN2(i, j) = v2(i)*true_normal(j)
      DenseMatrixFromVecVecT(v2, true_normal, VN2);
      Mult(work, VN2, IVN2);
      IVN2.Mult(nor, v2jump);

      //dvjum -> [[ (I-n_i n_j)(v_i n_j+) ]]
      Vector dvjump = v1jump;
             dvjump -= v2jump;

      Vector dvxjump = vx1jump;
             dvxjump -= vx2jump;
      Vector dvyjump = vy1jump;
             dvyjump -= vy2jump;


      const int idx1 = Trans.ElementNo*nqp_face*2 + q + 0*nqp_face,
                idx2 = Trans.ElementNo*nqp_face*2 + q + 1*nqp_face;

      double rhocs1 = cfqdata.rhocs(idx1),
             rhocs2 = cfqdata.rhocs(idx2),
             rhohdt1 = cfqdata.rho(idx1)*cfqdata.h(idx1)/(*dt),
             rhohdt2 = cfqdata.rho(idx2)*cfqdata.h(idx2)/(*dt);
//      double tau = 2.0*rhocs1*rhocs2/(rhocs1 + rhocs2);
      double tau = 2.0*rhohdt1*rhohdt2/(rhohdt1 + rhohdt2);

//      dvxjump = dvjump;
//      dvyjump = dvjump;

      // 1st element.
      {
         el_1.CalcShape(ip_e1, h1_shape);
         double w1 = alpha*tau*ip_f.weight / nor.Norml2();
         //We divide by nor.Norml2() here because we have included it twice
         //through the jump terms
         for (int j = 0; j < h1_shape.Size(); j++)
         {
             Vector psi(dim);
             psi = 0.;
             psi(0) = h1_shape(j);
             DenseMatrixFromVecVecT(psi, true_normal, PsiN1);
             Mult(work, PsiN1, IPsiN1);
             IPsiN1.Mult(nor, psi1jump);
             elvect(j) += w1 * (psi1jump*dvxjump);

             psi = 0.;
             psi(1) = h1_shape(j);
             DenseMatrixFromVecVecT(psi, true_normal, PsiN1);
             Mult(work, PsiN1, IPsiN1);
             IPsiN1.Mult(nor, psi1jump);
             elvect(j + h1dofs_cnt) += w1 * (psi1jump*dvyjump);
         }
      }

      // 2nd element
      {
         el_2.CalcShape(ip_e2, h1_shape);
         double w2 = alpha*tau*ip_f.weight / nor.Norml2();
         for (int j = 0; j < h1_shape.Size(); j++)
         {
             Vector psi(dim);
             psi = 0.;
             psi(0) = h1_shape(j);
             DenseMatrixFromVecVecT(psi, true_normal, PsiN2);
             Mult(work, PsiN2, IPsiN2);
             IPsiN2.Mult(nor, psi2jump);
             elvect(h1dofs_cnt * dim + j) -= w2 * (psi2jump*dvxjump);

             psi = 0.;
             psi(1) = h1_shape(j);
             DenseMatrixFromVecVecT(psi, true_normal, PsiN2);
             Mult(work, PsiN2, IPsiN2);
             IPsiN2.Mult(nor, psi2jump);
             elvect(h1dofs_cnt * dim + j + h1dofs_cnt) -= w2 * (psi2jump*dvyjump);
         }
      }
   }
   //elvect *= 0.0;
}

void VelocityBoundaryLFI::AssembleRHSElementVect(const FiniteElement &el,
                                                 FaceElementTransformations &Trans,
                                                 Vector &elvect)
{
   const int h1dofs_cnt = el.GetDof();
   const int dim = el.GetDim();

   elvect.SetSize(h1dofs_cnt * dim);
   elvect = 0.0;

   Vector h1_shape(h1dofs_cnt);

   const int ir_order =
      2*el.GetOrder() + Trans.OrderW();
   const IntegrationRule *ir = &IntRules.Get(Trans.GetGeometryType(), ir_order);
   const int nqp_face = ir->GetNPoints();

   Vector nor(dim);
   for (int q = 0; q  < nqp_face; q++)
   {
      const IntegrationPoint &ip_f = ir->IntPoint(q);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip_f);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &ip_e1 = Trans.GetElement1IntPoint();

      // The normal includes the Jac scaling.
      // The orientation is taken into account in the processing of element 1.
      if (dim == 1)
      {
         nor(0) = (2*ip_e1.x - 1.0 ) * Trans.Weight();
      }
      else { CalcOrtho(Trans.Jacobian(), nor); }

      const double pr = p.GetValue(Trans.GetElement1Transformation(), ip_e1);
      // 1st element.
      {
         el.CalcShape(ip_e1, h1_shape);
         for (int j = 0; j < h1_shape.Size(); j++)
         {
             for (int d = 0; d < dim; d++)
             {
                elvect(j + d*h1dofs_cnt) += ip_f.weight * h1_shape(j) * nor(d) * pr;
             }
         }
      }
   }
}

void EnergyInterfaceIntegrator::AssembleRHSElementVect(const FiniteElement &el_1,
                                                       const FiniteElement &el_2,
                                                       FaceElementTransformations &Trans,
                                                       Vector &elvect)
{
   const int l2dofs_cnt = el_1.GetDof();
   const int dim = el_1.GetDim();

   elvect.SetSize(l2dofs_cnt * 2);
   if (Trans.Elem2No < 0)
   {
      elvect.SetSize(l2dofs_cnt);
   }
   elvect = 0.0;

   // Must be done after elvect.SetSize().
   if (Trans.Attribute != 77) { return; }

   Vector l2_shape(l2dofs_cnt);

   const int ir_order =
      2*el_1.GetOrder() + Trans.OrderW();
   const IntegrationRule *ir = &IntRules.Get(Trans.GetGeometryType(), ir_order);
   const int nqp_face = ir->GetNPoints();

   // grad_p at all quad points, on both sides.
   const FiniteElement &el_p = *p.ParFESpace()->GetFE(0);
   const FiniteElement &el_v = *v.ParFESpace()->GetFE(0);
   const int dof_p = el_p.GetDof(), dof_v = el_v.GetDof();
   DenseMatrix p_grad_e_1(dof_p, dim), p_grad_e_2(dof_p, dim);
   if (Trans.Elem2No > 0)
   {
      GradAtLocalDofs(Trans.GetElement2Transformation(), p, p_grad_e_2);
   }
   GradAtLocalDofs(Trans.GetElement1Transformation(), p, p_grad_e_1);

   Vector nor(dim);

   Vector p_grad_q1(dim), p_grad_q2(dim), d_q1(dim), d_q2(dim),
         shape_p(dof_p), v_vals(dim), shape_v(dof_v);

   DenseTensor v_strain_e_1(dof_v, dim, dim), v_strain_e_2(dof_v, dim, dim);
   if (Trans.Elem2No > 0)
   {
       StrainTensorAtLocalDofs(Trans.GetElement2Transformation(), v, v_strain_e_2);
   }
   StrainTensorAtLocalDofs(Trans.GetElement1Transformation(), v, v_strain_e_1);

   DenseMatrix v_strain_q1(dim), v_strain_q2(dim);

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

      // 1st element stuff.
      el_p.CalcShape(ip_e1, shape_p);
      const double p1 = p.GetValue(Trans.GetElement1Transformation(), ip_e1);

      Vector v1(dim), v2(dim);
      v.GetVectorValue(Trans.GetElement1Transformation(), ip_e1, v1);
      el_v.CalcShape(ip_e1, shape_v);
      for (int d = 0; d < dim; d++) {
          DenseMatrix v_grad_e_1(v_strain_e_1.GetData(d), dof_v, dim);
          Vector v_grad_q1(v_strain_q1.GetData()+d*dim, dim);
          v_grad_e_1.MultTranspose(shape_v, v_grad_q1);
      }
      v_strain_q1.Transpose(); //du_i/dx_j

      // 2nd element stuff.
      el_p.CalcShape(ip_e2, shape_p);
      const double p2 = p.GetValue(Trans.GetElement2Transformation(), ip_e2);

      v.GetVectorValue(Trans.GetElement2Transformation(), ip_e2, v2);
      el_v.CalcShape(ip_e2, shape_v);
      for (int d = 0; d < dim; d++) {
          DenseMatrix v_grad_e_2(v_strain_e_2.GetData(d), dof_v, dim);
          Vector v_grad_q2(v_strain_q2.GetData()+d*dim, dim);
          v_grad_e_2.MultTranspose(shape_v, v_grad_q2);
      }
      v_strain_q2.Transpose();

      //v.GetVectorValue(Trans, ip_f, v_vals);
      double gradv_d_n_n_jump = 0.;
      {
        d_q1 = nor;
        Vector true_normal = d_q1;
        true_normal *= d_q1.Norml2() == 0. ? 0 : 1./d_q1.Norml2();
        double gradv_d_n = v1*true_normal;
        Vector gradv_d_n_n1 = true_normal;
        gradv_d_n_n1 *= gradv_d_n;

        d_q2 = nor;
        true_normal = d_q2;
        true_normal *= d_q2.Norml2() == 0. ? 0 : 1./d_q2.Norml2();
        gradv_d_n = v2*true_normal;
        Vector gradv_d_n_n2 = true_normal;
        gradv_d_n_n2 *= gradv_d_n;

        gradv_d_n_n_jump = gradv_d_n_n1*nor - gradv_d_n_n2*nor;
      }

      // - <[[(v +nabla v d) . n)n]], {{p phi}}
      // 1st element.
      {
         // L2 shape functions in the 1st element.
         el_1.CalcShape(ip_e1, l2_shape);

         Vector l2_shape_p_avg = l2_shape;
         l2_shape_p_avg *= 0.5*p1;
         l2_shape_p_avg *= gradv_d_n_n_jump;
         l2_shape_p_avg *= ip_f.weight;
         Vector elvect_temp(elvect.GetData(), l2dofs_cnt);
         elvect_temp.Add(-1., l2_shape_p_avg);
      }

      // 2nd element
      {
         el_2.CalcShape(ip_e2, l2_shape);

         Vector l2_shape_p_avg = l2_shape;
         l2_shape_p_avg *= 0.5*p2;
         l2_shape_p_avg *= gradv_d_n_n_jump;
         l2_shape_p_avg *= ip_f.weight;
         Vector elvect_temp(elvect.GetData()+l2dofs_cnt, l2dofs_cnt);
         elvect_temp.Add(-1., l2_shape_p_avg);
      }
   }
}


void EnergyStabilizerLFI::AssembleRHSElementVect(const FiniteElement &el_1,
                                                 const FiniteElement &el_2,
                                                 FaceElementTransformations &Trans,
                                                 Vector &elvect)
{
   const int l2dofs_cnt = el_1.GetDof();
   const int dim = el_1.GetDim();

   elvect.SetSize(l2dofs_cnt * 2);
   if (Trans.Elem2No < 0)
   {
      elvect.SetSize(l2dofs_cnt);
   }
   elvect = 0.0;

   // Must be done after elvect.SetSize().
   if (Trans.Attribute != 77) { return; }

   Vector l2_shape(l2dofs_cnt);

   const int ir_order =
      2*el_1.GetOrder() + Trans.OrderW();
   const IntegrationRule *ir = &IntRules.Get(Trans.GetGeometryType(), ir_order);
   const int nqp_face = ir->GetNPoints();

   // grad_p at all quad points, on both sides.
   const FiniteElement &el_p = *p.ParFESpace()->GetFE(0);

   Vector nor(dim);

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

      // 1st element stuff.
      const double p1 = p.GetValue(Trans.GetElement1Transformation(), ip_e1);

      // 2nd element stuff.
      const double p2 = p.GetValue(Trans.GetElement2Transformation(), ip_e2);

      const double h1 = Trans.GetElement1Transformation().Weight()/nor.Norml2();
      const double h2 = Trans.GetElement2Transformation().Weight()/nor.Norml2();

      double hdt1 = h1/(*dt),
             hdt2 = h2/(*dt);
      double tau = scale*2.0*hdt1*hdt2/(hdt1 + hdt2);

      double p_jump = p1-p2;

      // + <tau[[p]], [[phi]]>
      // 1st element.
      {
         // L2 shape functions in the 1st element.
         el_1.CalcShape(ip_e1, l2_shape);

         Vector l2_shape_jump = l2_shape;
         l2_shape_jump *= p_jump;
         l2_shape_jump *= ip_f.weight;
         l2_shape_jump *= nor.Norml2();
         l2_shape_jump *= tau;
         Vector elvect_temp(elvect.GetData(), l2dofs_cnt);
         elvect_temp.Add(1., l2_shape_jump);
      }

      // 2nd element
      {
         el_2.CalcShape(ip_e2, l2_shape);

         Vector l2_shape_jump = l2_shape;
         l2_shape_jump *= -1.0;
         l2_shape_jump *= p_jump;
         l2_shape_jump *= ip_f.weight;
         l2_shape_jump *= nor.Norml2();
         l2_shape_jump *= tau;
         Vector elvect_temp(elvect.GetData()+l2dofs_cnt, l2dofs_cnt);
         elvect_temp.Add(1., l2_shape_jump);
      }
   }
}

void EnergyBoundaryLFI::AssembleRHSElementVect(const FiniteElement &el,
                                               FaceElementTransformations &Trans,
                                               Vector &elvect)
{

   const int l2dofs_cnt = el.GetDof();
   const int dim = el.GetDim();

   elvect.SetSize(l2dofs_cnt);
   elvect = 0.0;

   Vector l2_shape(l2dofs_cnt);

   const int ir_order =
      4*el.GetOrder() + Trans.OrderW();
   const IntegrationRule *ir = &IntRules.Get(Trans.GetGeometryType(), ir_order);
   const int nqp_face = ir->GetNPoints();

   // grad_p at all quad points, on both sides.
   const FiniteElement &el_p = *p.ParFESpace()->GetFE(0);
   const int dof_p = el_p.GetDof();
   Vector nor(dim), v1(dim);

   for (int q = 0; q  < nqp_face; q++)
   {
      const IntegrationPoint &ip_f = ir->IntPoint(q);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip_f);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &ip_e1 = Trans.GetElement1IntPoint();

      // The normal includes the Jac scaling.
      // The orientation is taken into account in the processing of element 1.
      if (dim == 1)
      {
         nor(0) = (2*ip_e1.x - 1.0 ) * Trans.Weight();
      }
      else { CalcOrtho(Trans.Jacobian(), nor); }

      const double pr = p.GetValue(Trans.GetElement1Transformation(), ip_e1);
      v.GetVectorValue(Trans.GetElement1Transformation(), ip_e1, v1);
      // 1st element.
      {
         el.CalcShape(ip_e1, l2_shape);
         l2_shape *= ip_f.weight * pr * (v1*nor);
         elvect += l2_shape;
      }
   }
}

void InitSod2Mat(ParGridFunction &rho, ParGridFunction &v,
                 ParGridFunction &e, ParGridFunction &gamma)
{
   v = 0.0;

   ParFiniteElementSpace &pfes = *rho.ParFESpace();
   const int NE    = pfes.GetNE();
   const int ndofs = rho.Size() / NE;
   double r, g, p;
   for (int i = 0; i < NE; i++)
   {
      if (pfes.GetParMesh()->GetAttribute(i) == 1)
      {
         r = 1.000; g = 2.0; p = 2.0;
      }
      else
      {
         r = 0.125; g = 1.4; p = 0.1;
      }

      gamma(i) = g;
      for (int j = 0; j < ndofs; j++)
      {
         rho(i*ndofs + j) = r;
         e(i*ndofs + j)   = p / r / (g - 1.0);
      }
   }
}

void InitWaterAir(ParGridFunction &rho, ParGridFunction &v,
                  ParGridFunction &e, ParGridFunction &gamma)
{
   v = 0.0;

   ParFiniteElementSpace &pfes = *rho.ParFESpace();
   const int NE    = pfes.GetNE();
   const int ndofs = rho.Size() / NE;
   double r, g, p;
   for (int i = 0; i < NE; i++)
   {
      if (pfes.GetParMesh()->GetAttribute(i) == 1)
      {
         r = 1000; g = 4.4; p = 1.e9;
         double A = 6.0e8;
         gamma(i) = g;
         for (int j = 0; j < ndofs; j++)
         {
            rho(i*ndofs + j) = r;
            e(i*ndofs + j)   = (p + g*A) / r / (g - 1.0);
         }
      }
      else
      {
         r = 50; g = 1.4; p = 1.e5;
         gamma(i) = g;
         for (int j = 0; j < ndofs; j++)
         {
            rho(i*ndofs + j) = r;
            e(i*ndofs + j)   = p / r / (g - 1.0);
         }
      }
   }
}

void InitTriPoint2Mat(ParGridFunction &rho, ParGridFunction &v,
                      ParGridFunction &e, ParGridFunction &gamma)
{
   MFEM_VERIFY(rho.ParFESpace()->GetMesh()->Dimension() == 2, "2D only.");

   v = 0.0;
   ParFiniteElementSpace &pfes = *rho.ParFESpace();
   const int NE    = pfes.GetNE();
   const int ndofs = rho.Size() / NE;
   double r, g, p;
   for (int i = 0; i < NE; i++)
   {
      if (pfes.GetParMesh()->GetAttribute(i) == 1)
      {
         r = 1.0; g = 1.5; p = 1.0;
      }
      else
      {
         p = 0.1;
         Vector center(2);
         pfes.GetParMesh()->GetElementCenter(i, center);
         r = (center(1) < 1.5) ? 1.0 : 0.125;
         g = (center(1) < 1.5) ? 1.4 : 1.5;
      }

      gamma(i) = g;
      for (int j = 0; j < ndofs; j++)
      {
         rho(i*ndofs + j) = r;
         e(i*ndofs + j)   = p / r / (g - 1.0);
      }
   }
}


} // namespace hydrodynamics

} // namespace mfem
