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

#include "laghos_assembly.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{


namespace hydrodynamics
{

const Tensors1D *tensors1D = NULL;

Tensors1D::Tensors1D(int H1order, int L2order, int nqp1D)
   : HQshape1D(H1order + 1, nqp1D),
     HQgrad1D(H1order + 1, nqp1D),
     LQshape1D(L2order + 1, nqp1D)
{
   // In this miniapp we assume:
   // - Gauss-Legendre quadrature points.
   // - Gauss-Lobatto continuous kinematic basis.
   // - Bernstein discontinuous thermodynamic basis.

   const double *quad1D_pos = poly1d.GetPoints(nqp1D - 1,
                                               Quadrature1D::GaussLegendre);
   Poly_1D::Basis &basisH1 = poly1d.GetBasis(H1order,
                                             Quadrature1D::GaussLobatto);
   Vector col, grad_col;
   for (int q = 0; q < nqp1D; q++)
   {
      HQshape1D.GetColumnReference(q, col);
      HQgrad1D.GetColumnReference(q, grad_col);
      basisH1.Eval(quad1D_pos[q], col, grad_col);
   }
   for (int q = 0; q < nqp1D; q++)
   {
      LQshape1D.GetColumnReference(q, col);
      poly1d.CalcBernstein(L2order, quad1D_pos[q], col);
   }
}

void DensityIntegrator::AssembleRHSElementVect(const FiniteElement &fe,
                                               ElementTransformation &Tr,
                                               Vector &elvect)
{
   const int ip_cnt = IntRule->GetNPoints();
   Vector shape(fe.GetDof());

   elvect.SetSize(fe.GetDof());
   elvect = 0.0;

   for (int q = 0; q < ip_cnt; q++)
   {
      const IntegrationPoint &ip = IntRule->IntPoint(q);

      fe.CalcShape(ip, shape);
      // Note that rhoDetJ = rho0DetJ0.
      shape *= ip.weight * quad_data.rho0DetJ0(Tr.ElementNo*ip_cnt + q);
      elvect += shape;
   }
}

void ForceIntegrator::AssembleElementMatrix2(const FiniteElement &trial_fe,
                                             const FiniteElement &test_fe,
                                             ElementTransformation &Trans,
                                             DenseMatrix &elmat)
{
   const int ip_cnt = IntRule->GetNPoints();
   const int dim = trial_fe.GetDim();
   const int zone_id = Trans.ElementNo;
   const int h1dofs_cnt = test_fe.GetDof();
   const int l2dofs_cnt = trial_fe.GetDof();

   elmat.SetSize(h1dofs_cnt*dim, l2dofs_cnt);
   elmat = 0.0;

   DenseMatrix vshape(h1dofs_cnt, dim), dshape(h1dofs_cnt, dim),
               loc_force(h1dofs_cnt, dim), Jinv(dim);
   Vector shape(l2dofs_cnt), Vloc_force(loc_force.Data(), h1dofs_cnt*dim);

   for (int q = 0; q < ip_cnt; q++)
   {
      const DenseMatrix &J = quad_data.Jac(zone_id*ip_cnt + q);
      const IntegrationPoint &ip = IntRule->IntPoint(q);

      test_fe.CalcDShape(ip, vshape);
      CalcInverse(J, Jinv);
      Mult(vshape, Jinv, dshape);

      MultABt(dshape, quad_data.stress(zone_id*ip_cnt + q), loc_force);
      trial_fe.CalcShape(ip, shape);

      AddMult_a_VWt(ip.weight * J.Det(), Vloc_force, shape, elmat);
   }
}

void ForcePAOperator::Mult(const Vector &vecL2, Vector &vecH1) const
{
   if      (dim == 2) { MultQuad(vecL2, vecH1); }
   else if (dim == 3) { MultHex(vecL2, vecH1); }
   else { MFEM_ABORT("Unsupported dimension"); }
}

void ForcePAOperator::MultTranspose(const Vector &vecH1, Vector &vecL2) const
{
   if      (dim == 2) { MultTransposeQuad(vecH1, vecL2); }
   else if (dim == 3) { MultTransposeHex(vecH1, vecL2); }
   else { MFEM_ABORT("Unsupported dimension"); }
}

// Force matrix action on quadrilateral elements in 2D
void ForcePAOperator::MultQuad(const Vector &vecL2, Vector &vecH1) const
{
   const int nH1dof1D = tensors1D->HQshape1D.Height(),
             nL2dof1D = tensors1D->LQshape1D.Height(),
             nqp1D    = tensors1D->HQshape1D.Width(),
             nqp      =  nqp1D * nqp1D,
             nH1dof   = nH1dof1D * nH1dof1D;
   Array<int> h1dofs, l2dofs;
   Vector e(nL2dof1D * nL2dof1D);
   DenseMatrix E(e.GetData(), nL2dof1D, nL2dof1D);
   DenseMatrix LQ(nL2dof1D, nqp1D), HQ(nH1dof1D, nqp1D), QQ(nqp1D, nqp1D),
               HHx(nH1dof1D, nH1dof1D), HHy(nH1dof1D, nH1dof1D);
   // Quadrature data for a specific direction.
   DenseMatrix QQd(nqp1D, nqp1D);
   double *data_qd = QQd.GetData(), *data_q = QQ.GetData();

   const H1_QuadrilateralElement *fe =
      dynamic_cast<const H1_QuadrilateralElement *>(H1FESpace.GetFE(0));
   const Array<int> &dof_map = fe->GetDofMap();

   vecH1 = 0.0;
   for (int z = 0; z < nzones; z++)
   {
      // Note that the local numbering for L2 is the tensor numbering.
      L2FESpace.GetElementDofs(z, l2dofs);
      vecL2.GetSubVector(l2dofs, e);

      // LQ_j2_k1 = E_j1_j2 LQs_j1_k1  -- contract in x direction.
      // QQ_k1_k2 = LQ_j2_k1 LQs_j2_k2 -- contract in y direction.
      MultAtB(E, tensors1D->LQshape1D, LQ);
      MultAtB(LQ, tensors1D->LQshape1D, QQ);

      // Iterate over the components (x and y) of the result.
      for (int c = 0; c < 2; c++)
      {
         // QQd_k1_k2 *= stress_k1_k2(c,0)  -- stress that scales d[v_c]_dx.
         // HQ_i2_k1   = HQs_i2_k2 QQ_k1_k2 -- contract in y direction.
         // HHx_i1_i2  = HQg_i1_k1 HQ_i2_k1 -- gradients in x direction.
         double *d = quad_data->stressJinvT(c).GetData() + z*nqp;
         for (int q = 0; q < nqp; q++) { data_qd[q] = data_q[q] * d[q]; };
         MultABt(tensors1D->HQshape1D, QQd, HQ);
         MultABt(tensors1D->HQgrad1D, HQ, HHx);

         // QQd_k1_k2 *= stress_k1_k2(c,1) -- stress that scales d[v_c]_dy.
         // HQ_i2_k1  = HQg_i2_k2 QQ_k1_k2 -- gradients in y direction.
         // HHy_i1_i2 = HQ_i1_k1 HQ_i2_k1  -- contract in x direction.
         d = quad_data->stressJinvT(c).GetData() + 1*nzones*nqp + z*nqp;
         for (int q = 0; q < nqp; q++) { data_qd[q] = data_q[q] * d[q]; };
         MultABt(tensors1D->HQgrad1D, QQd, HQ);
         MultABt(tensors1D->HQshape1D, HQ, HHy);

         // Set the c-component of the result.
         H1FESpace.GetElementVDofs(z, h1dofs);
         for (int i1 = 0; i1 < nH1dof1D; i1++)
         {
            for (int i2 = 0; i2 < nH1dof1D; i2++)
            {
               // Transfer from the mfem's H1 local numbering to the tensor
               // structure numbering.
               const int idx = i2 * nH1dof1D + i1;
               vecH1[h1dofs[c*nH1dof + dof_map[idx]]] +=
                  HHx(i1, i2) + HHy(i1, i2);
            }
         }
      }
   }
}

// Force matrix action on hexahedral elements in 3D
void ForcePAOperator::MultHex(const Vector &vecL2, Vector &vecH1) const
{
   const int nH1dof1D = tensors1D->HQshape1D.Height(),
             nL2dof1D = tensors1D->LQshape1D.Height(),
             nqp1D    = tensors1D->HQshape1D.Width(),
             nqp      = nqp1D * nqp1D * nqp1D,
             nH1dof   = nH1dof1D * nH1dof1D * nH1dof1D;
   Array<int> h1dofs, l2dofs;

   Vector e(nL2dof1D * nL2dof1D * nL2dof1D);
   DenseMatrix E(e.GetData(), nL2dof1D*nL2dof1D, nL2dof1D);

   DenseMatrix HH_Q(nH1dof1D * nH1dof1D, nqp1D),
               H_HQ(HH_Q.GetData(), nH1dof1D, nH1dof1D*nqp1D),
               Q_HQ(nqp1D, nH1dof1D*nqp1D);
   DenseMatrix LL_Q(nL2dof1D * nL2dof1D, nqp1D),
               L_LQ(LL_Q.GetData(), nL2dof1D, nL2dof1D*nqp1D),
               Q_LQ(nqp1D, nL2dof1D*nqp1D);
   DenseMatrix QQ_Q(nqp1D * nqp1D, nqp1D), QQ_Qc(nqp1D * nqp1D, nqp1D);
   double *qqq = QQ_Q.GetData(), *qqqc = QQ_Qc.GetData();
   DenseMatrix HHHx(nH1dof1D * nH1dof1D, nH1dof1D),
               HHHy(nH1dof1D * nH1dof1D, nH1dof1D),
               HHHz(nH1dof1D * nH1dof1D, nH1dof1D);

   const H1_HexahedronElement *fe =
      dynamic_cast<const H1_HexahedronElement *>(H1FESpace.GetFE(0));
   const Array<int> &dof_map = fe->GetDofMap();

   vecH1 = 0.0;
   for (int z = 0; z < nzones; z++)
   {
      // Note that the local numbering for L2 is the tensor numbering.
      L2FESpace.GetElementDofs(z, l2dofs);
      vecL2.GetSubVector(l2dofs, e);

      // LLQ_j1_j2_k3  = E_j1_j2_j3 LQs_j3_k3   -- contract in z direction.
      // QLQ_k1_j2_k3  = LQs_j1_k1 LLQ_j1_j2_k3 -- contract in x direction.
      // QQQ_k1_k2_k3  = QLQ_k1_j2_k3 LQs_j2_k2 -- contract in y direction.
      // The last step does some reordering (it's not product of matrices).
      mfem::Mult(E, tensors1D->LQshape1D, LL_Q);
      MultAtB(tensors1D->LQshape1D, L_LQ, Q_LQ);
      for (int k1 = 0; k1 < nqp1D; k1++)
      {
         for (int k2 = 0; k2 < nqp1D; k2++)
         {
            for (int k3 = 0; k3 < nqp1D; k3++)
            {
               QQ_Q(k1 + nqp1D*k2, k3) = 0.0;
               for (int j2 = 0; j2 < nL2dof1D; j2++)
               {
                  QQ_Q(k1 + nqp1D*k2, k3) +=
                     Q_LQ(k1, j2 + k3*nL2dof1D) * tensors1D->LQshape1D(j2, k2);
               }
            }
         }
      }

      // Iterate over the components (x, y, z) of the result.
      for (int c = 0; c < 3; c++)
      {
         // QQQc_k1_k2_k3 *= stress_k1_k2_k3(c,0) -- stress scaling d[v_c]_dx.
         double *d = quad_data->stressJinvT(c).GetData() + z*nqp;
         for (int q = 0; q < nqp; q++) { qqqc[q] = qqq[q] * d[q]; };

         // QHQ_k1_i2_k3  = QQQc_k1_k2_k3 HQs_i2_k2 -- contract  in y direction.
         // This first step does some reordering (it's not product of matrices).
         // HHQ_i1_i2_k3  = HQg_i1_k1 QHQ_k1_i2_k3  -- gradients in x direction.
         // HHHx_i1_i2_i3 = HHQ_i1_i2_k3 HQs_i3_k3  -- contract  in z direction.
         for (int k1 = 0; k1 < nqp1D; k1++)
         {
            for (int i2 = 0; i2 < nH1dof1D; i2++)
            {
               for (int k3 = 0; k3 < nqp1D; k3++)
               {
                  Q_HQ(k1, i2 + nH1dof1D*k3) = 0.0;
                  for (int k2 = 0; k2 < nqp1D; k2++)
                  {
                     Q_HQ(k1, i2 + nH1dof1D*k3) +=
                        QQ_Qc(k1 + nqp1D*k2, k3) * tensors1D->HQshape1D(i2, k2);
                  }
               }
            }
         }
         mfem::Mult(tensors1D->HQgrad1D, Q_HQ, H_HQ);
         MultABt(HH_Q, tensors1D->HQshape1D, HHHx);

         // QQQc_k1_k2_k3 *= stress_k1_k2_k3(c,1) -- stress scaling d[v_c]_dy.
         d = quad_data->stressJinvT(c).GetData() + 1*nzones*nqp + z*nqp;
         for (int q = 0; q < nqp; q++) { qqqc[q] = qqq[q] * d[q]; };

         // QHQ_k1_i2_k3  = QQQc_k1_k2_k3 HQg_i2_k2 -- gradients in y direction.
         // This first step does some reordering (it's not product of matrices).
         // HHQ_i1_i2_k3  = HQs_i1_k1 QHQ_k1_i2_k3  -- contract  in x direction.
         // HHHy_i1_i2_i3 = HHQ_i1_i2_k3 HQs_i3_k3  -- contract  in z direction.
         for (int k1 = 0; k1 < nqp1D; k1++)
         {
            for (int i2 = 0; i2 < nH1dof1D; i2++)
            {
               for (int k3 = 0; k3 < nqp1D; k3++)
               {
                  Q_HQ(k1, i2 + nH1dof1D*k3) = 0.0;
                  for (int k2 = 0; k2 < nqp1D; k2++)
                  {
                     Q_HQ(k1, i2 + nH1dof1D*k3) +=
                        QQ_Qc(k1 + nqp1D*k2, k3) * tensors1D->HQgrad1D(i2, k2);
                  }
               }
            }
         }
         mfem::Mult(tensors1D->HQshape1D, Q_HQ, H_HQ);
         MultABt(HH_Q, tensors1D->HQshape1D, HHHy);

         // QQQc_k1_k2_k3 *= stress_k1_k2_k3(c,2) -- stress scaling d[v_c]_dz.
         d = quad_data->stressJinvT(c).GetData() + 2*nzones*nqp + z*nqp;
         for (int q = 0; q < nqp; q++) { qqqc[q] = qqq[q] * d[q]; };

         // QHQ_k1_i2_k3  = QQQc_k1_k2_k3 HQg_i2_k2 -- contract  in y direction.
         // This first step does some reordering (it's not product of matrices).
         // HHQ_i1_i2_k3  = HQs_i1_k1 QHQ_k1_i2_k3  -- contract  in x direction.
         // HHHz_i1_i2_i3 = HHQ_i1_i2_k3 HQs_i3_k3  -- gradients in z direction.
         for (int k1 = 0; k1 < nqp1D; k1++)
         {
            for (int i2 = 0; i2 < nH1dof1D; i2++)
            {
               for (int k3 = 0; k3 < nqp1D; k3++)
               {
                  Q_HQ(k1, i2 + nH1dof1D*k3) = 0.0;
                  for (int k2 = 0; k2 < nqp1D; k2++)
                  {
                     Q_HQ(k1, i2 + nH1dof1D*k3) +=
                        QQ_Qc(k1 + nqp1D*k2, k3) * tensors1D->HQshape1D(i2, k2);
                  }
               }
            }
         }
         mfem::Mult(tensors1D->HQshape1D, Q_HQ, H_HQ);
         MultABt(HH_Q, tensors1D->HQgrad1D, HHHz);

         // Set the c-component of the result.
         H1FESpace.GetElementVDofs(z, h1dofs);
         for (int i1 = 0; i1 < nH1dof1D; i1++)
         {
            for (int i2 = 0; i2 < nH1dof1D; i2++)
            {
               for (int i3 = 0; i3 < nH1dof1D; i3++)
               {
                  // Transfer from the mfem's H1 local numbering to the tensor
                  // structure numbering.
                  const int idx = i3*nH1dof1D*nH1dof1D + i2*nH1dof1D + i1;
                  vecH1[h1dofs[c*nH1dof + dof_map[idx]]] +=
                     HHHx(i1 + i2*nH1dof1D, i3) +
                     HHHy(i1 + i2*nH1dof1D, i3) +
                     HHHz(i1 + i2*nH1dof1D, i3);
               }
            }
         }
      }
   }
}

// Transpose force matrix action on quadrilateral elements in 2D
void ForcePAOperator::MultTransposeQuad(const Vector &vecH1,
                                        Vector &vecL2) const
{
   const int nH1dof1D = tensors1D->HQshape1D.Height(),
             nL2dof1D = tensors1D->LQshape1D.Height(),
             nqp1D    = tensors1D->HQshape1D.Width(),
             nqp      = nqp1D * nqp1D,
             nH1dof   = nH1dof1D * nH1dof1D;
   Array<int> h1dofs, l2dofs;
   Vector v(nH1dof * 2), e(nL2dof1D * nL2dof1D);
   DenseMatrix V, E(e.GetData(), nL2dof1D, nL2dof1D);
   DenseMatrix HQ(nH1dof1D, nqp1D), LQ(nL2dof1D, nqp1D),
               QQc(nqp1D, nqp1D), QQ(nqp1D, nqp1D);
   double *qqc = QQc.GetData();

   const H1_QuadrilateralElement *fe =
      dynamic_cast<const H1_QuadrilateralElement *>(H1FESpace.GetFE(0));
   const Array<int> &dof_map = fe->GetDofMap();

   for (int z = 0; z < nzones; z++)
   {
      H1FESpace.GetElementVDofs(z, h1dofs);

      // Form (stress:grad_v) at all quadrature points.
      QQ = 0.0;
      for (int c = 0; c < 2; c++)
      {
         // Transfer from the mfem's H1 local numbering to the tensor structure
         // numbering.
         for (int j = 0; j < nH1dof; j++)
         {
            v[c*nH1dof + j] = vecH1[h1dofs[c*nH1dof + dof_map[j]]];
         }
         // Connect to [v_c], i.e., the c-component of v.
         V.UseExternalData(v.GetData() + c*nH1dof, nH1dof1D, nH1dof1D);

         // HQ_i2_k1   = V_i1_i2 HQg_i1_k1  -- gradients in x direction.
         // QQc_k1_k2  = HQ_i2_k1 HQs_i2_k2 -- contract  in y direction.
         // QQc_k1_k2 *= stress_k1_k2(c,0)  -- stress that scales d[v_c]_dx.
         MultAtB(V, tensors1D->HQgrad1D, HQ);
         MultAtB(HQ, tensors1D->HQshape1D, QQc);
         double *d = quad_data->stressJinvT(c).GetData() + z*nqp;
         for (int q = 0; q < nqp; q++) { qqc[q] *= d[q]; }
         // Add the (stress(c,0) * d[v_c]_dx) part of (stress:grad_v).
         QQ += QQc;

         // HQ_i2_k1   = V_i1_i2 HQs_i1_k1  -- contract  in x direction.
         // QQc_k1_k2  = HQ_i2_k1 HQg_i2_k2 -- gradients in y direction.
         // QQc_k1_k2 *= stress_k1_k2(c,1)  -- stress that scales d[v_c]_dy.
         MultAtB(V, tensors1D->HQshape1D, HQ);
         MultAtB(HQ, tensors1D->HQgrad1D, QQc);
         d = quad_data->stressJinvT(c).GetData() + 1*nzones*nqp + z*nqp;
         for (int q = 0; q < nqp; q++) { qqc[q] *= d[q]; }
         // Add the (stress(c,1) * d[v_c]_dy) part of (stress:grad_v).
         QQ += QQc;
      }

      // LQ_j1_k2 = LQs_j1_k1 QQ_k1_k2 -- contract in x direction.
      // E_j1_j2  = LQ_j1_k2 LQs_j2_k2 -- contract in y direction.
      mfem::Mult(tensors1D->LQshape1D, QQ, LQ);
      MultABt(LQ, tensors1D->LQshape1D, E);

      L2FESpace.GetElementDofs(z, l2dofs);
      vecL2.SetSubVector(l2dofs, e);
   }
}

// Transpose force matrix action on hexahedral elements in 3D
void ForcePAOperator::MultTransposeHex(const Vector &vecH1, Vector &vecL2) const
{
   const int nH1dof1D = tensors1D->HQshape1D.Height(),
             nL2dof1D = tensors1D->LQshape1D.Height(),
             nqp1D    = tensors1D->HQshape1D.Width(),
             nqp      = nqp1D * nqp1D * nqp1D,
             nH1dof   = nH1dof1D * nH1dof1D * nH1dof1D;
   Array<int> h1dofs, l2dofs;

   Vector v(nH1dof * 3), e(nL2dof1D * nL2dof1D * nL2dof1D);
   DenseMatrix V, E(e.GetData(), nL2dof1D * nL2dof1D, nL2dof1D);

   DenseMatrix HH_Q(nH1dof1D * nH1dof1D, nqp1D),
               H_HQ(HH_Q.GetData(), nH1dof1D, nH1dof1D * nqp1D),
               Q_HQ(nqp1D, nH1dof1D*nqp1D);
   DenseMatrix LL_Q(nL2dof1D * nL2dof1D, nqp1D),
               L_LQ(LL_Q.GetData(), nL2dof1D, nL2dof1D * nqp1D),
               Q_LQ(nqp1D, nL2dof1D*nqp1D);
   DenseMatrix QQ_Q(nqp1D * nqp1D, nqp1D),  QQ_Qc(nqp1D * nqp1D, nqp1D);
   double *qqqc = QQ_Qc.GetData();

   const H1_HexahedronElement *fe =
      dynamic_cast<const H1_HexahedronElement *>(H1FESpace.GetFE(0));
   const Array<int> &dof_map = fe->GetDofMap();

   for (int z = 0; z < nzones; z++)
   {
      H1FESpace.GetElementVDofs(z, h1dofs);

      // Form (stress:grad_v) at all quadrature points.
      QQ_Q = 0.0;
      for (int c = 0; c < 3; c++)
      {
         // Transfer from the mfem's H1 local numbering to the tensor structure
         // numbering.
         for (int j = 0; j < nH1dof; j++)
         {
            v[c*nH1dof + j] = vecH1[h1dofs[c*nH1dof + dof_map[j]]];
         }
         // Connect to [v_c], i.e., the c-component of v.
         V.UseExternalData(v.GetData() + c*nH1dof, nH1dof1D*nH1dof1D, nH1dof1D);

         // HHQ_i1_i2_k3  = V_i1_i2_i3 HQs_i3_k3   -- contract  in z direction.
         // QHQ_k1_i2_k3  = HQg_i1_k1 HHQ_i1_i2_k3 -- gradients in x direction.
         // QQQc_k1_k2_k3 = QHQ_k1_i2_k3 HQs_i2_k2 -- contract  in y direction.
         // The last step does some reordering (it's not product of matrices).
         mfem::Mult(V, tensors1D->HQshape1D, HH_Q);
         MultAtB(tensors1D->HQgrad1D, H_HQ, Q_HQ);
         for (int k1 = 0; k1 < nqp1D; k1++)
         {
            for (int k2 = 0; k2 < nqp1D; k2++)
            {
               for (int k3 = 0; k3 < nqp1D; k3++)
               {
                  QQ_Qc(k1 + nqp1D*k2, k3) = 0.0;
                  for (int i2 = 0; i2 < nH1dof1D; i2++)
                  {
                     QQ_Qc(k1 + nqp1D*k2, k3) += Q_HQ(k1, i2 + k3*nH1dof1D) *
                                                 tensors1D->HQshape1D(i2, k2);
                  }
               }
            }
         }
         // QQQc_k1_k2_k3 *= stress_k1_k2_k3(c,0) -- stress scaling d[v_c]_dx.
         double *d = quad_data->stressJinvT(c).GetData() + z*nqp;
         for (int q = 0; q < nqp; q++) { qqqc[q] *= d[q]; };
         // Add the (stress(c,0) * d[v_c]_dx) part of (stress:grad_v).
         QQ_Q += QQ_Qc;

         // HHQ_i1_i2_k3  = V_i1_i2_i3 HQs_i3_k3   -- contract  in z direction.
         // QHQ_k1_i2_k3  = HQs_i1_k1 HHQ_i1_i2_k3 -- contract  in x direction.
         // QQQc_k1_k2_k3 = QHQ_k1_i2_k3 HQg_i2_k2 -- gradients in y direction.
         // The last step does some reordering (it's not product of matrices).
         mfem::Mult(V, tensors1D->HQshape1D, HH_Q);
         MultAtB(tensors1D->HQshape1D, H_HQ, Q_HQ);
         for (int k1 = 0; k1 < nqp1D; k1++)
         {
            for (int k2 = 0; k2 < nqp1D; k2++)
            {
               for (int k3 = 0; k3 < nqp1D; k3++)
               {
                  QQ_Qc(k1 + nqp1D*k2, k3) = 0.0;
                  for (int i2 = 0; i2 < nH1dof1D; i2++)
                  {
                     QQ_Qc(k1 + nqp1D*k2, k3) += Q_HQ(k1, i2 + k3*nH1dof1D) *
                                                 tensors1D->HQgrad1D(i2, k2);
                  }
               }
            }
         }
         // QQQc_k1_k2_k3 *= stress_k1_k2_k3(c,1) -- stress scaling d[v_c]_dy.
         d = quad_data->stressJinvT(c).GetData() + 1*nzones*nqp + z*nqp;
         for (int q = 0; q < nqp; q++) { qqqc[q] *= d[q]; };
         // Add the (stress(c,1) * d[v_c]_dy) part of (stress:grad_v).
         QQ_Q += QQ_Qc;

         // HHQ_i1_i2_k3  = V_i1_i2_i3 HQg_i3_k3   -- gradients in z direction.
         // QHQ_k1_i2_k3  = HQs_i1_k1 HHQ_i1_i2_k3 -- contract  in x direction.
         // QQQc_k1_k2_k3 = QHQ_k1_i2_k3 HQs_i2_k2 -- contract  in y direction.
         // The last step does some reordering (it's not product of matrices).
         mfem::Mult(V, tensors1D->HQgrad1D, HH_Q);
         MultAtB(tensors1D->HQshape1D, H_HQ, Q_HQ);
         for (int k1 = 0; k1 < nqp1D; k1++)
         {
            for (int k2 = 0; k2 < nqp1D; k2++)
            {
               for (int k3 = 0; k3 < nqp1D; k3++)
               {
                  QQ_Qc(k1 + nqp1D*k2, k3) = 0.0;
                  for (int i2 = 0; i2 < nH1dof1D; i2++)
                  {
                     QQ_Qc(k1 + nqp1D*k2, k3) += Q_HQ(k1, i2 + k3*nH1dof1D) *
                                                 tensors1D->HQshape1D(i2, k2);
                  }
               }
            }
         }
         // QQQc_k1_k2_k3 *= stress_k1_k2_k3(c,2) -- stress scaling d[v_c]_dz.
         d = quad_data->stressJinvT(c).GetData() + 2*nzones*nqp + z*nqp;
         for (int q = 0; q < nqp; q++) { qqqc[q] *= d[q]; };
         // Add the (stress(c,2) * d[v_c]_dz) part of (stress:grad_v).
         QQ_Q += QQ_Qc;
      }

      // QLQ_k1_j2_k3 = QQQ_k1_k2_k3 LQs_j2_k2 -- contract in y direction.
      // This first step does some reordering (it's not product of matrices).
      // LLQ_j1_j2_k3 = LQs_j1_k1 QLQ_k1_j2_k3 -- contract in x direction.
      // E_j1_j2_i3   = LLQ_j1_j2_k3 LQs_j3_k3 -- contract in z direction.
      for (int k1 = 0; k1 < nqp1D; k1++)
      {
         for (int j2 = 0; j2 < nL2dof1D; j2++)
         {
            for (int k3 = 0; k3 < nqp1D; k3++)
            {
               Q_LQ(k1, j2 + nL2dof1D*k3) = 0.0;
               for (int k2 = 0; k2 < nqp1D; k2++)
               {
                  Q_LQ(k1, j2 + nL2dof1D*k3) +=
                     QQ_Q(k1 + nqp1D*k2, k3) * tensors1D->LQshape1D(j2, k2);
               }
            }
         }
      }
      mfem::Mult(tensors1D->LQshape1D, Q_LQ, L_LQ);
      MultABt(LL_Q, tensors1D->LQshape1D, E);

      L2FESpace.GetElementDofs(z, l2dofs);
      vecL2.SetSubVector(l2dofs, e);
   }
}

void MassPAOperator::Mult(const Vector &x, Vector &y) const
{
   if (ess_tdofs)
   {
      Vector xx = x;
      for (int i = 0; i < ess_tdofs->Size(); i++)
      {
         const int idx = (*ess_tdofs)[i];
         xx(idx) = 0.0;
      }
      x_gf.Distribute(xx);
   }
   else { x_gf.Distribute(x); }

   if      (dim == 2) { MultQuad(x_gf, y_gf); }
   else if (dim == 3) { MultHex(x_gf, y_gf); }
   else { MFEM_ABORT("Unsupported dimension"); }
   FESpace.Dof_TrueDof_Matrix()->MultTranspose(y_gf, y);

   if (ess_tdofs)
   {
      for (int i = 0; i < ess_tdofs->Size(); i++)
      {
         const int idx = (*ess_tdofs)[i];
         y(idx) = 0.0;
      }
   }
}

// Mass matrix action on quadrilateral elements in 2D
void MassPAOperator::MultQuad(const Vector &x, Vector &y) const
{
   // Are we working with the velocity or energy mass matrix?
   const H1_QuadrilateralElement *fe_H1 =
      dynamic_cast<const H1_QuadrilateralElement *>(FESpace.GetFE(0));
   const DenseMatrix &DQs = (fe_H1) ? tensors1D->HQshape1D
                            : tensors1D->LQshape1D;

   const int ndof1D = DQs.Height(), nqp1D = DQs.Width();
   DenseMatrix DQ(ndof1D, nqp1D), QQ(nqp1D, nqp1D);
   Vector xz(ndof1D * ndof1D), yz(ndof1D * ndof1D);
   DenseMatrix X(xz.GetData(), ndof1D, ndof1D),
               Y(yz.GetData(), ndof1D, ndof1D);
   Array<int> dofs;
   double *qq = QQ.GetData();
   const int nqp = nqp1D * nqp1D;

   y.SetSize(x.Size());
   y = 0.0;

   for (int z = 0; z < nzones; z++)
   {
      FESpace.GetElementDofs(z, dofs);
      if (fe_H1)
      {
         // Transfer from the mfem's H1 local numbering to the tensor structure
         // numbering.
         const Array<int> &dof_map = fe_H1->GetDofMap();
         for (int j = 0; j < xz.Size(); j++)
         {
            xz[j] = x[dofs[dof_map[j]]];
         }
      }
      else { x.GetSubVector(dofs, xz); }

      // DQ_i1_k2 = X_i1_i2 DQs_i2_k2  -- contract in y direction.
      // QQ_k1_k2 = DQs_i1_k1 DQ_i1_k2 -- contract in x direction.
      mfem::Mult(X, DQs, DQ);
      MultAtB(DQs, DQ, QQ);

      // QQ_k1_k2 *= quad_data_k1_k2 -- scaling with quadrature values.
      double *d = quad_data->rhoDetJw.GetData() + z*nqp;
      for (int q = 0; q < nqp; q++) { qq[q] *= d[q]; }

      // DQ_i1_k2 = DQs_i1_k1 QQ_k1_k2 -- contract in x direction.
      // Y_i1_i2  = DQ_i1_k2 DQs_i2_k2 -- contract in y direction.
      mfem::Mult(DQs, QQ, DQ);
      MultABt(DQ, DQs, Y);

      if (fe_H1)
      {
         const Array<int> &dof_map = fe_H1->GetDofMap();
         for (int j = 0; j < yz.Size(); j++)
         {
            y[dofs[dof_map[j]]] += yz[j];
         }
      }
      else { y.SetSubVector(dofs, yz); }
   }
}

// Mass matrix action on hexahedral elements in 3D
void MassPAOperator::MultHex(const Vector &x, Vector &y) const
{
   // Are we working with the velocity or energy mass matrix?
   const H1_HexahedronElement *fe_H1 =
      dynamic_cast<const H1_HexahedronElement *>(FESpace.GetFE(0));
   const DenseMatrix &DQs = (fe_H1) ? tensors1D->HQshape1D
                            : tensors1D->LQshape1D;

   const int ndof1D = DQs.Height(), nqp1D = DQs.Width();
   DenseMatrix DD_Q(ndof1D * ndof1D, nqp1D);
   DenseMatrix D_DQ(DD_Q.GetData(), ndof1D, ndof1D*nqp1D);
   DenseMatrix Q_DQ(nqp1D, ndof1D*nqp1D);
   DenseMatrix QQ_Q(nqp1D*nqp1D, nqp1D);
   double *qqq = QQ_Q.GetData();
   Vector xz(ndof1D * ndof1D * ndof1D), yz(ndof1D * ndof1D * ndof1D);
   DenseMatrix X(xz.GetData(), ndof1D*ndof1D, ndof1D),
               Y(yz.GetData(), ndof1D*ndof1D, ndof1D);
   const int nqp = nqp1D * nqp1D * nqp1D;
   Array<int> dofs;

   y.SetSize(x.Size());
   y = 0.0;

   for (int z = 0; z < nzones; z++)
   {
      FESpace.GetElementDofs(z, dofs);
      if (fe_H1)
      {
         // Transfer from the mfem's H1 local numbering to the tensor structure
         // numbering.
         const Array<int> &dof_map = fe_H1->GetDofMap();
         for (int j = 0; j < xz.Size(); j++)
         {
            xz[j] = x[dofs[dof_map[j]]];
         }
      }
      else { x.GetSubVector(dofs, xz); }

      // DDQ_i1_i2_k3  = X_i1_i2_i3 DQs_i3_k3   -- contract in z direction.
      // QDQ_k1_i2_k3  = DQs_i1_k1 DDQ_i1_i2_k3 -- contract in x direction.
      // QQQ_k1_k2_k3  = QDQ_k1_i2_k3 DQs_i2_k2 -- contract in y direction.
      // The last step does some reordering (it's not product of matrices).
      mfem::Mult(X, DQs, DD_Q);
      MultAtB(DQs, D_DQ, Q_DQ);
      for (int k1 = 0; k1 < nqp1D; k1++)
      {
         for (int k2 = 0; k2 < nqp1D; k2++)
         {
            for (int k3 = 0; k3 < nqp1D; k3++)
            {
               QQ_Q(k1 + nqp1D*k2, k3) = 0.0;
               for (int i2 = 0; i2 < ndof1D; i2++)
               {
                  QQ_Q(k1 + nqp1D*k2, k3) +=
                     Q_DQ(k1, i2 + k3*ndof1D) * DQs(i2, k2);
               }
            }
         }
      }

      // QQQ_k1_k2_k3 *= quad_data_k1_k2_k3 -- scaling with quadrature values.
      double *d = quad_data->rhoDetJw.GetData() + z*nqp;
      for (int q = 0; q < nqp; q++) { qqq[q] *= d[q]; }

      // QDQ_k1_i2_k3 = QQQ_k1_k2_k3 DQs_i2_k2 -- contract in y direction.
      // This first step does some reordering (it's not product of matrices).
      // DDQ_i1_i2_k3 = DQs_i1_k1 QDQ_k1_i2_k3 -- contract in x direction.
      // Y_i1_i2_i3   = DDQ_i1_i2_k3 DQs_i3_k3 -- contract in z direction.
      for (int k1 = 0; k1 < nqp1D; k1++)
      {
         for (int i2 = 0; i2 < ndof1D; i2++)
         {
            for (int k3 = 0; k3 < nqp1D; k3++)
            {
               Q_DQ(k1, i2 + ndof1D*k3) = 0.0;
               for (int k2 = 0; k2 < nqp1D; k2++)
               {
                  Q_DQ(k1, i2 + ndof1D*k3) +=
                     QQ_Q(k1 + nqp1D*k2, k3) * DQs(i2, k2);
               }
            }
         }
      }
      mfem::Mult(DQs, Q_DQ, D_DQ);
      MultABt(DD_Q, DQs, Y);

      if (fe_H1)
      {
         const Array<int> &dof_map = fe_H1->GetDofMap();
         for (int j = 0; j < yz.Size(); j++)
         {
            y[dofs[dof_map[j]]] += yz[j];
         }
      }
      else { y.SetSubVector(dofs, yz); }
   }
}

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI
