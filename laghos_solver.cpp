// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-XXXXXX. All Rights
// reserved. See file LICENSE for details.
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
// testbed platforms, in support of the nation’s exascale computing imperative.

#include "laghos_solver.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{

namespace miniapps
{

void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    ParGridFunction &gf, const char *title,
                    int x, int y, int w, int h, bool vec)
{
   ParMesh &pmesh = *gf.ParFESpace()->GetParMesh();
   MPI_Comm comm = pmesh.GetComm();

   int num_procs, myid;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   bool newly_opened = false;
   int connection_failed;

   do
   {
      if (myid == 0)
      {
         if (!sock.is_open() || !sock)
         {
            sock.open(vishost, visport);
            sock.precision(8);
            newly_opened = true;
         }
         sock << "solution\n";
      }

      pmesh.PrintAsOne(sock);
      gf.SaveAsOne(sock);

      if (myid == 0 && newly_opened)
      {
         sock << "window_title '" << title << "'\n"
              << "window_geometry "
              << x << " " << y << " " << w << " " << h << "\n"
              << "keys maaAc";
         if ( vec ) { sock << "vvv"; }
         sock << endl;
      }

      if (myid == 0)
      {
         connection_failed = !sock && !newly_opened;
      }
      MPI_Bcast(&connection_failed, 1, MPI_INT, 0, comm);
   }
   while (connection_failed);
}

} // namespace miniapps

namespace hydrodynamics
{

const Tensors1D *tensors1D;

Tensors1D::Tensors1D(int H1order, int L2order, int nqp1D)
   : HQshape1D(H1order + 1, nqp1D),
     HQgrad1D(H1order + 1, nqp1D),
     LQshape1D(L2order + 1, nqp1D)
{
   // In this miniapp we assume:
   //   Gauss-Legendre quadrature points.
   //   Gauss-Lobatto kinematic basis.
   //   Bernstein thermodynamic basis.

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

ForcePAOperator::ForcePAOperator(QuadratureData *quad_data_,
                                 ParFiniteElementSpace &h1fes,
                                 ParFiniteElementSpace &l2fes)
   : dim(h1fes.GetMesh()->Dimension()),
     nzones(h1fes.GetMesh()->GetNE()),
     quad_data(quad_data_),
     H1FESpace(h1fes), L2FESpace(l2fes)
{ }

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

void ForcePAOperator::MultTransposeQuad(const Vector &vecH1, Vector &vecL2) const
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
      xg.Distribute(xx);
   }
   else { xg.Distribute(x); }

   if      (dim == 2) { MultQuad(xg, yg); }
   else if (dim == 3) { MultHex(xg, yg); }
   else { MFEM_ABORT("Unsupported dimension"); }
   FESpace.Dof_TrueDof_Matrix()->MultTranspose(yg, y);

   if (ess_tdofs)
   {
      for (int i = 0; i < ess_tdofs->Size(); i++)
      {
         const int idx = (*ess_tdofs)[i];
         y(idx) = 0.0;
      }
   }
}

MassPAOperator::MassPAOperator(QuadratureData *quad_data_,
                               ParFiniteElementSpace &fes)
   : Operator(fes.TrueVSize()),
     dim(fes.GetMesh()->Dimension()),
     nzones(fes.GetMesh()->GetNE()),
     quad_data(quad_data_),
     FESpace(fes), ess_tdofs(NULL), xg(&fes), yg(&fes)
{ }

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

LagrangianHydroOperator::LagrangianHydroOperator(int size,
                                                 ParFiniteElementSpace &h1_fes,
                                                 ParFiniteElementSpace &l2_fes,
                                                 Array<int> &essential_tdofs,
                                                 ParGridFunction &rho0,
                                                 int source_type_, double cfl_,
                                                 double gamma_, bool visc,
                                                 bool pa)
   : TimeDependentOperator(size),
     H1FESpace(h1_fes), L2FESpace(l2_fes),
     H1compFESpace(h1_fes.GetParMesh(), h1_fes.FEColl(), 1),
     ess_tdofs(essential_tdofs),
     dim(h1_fes.GetMesh()->Dimension()),
     zones_cnt(h1_fes.GetMesh()->GetNE()),
     l2dofs_cnt(l2_fes.GetFE(0)->GetDof()),
     h1dofs_cnt(h1_fes.GetFE(0)->GetDof()),
     source_type(source_type_), cfl(cfl_), gamma(gamma_),
     use_viscosity(visc), p_assembly(pa),
     Mv(&h1_fes), Me_inv(l2dofs_cnt, l2dofs_cnt, zones_cnt),
     integ_rule(IntRules.Get(h1_fes.GetMesh()->GetElementBaseGeometry(),
                             3*h1_fes.GetOrder(0) + l2_fes.GetOrder(0) - 1)),
     quad_data(dim, zones_cnt, integ_rule.GetNPoints()),
     quad_data_is_current(false),
     Force(&l2_fes, &h1_fes), ForcePA(&quad_data, h1_fes, l2_fes)
{
   GridFunctionCoefficient rho_coeff(&rho0);

   // Standard local assembly and inversion for energy mass matrices.
   DenseMatrix Me(l2dofs_cnt);
   DenseMatrixInverse inv(&Me);
   MassIntegrator mi(rho_coeff, &integ_rule);
   for (int i = 0; i < zones_cnt; i++)
   {
      mi.AssembleElementMatrix(*l2_fes.GetFE(i),
                               *l2_fes.GetElementTransformation(i), Me);
      inv.Factor();
      inv.GetInverseMatrix(Me_inv(i));
   }

   // Standard assembly for the velocity mass matrix.
   VectorMassIntegrator *vmi = new VectorMassIntegrator(rho_coeff, &integ_rule);
   Mv.AddDomainIntegrator(vmi);
   Mv.Assemble();

   // Values of rho0DetJ0 and Jac0inv at all quadrature points.
   const int nqp = integ_rule.GetNPoints();
   Vector rho_vals(nqp);
   for (int i = 0; i < zones_cnt; i++)
   {
      rho0.GetValues(i, integ_rule, rho_vals);
      ElementTransformation *T = h1_fes.GetElementTransformation(i);
      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = integ_rule.IntPoint(q);
         T->SetIntPoint(&ip);

         DenseMatrixInverse Jinv(T->Jacobian());
         Jinv.GetInverseMatrix(quad_data.Jac0inv(i*nqp + q));

         const double rho0DetJ0 = T->Weight() * rho_vals(q);
         quad_data.rho0DetJ0(i*nqp + q) = rho0DetJ0;
         quad_data.rhoDetJw(i*nqp + q)  = rho0DetJ0 *
                                          integ_rule.IntPoint(q).weight;
      }
   }

   // Initial local mesh size (assumes similar cells).
   double loc_area = 0.0, glob_area;
   int glob_z_cnt;
   ParMesh *pm = H1FESpace.GetParMesh();
   for (int i = 0; i < zones_cnt; i++) { loc_area += pm->GetElementVolume(i); }
   MPI_Allreduce(&loc_area, &glob_area, 1, MPI_DOUBLE, MPI_SUM, pm->GetComm());
   MPI_Allreduce(&zones_cnt, &glob_z_cnt, 1, MPI_INT, MPI_SUM, pm->GetComm());
   switch (pm->GetElementBaseGeometry(0))
   {
      case Geometry::SQUARE:
         quad_data.h0 = sqrt(glob_area / glob_z_cnt); break;
      case Geometry::TRIANGLE:
         quad_data.h0 = sqrt(2.0 * glob_area / glob_z_cnt); break;
      case Geometry::CUBE:
         quad_data.h0 = pow(glob_area / glob_z_cnt, 1.0/3.0); break;
      case Geometry::TETRAHEDRON:
         quad_data.h0 = pow(6.0 * glob_area / glob_z_cnt, 1.0/3.0); break;
      default: MFEM_ABORT("Unknown zone type!");
   }
   quad_data.h0 /= (double) H1FESpace.GetOrder(0);

   ForceIntegrator *fi = new ForceIntegrator(quad_data);
   fi->SetIntRule(&integ_rule);
   Force.AddDomainIntegrator(fi);
   // Make a dummy assembly to figure out the sparsity.
   for (int i = 0; i < zones_cnt * nqp; i++)
   { quad_data.Jac(i).Diag(1.0, dim); }
   Force.Assemble(0);
   Force.Finalize(0);

   tensors1D = new Tensors1D(H1FESpace.GetFE(0)->GetOrder(),
                             L2FESpace.GetFE(0)->GetOrder(),
                             int(floor(0.7 + pow(nqp, 1.0 / dim))));
}

void LagrangianHydroOperator::Mult(const Vector &S, Vector &dS_dt) const
{
   dS_dt = 0.0;

   // Make sure that the mesh positions correspond to the ones in S. This is
   // needed only because some mfem time integrators don't update the solution
   // vector at every intermediate stage (hence they don't change the mesh).
   Vector* sptr = (Vector*) &S;
   ParGridFunction x;
   x.MakeRef(&H1FESpace, *sptr, 0);
   H1FESpace.GetParMesh()->NewNodes(x, false);

   UpdateQuadratureData(S);

   // The big BlockVector stores the fields as follows:
   //    Position
   //    Velocity
   //    Specific Internal Energy

   const int Vsize_l2 = L2FESpace.GetVSize();
   const int Vsize_h1 = H1FESpace.GetVSize();

   ParGridFunction v, e;
   v.MakeRef(&H1FESpace, *sptr, Vsize_h1);
   e.MakeRef(&L2FESpace, *sptr, Vsize_h1*2);

   ParGridFunction dx, dv, de;
   dx.MakeRef(&H1FESpace, dS_dt, 0);
   dv.MakeRef(&H1FESpace, dS_dt, Vsize_h1);
   de.MakeRef(&L2FESpace, dS_dt, Vsize_h1*2);

   // Set dx_dt = v (explicit);
   dx = v;

   if (!p_assembly)
   {
      Force = 0.0;
      Force.Assemble();
   }

   // Solve for velocity.
   Vector one(Vsize_l2), rhs(Vsize_h1), B, X; one = 1.0;
   if (p_assembly)
   {
      ForcePA.Mult(one, rhs); rhs.Neg();

      // Partial assembly solve for each velocity component.
      MassPAOperator VMassPA(&quad_data, H1compFESpace);
      const int size = H1compFESpace.GetVSize();
      for (int c = 0; c < dim; c++)
      {
         Vector rhs_c(rhs.GetData() + c*size, size),
                dv_c(dv.GetData() + c*size, size);

         Array<int> c_tdofs;
         Array<int> ess_bdr(H1FESpace.GetParMesh()->bdr_attributes.Max());
         // Attributes 1/2/3 correspond to fixed-x/y/z boundaries, i.e.,
         // we must enforce v_x/y/z = 0 for the velocity components.
         ess_bdr = 0; ess_bdr[c] = 1;
         // True dofs as if there's only one component.
         H1compFESpace.GetEssentialTrueDofs(ess_bdr, c_tdofs);

         dv_c = 0.0;
         Vector B(H1compFESpace.TrueVSize()), X(H1compFESpace.TrueVSize());
         H1compFESpace.Dof_TrueDof_Matrix()->MultTranspose(rhs_c, B);
         H1compFESpace.GetRestrictionMatrix()->Mult(dv_c, X);

         VMassPA.EliminateRHS(c_tdofs, B);

         CGSolver cg(H1FESpace.GetParMesh()->GetComm());
         cg.SetOperator(VMassPA);
         cg.SetRelTol(1e-8);
         cg.SetAbsTol(0.0);
         cg.SetMaxIter(200);
         cg.SetPrintLevel(0);
         cg.Mult(B, X);
         H1compFESpace.Dof_TrueDof_Matrix()->Mult(X, dv_c);
      }
   }
   else
   {
      Force.Mult(one, rhs); rhs.Neg();
      HypreParMatrix A;
      dv = 0.0;
      Mv.FormLinearSystem(ess_tdofs, dv, rhs, A, X, B);
      CGSolver cg(H1FESpace.GetParMesh()->GetComm());
      cg.SetOperator(A);
      cg.SetRelTol(1e-8); cg.SetAbsTol(0.0);
      cg.SetMaxIter(200);
      cg.SetPrintLevel(0);
      cg.Mult(B, X);
      Mv.RecoverFEMSolution(X, rhs, dv);
   }

   // Solve for energy, assemble the energy source if such exists.
   LinearForm *e_source = NULL;
   if (source_type == 1) // 2D Taylor-Green.
   {
      e_source = new LinearForm(&L2FESpace);
      TaylorCoefficient coeff;
      DomainLFIntegrator *d = new DomainLFIntegrator(coeff, &integ_rule);
      e_source->AddDomainIntegrator(d);
      e_source->Assemble();
   }
   if (p_assembly)
   {
      Vector rhs(Vsize_l2);
      ForcePA.MultTranspose(v, rhs);

      if (e_source) { rhs += *e_source; }

      MassPAOperator EMassPA(&quad_data, L2FESpace);
      CGSolver cg(L2FESpace.GetParMesh()->GetComm());
      cg.SetOperator(EMassPA);
      cg.SetRelTol(1e-8);
      cg.SetAbsTol(0.0);
      cg.SetMaxIter(200);
      cg.SetPrintLevel(0);
      cg.Mult(rhs, de);
   }
   else
   {
      Array<int> l2dofs, h1dofs;
      DenseMatrix loc_Force(h1dofs_cnt * dim, l2dofs_cnt);
      Vector v_vals(h1dofs_cnt * dim), e_rhs(l2dofs_cnt), de_loc(l2dofs_cnt);
      for (int i = 0; i < zones_cnt; i++)
      {
         H1FESpace.GetElementVDofs(i, h1dofs);
         L2FESpace.GetElementDofs(i, l2dofs);
         Force.SpMat().GetSubMatrix(h1dofs, l2dofs, loc_Force);
         v.GetSubVector(h1dofs, v_vals);

         loc_Force.MultTranspose(v_vals, e_rhs);
         if (e_source)
         {
            e_source->GetSubVector(l2dofs, de_loc); // Use de_loc as temporary.
            e_rhs += de_loc;
         }
         Me_inv(i).Mult(e_rhs, de_loc);
         de.SetSubVector(l2dofs, de_loc);
      }
   }

   delete e_source;
   quad_data_is_current = false;
}

double LagrangianHydroOperator::GetTimeStepEstimate(const Vector &S) const
{
   Vector* sptr = (Vector*) &S;
   ParGridFunction x;
   x.MakeRef(&H1FESpace, *sptr, 0);
   H1FESpace.GetParMesh()->NewNodes(x, false);
   UpdateQuadratureData(S);

   double glob_dt_est;
   MPI_Allreduce(&quad_data.dt_est, &glob_dt_est, 1, MPI_DOUBLE, MPI_MIN,
                 H1FESpace.GetParMesh()->GetComm());
   return glob_dt_est;
}

void LagrangianHydroOperator::ResetTimeStepEstimate() const
{
   quad_data.dt_est = numeric_limits<double>::infinity();
}

void LagrangianHydroOperator::ComputeDensity(ParGridFunction &rho)
{
   rho.SetSpace(&L2FESpace);

   DenseMatrix Mrho(l2dofs_cnt);
   Vector rhs(l2dofs_cnt), rho_z(l2dofs_cnt);
   Array<int> dofs(l2dofs_cnt);
   DenseMatrixInverse inv(&Mrho);
   MassIntegrator mi(&integ_rule);
   DensityIntegrator di(quad_data);
   di.SetIntRule(&integ_rule);
   for (int i = 0; i < zones_cnt; i++)
   {
      di.AssembleRHSElementVect(*L2FESpace.GetFE(i),
                                *L2FESpace.GetElementTransformation(i), rhs);
      mi.AssembleElementMatrix(*L2FESpace.GetFE(i),
                               *L2FESpace.GetElementTransformation(i), Mrho);
      inv.Factor();
      inv.Mult(rhs, rho_z);
      L2FESpace.GetElementDofs(i, dofs);
      rho.SetSubVector(dofs, rho_z);
   }
}

LagrangianHydroOperator::~LagrangianHydroOperator()
{
   delete tensors1D;
}

void LagrangianHydroOperator::UpdateQuadratureData(const Vector &S) const
{
   if (quad_data_is_current) { return; }

   const int dim = H1FESpace.GetParMesh()->Dimension();
   const int nqp = integ_rule.GetNPoints();

   ParGridFunction e, v;
   Vector* sptr = (Vector*) &S;
   v.MakeRef(&H1FESpace, *sptr, H1FESpace.GetVSize());
   e.MakeRef(&L2FESpace, *sptr, 2*H1FESpace.GetVSize());
   Vector e_vals;
   DenseMatrix Jpi(dim), sgrad_v(dim), Jinv(dim), stressJiT(dim);
   DenseMatrix v_vals;

   for (int i = 0; i < zones_cnt; i++)
   {
      ElementTransformation *T = H1FESpace.GetElementTransformation(i);
      e.GetValues(i, integ_rule, e_vals);
      v.GetVectorValues(*T, integ_rule, v_vals);
      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = integ_rule.IntPoint(q);
         T->SetIntPoint(&ip);
         const DenseMatrix &Jpr = T->Jacobian();

         quad_data.Jac(i*nqp + q) = Jpr;
         const double detJ = T->Weight();
         MFEM_VERIFY(detJ > 0.0, "Bad Jacobian determinant: " << detJ);

         DenseMatrix &s = quad_data.stress(i*nqp + q);
         s = 0.0;
         const double rho = quad_data.rho0DetJ0(i*nqp + q) / detJ;
         const double e   = max(0.0, e_vals(q));
         for (int d = 0; d < dim; d++)
         {
            s(d, d) = - (gamma - 1.0) * rho * e;
         }

         // Length scale at the point. The first eigenvector of the symmetric
         // velocity gradient gives the direction of maximal compression. This
         // is used to define the relative change of the initial length scale.
         v.GetVectorGradient(*T, sgrad_v);
         sgrad_v.Symmetrize();
         double eig_val_data[3], eig_vec_data[9];
         sgrad_v.CalcEigenvalues(eig_val_data, eig_vec_data);
         Vector compr_dir(eig_vec_data, dim);
         // Computes the initial->physical transformation Jacobian.
         mfem::Mult(Jpr, quad_data.Jac0inv(i*nqp + q), Jpi);
         Vector ph_dir(dim); Jpi.Mult(compr_dir, ph_dir);
         // Change of the initial mesh size in the compression direction.
         const double h = quad_data.h0 * ph_dir.Norml2() / (compr_dir.Norml2());

         // Time step estimate at the point.
         const double sound_speed = sqrt(gamma * (gamma-1.0) * e);
         quad_data.dt_est = min(quad_data.dt_est, cfl * h / sound_speed);

         if (use_viscosity)
         {
            // Measure of maximal compression.
            const double mu = eig_val_data[0];
            double visc_coeff = 2.0 * rho * h * h * fabs(mu);
            if (mu < 0.0) { visc_coeff += 0.5 * rho * h * sound_speed; }
            s.Add(visc_coeff, sgrad_v);
         }

         // Quadrature data for partial assembly of the force operator.
         CalcInverse(Jpr, Jinv);
         MultABt(s, Jinv, stressJiT);
         stressJiT *= integ_rule.IntPoint(q).weight * detJ;
         for (int vd = 0 ; vd < dim; vd++)
         {
            for (int gd = 0; gd < dim; gd++)
            {
              quad_data.stressJinvT(vd)(i*nqp + q, gd) = stressJiT(vd, gd);
            }
         }
      }
   }

   quad_data_is_current = true;
}

} // namespace electromagnetics

} // namespace mfem

#endif // MFEM_USE_MPI
