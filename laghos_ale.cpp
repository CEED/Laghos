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

#include "laghos_ale.hpp"
#include "laghos_assembly.hpp"

using namespace std;
namespace mfem
{

namespace hydrodynamics
{

RemapAdvector::RemapAdvector(const ParMesh &m, int order_v, int order_e)
   : pmesh(m, true), dim(pmesh.Dimension()),
     fec_L2(order_e, pmesh.Dimension(), BasisType::Positive),
     fec_H1(order_v, pmesh.Dimension()),
     pfes_L2(&pmesh, &fec_L2, 1),
     pfes_H1(&pmesh, &fec_H1, pmesh.Dimension()),
     offsets(5), S(),
     d(), v(), rho(), e()
{
   const int vsize_H1 = pfes_H1.GetVSize(), vsize_L2 = pfes_L2.GetVSize();

   // Arrangement: distance (dim), velocity (dim), density (1), energy (1).
   offsets[0] = 0;
   offsets[1] = vsize_H1;
   offsets[2] = 2 * vsize_H1;
   offsets[3] = 2 * vsize_H1 + vsize_L2;
   offsets[4] = 2 * vsize_H1 + vsize_L2;
   S.Update(offsets);

   d.MakeRef(&pfes_H1, S, offsets[0]);
   v.MakeRef(&pfes_H1, S, offsets[1]);
   rho.MakeRef(&pfes_L2, S, offsets[2]);
   e.MakeRef(&pfes_L2, S, offsets[3]);
}

void RemapAdvector::InitFromLagr(const Vector &nodes0,
                                 const ParGridFunction &dist,
                                 const ParGridFunction &vel,
                                 const IntegrationRule &rho_ir,
                                 const Vector &rhoDetJw)
{
   x0 = nodes0;
   d  = dist;
   v  = vel;

   SolutionMover mover(rho_ir);
   mover.MoveDensityLR(rhoDetJw, rho);
}

void RemapAdvector::ComputeAtNewPosition(const Vector &new_nodes)
{
   const int vsize_H1 = pfes_H1.GetVSize();

   // This will be used to move the positions.
   GridFunction *x = pmesh.GetNodes();
   *x = x0;

   // Velocity of the positions.
   GridFunction u(x->FESpace());
   subtract(new_nodes, x0, u);

   // Define a scalar FE space for the solution, and the advection operator.
   ParFiniteElementSpace pfes_H1(&pmesh, &fec_H1, 1);
   ParFiniteElementSpace pfes_L2(&pmesh, &fec_L2, 1);
   AdvectorOper oper(S.Size(), x0, u, pfes_H1, pfes_L2);
   ode_solver.Init(oper);

   // Compute some time step [mesh_size / speed].
   double h_min = std::numeric_limits<double>::infinity();
   for (int i = 0; i < pmesh.GetNE(); i++)
   {
      h_min = std::min(h_min, pmesh.GetElementSize(i));
   }
   double v_max = 0.0;
   const int s = vsize_H1 / dim;

   for (int i = 0; i < s; i++)
   {
      double vel = 0.;
      for (int j = 0; j < dim; j++)
      {
         vel += u(i+j*s)*u(i+j*s);
      }
      v_max = std::max(v_max, vel);
   }

   double v_loc = v_max, h_loc = h_min;
   MPI_Allreduce(&v_loc, &v_max, 1, MPI_DOUBLE, MPI_MAX, pfes_H1.GetComm());
   MPI_Allreduce(&h_loc, &h_min, 1, MPI_DOUBLE, MPI_MIN, pfes_H1.GetComm());

   if (v_max == 0.0) // No need to change the field.
   {
      return;
   }

   v_max = std::sqrt(v_max);
   double dt = 0.5 * h_min / v_max;

   double t = 0.0;
   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= 1.0)
      {
         dt = 1.0 - t;
         last_step = true;
      }
      ode_solver.Step(S, t, dt);
   }
}

void RemapAdvector::TransferToLagr(ParGridFunction &dist,
                                   ParGridFunction &vel,
                                   const IntegrationRule &rho_ir,
                                   Vector &rhoDetJw)
{
   dist = d;
   vel  = v;
}

AdvectorOper::AdvectorOper(int size, const Vector &x_start,
                           GridFunction &velocity,
                           ParFiniteElementSpace &pfes_H1,
                           ParFiniteElementSpace &pfes_L2)
   : TimeDependentOperator(size),
     x0(x_start), x_now(*pfes_H1.GetMesh()->GetNodes()),
     u(velocity), u_coeff(&u),
     M(&pfes_H1), K(&pfes_H1),
     M_L2(&pfes_L2), K_L2(&pfes_L2)
{
   M.AddDomainIntegrator(new MassIntegrator);
   M.Assemble(0);
   M.Finalize(0);

   K.AddDomainIntegrator(new ConvectionIntegrator(u_coeff));
   K.Assemble(0);
   K.Finalize(0);

   M_L2.AddDomainIntegrator(new MassIntegrator);
   M_L2.Assemble(0);
   M_L2.Finalize(0);

   K_L2.AddDomainIntegrator(new ConvectionIntegrator(u_coeff));
   DGTraceIntegrator *dgt_i = new DGTraceIntegrator(u_coeff, -1.0, -0.5);
   DGTraceIntegrator *dgt_b = new DGTraceIntegrator(u_coeff, -1.0, -0.5);
   K_L2.AddInteriorFaceIntegrator(new TransposeIntegrator(dgt_i));
   K_L2.AddBdrFaceIntegrator(new TransposeIntegrator(dgt_b));
   K_L2.KeepNbrBlock(true);
   K_L2.Assemble(0);
   K_L2.Finalize(0);
}

void AdvectorOper::Mult(const Vector &U, Vector &dU) const
{
   ParFiniteElementSpace &pfes_H1 = *K.ParFESpace(),
                         &pfes_L2 = *K_L2.ParFESpace();
   const int dim     = pfes_H1.GetMesh()->Dimension();
   const int size_H1 = pfes_H1.GetVSize(), size_L2 = pfes_L2.GetVSize();

   // Move the mesh.
   const double t = GetTime();
   add(x0, t, u, x_now);

   dU = 0.0;

   // Assemble on the H1 forms on the new mesh.
   K.BilinearForm::operator=(0.0);
   K.Assemble();
   M.BilinearForm::operator=(0.0);
   M.Assemble();
   OperatorHandle Mop;
   Mop.Reset(M.ParallelAssemble());

   // Arrangement: distance (dim), velocity (dim), density (1), energy (1).
   Vector *Uptr = const_cast<Vector *>(&U);

   HypreSmoother prec;
   prec.SetType(HypreSmoother::Jacobi, 1);
   CGSolver lin_solver(M.ParFESpace()->GetParMesh()->GetComm());
   lin_solver.SetPreconditioner(prec);
   lin_solver.SetOperator(*Mop);
   lin_solver.SetRelTol(1e-8);
   lin_solver.SetAbsTol(0.0);
   lin_solver.SetMaxIter(100);
   lin_solver.SetPrintLevel(0);

   const Operator *P_H1 = pfes_H1.GetProlongationMatrix();
   Vector dist, d_dist, v, d_v;
   Vector rhs_H1(size_H1),
          RHS_H1(P_H1->Width()),
          X(P_H1->Width());
   for (int d = 0; d < dim; d++)
   {
      // Distance component.
      dist.MakeRef(*Uptr, d * size_H1, size_H1);
      d_dist.MakeRef(dU,  d * size_H1, size_H1);
      K.Mult(dist, rhs_H1);
      P_H1->MultTranspose(rhs_H1, RHS_H1);
      X = 0.0;
      lin_solver.Mult(RHS_H1, X);
      P_H1->Mult(X, d_dist);

      // Velocity component.
      v.MakeRef(*Uptr, (dim + d) * size_H1, size_H1);
      d_v.MakeRef(dU,  (dim + d) * size_H1, size_H1);
      K.Mult(v, rhs_H1);
      P_H1->MultTranspose(rhs_H1, RHS_H1);
      X = 0.0;
      lin_solver.Mult(RHS_H1, X);
      P_H1->Mult(X, d_v);
   }

   // Assemble the L2 forms on the new mesh.
   K_L2.BilinearForm::operator=(0.0);
   K_L2.Assemble();
   M_L2.BilinearForm::operator=(0.0);
   M_L2.Assemble();

   // Density remap.
   Vector rho, d_rho;
   rho.MakeRef(*Uptr, 2 * dim * size_H1, size_L2);
   d_rho.MakeRef(dU,  2 * dim * size_H1, size_L2);
   Vector lumpedM; M_L2.SpMat().GetDiag(lumpedM);
   DiscreteUpwindLOSolver lo_solver(pfes_L2, K_L2.SpMat(), lumpedM);
   lo_solver.CalcLOSolution(rho, d_rho);
}

void SolutionMover::MoveDensityLR(const Vector &quad_rho, ParGridFunction &rho)
{
   ParMesh &pmesh = *rho.ParFESpace()->GetParMesh();
   L2_FECollection fec0(0, pmesh.Dimension());
   ParFiniteElementSpace pfes0(&pmesh, &fec0);
   ParGridFunction rho_min(&pfes0), rho_max(&pfes0);

   // Local max / min.
   const int NE = pmesh.GetNE(), nqp = ir_rho.GetNPoints();
   for (int k = 0; k < NE; k++)
   {
      ElementTransformation &T = *pmesh.GetElementTransformation(k);
      rho_min(k) =   std::numeric_limits<double>::infinity();
      rho_max(k) = - std::numeric_limits<double>::infinity();

      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir_rho.IntPoint(q);
         T.SetIntPoint(&ip);
         const double detJ = T.Jacobian().Det();
         const double rho = quad_rho(k * nqp + q) / detJ / ip.weight;
         rho_min(k) = min(rho_min(k), rho);
         rho_max(k) = max(rho_max(k), rho);
      }
   }

   // One-level face neighbors max / min.
   rho_min.ExchangeFaceNbrData();
   rho_max.ExchangeFaceNbrData();
   const Vector &rho_min_nbr = rho_min.FaceNbrData(),
                &rho_max_nbr = rho_max.FaceNbrData();
   const Table &el_to_el = pmesh.ElementToElementTable();
   Array<int> face_nbr_el;
   for (int k = 0; k < NE; k++)
   {
      el_to_el.GetRow(k, face_nbr_el);
      for (int n = 0; n < face_nbr_el.Size(); n++)
      {
         if (face_nbr_el[n] < NE)
         {
            // Local neighbor.
            rho_min(k) = min(rho_min(k), rho_min(face_nbr_el[n]));
            rho_max(k) = max(rho_max(k), rho_max(face_nbr_el[n]));
         }
         else
         {
            // MPI face neighbor.
            rho_min(k) = min(rho_min(k), rho_min_nbr(face_nbr_el[n] - NE));
            rho_max(k) = max(rho_max(k), rho_max_nbr(face_nbr_el[n] - NE));
         }
      }
   }

   // HO solution - FCT_Project.
   const int dof_cnt = rho.Size() / NE;
   DenseMatrix M(dof_cnt), F(dof_cnt);
   Vector rhs(dof_cnt), rho_HO(dof_cnt), rho_z(dof_cnt), ML(dof_cnt),
          beta(dof_cnt), z(dof_cnt), gp(dof_cnt), gm(dof_cnt);
   Array<int> dofs(dof_cnt);
   DenseMatrixInverse inv(&M);
   MassIntegrator mi(&ir_rho);
   DensityIntegrator di(quad_rho);
   di.SetIntRule(&ir_rho);
   for (int k = 0; k < NE; k++)
   {
      const FiniteElement &fe = *rho.ParFESpace()->GetFE(k);
      ElementTransformation &T = *pmesh.GetElementTransformation(k);
      di.AssembleRHSElementVect(fe, T, rhs);
      mi.AssembleElementMatrix(fe, T, M);
      M.GetRowSums(ML);

      inv.Factor();
      inv.Mult(rhs, rho_HO);

      const double rho_avg = rhs.Sum() / ML.Sum();

      beta = ML;
      beta /= beta.Sum();

      // The low order flux correction.
      for (int i = 0; i < dof_cnt; i++) { z(i) = rhs(i) - ML(i) * rho_avg; }

      for (int i = 1; i < dof_cnt; i++)
      {
         for (int j = 0; j < i; j++)
         {
            F(i, j) = M(i, j) * (rho_HO(i) - rho_HO(j)) +
                      (beta(j) * z(i) - beta(i) * z(j));
         }
      }

      gp = 0.0;
      gm = 0.0;
      for (int i = 1; i < dof_cnt; i++)
      {
         for (int j = 0; j < i; j++)
         {
            double fij = F(i, j);
            if (fij >= 0.0)
            {
               gp(i) += fij;
               gm(j) -= fij;
            }
            else
            {
               gm(i) += fij;
               gp(j) -= fij;
            }
         }
      }

      rho_z = rho_avg;

      for (int i = 0; i < dof_cnt; i++)
      {
         double rp = max(ML(i) * (rho_max(k) - rho_z(i)), 0.0);
         double rm = min(ML(i) * (rho_min(k) - rho_z(i)), 0.0);
         double sp = gp(i), sm = gm(i);

         gp(i) = (rp < sp) ? rp / sp : 1.0;
         gm(i) = (rm > sm) ? rm / sm : 1.0;
      }

      for (int i = 1; i < dof_cnt; i++)
      {
         for (int j = 0; j < i; j++)
         {
            double fij = F(i, j), aij;

            if (fij >= 0.0)
            {
               aij = min(gp(i), gm(j));
            }
            else
            {
               aij = min(gm(i), gp(j));
            }

            fij *= aij;
            rho_z(i) += fij / ML(i);
            rho_z(j) -= fij / ML(j);
         }
      }

      rho.ParFESpace()->GetElementDofs(k, dofs);
      rho.SetSubVector(dofs, rho_z);
   }
}

void LocalInverseHOSolver::CalcHOSolution(const Vector &u, Vector &du) const
{
   ParFiniteElementSpace &pfes = *M.ParFESpace();
   Vector rhs(u.Size());
   HypreParMatrix *K_mat = K.ParallelAssemble(&K.SpMat());
   K_mat->Mult(u, rhs);

   const int ne = pfes.GetMesh()->GetNE();
   const int nd = pfes.GetFE(0)->GetDof();
   DenseMatrix M_loc(nd);
   DenseMatrixInverse M_loc_inv(&M_loc);
   Vector rhs_loc(nd), du_loc(nd);
   Array<int> dofs;
   for (int i = 0; i < ne; i++)
   {
      pfes.GetElementDofs(i, dofs);
      rhs.GetSubVector(dofs, rhs_loc);
      M.SpMat().GetSubMatrix(dofs, dofs, M_loc);
      M_loc_inv.Factor();
      M_loc_inv.Mult(rhs_loc, du_loc);
      du.SetSubVector(dofs, du_loc);
   }

   delete K_mat;
}

DiscreteUpwindLOSolver::DiscreteUpwindLOSolver(ParFiniteElementSpace &space,
                                               const SparseMatrix &adv,
                                               const Vector &Mlump)
   : pfes(space), K(adv), D(adv), K_smap(), M_lumped(Mlump)
{
   // Assuming it is finalized.
   const int *I = K.GetI(), *J = K.GetJ(), n = K.Size();
   K_smap.SetSize(I[n]);
   for (int row = 0, j = 0; row < n; row++)
   {
      for (int end = I[row+1]; j < end; j++)
      {
         int col = J[j];
         // Find the offset, _j, of the (col,row) entry and store it in smap[j].
         for (int _j = I[col], _end = I[col+1]; true; _j++)
         {
            MFEM_VERIFY(_j != _end, "Can't find the symmetric entry!");

            if (J[_j] == row) { K_smap[j] = _j; break; }
         }
      }
   }
}

void DiscreteUpwindLOSolver::CalcLOSolution(const Vector &u, Vector &du) const
{
   const int ndof = pfes.GetFE(0)->GetDof();
   Vector alpha(ndof); alpha = 0.0;

   ComputeDiscreteUpwindMatrix();
   ParGridFunction u_gf(&pfes);
   u_gf = u;

   ApplyDiscreteUpwindMatrix(u_gf, du);

   const int s = du.Size();
   for (int i = 0; i < s; i++) { du(i) /= M_lumped(i); }
}

void DiscreteUpwindLOSolver::ComputeDiscreteUpwindMatrix() const
{
   const int *Ip = K.HostReadI(), *Jp = K.HostReadJ(), n = K.Size();

   const double *Kp = K.HostReadData();

   double *Dp = D.HostReadWriteData();
   D.HostReadWriteI(); D.HostReadWriteJ();

   for (int i = 0, k = 0; i < n; i++)
   {
      double rowsum = 0.;
      for (int end = Ip[i+1]; k < end; k++)
      {
         int j = Jp[k];
         double kij = Kp[k];
         double kji = Kp[K_smap[k]];
         double dij = fmax(fmax(0.0,-kij),-kji);
         Dp[k] = kij + dij;
         Dp[K_smap[k]] = kji + dij;
         if (i != j) { rowsum += dij; }
      }
      D(i,i) = K(i,i) - rowsum;
   }
}

void DiscreteUpwindLOSolver::ApplyDiscreteUpwindMatrix(ParGridFunction &u,
                                                       Vector &du) const
{
   const int s = u.Size();
   const int *D_I = D.HostReadI(), *D_J = D.HostReadJ();
   const double *D_data = D.HostReadData();

   u.ExchangeFaceNbrData();
   const Vector &u_np = u.FaceNbrData();

   for (int i = 0; i < s; i++)
   {
      du(i) = 0.0;
      for (int k = D_I[i]; k < D_I[i + 1]; k++)
      {
         int j = D_J[k];
         double u_j  = (j < s) ? u(j) : u_np[j - s];
         double d_ij = D_data[k];
         du(i) += d_ij * u_j;
      }
   }
}

} // namespace hydrodynamics

} // namespace mfem
