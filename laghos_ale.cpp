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
     d(), v(), rho(), e(), x0()
{
   const int vsize_H1 = pfes_H1.GetVSize(), vsize_L2 = pfes_L2.GetVSize();

   // Arrangement: distance (dim), velocity (dim), density (1), energy (1).
   offsets[0] = 0;
   offsets[1] = vsize_H1;
   offsets[2] = offsets[1] + vsize_H1;
   offsets[3] = offsets[2] + vsize_L2;
   offsets[4] = offsets[3] + vsize_L2;
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
                                 const Vector &rhoDetJw,
                                 const ParGridFunction &energy)
{
   x0 = nodes0;
   d  = dist;
   v  = vel;
   e  = energy;

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
   AdvectorOper oper(S.Size(), x0, u, rho, pfes_H1, pfes_L2);
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
   double dt = 0.1 * h_min / v_max;

   double t = 0.0;
   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= 1.0)
      {
         dt = 1.0 - t;
         last_step = true;
      }

      oper.SetDt(dt);
      ode_solver.Step(S, t, dt);
   }
}

void RemapAdvector::TransferToLagr(ParGridFunction &dist,
                                   ParGridFunction &vel,
                                   const IntegrationRule &ir_rho,
                                   Vector &rhoDetJw,
                                   ParGridFunction &rho0,
                                   ParGridFunction &energy)
{
   dist = d;
   vel  = v;

   rho0 = rho;
   const int NE = pfes_L2.GetNE(), nqp = ir_rho.GetNPoints();
   Vector rho_vals(nqp);
   for (int k = 0; k < NE; k++)
   {
      // Must use the space of the results.
      ElementTransformation &T = *vel.ParFESpace()->GetElementTransformation(k);
      rho.GetValues(T, ir_rho, rho_vals);
      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir_rho.IntPoint(q);
         T.SetIntPoint(&ip);
         rhoDetJw(k*nqp + q) = rho_vals(q) * T.Jacobian().Det() * ip.weight;
      }
   }

   energy = e;
}

AdvectorOper::AdvectorOper(int size, const Vector &x_start,
                           GridFunction &velocity, GridFunction &rho,
                           ParFiniteElementSpace &pfes_H1,
                           ParFiniteElementSpace &pfes_L2)
   : TimeDependentOperator(size),
     x0(x_start), x_now(*pfes_H1.GetMesh()->GetNodes()),
     u(velocity),
     u_coeff(&u), rho_coeff(&rho), rho_u_coeff(rho_coeff, u_coeff),
     M_H1(&pfes_H1), K_H1(&pfes_H1),
     Mr_H1(&pfes_H1), Kr_H1(&pfes_H1),
     M_L2(&pfes_L2), M_L2_Lump(&pfes_L2), K_L2(&pfes_L2),
     Mr_L2(&pfes_L2),  Mr_L2_Lump(&pfes_L2), Kr_L2(&pfes_L2)
{
   M_H1.AddDomainIntegrator(new MassIntegrator);
   M_H1.Assemble(0);
   M_H1.Finalize(0);

   K_H1.AddDomainIntegrator(new ConvectionIntegrator(u_coeff));
   K_H1.Assemble(0);
   K_H1.Finalize(0);

   Mr_H1.AddDomainIntegrator(new MassIntegrator(rho_coeff));
   Mr_H1.Assemble(0);
   Mr_H1.Finalize(0);

   Kr_H1.AddDomainIntegrator(new ConvectionIntegrator(rho_u_coeff));
   Kr_H1.Assemble(0);
   Kr_H1.Finalize(0);

   M_L2.AddDomainIntegrator(new MassIntegrator);
   M_L2.Assemble(0);
   M_L2.Finalize(0);

   M_L2_Lump.AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator));
   M_L2_Lump.Assemble();
   M_L2_Lump.Finalize();

   K_L2.AddDomainIntegrator(new ConvectionIntegrator(u_coeff));
   DGTraceIntegrator *dgt_i = new DGTraceIntegrator(u_coeff, -1.0, -0.5);
   DGTraceIntegrator *dgt_b = new DGTraceIntegrator(u_coeff, -1.0, -0.5);
   K_L2.AddInteriorFaceIntegrator(new TransposeIntegrator(dgt_i));
   K_L2.AddBdrFaceIntegrator(new TransposeIntegrator(dgt_b));
   K_L2.KeepNbrBlock(true);
   K_L2.Assemble(0);
   K_L2.Finalize(0);

   Mr_L2.AddDomainIntegrator(new MassIntegrator(rho_coeff));
   Mr_L2.Assemble(0);
   Mr_L2.Finalize(0);

   auto *minteg = new MassIntegrator(rho_coeff);
   Mr_L2_Lump.AddDomainIntegrator(new LumpedIntegrator(minteg));
   Mr_L2_Lump.Assemble();
   Mr_L2_Lump.Finalize();

   Kr_L2.AddDomainIntegrator(new ConvectionIntegrator(rho_u_coeff));
   DGTraceIntegrator *dgt_ir = new DGTraceIntegrator(rho_u_coeff, -1.0, -0.5);
   DGTraceIntegrator *dgt_br = new DGTraceIntegrator(rho_u_coeff, -1.0, -0.5);
   Kr_L2.AddInteriorFaceIntegrator(new TransposeIntegrator(dgt_ir));
   Kr_L2.AddBdrFaceIntegrator(new TransposeIntegrator(dgt_br));
   Kr_L2.KeepNbrBlock(true);
   Kr_L2.Assemble(0);
   Kr_L2.Finalize(0);
}

void AdvectorOper::Mult(const Vector &U, Vector &dU) const
{
   ParFiniteElementSpace &pfes_H1 = *K_H1.ParFESpace(),
                         &pfes_L2 = *K_L2.ParFESpace();
   const int dim     = pfes_H1.GetMesh()->Dimension();
   const int NE      = pfes_H1.GetNE();
   const int size_H1 = pfes_H1.GetVSize(), size_L2 = pfes_L2.GetVSize();

   // Move the mesh.
   const double t = GetTime();
   add(x0, t, u, x_now);

   dU = 0.0;

   // Arrangement: distance (dim), velocity (dim), density (1), energy (1).
   Vector *Uptr = const_cast<Vector *>(&U);

   // Solver for H1 fields (no monotonicity).
   HypreSmoother prec;
   prec.SetType(HypreSmoother::Jacobi, 1);
   CGSolver lin_solver(pfes_H1.GetComm());
   lin_solver.SetPreconditioner(prec);
   lin_solver.SetRelTol(1e-8);
   lin_solver.SetAbsTol(0.0);
   lin_solver.SetMaxIter(100);
   lin_solver.SetPrintLevel(0);
   OperatorHandle Mass_oper;

   const Operator *P_H1 = pfes_H1.GetProlongationMatrix();
   Vector rhs_H1(size_H1), RHS_H1(P_H1->Width()), X(P_H1->Width());

   // Distance remap.
   M_H1.BilinearForm::operator=(0.0);
   M_H1.Assemble();
   K_H1.BilinearForm::operator=(0.0);
   K_H1.Assemble();
   Mass_oper.Reset(M_H1.ParallelAssemble());
   lin_solver.SetOperator(*Mass_oper);
   Vector dist, d_dist;
   for (int d = 0; d < dim; d++)
   {
      // Distance component.
      dist.MakeRef(*Uptr, d * size_H1, size_H1);
      d_dist.MakeRef(dU,  d * size_H1, size_H1);
      K_H1.Mult(dist, rhs_H1);
      P_H1->MultTranspose(rhs_H1, RHS_H1);
      X = 0.0;
      lin_solver.Mult(RHS_H1, X);
      P_H1->Mult(X, d_dist);
   }

   // Velocity remap.
   Mr_H1.BilinearForm::operator=(0.0);
   Mr_H1.Assemble();
   Kr_H1.BilinearForm::operator=(0.0);
   Kr_H1.Assemble();
   Mass_oper.Reset(Mr_H1.ParallelAssemble());
   lin_solver.SetOperator(*Mass_oper);
   Vector v, d_v;
   for (int d = 0; d < dim; d++)
   {
      // Velocity component.
      v.MakeRef(*Uptr, (dim + d) * size_H1, size_H1);
      d_v.MakeRef(dU,  (dim + d) * size_H1, size_H1);
      Kr_H1.Mult(v, rhs_H1);
      P_H1->MultTranspose(rhs_H1, RHS_H1);
      X = 0.0;
      lin_solver.Mult(RHS_H1, X);
      P_H1->Mult(X, d_v);
   }

   // Density remap.
   K_L2.BilinearForm::operator=(0.0);
   K_L2.Assemble();
   M_L2.BilinearForm::operator=(0.0);
   M_L2.Assemble();
   M_L2_Lump.BilinearForm::operator=(0.0);
   M_L2_Lump.Assemble();
   Vector rho, d_rho, d_rho_HO(size_L2), d_rho_LO(size_L2);
   rho.MakeRef(*Uptr, 2 * dim * size_H1, size_L2);
   d_rho.MakeRef(dU,  2 * dim * size_H1, size_L2);
   d_rho = 0.0;
   Vector lumpedM; M_L2_Lump.SpMat().GetDiag(lumpedM);
   DiscreteUpwindLOSolver lo_solver(pfes_L2, K_L2.SpMat(), lumpedM);
   lo_solver.CalcLOSolution(rho, d_rho_LO);
   LocalInverseHOSolver ho_solver(M_L2, K_L2);
   ho_solver.CalcHOSolution(rho, d_rho_HO);
   Vector el_min(NE), el_max(NE),
          rho_min(size_L2), rho_max(size_L2);
   ParGridFunction rho_gf(&pfes_L2);
   rho_gf = rho;
   rho_gf.ExchangeFaceNbrData();
   ComputeElementsMinMax(rho_gf, el_min, el_max);
   ComputeSparsityBounds(pfes_L2, el_min, el_max, rho_min, rho_max);
   FluxBasedFCT fct_solver(pfes_L2, dt,
                           K_L2.SpMat(), lo_solver.GetKmap(), M_L2.SpMat());
   fct_solver.CalcFCTSolution(rho_gf, lumpedM, d_rho_HO, d_rho_LO,
                              rho_min, rho_max, d_rho);

   // Energy remap.
   Kr_L2.BilinearForm::operator=(0.0);
   Kr_L2.Assemble();
   Mr_L2.BilinearForm::operator=(0.0);
   Mr_L2.Assemble();
   Mr_L2_Lump.BilinearForm::operator=(0.0);
   Mr_L2_Lump.Assemble();
   Vector e, d_e, d_e_HO(size_L2), d_e_LO(size_L2);
   e.MakeRef(*Uptr, 2 * dim * size_H1 + size_L2, size_L2);
   d_e.MakeRef(dU,  2 * dim * size_H1 + size_L2, size_L2);
   Vector lumped;
   Mr_L2_Lump.SpMat().GetDiag(lumped);
   DiscreteUpwindLOSolver lo_e_solver(pfes_L2, Kr_L2.SpMat(), lumped);
   lo_e_solver.CalcLOSolution(e, d_e_LO);
   LocalInverseHOSolver ho_e_solver(Mr_L2, Kr_L2);
   ho_e_solver.CalcHOSolution(e, d_e_HO);
   Vector e_min(size_L2), e_max(size_L2);
   ParGridFunction e_gf(&pfes_L2);
   e_gf = e;
   e_gf.ExchangeFaceNbrData();
   ComputeElementsMinMax(e_gf, el_min, el_max);
   ComputeSparsityBounds(pfes_L2, el_min, el_max, e_min, e_max);
   FluxBasedFCT fct_e_solver(pfes_L2, dt, Kr_L2.SpMat(),
                             lo_e_solver.GetKmap(), Mr_L2.SpMat());
   fct_e_solver.CalcFCTSolution(e_gf, lumpedM, d_e_HO, d_e_LO,
                                e_min, e_max, d_e);
}

double AdvectorOper::Momentum(ParGridFunction &v, double t)
{
   add(x0, t, u, x_now);

   Mr_H1.BilinearForm::operator=(0.0);
   Mr_H1.Assemble();

   Vector one(Mr_H1.SpMat().Height());
   one = 1.0;
   double loc_m  = Mr_H1.InnerProduct(one, v);

   double glob_m;
   MPI_Allreduce(&loc_m, &glob_m, 1, MPI_DOUBLE, MPI_SUM,
                 Mr_H1.ParFESpace()->GetComm());
   return glob_m;
}

double AdvectorOper::Distance(ParGridFunction &d, double t)
{
   add(x0, t, u, x_now);

   M_H1.BilinearForm::operator=(0.0);
   M_H1.Assemble();

   Vector one(M_H1.SpMat().Height());
   one = 1.0;
   double loc_d  = M_H1.InnerProduct(one, d);

   double glob_d;
   MPI_Allreduce(&loc_d, &glob_d, 1, MPI_DOUBLE, MPI_SUM,
                 M_H1.ParFESpace()->GetComm());
   return glob_d;
}

double AdvectorOper::Energy(ParGridFunction &e, double t)
{
   add(x0, t, u, x_now);

   Mr_L2.BilinearForm::operator=(0.0);
   Mr_L2.Assemble();

   Vector one(Mr_L2.SpMat().Height());
   one = 1.0;
   double loc_e = Mr_L2.InnerProduct(one, e);

   double glob_e;
   MPI_Allreduce(&loc_e, &glob_e, 1, MPI_DOUBLE, MPI_SUM,
                 Mr_L2.ParFESpace()->GetComm());
   return glob_e;
}

void AdvectorOper::ComputeElementsMinMax(const ParGridFunction &u,
                                         Vector &el_min, Vector &el_max) const
{
   ParFiniteElementSpace &pfes = *u.ParFESpace();
   const int NE = pfes.GetNE(), ndof = pfes.GetFE(0)->GetDof();
   int dof_id;
   for (int k = 0; k < NE; k++)
   {
      el_min(k) = numeric_limits<double>::infinity();
      el_max(k) = -numeric_limits<double>::infinity();

      for (int i = 0; i < ndof; i++)
      {
         dof_id = k*ndof + i;

         el_min(k) = min(el_min(k), u(dof_id));
         el_max(k) = max(el_max(k), u(dof_id));
      }
   }
}

void AdvectorOper::ComputeSparsityBounds(const ParFiniteElementSpace &pfes,
                                         const Vector &el_min,
                                         const Vector &el_max,
                                         Vector &u_min, Vector &u_max) const
{
   ParMesh *pmesh = pfes.GetParMesh();
   L2_FECollection fec_bounds(0, pmesh->Dimension());
   ParFiniteElementSpace pfes_bounds(pmesh, &fec_bounds);
   ParGridFunction x_min(&pfes_bounds), x_max(&pfes_bounds);
   const int NE = pmesh->GetNE();
   const int ndofs = u_min.Size() / NE;

   x_min = el_min;
   x_max = el_max;

   x_min.ExchangeFaceNbrData(); x_max.ExchangeFaceNbrData();
   const Vector &minv = x_min.FaceNbrData(), &maxv = x_max.FaceNbrData();
   const Table &el_to_el = pmesh->ElementToElementTable();
   Array<int> face_nbr_el;
   for (int i = 0; i < NE; i++)
   {
      double el_min = x_min(i), el_max = x_max(i);

      el_to_el.GetRow(i, face_nbr_el);
      for (int n = 0; n < face_nbr_el.Size(); n++)
      {
         if (face_nbr_el[n] < NE)
         {
            // Local neighbor.
            el_min = std::min(el_min, x_min(face_nbr_el[n]));
            el_max = std::max(el_max, x_max(face_nbr_el[n]));
         }
         else
         {
            // MPI face neighbor.
            el_min = std::min(el_min, minv(face_nbr_el[n] - NE));
            el_max = std::max(el_max, maxv(face_nbr_el[n] - NE));
         }
      }

      for (int j = 0; j < ndofs; j++)
      {
         u_min(i*ndofs + j) = el_min;
         u_max(i*ndofs + j) = el_max;
      }
   }
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
   ComputeDiscreteUpwindMatrix();
   ParGridFunction u_gf(&pfes);
   u_gf = u;

   ApplyDiscreteUpwindMatrix(u_gf, du);

   const int s = du.Size();
   for (int i = 0; i < s; i++) { du(i) /= M_lumped(i); }
}

void DiscreteUpwindLOSolver::ComputeDiscreteUpwindMatrix() const
{
   const int *I = K.HostReadI(), *J = K.HostReadJ(), n = K.Size();

   const double *K_data = K.HostReadData();

   double *D_data = D.HostReadWriteData();
   D.HostReadWriteI(); D.HostReadWriteJ();

   for (int i = 0, k = 0; i < n; i++)
   {
      double rowsum = 0.;
      for (int end = I[i+1]; k < end; k++)
      {
         int j = J[k];
         double kij = K_data[k];
         double kji = K_data[K_smap[k]];
         double dij = fmax(fmax(0.0,-kij),-kji);
         D_data[k] = kij + dij;
         D_data[K_smap[k]] = kji + dij;
         if (i != j) { rowsum += dij; }
      }
      D(i,i) = K(i,i) - rowsum;
   }
}

void DiscreteUpwindLOSolver::ApplyDiscreteUpwindMatrix(ParGridFunction &u,
                                                       Vector &du) const
{
   const int s = u.Size();
   const int *I = D.HostReadI(), *J = D.HostReadJ();
   const double *D_data = D.HostReadData();

   u.ExchangeFaceNbrData();
   const Vector &u_np = u.FaceNbrData();

   for (int i = 0; i < s; i++)
   {
      du(i) = 0.0;
      for (int k = I[i]; k < I[i + 1]; k++)
      {
         int j = J[k];
         double u_j  = (j < s) ? u(j) : u_np[j - s];
         double d_ij = D_data[k];
         du(i) += d_ij * u_j;
      }
   }
}

void FluxBasedFCT::CalcFCTSolution(const ParGridFunction &u, const Vector &m,
                                   const Vector &du_ho, const Vector &du_lo,
                                   const Vector &u_min, const Vector &u_max,
                                   Vector &du) const
{
   // Construct the flux matrix (it gets recomputed every time).
   ComputeFluxMatrix(u, du_ho, flux_ij);

   // Iterated FCT correction.
   Vector du_lo_fct(du_lo);
   for (int fct_iter = 0; fct_iter < 1; fct_iter++)
   {
      // Compute sums of incoming/outgoing fluxes at each DOF.
      AddFluxesAtDofs(flux_ij, gp, gm);

      // Compute the flux coefficients (aka alphas) into gp and gm.
      ComputeFluxCoefficients(u, du_lo_fct, m, u_min, u_max, gp, gm);

      // Apply the alpha coefficients to get the final solution.
      // Update the flux matrix for iterative FCT (when iter_cnt > 1).
      UpdateSolutionAndFlux(du_lo_fct, m, gp, gm, flux_ij, du);

      du_lo_fct = du;
   }
}

void FluxBasedFCT::ComputeFluxMatrix(const ParGridFunction &u,
                                     const Vector &du_ho,
                                     SparseMatrix &flux_mat) const
{
   const int s = u.Size();
   double *flux_data = flux_mat.HostReadWriteData();
   flux_mat.HostReadI(); flux_mat.HostReadJ();
   const int *K_I = K.HostReadI(), *K_J = K.HostReadJ();
   const double *K_data = K.HostReadData();
   const double *u_np = u.FaceNbrData().HostRead();
   u.HostRead();
   du_ho.HostRead();
   for (int i = 0; i < s; i++)
   {
      for (int k = K_I[i]; k < K_I[i + 1]; k++)
      {
         int j = K_J[k];
         if (j <= i) { continue; }

         double kij  = K_data[k], kji = K_data[K_smap[k]];
         double dij  = max(max(0.0, -kij), -kji);
         double u_ij = (j < s) ? u(i) - u(j)
                       : u(i) - u_np[j - s];

         flux_data[k] = dt * dij * u_ij;
      }
   }

   const int NE = pfes.GetMesh()->GetNE();
   const int ndof = s / NE;
   Array<int> dofs;
   DenseMatrix Mz(ndof);
   Vector du_z(ndof);
   for (int k = 0; k < NE; k++)
   {
      pfes.GetElementDofs(k, dofs);
      M.GetSubMatrix(dofs, dofs, Mz);
      du_ho.GetSubVector(dofs, du_z);
      for (int i = 0; i < ndof; i++)
      {
         int j = 0;
         for (; j <= i; j++) { Mz(i, j) = 0.0; }
         for (; j < ndof; j++) { Mz(i, j) *= dt * (du_z(i) - du_z(j)); }
      }
      flux_mat.AddSubMatrix(dofs, dofs, Mz, 0);
   }
}

// Compute sums of incoming fluxes for every DOF.
void FluxBasedFCT::AddFluxesAtDofs(const SparseMatrix &flux_mat,
                                   Vector &flux_pos, Vector &flux_neg) const
{
   const int s = flux_pos.Size();
   const double *flux_data = flux_mat.GetData();
   const int *flux_I = flux_mat.GetI(), *flux_J = flux_mat.GetJ();
   flux_pos = 0.0;
   flux_neg = 0.0;
   flux_pos.HostReadWrite();
   flux_neg.HostReadWrite();
   for (int i = 0; i < s; i++)
   {
      for (int k = flux_I[i]; k < flux_I[i + 1]; k++)
      {
         int j = flux_J[k];

         // The skipped fluxes will be added when the outer loop is at j as
         // the flux matrix is always symmetric.
         if (j <= i) { continue; }

         const double f_ij = flux_data[k];

         if (f_ij >= 0.0)
         {
            flux_pos(i) += f_ij;
            // Modify j if it's on the same MPI task (prevents x2 counting).
            if (j < s) { flux_neg(j) -= f_ij; }
         }
         else
         {
            flux_neg(i) += f_ij;
            // Modify j if it's on the same MPI task (prevents x2 counting).
            if (j < s) { flux_pos(j) -= f_ij; }
         }
      }
   }
}

// Compute the so-called alpha coefficients that scale the fluxes into gp, gm.
void FluxBasedFCT::
ComputeFluxCoefficients(const Vector &u, const Vector &du_lo, const Vector &m,
                        const Vector &u_min, const Vector &u_max,
                        Vector &coeff_pos, Vector &coeff_neg) const
{
   const int s = u.Size();
   for (int i = 0; i < s; i++)
   {
      const double u_lo = u(i) + dt * du_lo(i);
      const double max_pos_diff = max((u_max(i) - u_lo) * m(i), 0.0),
                   min_neg_diff = min((u_min(i) - u_lo) * m(i), 0.0);
      const double sum_pos = coeff_pos(i), sum_neg = coeff_neg(i);

      coeff_pos(i) = (sum_pos > max_pos_diff) ? max_pos_diff / sum_pos : 1.0;
      coeff_neg(i) = (sum_neg < min_neg_diff) ? min_neg_diff / sum_neg : 1.0;
   }
}

void FluxBasedFCT::
UpdateSolutionAndFlux(const Vector &du_lo, const Vector &m,
                      ParGridFunction &coeff_pos, ParGridFunction &coeff_neg,
                      SparseMatrix &flux_mat, Vector &du) const
{
   Vector &a_pos_n = coeff_pos.FaceNbrData(),
          &a_neg_n = coeff_neg.FaceNbrData();
   coeff_pos.ExchangeFaceNbrData();
   coeff_neg.ExchangeFaceNbrData();

   du = du_lo;

   coeff_pos.HostReadWrite();
   coeff_neg.HostReadWrite();
   du.HostReadWrite();

   double *flux_data = flux_mat.HostReadWriteData();
   const int *flux_I = flux_mat.HostReadI(), *flux_J = flux_mat.HostReadJ();
   const int s = du.Size();
   for (int i = 0; i < s; i++)
   {
      for (int k = flux_I[i]; k < flux_I[i + 1]; k++)
      {
         int j = flux_J[k];
         if (j <= i) { continue; }

         double fij = flux_data[k], a_ij;
         if (fij >= 0.0)
         {
            a_ij = (j < s) ? min(coeff_pos(i), coeff_neg(j))
                   : min(coeff_pos(i), a_neg_n(j - s));
         }
         else
         {
            a_ij = (j < s) ? min(coeff_neg(i), coeff_pos(j))
                   : min(coeff_neg(i), a_pos_n(j - s));
         }
         fij *= a_ij;

         du(i) += fij / m(i) / dt;
         if (j < s) { du(j) -= fij / m(j) / dt; }

         flux_data[k] -= fij;
      }
   }
}

} // namespace hydrodynamics

} // namespace mfem
