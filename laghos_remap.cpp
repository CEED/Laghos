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

#include "laghos_remap.hpp"
#include "laghos_assembly.hpp"

using namespace std;
namespace mfem
{

namespace hydrodynamics
{

void InterpolationRemap::Remap(const ParGridFunction &source,
                               const ParGridFunction &x_new,
                               ParGridFunction &interpolated)
{
   if (source.ParFESpace()->GetMyRank() == 0)
   {
      std::cout << "--- Interpolation remap of velocity." << std::endl;
   }

   ParMesh &pmesh_src = *source.ParFESpace()->GetParMesh();
   ParFiniteElementSpace &pfes_tgt = *interpolated.ParFESpace();

   const int dim = pmesh_src.Dimension();
   MFEM_VERIFY(dim > 1, "Interpolation remap works only in 2D and 3D.");

   const int NE = pmesh_src.GetNE();
   const int nsp = interpolated.ParFESpace()->GetFE(0)->GetNodes().GetNPoints();
   const int ncomp = source.VectorDim();

   // Generate list of points where the grid function will be evaluated.
   Vector vxyz = x_new;

   // vxyz.SetSize(nsp * NE * dim);
   // for (int e = 0; e < NE; e++)
   // {
   //    const IntegrationRule &ir = pfes_tgt.GetFE(e)->GetNodes();

   //    // Transformation of the element with the new coordinates.
   //    IsoparametricTransformation Tr;
   //    pmesh_src.GetElementTransformation(e, x_new, &Tr);

   //    // Node positions of the interpolated f-n (new element coordinates).
   //    DenseMatrix pos_target_nodes;
   //    Tr.Transform(ir, pos_target_nodes);
   //    Vector rowx(vxyz.GetData() + e*nsp, nsp),
   //           rowy(vxyz.GetData() + e*nsp + NE*nsp, nsp), rowz;
   //    if (dim == 3)
   //    {
   //       rowz.SetDataAndSize(vxyz.GetData() + e*nsp + 2*NE*nsp, nsp);
   //    }
   //    pos_target_nodes.GetRow(0, rowx);
   //    pos_target_nodes.GetRow(1, rowy);
   //    if (dim == 3) { pos_target_nodes.GetRow(2, rowz); }
   // }

   const int nodes_cnt = vxyz.Size() / dim;

   // Evaluate source grid function.
   Vector interp_vals(ncomp * nodes_cnt);
   FindPointsGSLIB finder(pfes_tgt.GetComm());
   finder.Setup(pmesh_src, 1.0);
   finder.SetDistanceToleranceForPointsFoundOnBoundary(0.05);
   finder.SetDefaultInterpolationValue(-1.0);
   finder.FindPoints(vxyz);
   const Array<unsigned int> &codes = finder.GetCode();
   int cnt = 0;
   for (int i = 0; i < nodes_cnt; i++)
   {
      if (codes[i] == 2)
      {
         cnt++;
         cout << vxyz(i) << " " << vxyz(i + nodes_cnt) << endl;
      }
   }
   std::cout << cnt << std::endl;
   MFEM_VERIFY(cnt == 0, "Points not found");
   finder.Interpolate(source, interp_vals);

   interpolated = interp_vals;
}

RemapAdvector::RemapAdvector(const ParMesh &m, int order_v, int order_e,
                             double cfl, bool remap_vel)
   : pmesh(m, true), dim(pmesh.Dimension()),
     fec_L2(order_e, pmesh.Dimension(), BasisType::Positive),
     fec_H1(order_v, pmesh.Dimension()),
     pfes_L2(&pmesh, &fec_L2, 1),
     pfes_H1(&pmesh, &fec_H1, pmesh.Dimension()),
     remap_v(remap_vel),
     cfl_factor(cfl),
     offsets(4), S(),
     v(), rho(), e(), x0()
{
   const int vsize_H1 = pfes_H1.GetVSize(), vsize_L2 = pfes_L2.GetVSize();

   // Arrangement: velocity (dim), density (1), energy (1).
   offsets[0] = 0;
   offsets[1] = vsize_H1;
   offsets[2] = offsets[1] + vsize_L2;
   offsets[3] = offsets[2] + vsize_L2;
   S.Update(offsets);

   v.MakeRef(&pfes_H1, S, offsets[0]);
   rho.MakeRef(&pfes_L2, S, offsets[1]);
   e.MakeRef(&pfes_L2, S, offsets[2]);
}

void RemapAdvector::InitFromLagr(const Vector &nodes0,
                                 const ParGridFunction &vel,
                                 const IntegrationRule &rho_ir,
                                 const Vector &rhoDetJw,
                                 const ParGridFunction &energy)
{
   x0 = nodes0;
   v  = vel;
   e  = energy;

   e_max = e.Max();
   MPI_Allreduce(MPI_IN_PLACE, &e_max, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());

   // Get densities as GridFunctions.
   SolutionMover mover(rho_ir);
   mover.MoveDensityLR(rhoDetJw, rho);
}

void RemapAdvector::ComputeAtNewPosition(const Vector &new_nodes,
                                         const Array<int> &ess_tdofs)
{
   const int vsize_H1 = pfes_H1.GetVSize();

   // This will be used to move the positions.
   GridFunction *x = pmesh.GetNodes();
   *x = x0;

   // Velocity of the positions.
   ParGridFunction u(&pfes_H1);
   subtract(new_nodes, x0, u);

   // Define scalar FE spaces for the solution, and the advection operator.
   ParFiniteElementSpace pfes_H1_s(&pmesh, pfes_H1.FEColl(), 1);
   AdvectorOper oper(S.Size(), x0, ess_tdofs, u, rho,
                     pfes_H1, pfes_H1_s, pfes_L2);
   oper.SetVelocityRemap(remap_v);
   ode_solver.Init(oper);

   // Compute some time step [mesh_size / speed].
   double h_min = std::numeric_limits<double>::infinity();
   for (int k = 0; k < pmesh.GetNE(); k++)
   {
      h_min = std::min(h_min, pmesh.GetElementSize(k));
   }
   double u_max = 0.0;
   const int s = vsize_H1 / dim;

   for (int i = 0; i < s; i++)
   {
      double vel = 0.;
      for (int j = 0; j < dim; j++)
      {
         vel += u(i+j*s)*u(i+j*s);
      }
      u_max = std::max(u_max, vel);
   }

   double v_loc = u_max, h_loc = h_min;
   MPI_Allreduce(&v_loc, &u_max, 1, MPI_DOUBLE, MPI_MAX, pfes_H1.GetComm());
   MPI_Allreduce(&h_loc, &h_min, 1, MPI_DOUBLE, MPI_MIN, pfes_H1.GetComm());

   if (u_max == 0.0) { return; } // No need to change the fields.

   u_max = std::sqrt(u_max);
   double dt = cfl_factor * h_min / u_max;

   double t = 0.0;
   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= 1.0)
      {
         dt = 1.0 - t;
         last_step = true;
      }

      if (pmesh.GetMyRank() == 0) { cout << "." << flush; }

      oper.SetDt(dt);
      ode_solver.Step(S, t, dt);

      double e_max_new = e.Max();
      MPI_Allreduce(MPI_IN_PLACE, &e_max_new, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
      if (e_max_new > e_max)
      {
         cout << e_max << " " << e_max_new << endl;
         MFEM_ABORT("\n e_1 max remap violation");
      }
   }
   if (pmesh.GetMyRank() == 0) { cout << endl; }
}

void RemapAdvector::TransferToLagr(ParGridFunction &rho0_gf,
                                   ParGridFunction &vel,
                                   const IntegrationRule &ir_rho,
                                   Vector &rhoDetJw,
                                   const IntegrationRule &ir_rho_b,
                                   Vector &rhoDetJ_be,
                                   ParGridFunction &energy)
{
   // This is used to update the mass matrices.
   rho0_gf = rho;

   // Just copy velocity.
   vel = v;

   ParMesh &pmesh_lagr = *vel.ParFESpace()->GetParMesh();
   const int NE  = pmesh_lagr.GetNE(),
             NBE = pmesh_lagr.GetNBE();
   const int nqp = ir_rho.GetNPoints();

   Vector rho_vals(nqp);
   for (int k = 0; k < NE; k++)
   {
      // Must use the space of the results.
      ElementTransformation &T = *pmesh_lagr.GetElementTransformation(k);
      rho.GetValues(T, ir_rho, rho_vals);
      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir_rho.IntPoint(q);
         T.SetIntPoint(&ip);
         rhoDetJw(k*nqp + q) = rho_vals(q) * T.Weight() * ip.weight;
      }
   }

   for (int be = 0; be < NBE; be++)
   {
      int b_nqp = ir_rho_b.GetNPoints();
      auto b_face_tr = pmesh_lagr.GetBdrFaceTransformations(be);
      if (b_face_tr == nullptr) { continue; }
      for (int q = 0; q < b_nqp; q++)
      {
         const IntegrationPoint &ip_f = ir_rho_b.IntPoint(q);
         b_face_tr->SetAllIntPoints(&ip_f);
         ElementTransformation &tr_el = b_face_tr->GetElement1Transformation();
         double detJ = tr_el.Weight();
         MFEM_VERIFY(detJ > 0, "Negative detJ at a face! " << detJ);
         rhoDetJ_be(be * b_nqp + q) = detJ * rho0_gf.GetValue(tr_el);
      }
   }

   // Just copy energy.
   energy = e;
}

AdvectorOper::AdvectorOper(int size, const Vector &x_start,
                           const Array<int> &v_ess_td,
                           ParGridFunction &mesh_vel,
                           ParGridFunction &rho,
                           ParFiniteElementSpace &pfes_H1,
                           ParFiniteElementSpace &pfes_H1_s,
                           ParFiniteElementSpace &pfes_L2)
   : TimeDependentOperator(size),
     x0(x_start), x_now(*pfes_H1.GetMesh()->GetNodes()),
     v_ess_tdofs(v_ess_td),
     u(mesh_vel), u_coeff(&u),
     rho_coeff(&rho),
     rho_u_coeff(rho_coeff, u_coeff),
     Mr_H1(&pfes_H1), Kr_H1(&pfes_H1_s),
     M_L2(&pfes_L2), M_L2_Lump(&pfes_L2), K_L2(&pfes_L2),
     Mr_L2(&pfes_L2),  Mr_L2_Lump(&pfes_L2), Kr_L2(&pfes_L2)
{
   Mr_H1.AddDomainIntegrator(new VectorMassIntegrator(rho_coeff));
   Mr_H1.Assemble(0);
   Mr_H1.Finalize(0);

   Kr_H1.AddDomainIntegrator(new ConvectionIntegrator(rho_u_coeff));
   Kr_H1.Assemble(0);
   Kr_H1.Finalize(0);

   M_L2.AddDomainIntegrator(new MassIntegrator);
   M_L2.Assemble(0);
   M_L2.Finalize(0);

   M_L2_Lump.AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator));
   M_L2_Lump.Assemble(0);
   M_L2_Lump.Finalize(0);

   K_L2.AddDomainIntegrator(new ConvectionIntegrator(u_coeff));
   auto dgt_i = new DGTraceIntegrator(u_coeff, -1.0, -0.5);
   auto dgt_b = new DGTraceIntegrator(u_coeff, -1.0, -0.5);
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
   auto dgt_ir_1 = new DGTraceIntegrator(rho_u_coeff, -1.0, -0.5);
   auto dgt_br_1 = new DGTraceIntegrator(rho_u_coeff, -1.0, -0.5);
   Kr_L2.AddInteriorFaceIntegrator(new TransposeIntegrator(dgt_ir_1));
   Kr_L2.AddBdrFaceIntegrator(new TransposeIntegrator(dgt_br_1));
   Kr_L2.KeepNbrBlock(true);
   // In parallel, the assembly of Kr_L2 needs to see values from MPI-neighbors.
   // That is, the rho_coeff must be evaluated in MPI-neighbor zones.
   rho.ExchangeFaceNbrData();
   Kr_L2.Assemble(0);
   Kr_L2.Finalize(0);
}

void AdvectorOper::Mult(const Vector &U, Vector &dU) const
{
   ParFiniteElementSpace &pfes_H1 = *Mr_H1.ParFESpace(),
                         &pfes_L2 = *M_L2.ParFESpace();
   const int dim     = pfes_H1.GetMesh()->Dimension();
   const int NE      = pfes_H1.GetNE();
   const int dofs_h1 = pfes_H1.GetVSize() / dim, size_L2 = pfes_L2.GetVSize();

   // Move the mesh.
   const double t = GetTime();
   add(x0, t, u, x_now);

   dU = 0.0;

   // Arrangement: interface (1), velocity (dim), density (1), energy (1).
   Vector *U_ptr = const_cast<Vector *>(&U);

   if (remap_v)
   {
      // Solver for H1 fields (no monotonicity).
      HypreSmoother prec;
      prec.SetType(HypreSmoother::Jacobi, 1);
      CGSolver lin_solver(pfes_H1.GetComm());
      lin_solver.SetPreconditioner(prec);
      lin_solver.SetRelTol(1e-8);
      lin_solver.SetAbsTol(0.0);
      lin_solver.SetMaxIter(100);
      lin_solver.SetPrintLevel(0);

      // Velocity remap.
      Mr_H1.BilinearForm::operator=(0.0);
      Mr_H1.Assemble();
      Kr_H1.BilinearForm::operator=(0.0);
      Kr_H1.Assemble();
      HypreParMatrix *A = Mr_H1.ParallelAssemble();
      lin_solver.SetOperator(*A);
      Vector v, d_v, rhs_v(dofs_h1*dim);
      v.MakeRef(*U_ptr, 0, dofs_h1*dim);
      d_v.MakeRef(dU,   0, dofs_h1*dim);
      Vector v_comp, rhs_v_comp;
      for (int d = 0; d < dim; d++)
      {
         v_comp.MakeRef(v, d * dofs_h1, dofs_h1);
         rhs_v_comp.MakeRef(rhs_v, d * dofs_h1, dofs_h1);
         Kr_H1.Mult(v_comp, rhs_v_comp);
      }
      const Operator *P_v = pfes_H1.GetProlongationMatrix();
      Vector RHS_V(P_v->Width()), X_V(P_v->Width());
      P_v->MultTranspose(rhs_v, RHS_V);
      X_V = 0.0;
      OperatorHandle M_elim;
      //M_elim.EliminateRowsCols(Mass_oper, v_ess_tdofs);
      //Mass_oper.EliminateBC(M_elim, v_ess_tdofs, X_V, RHS_V);
      lin_solver.Mult(RHS_V, X_V);
      P_v->Mult(X_V, d_v);
   }

   Vector el_min(NE), el_max(NE);

   // Density remap.
   K_L2.BilinearForm::operator=(0.0);
   K_L2.Assemble();
   M_L2.BilinearForm::operator=(0.0);
   M_L2.Assemble();
   M_L2_Lump.BilinearForm::operator=(0.0);
   M_L2_Lump.Assemble();
   Vector rho, d_rho, d_rho_HO(size_L2), d_rho_LO(size_L2);
   Vector lumpedM; M_L2_Lump.SpMat().GetDiag(lumpedM);
   DiscreteUpwindLOSolver lo_solver(pfes_L2, K_L2.SpMat(), lumpedM);
   LocalInverseHOSolver ho_solver(M_L2, K_L2);
   Vector rho_min(size_L2), rho_max(size_L2);
   ParGridFunction rho_gf(&pfes_L2);
   FluxBasedFCT fct_solver(pfes_L2, dt,
                           K_L2.SpMat(), lo_solver.GetKmap(), M_L2.SpMat());
   rho.MakeRef(*U_ptr, dim * dofs_h1, size_L2);
   d_rho.MakeRef(dU,   dim * dofs_h1, size_L2);
   lo_solver.CalcLOSolution(rho, d_rho_LO);
   ho_solver.CalcHOSolution(rho, d_rho_HO);
   rho_gf = rho;
   rho_gf.ExchangeFaceNbrData();
   ComputeElementsMinMax(rho_gf, el_min, el_max);
   ComputeSparsityBounds(pfes_L2, el_min, el_max, rho_min, rho_max);
   fct_solver.CalcFCTSolution(rho_gf, lumpedM, d_rho_HO, d_rho_LO,
                              rho_min, rho_max, d_rho);

   // Energy remap.
   auto rho_gf_const = dynamic_cast<const ParGridFunction *>
                       (rho_coeff.GetGridFunction());
   auto rho_pgf = const_cast<ParGridFunction *>(rho_gf_const);
   rho_pgf->ExchangeFaceNbrData();
   Kr_L2.BilinearForm::operator=(0.0);
   Kr_L2.Assemble();
   Mr_L2.BilinearForm::operator=(0.0);
   Mr_L2.Assemble();
   Mr_L2_Lump.BilinearForm::operator=(0.0);
   Mr_L2_Lump.Assemble();
   Vector e, d_e, d_e_HO(size_L2), d_e_LO(size_L2), Me_lumped;
   Vector e_min(size_L2), e_max(size_L2);
   ParGridFunction e_gf(&pfes_L2);
   Mr_L2_Lump.SpMat().GetDiag(Me_lumped);
   DiscreteUpwindLOSolver lo_e_solver(pfes_L2, Kr_L2.SpMat(), Me_lumped);
   LocalInverseHOSolver ho_e_solver(Mr_L2, Kr_L2);
   FluxBasedFCT fct_e_solver(pfes_L2, dt, Kr_L2.SpMat(),
                             lo_e_solver.GetKmap(), Mr_L2.SpMat());
   e.MakeRef(*U_ptr, dim * dofs_h1 + size_L2, size_L2);
   d_e.MakeRef(dU,   dim * dofs_h1 + size_L2, size_L2);
   lo_e_solver.CalcLOSolution(e, d_e_LO);
   ho_e_solver.CalcHOSolution(e, d_e_HO);
   e_gf = e;
   e_gf.ExchangeFaceNbrData();
   ComputeElementsMinMax(e_gf, el_min, el_max);
   ComputeSparsityBounds(pfes_L2, el_min, el_max, e_min, e_max);
   fct_e_solver.CalcFCTSolution(e_gf, Me_lumped, d_e_HO, d_e_LO,
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

void AdvectorOper::ComputeElementsMinMax(const ParGridFunction &gf,
                                         Vector &el_min, Vector &el_max) const
{
   ParFiniteElementSpace &pfes = *gf.ParFESpace();
   const int NE = pfes.GetNE(), ndof = pfes.GetFE(0)->GetDof();
   for (int k = 0; k < NE; k++)
   {
      el_min(k) = numeric_limits<double>::infinity();
      el_max(k) = -numeric_limits<double>::infinity();

      for (int i = 0; i < ndof; i++)
      {
         el_min(k) = min(el_min(k), gf(k*ndof + i));
         el_max(k) = max(el_max(k), gf(k*ndof + i));
      }
   }
}

void AdvectorOper::ComputeSparsityBounds(const ParFiniteElementSpace &pfes,
                                         const Vector &el_min,
                                         const Vector &el_max,
                                         Vector &dof_min, Vector &dof_max) const
{
   ParMesh *pmesh = pfes.GetParMesh();
   L2_FECollection fec_bounds(0, pmesh->Dimension());
   ParFiniteElementSpace pfes_bounds(pmesh, &fec_bounds);
   ParGridFunction el_min_gf(&pfes_bounds), el_max_gf(&pfes_bounds);
   const int NE = pmesh->GetNE(), ndofs = dof_min.Size() / NE;

   el_min_gf = el_min;
   el_max_gf = el_max;

   el_min_gf.ExchangeFaceNbrData(); el_max_gf.ExchangeFaceNbrData();
   const Vector &min_nbr = el_min_gf.FaceNbrData(),
                &max_nbr = el_max_gf.FaceNbrData();
   const Table &el_to_el = pmesh->ElementToElementTable();
   Array<int> face_nbr_el;
   for (int k = 0; k < NE; k++)
   {
      double k_min = el_min_gf(k), k_max = el_max_gf(k);

      el_to_el.GetRow(k, face_nbr_el);
      for (int n = 0; n < face_nbr_el.Size(); n++)
      {
         if (face_nbr_el[n] < NE)
         {
            // Local neighbor.
            k_min = std::min(k_min, el_min_gf(face_nbr_el[n]));
            k_max = std::max(k_max, el_max_gf(face_nbr_el[n]));
         }
         else
         {
            // MPI face neighbor.
            k_min = std::min(k_min, min_nbr(face_nbr_el[n] - NE));
            k_max = std::max(k_max, max_nbr(face_nbr_el[n] - NE));
         }
      }

      for (int j = 0; j < ndofs; j++)
      {
         dof_min(k*ndofs + j) = k_min;
         dof_max(k*ndofs + j) = k_max;
      }
   }
}

void SolutionMover::MoveDensityLR(const Vector &quad_rho,
                                  ParGridFunction &rho)
{
   ParMesh &pmesh = *rho.ParFESpace()->GetParMesh();
   L2_FECollection fec0(0, pmesh.Dimension());
   ParFiniteElementSpace pfes0(&pmesh, &fec0);
   ParGridFunction rho_min_loc(&pfes0), rho_max_loc(&pfes0);

   // Local max / min.
   const int NE = pmesh.GetNE(), nqp = ir_rho.GetNPoints();
   for (int k = 0; k < NE; k++)
   {
      ElementTransformation &T = *pmesh.GetElementTransformation(k);
      rho_min_loc(k) =   std::numeric_limits<double>::infinity();
      rho_max_loc(k) = - std::numeric_limits<double>::infinity();

      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir_rho.IntPoint(q);
         T.SetIntPoint(&ip);
         const double detJ = T.Jacobian().Det();
         const double rho = quad_rho(k * nqp + q) / detJ / ip.weight;

         rho_min_loc(k) = std::min(rho_min_loc(k), rho);
         rho_max_loc(k) = std::max(rho_max_loc(k), rho);
      }
   }

   Vector rho_min(rho_min_loc), rho_max(rho_max_loc);

   // One-level face neighbors max / min.
   rho_min_loc.ExchangeFaceNbrData();
   rho_max_loc.ExchangeFaceNbrData();
   const Vector &rho_min_nbr = rho_min_loc.FaceNbrData(),
                &rho_max_nbr = rho_max_loc.FaceNbrData();
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
            rho_min(k) = std::min(rho_min(k), rho_min_loc(face_nbr_el[n]));
            rho_max(k) = std::max(rho_max(k), rho_max_loc(face_nbr_el[n]));
         }
         else
         {
            // MPI face neighbor.
            rho_min(k) = std::min(rho_min(k), rho_min_nbr(face_nbr_el[n] - NE));
            rho_max(k) = std::max(rho_max(k), rho_max_nbr(face_nbr_el[n] - NE));
         }
      }
   }

   // HO solution - FCT_Project.
   const int dof_cnt = rho.Size() / NE;
   DenseMatrix M(dof_cnt), F(dof_cnt);
   DenseMatrixInverse M_inv(&M);
   Vector rhs(dof_cnt), rho_HO(dof_cnt), rho_z(dof_cnt), ML(dof_cnt),
          beta(dof_cnt), z(dof_cnt), gp(dof_cnt), gm(dof_cnt);
   Array<int> dofs(dof_cnt);
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

      M_inv.Factor();
      M_inv.Mult(rhs, rho_HO);

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

   const int NE = pfes.GetMesh()->GetNE();
   const int nd = pfes.GetFE(0)->GetDof();
   DenseMatrix M_loc(nd);
   DenseMatrixInverse M_loc_inv(&M_loc);
   Vector rhs_loc(nd), du_loc(nd);
   Array<int> dofs;
   for (int k = 0; k < NE; k++)
   {
      pfes.GetElementDofs(k, dofs);
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
