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
#include "extrapolator.hpp"

using namespace std;
namespace mfem
{

namespace hydrodynamics
{

RemapAdvector::RemapAdvector(const ParMesh &m, int order_v, int order_e,
                             double cfl)
   : pmesh(m, true), dim(pmesh.Dimension()),
     fec_L2(order_e, pmesh.Dimension(), BasisType::Positive),
     fec_H1(order_v, pmesh.Dimension()),
     pfes_L2(&pmesh, &fec_L2, 1),
     pfes_H1(&pmesh, &fec_H1, pmesh.Dimension()),
     pfes_H1_s(&pmesh, &fec_H1, 1),
     cfl_factor(cfl),
     offsets(7), S(),
     xi(), v(), rho_1(), rho_2(), e_1(), e_2(), x0()
{
   const int vsize_H1 = pfes_H1.GetVSize(), vsize_L2 = pfes_L2.GetVSize();

   // Arrangement: level_set (1), velocity (dim), density (1), energy (1).
   offsets[0] = 0;
   offsets[1] = vsize_H1 / dim;
   offsets[2] = offsets[1] + vsize_H1;
   offsets[3] = offsets[2] + vsize_L2;
   offsets[4] = offsets[3] + vsize_L2;
   offsets[5] = offsets[4] + vsize_L2;
   offsets[6] = offsets[5] + vsize_L2;
   S.Update(offsets);

   xi.MakeRef(&pfes_H1_s, S, offsets[0]);
   v.MakeRef(&pfes_H1, S, offsets[1]);
   rho_1.MakeRef(&pfes_L2, S, offsets[2]);
   rho_2.MakeRef(&pfes_L2, S, offsets[3]);
   e_1.MakeRef(&pfes_L2, S, offsets[4]);
   e_2.MakeRef(&pfes_L2, S, offsets[5]);
}

void RemapAdvector::InitFromLagr(const Vector &nodes0,
                                 const ParGridFunction &interface,
                                 const ParGridFunction &vel,
                                 const IntegrationRule &rho_ir,
                                 const Vector &rhoDetJw_1,
                                 const Vector &rhoDetJw_2,
                                 const MaterialData &mat_data)
{
   x0 = nodes0;
   xi = interface;
   v  = vel;

   Extrapolator xtrap;
   xtrap.xtrap_type     = Extrapolator::ASLAM;
   xtrap.advection_mode = AdvectionOper::LO;
   xtrap.xtrap_degree   = 0;
   ParGridFunction lset_1(mat_data.level_set), lset_2(mat_data.level_set);
   // First material is for level_set < 0, so we need to flip.
   lset_1 *= -1;
   GridFunctionCoefficient lset_1_coeff(&lset_1), lset_2_coeff(&lset_2);

   // Extrapolate e_1.
   xtrap.Extrapolate(lset_1_coeff, mat_data.alpha_1,
                     mat_data.e_1, 5.0, e_1);
   // Extrapolate e_2.
   xtrap.Extrapolate(lset_2_coeff, mat_data.alpha_2,
                     mat_data.e_2, 5.0, e_2);

   e_1_max = e_1.Max();
   MPI_Allreduce(MPI_IN_PLACE, &e_1_max, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
   e_2_max = e_2.Max();
   MPI_Allreduce(MPI_IN_PLACE, &e_2_max, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());

   // Get densities as GridFunctions.
   SolutionMover mover(rho_ir);
   ParGridFunction r1(mat_data.e_1.ParFESpace()),
                   r2(mat_data.e_1.ParFESpace());
   mover.MoveDensityLR(rhoDetJw_1, r1);
   mover.MoveDensityLR(rhoDetJw_2, r2);

   // Extrapolate rho_1.
   xtrap.Extrapolate(lset_1_coeff, mat_data.alpha_1,
                     r1, 5.0, rho_1);
   // Extrapolate rho_2.
   xtrap.Extrapolate(lset_2_coeff, mat_data.alpha_2,
                     r2, 5.0, rho_2);
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
   AdvectorOper oper(S.Size(), x0, ess_tdofs, u, rho_1, rho_2,
                     pfes_H1, pfes_H1_s, pfes_L2);
   ode_solver.Init(oper);

   // Compute some time step [mesh_size / speed].
   double h_min = std::numeric_limits<double>::infinity();
   for (int k = 0; k < pmesh.GetNE(); k++)
   {
      h_min = std::min(h_min, pmesh.GetElementSize(k));
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

   if (v_max == 0.0) // No need to change the fields.
   {
      return;
   }

   v_max = std::sqrt(v_max);
   double dt = cfl_factor * h_min / v_max;

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

      double e_1_max_new = e_1.Max();
      MPI_Allreduce(MPI_IN_PLACE, &e_1_max_new, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
      double e_2_max_new = e_2.Max();
      MPI_Allreduce(MPI_IN_PLACE, &e_2_max_new, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
      if (e_1_max_new > e_1_max)
      {
         cout << e_1_max << " " << e_1_max_new << endl;
         MFEM_ABORT("\n e_1 max remap violation");
      }
      if (e_2_max_new > e_2_max)
      {
         cout << endl << e_2_max << " " << e_2_max_new << endl;
         MFEM_ABORT("e_2 max remap violation");
      }
   }
   if (pmesh.GetMyRank() == 0) { cout << endl; }
}

void RemapAdvector::TransferToLagr(ParGridFunction &vel,
                                   const IntegrationRule &ir_rho,
                                   Vector &rhoDetJw_1, Vector &rhoDetJw_2,
                                   MaterialData &mat_data, SIMarker &marker)
{
   mat_data.level_set = xi;
   vel = v;

   // Re-mark the elements and faces.
   ParMesh &pmesh_lagr = *vel.ParFESpace()->GetParMesh();
   const int NE = pmesh_lagr.GetNE();
   for (int e = 0; e < NE; e++)
   {
      int mat_id = marker.GetMaterialID(e);
      pmesh_lagr.SetAttribute(e, mat_id);
      marker.mat_attr(e) = mat_id;
   }
   marker.MarkFaceAttributes();

   // Update the volume fractions.
   UpdateAlpha(mat_data.level_set, mat_data.alpha_1, mat_data.alpha_2);

   // This will affect the update of mass matrices.
   mat_data.rho0_1 = rho_1;
   mat_data.rho0_2 = rho_2;

   const int nqp = ir_rho.GetNPoints();
   Vector rho_1_vals(nqp), rho_2_vals(nqp);
   for (int k = 0; k < NE; k++)
   {
      // Must use the space of the results.
      ElementTransformation &T = *pmesh_lagr.GetElementTransformation(k);
      rho_1.GetValues(T, ir_rho, rho_1_vals);
      rho_2.GetValues(T, ir_rho, rho_2_vals);
      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir_rho.IntPoint(q);
         T.SetIntPoint(&ip);
         rhoDetJw_1(k*nqp + q) = rho_1_vals(q) * mat_data.alpha_1(k) *
                                 T.Jacobian().Det() * ip.weight;
         rhoDetJw_2(k*nqp + q) = rho_2_vals(q) * mat_data.alpha_2(k) *
                                 T.Jacobian().Det() * ip.weight;
      }
   }

   mat_data.e_1 = e_1;
   mat_data.e_2 = e_2;
   const int ndofs = mat_data.e_1.Size() / NE;
   for (int e = 0; e < NE; e++)
   {
      const int attr = pmesh_lagr.GetAttribute(e);

      if (attr == 10)
      {
         for (int i = 0; i < ndofs; i++) { mat_data.e_2(e*ndofs + i) = 0.0; }
      }
      if (attr == 20)
      {
         for (int i = 0; i < ndofs; i++) { mat_data.e_1(e*ndofs + i) = 0.0; }
      }
   }

   mat_data.p_1->UpdateRho0Alpha0(mat_data.alpha_1, mat_data.rho0_1);
   mat_data.p_2->UpdateRho0Alpha0(mat_data.alpha_2, mat_data.rho0_2);
   mat_data.p_1->UpdatePressure(mat_data.alpha_1, mat_data.e_1);
   mat_data.p_2->UpdatePressure(mat_data.alpha_2, mat_data.e_2);
}

AdvectorOper::AdvectorOper(int size, const Vector &x_start,
                           const Array<int> &v_ess_td,
                           ParGridFunction &mesh_vel,
                           ParGridFunction &rho_1, ParGridFunction &rho_2,
                           ParFiniteElementSpace &pfes_H1,
                           ParFiniteElementSpace &pfes_H1_s,
                           ParFiniteElementSpace &pfes_L2)
   : TimeDependentOperator(size),
     fec_alpha(0, pfes_H1.GetMesh()->Dimension()),
     pfes_alpha(pfes_H1.GetParMesh(), &fec_alpha, 1),
     alpha_1(&pfes_alpha), alpha_2(&pfes_alpha),
     x0(x_start), x_now(*pfes_H1.GetMesh()->GetNodes()),
     v_ess_tdofs(v_ess_td),
     u(mesh_vel), u_coeff(&u),
     rho_coeff(alpha_1, alpha_2, rho_1, rho_2),
     rho_1_coeff(&rho_1), rho_2_coeff(&rho_2),
     rho_u_coeff(rho_coeff, u_coeff),
     rho_1_u_coeff(rho_1_coeff, u_coeff), rho_2_u_coeff(rho_2_coeff, u_coeff),
     M_H1(&pfes_H1_s), K_H1(&pfes_H1_s),
     Mr_H1(&pfes_H1), Kr_H1(&pfes_H1_s),
     M_L2(&pfes_L2), M_L2_Lump(&pfes_L2), K_L2(&pfes_L2),
     Mr_1_L2(&pfes_L2),  Mr_1_L2_Lump(&pfes_L2), Kr_1_L2(&pfes_L2),
     Mr_2_L2(&pfes_L2),  Mr_2_L2_Lump(&pfes_L2), Kr_2_L2(&pfes_L2)
{
   M_H1.AddDomainIntegrator(new MassIntegrator);
   M_H1.Assemble(0);
   M_H1.Finalize(0);

   K_H1.AddDomainIntegrator(new ConvectionIntegrator(u_coeff));
   K_H1.Assemble(0);
   K_H1.Finalize(0);

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

//   Mr_1_L2.AddDomainIntegrator(new MassIntegrator(rho_1_coeff));
   Mr_1_L2.AddDomainIntegrator(new MassIntegrator());
   Mr_1_L2.Assemble(0);
   Mr_1_L2.Finalize(0);

//   auto *minteg_1 = new MassIntegrator(rho_1_coeff);
   auto *minteg_1 = new MassIntegrator();
   Mr_1_L2_Lump.AddDomainIntegrator(new LumpedIntegrator(minteg_1));
   Mr_1_L2_Lump.Assemble();
   Mr_1_L2_Lump.Finalize();

//   Kr_1_L2.AddDomainIntegrator(new ConvectionIntegrator(rho_1_u_coeff));
//   auto dgt_ir_1 = new DGTraceIntegrator(rho_1_coeff, u_coeff, -1.0, -0.5);
//   auto dgt_br_1 = new DGTraceIntegrator(rho_1_coeff, u_coeff, -1.0, -0.5);
   Kr_1_L2.AddDomainIntegrator(new ConvectionIntegrator(u_coeff));
   auto dgt_ir_1 = new DGTraceIntegrator(u_coeff, -1.0, -0.5);
   auto dgt_br_1 = new DGTraceIntegrator(u_coeff, -1.0, -0.5);
   Kr_1_L2.AddInteriorFaceIntegrator(new TransposeIntegrator(dgt_ir_1));
   Kr_1_L2.AddBdrFaceIntegrator(new TransposeIntegrator(dgt_br_1));
   Kr_1_L2.KeepNbrBlock(true);
   // In parallel, the assembly of Kr_L2 needs to see values from MPI-neighbors.
   // That is, the rho_coeff must be evaluated in MPI-neighbor zones.
   rho_1.ExchangeFaceNbrData();
   Kr_1_L2.Assemble(0);
   Kr_1_L2.Finalize(0);

//   Mr_2_L2.AddDomainIntegrator(new MassIntegrator(rho_2_coeff));
   Mr_2_L2.AddDomainIntegrator(new MassIntegrator());
   Mr_2_L2.Assemble(0);
   Mr_2_L2.Finalize(0);

//   auto *minteg_2 = new MassIntegrator(rho_2_coeff);
   auto *minteg_2 = new MassIntegrator();
   Mr_2_L2_Lump.AddDomainIntegrator(new LumpedIntegrator(minteg_2));
   Mr_2_L2_Lump.Assemble();
   Mr_2_L2_Lump.Finalize();

//   Kr_2_L2.AddDomainIntegrator(new ConvectionIntegrator(rho_2_u_coeff));
//   auto dgt_ir_2 = new DGTraceIntegrator(rho_2_coeff, u_coeff, -1.0, -0.5);
//   auto dgt_br_2 = new DGTraceIntegrator(rho_2_coeff, u_coeff, -1.0, -0.5);
   Kr_2_L2.AddDomainIntegrator(new ConvectionIntegrator(u_coeff));
   auto dgt_ir_2 = new DGTraceIntegrator(u_coeff, -1.0, -0.5);
   auto dgt_br_2 = new DGTraceIntegrator(u_coeff, -1.0, -0.5);
   Kr_2_L2.AddInteriorFaceIntegrator(new TransposeIntegrator(dgt_ir_2));
   Kr_2_L2.AddBdrFaceIntegrator(new TransposeIntegrator(dgt_br_2));
   Kr_2_L2.KeepNbrBlock(true);
   // In parallel, the assembly of Kr_L2 needs to see values from MPI-neighbors.
   // That is, the rho_coeff must be evaluated in MPI-neighbor zones.
   rho_2.ExchangeFaceNbrData();
   Kr_2_L2.Assemble(0);
   Kr_2_L2.Finalize(0);
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

   // Arrangement: interface (1), velocity (dim), density (1), energy (1).
   Vector *U_ptr = const_cast<Vector *>(&U);

   // Update alphas.
   ParGridFunction xi_gf(M_H1.ParFESpace());
   Vector xi;
   xi.MakeRef(*U_ptr, 0, size_H1);
   xi_gf = xi;
   UpdateAlpha(xi_gf, alpha_1, alpha_2);

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

   // Interface level set remap.
   M_H1.BilinearForm::operator=(0.0);
   M_H1.Assemble();
   K_H1.BilinearForm::operator=(0.0);
   K_H1.Assemble();
   Mass_oper.Reset(M_H1.ParallelAssemble());
   lin_solver.SetOperator(*Mass_oper);
   Vector d_xi;
   d_xi.MakeRef(dU,  0, size_H1);
   K_H1.Mult(xi, rhs_H1);
   P_H1->MultTranspose(rhs_H1, RHS_H1);
   X = 0.0;
   lin_solver.Mult(RHS_H1, X);
   P_H1->Mult(X, d_xi);

   // Velocity remap.
   ParFiniteElementSpace &pfes_v = *Mr_H1.ParFESpace();
   Mr_H1.BilinearForm::operator=(0.0);
   Mr_H1.Assemble();
   Kr_H1.BilinearForm::operator=(0.0);
   Kr_H1.Assemble();
   Mass_oper.Reset(Mr_H1.ParallelAssemble());
   lin_solver.SetOperator(*Mass_oper);
   Vector v, d_v, rhs_v(size_H1*dim);
   v.MakeRef(*U_ptr, size_H1, size_H1*dim);
   d_v.MakeRef(dU,   size_H1, size_H1*dim);
   Vector v_comp, rhs_v_comp;
   for (int d = 0; d < dim; d++)
   {
      v_comp.MakeRef(v, d * size_H1, size_H1);
      rhs_v_comp.MakeRef(rhs_v, d * size_H1, size_H1);
      Kr_H1.Mult(v_comp, rhs_v_comp);
   }
   const Operator *P_v = pfes_v.GetProlongationMatrix();
   Vector RHS_V(P_v->Width()), X_V(P_v->Width());
   P_v->MultTranspose(rhs_v, RHS_V);
   X_V = 0.0;
   OperatorHandle M_elim;
   M_elim.EliminateRowsCols(Mass_oper, v_ess_tdofs);
   Mass_oper.EliminateBC(M_elim, v_ess_tdofs, X_V, RHS_V);
   lin_solver.Mult(RHS_V, X_V);
   P_v->Mult(X_V, d_v);

   Vector el_min(NE), el_max(NE);

   // Density remap.
   K_L2.BilinearForm::operator=(0.0);
   K_L2.Assemble();
   M_L2.BilinearForm::operator=(0.0);
   M_L2.Assemble();
   M_L2_Lump.BilinearForm::operator=(0.0);
   M_L2_Lump.Assemble();
   Vector rho_1, rho_2, d_rho_1, d_rho_2, d_rho_HO(size_L2), d_rho_LO(size_L2);
   Vector lumpedM; M_L2_Lump.SpMat().GetDiag(lumpedM);
   DiscreteUpwindLOSolver lo_solver(pfes_L2, K_L2.SpMat(), lumpedM);
   LocalInverseHOSolver ho_solver(M_L2, K_L2);
   Vector rho_min(size_L2), rho_max(size_L2);
   ParGridFunction rho_gf(&pfes_L2);
   FluxBasedFCT fct_solver(pfes_L2, dt,
                           K_L2.SpMat(), lo_solver.GetKmap(), M_L2.SpMat());
   // Density 1.
   rho_1.MakeRef(*U_ptr, (1 + dim) * size_H1, size_L2);
   d_rho_1.MakeRef(dU,   (1 + dim) * size_H1, size_L2);
   lo_solver.CalcLOSolution(rho_1, d_rho_LO);
   ho_solver.CalcHOSolution(rho_1, d_rho_HO);
   rho_gf = rho_1;
   rho_gf.ExchangeFaceNbrData();
   ComputeElementsMinMax(rho_gf, el_min, el_max);
   ComputeSparsityBounds(pfes_L2, el_min, el_max, rho_min, rho_max);
   fct_solver.CalcFCTSolution(rho_gf, lumpedM, d_rho_HO, d_rho_LO,
                              rho_min, rho_max, d_rho_1);
   // Density 2.
   rho_2.MakeRef(*U_ptr, (1 + dim) * size_H1 + size_L2, size_L2);
   d_rho_2.MakeRef(dU,   (1 + dim) * size_H1 + size_L2, size_L2);
   lo_solver.CalcLOSolution(rho_2, d_rho_LO);
   ho_solver.CalcHOSolution(rho_2, d_rho_HO);
   rho_gf = rho_2;
   rho_gf.ExchangeFaceNbrData();
   ComputeElementsMinMax(rho_gf, el_min, el_max);
   ComputeSparsityBounds(pfes_L2, el_min, el_max, rho_min, rho_max);
   fct_solver.CalcFCTSolution(rho_gf, lumpedM, d_rho_HO, d_rho_LO,
                              rho_min, rho_max, d_rho_2);

   // Energy 1 remap.
   auto rho_1_gf_const = dynamic_cast<const ParGridFunction *>
                         (rho_1_coeff.GetGridFunction());
   auto rho_1_pgf = const_cast<ParGridFunction *>(rho_1_gf_const);
   rho_1_pgf->ExchangeFaceNbrData();
   Kr_1_L2.BilinearForm::operator=(0.0);
   Kr_1_L2.Assemble();
   Mr_1_L2.BilinearForm::operator=(0.0);
   Mr_1_L2.Assemble();
   Mr_1_L2_Lump.BilinearForm::operator=(0.0);
   Mr_1_L2_Lump.Assemble();
   Vector e_1, d_e_1, e_2, d_e_2, d_e_HO(size_L2), d_e_LO(size_L2), Me_lumped;
   Vector e_min(size_L2), e_max(size_L2);
   ParGridFunction e_gf(&pfes_L2);
   Mr_1_L2_Lump.SpMat().GetDiag(Me_lumped);
   DiscreteUpwindLOSolver lo_e_solver(pfes_L2, Kr_1_L2.SpMat(), Me_lumped);
   LocalInverseHOSolver ho_e_solver(Mr_1_L2, Kr_1_L2);
   FluxBasedFCT fct_e_solver(pfes_L2, dt, Kr_1_L2.SpMat(),
                             lo_e_solver.GetKmap(), Mr_1_L2.SpMat());
   e_1.MakeRef(*U_ptr, (1 + dim) * size_H1 + 2 * size_L2, size_L2);
   d_e_1.MakeRef(dU,   (1 + dim) * size_H1 + 2 * size_L2, size_L2);
   lo_e_solver.CalcLOSolution(e_1, d_e_LO);
   ho_e_solver.CalcHOSolution(e_1, d_e_HO);
   e_gf = e_1;
   e_gf.ExchangeFaceNbrData();
   ComputeElementsMinMax(e_gf, el_min, el_max);
   ComputeSparsityBounds(pfes_L2, el_min, el_max, e_min, e_max);
   fct_e_solver.CalcFCTSolution(e_gf, Me_lumped, d_e_HO, d_e_LO,
                                e_min, e_max, d_e_1);
   // Energy 2 remap.
   auto rho_2_gf_const = dynamic_cast<const ParGridFunction *>
                       (rho_2_coeff.GetGridFunction());
   auto rho_2_pgf = const_cast<ParGridFunction *>(rho_2_gf_const);
   rho_2_pgf->ExchangeFaceNbrData();
   Kr_2_L2.BilinearForm::operator=(0.0);
   Kr_2_L2.Assemble();
   Mr_2_L2.BilinearForm::operator=(0.0);
   Mr_2_L2.Assemble();
   Mr_2_L2_Lump.BilinearForm::operator=(0.0);
   Mr_2_L2_Lump.Assemble();
   Mr_2_L2_Lump.SpMat().GetDiag(Me_lumped);
   DiscreteUpwindLOSolver lo_e_2_solver(pfes_L2, Kr_2_L2.SpMat(), Me_lumped);
   LocalInverseHOSolver ho_e_2_solver(Mr_2_L2, Kr_2_L2);
   FluxBasedFCT fct_e_2_solver(pfes_L2, dt, Kr_2_L2.SpMat(),
                               lo_e_2_solver.GetKmap(), Mr_2_L2.SpMat());
   e_2.MakeRef(*U_ptr, (1 + dim) * size_H1 + 3 * size_L2, size_L2);
   d_e_2.MakeRef(dU,   (1 + dim) * size_H1 + 3 * size_L2, size_L2);
   lo_e_2_solver.CalcLOSolution(e_2, d_e_LO);
   ho_e_2_solver.CalcHOSolution(e_2, d_e_HO);
   e_gf = e_2;
   e_gf.ExchangeFaceNbrData();
   ComputeElementsMinMax(e_gf, el_min, el_max);
   ComputeSparsityBounds(pfes_L2, el_min, el_max, e_min, e_max);
   fct_e_2_solver.CalcFCTSolution(e_gf, Me_lumped, d_e_HO, d_e_LO,
                                  e_min, e_max, d_e_2);
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

double AdvectorOper::Interface(ParGridFunction &xi, double t)
{
   add(x0, t, u, x_now);

   M_H1.BilinearForm::operator=(0.0);
   M_H1.Assemble();

   Vector one(M_H1.SpMat().Height());
   one = 1.0;
   double loc_d  = M_H1.InnerProduct(one, xi);

   double glob_d;
   MPI_Allreduce(&loc_d, &glob_d, 1, MPI_DOUBLE, MPI_SUM,
                 M_H1.ParFESpace()->GetComm());
   return glob_d;
}

double AdvectorOper::Energy(ParGridFunction &e, double t)
{
   add(x0, t, u, x_now);

   Mr_1_L2.BilinearForm::operator=(0.0);
   Mr_1_L2.Assemble();

   Vector one(Mr_1_L2.SpMat().Height());
   one = 1.0;
   double loc_e = Mr_1_L2.InnerProduct(one, e);

   double glob_e;
   MPI_Allreduce(&loc_e, &glob_e, 1, MPI_DOUBLE, MPI_SUM,
                 Mr_1_L2.ParFESpace()->GetComm());
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
