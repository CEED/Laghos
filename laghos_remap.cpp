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
#include "laghos_solver.hpp"

using namespace std;
namespace mfem
{

namespace hydrodynamics
{

RemapAdvector::RemapAdvector(const ParMesh &m, int order_v, int order_e,
                             double cfl, bool remap_vel, const Array<int> &ess_tdofs)
   : pmesh(m, true), dim(pmesh.Dimension()),
     fec_L2(order_e, pmesh.Dimension(), BasisType::Positive),
     fec_H1(order_v, pmesh.Dimension(), BasisType::Positive),
     fec_H1Lag(order_v, pmesh.Dimension()),
     pfes_L2(&pmesh, &fec_L2, 1),
     pfes_H1(&pmesh, &fec_H1, pmesh.Dimension()),
     pfes_H1Lag(&pmesh, &fec_H1Lag, pmesh.Dimension()),
     v_ess_tdofs(ess_tdofs),
     //M_mixed(&pfes_H1, &pfes_H1Lag),
     //M_mixed(&pfes_H1Lag, &pfes_H1),
     remap_v(remap_vel),
     cfl_factor(cfl),
     offsets(4), S(),
     v(), rho(), e(), x0()
{
   const int vsize_H1 = pfes_H1.GetVSize(), vsize_L2 = pfes_L2.GetVSize();
   //const int dim = pmesh.Dimension();
   //const int ssice_H1 = pfes_H1.GetVSize() / dim;

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
   e  = energy;
   // project velocity field into Bernstein FE space via lumped L2 projection
   ParMixedBilinearForm M_mixed(&pfes_H1Lag, &pfes_H1);
   M_mixed.AddDomainIntegrator(new VectorMassIntegrator());
   M_mixed.Assemble(0);
   M_mixed.Finalize(0);

   HypreParMatrix *M_mixed_hpm = M_mixed.ParallelAssemble();

   ParBilinearForm M_lumped(&pfes_H1);
   M_lumped.AddDomainIntegrator(new LumpedIntegrator(new VectorMassIntegrator()));
   M_lumped.Assemble(0);
   M_lumped.Finalize(0);

   Vector lumped_vec(M_lumped.Height());
   M_lumped.SpMat().GetDiag(lumped_vec);
   GroupCommunicator &gcomm = pfes_H1.GroupComm();
   Array<double> lumpedmassmatrix_array(lumped_vec.GetData(), lumped_vec.Size());
   gcomm.Reduce<double>(lumpedmassmatrix_array, GroupCommunicator::Sum);
   gcomm.Bcast(lumpedmassmatrix_array);

   const Operator *P_lag = pfes_H1Lag.GetProlongationMatrix();
   const Operator *P_bern = pfes_H1.GetProlongationMatrix();
   Vector VEL(P_lag->Width()), B(P_bern->Width());
   Vector b(v.Size());
   P_lag->MultTranspose(vel, VEL);
   M_mixed_hpm->Mult(VEL, B);
   P_bern->Mult(B, b);

   int i_td, ess_tdof_index;
   for(int i = 0; i < v.Size(); i++)
   {  
      //v(i) = 0.0;
      i_td = pfes_H1.GetLocalTDofNumber(i);
      ess_tdof_index = v_ess_tdofs.Find(i_td);

      bool not_essential_tdof = (i_td != -1 && ess_tdof_index == -1);
      v(i) = not_essential_tdof  * b(i) / lumped_vec(i);
      if(!not_essential_tdof)
      {
         MFEM_VERIFY(abs(v(i)) < 1e-15, "bool multiplication weird");
      }
   } 
   Array<double> v_array(v.GetData(), v.Size());
   gcomm.Reduce<double>(v_array, GroupCommunicator::Sum);
   gcomm.Bcast(v_array);


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

   socketstream vis_v;
   char vishost[] = "localhost";
   int  visport   = 19916;
   int Wx = 0, Wy = 0; // window position
   int Ww = 350, Wh = 350; // window size
   int offx = Ww+10; // window offsets
   Wx += offx;
   Wy += offx;

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

      VisualizeField(vis_v, vishost, visport,
                                          v, "Remapped Velocity", Wx, Wy, Ww, Wh);
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
   
   ParMixedBilinearForm M_mixed(&pfes_H1, &pfes_H1Lag);
   M_mixed.AddDomainIntegrator(new VectorMassIntegrator());
   M_mixed.Assemble(0);
   M_mixed.Finalize(0);

   const Operator *P_lag = pfes_H1Lag.GetProlongationMatrix();
   const Operator *P_bern = pfes_H1.GetProlongationMatrix();
   HypreParMatrix *M_mixed_hpm = M_mixed.ParallelAssemble();
   Vector V(P_bern->Width()), B(P_bern->Width());
   Vector b(v.Size());
   P_bern->MultTranspose(v, V);
   M_mixed_hpm->Mult(V, B);
   P_lag->Mult(B, b);

   ParBilinearForm M_lumped(&pfes_H1Lag);
   M_lumped.AddDomainIntegrator(new LumpedIntegrator(new VectorMassIntegrator()));
   M_lumped.Assemble(0);
   M_lumped.Finalize(0);

   Vector lumped_vec(M_lumped.Height());
   M_lumped.SpMat().GetDiag(lumped_vec);
   GroupCommunicator &gcomm = pfes_H1Lag.GroupComm();
   Array<double> lumpedmassmatrix_array(lumped_vec.GetData(), lumped_vec.Size());
   gcomm.Reduce<double>(lumpedmassmatrix_array, GroupCommunicator::Sum);
   gcomm.Bcast(lumpedmassmatrix_array);

   MFEM_VERIFY(vel.Size() == lumped_vec.Size(), "lumped size wrong");
   MFEM_VERIFY(v.Size() == vel.Size(), "lumped size wrong");
   // Just copy velocity.
   int i_td, ess_tdof_index;
   for(int i = 0; i < v.Size(); i++)
   {
      i_td = pfes_H1.GetLocalTDofNumber(i);
      ess_tdof_index = v_ess_tdofs.Find(i_td);
      bool not_essential_tdof = (i_td != -1 && ess_tdof_index == -1);
      vel(i) = not_essential_tdof * b(i) / lumped_vec(i);

      if(!not_essential_tdof)
      {
         MFEM_VERIFY(abs(vel(i)) < 1e-15, "bool multiplication weird");
      }
      Array<double> vel_array(vel.GetData(), vel.Size());
      gcomm.Reduce<double>(vel_array, GroupCommunicator::Sum);
      gcomm.Bcast(vel_array);
   }
   //vel = v;



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
     u(mesh_vel), u_coeff(&u), u_hpr(&pfes_H1),
     rho_coeff(&rho),
     rho_u_coeff(rho_coeff, u_coeff),
     Mr_H1(&pfes_H1_s), Kr_H1(&pfes_H1_s), KrT_H1(&pfes_H1_s), lummpedMr_H1(&pfes_H1_s), 
     Cr_H1(&pfes_H1, &pfes_H1_s),
     M_L2(&pfes_L2), M_L2_Lump(&pfes_L2), K_L2(&pfes_L2),
     Mr_L2(&pfes_L2),  Mr_L2_Lump(&pfes_L2), Kr_L2(&pfes_L2)
{
   // no need for Vector Massmatrix
   //Mr_H1.AddDomainIntegrator(new VectorMassIntegrator(rho_coeff));
   Mr_H1.AddDomainIntegrator(new MassIntegrator(rho_coeff));
   Mr_H1.Assemble(0);
   Mr_H1.Finalize(0);

   // lumped Massmatrix
   lummpedMr_H1.AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator(rho_coeff)));
   lummpedMr_H1.Assemble(0);
   lummpedMr_H1.Finalize(0);
   //lummpedMr_H1.SpMat().GetDiag(lumpedMr_H1_vec);

   //Kr_H1.AddDomainIntegrator(new TransposeIntegrator(new ConvectionIntegrator(rho_u_coeff)));
   Kr_H1.AddDomainIntegrator(new ConvectionIntegrator(rho_u_coeff));
   Kr_H1.Assemble(0);
   Kr_H1.Finalize(0);

   KrT_H1.AddDomainIntegrator(new TransposeIntegrator(new ConvectionIntegrator(rho_u_coeff)));
   KrT_H1.Assemble(0);
   KrT_H1.Finalize(0);
   
   Cr_H1.AddDomainIntegrator(new DivergenceIntegrator(rho_coeff));
   Cr_H1.Assemble(0);
   Cr_H1.Finalize(0);

   /*
   global_to_local.SetSize(pfes_H1_s.GlobalTrueVSize());
   global_to_local = -1;
   int counter = 0;
   for(int g = 0; g < global_to_local.Size(); g++)
   {
      for(int i = 0; i < pfes_H1_s.GetVSize(); i++)
      {
         if(pfes_H1_s.GetGlobalTDofNumber(i) == g)
         {
            global_to_local[g] = i;
            counter++;
            break;
         }
      }
      if(counter == pfes_H1_s.GetVSize())
      {
         break;
      }
   }
   //*/

   for(int i = 0; i < pfes_H1_s.GetVSize(); i++)
   {
      int i_td = pfes_H1_s.GetLocalTDofNumber(i);
      if(i_td == -1){continue;}

      u_hpr(i_td) = u(i);
   }

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
   ParFiniteElementSpace &pfes_H1_s = *Mr_H1.ParFESpace(),
                         &pfes_L2 = *M_L2.ParFESpace();
   const int dim     = pfes_H1_s.GetMesh()->Dimension();
   const int NE      = pfes_H1_s.GetNE();
   const int dofs_h1 = pfes_H1_s.GetVSize(), size_L2 = pfes_L2.GetVSize();
   const int dofs_h1_glb = pfes_H1_s.GlobalTrueVSize();

   // Move the mesh.
   const double t = GetTime();
   add(x0, t, u, x_now);

   dU = 0.0;

   // Arrangement: interface (1), velocity (dim), density (1), energy (1).
   Vector *U_ptr = const_cast<Vector *>(&U);

   /*
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
      OperatorHandle Mass_oper;

      // Velocity remap.
      Mr_H1.BilinearForm::operator=(0.0);
      Mr_H1.Assemble();
      Kr_H1.BilinearForm::operator=(0.0);
      Kr_H1.Assemble();
      Mass_oper.Reset(Mr_H1.ParallelAssemble());
      lin_solver.SetOperator(*Mass_oper);
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
      M_elim.EliminateRowsCols(Mass_oper, v_ess_tdofs);
      Mass_oper.EliminateBC(M_elim, v_ess_tdofs, X_V, RHS_V);
      lin_solver.Mult(RHS_V, X_V);
      P_v->Mult(X_V, d_v);
   }
   //*/

   if (remap_v)
   {
      Mr_H1.BilinearForm::operator=(0.0);
      Mr_H1.Assemble();
      Kr_H1.BilinearForm::operator=(0.0);
      Kr_H1.Assemble();
      KrT_H1.BilinearForm::operator=(0.0);
      KrT_H1.Assemble();
      //Cr_H1.MixedBilinearForm::operator=(0.0);
      //Cr_H1.Assemble();

      //Kr_H1.SpMat().Print();
      //MFEM_ABORT("");
      
      lummpedMr_H1.BilinearForm::operator=(0.0);
      lummpedMr_H1.Assemble();
      lummpedMr_H1.SpMat().GetDiag(lumpedMr_H1_vec);

      // Sum up to get global entries of the lumped mass matrix
      //*
      GroupCommunicator &gcomm = lummpedMr_H1.ParFESpace()->GroupComm();
      Array<double> lumpedmassmatrix_array(lumpedMr_H1_vec.GetData(), lumpedMr_H1_vec.Size());
      gcomm.Reduce<double>(lumpedmassmatrix_array, GroupCommunicator::Sum);
      gcomm.Bcast(lumpedmassmatrix_array);
      //*/

      for(int i = 0; i < lumpedMr_H1_vec.Size(); i++)
      {
         MFEM_VERIFY(lumpedMr_H1_vec(i) > 1e-12, "lumped mass matrix entry negative or zero!");
      }

      // Get the global stencil of the local truedofs
      HypreParMatrix *M_hpm = Mr_H1.ParallelAssemble();
      HypreParMatrix *K_hpm = Kr_H1.ParallelAssemble();
      HypreParMatrix *KT_hpm = KrT_H1.ParallelAssemble();
      HypreParMatrix *C_hpm = Cr_H1.ParallelAssemble();
      SparseMatrix M_glb, K_glb, KT_glb, C_glb;
      M_hpm->MergeDiagAndOffd(M_glb);
      K_hpm->MergeDiagAndOffd(K_glb);
      KT_hpm->MergeDiagAndOffd(KT_glb);
      C_hpm->MergeDiagAndOffd(C_glb);
      
      Vector v, v_d, d_v;
      v.MakeRef(*U_ptr, 0, dofs_h1*dim);
      d_v.MakeRef(dU, 0, dofs_h1*dim);
      int scheme = 2;
      switch(scheme)
      {
         case 0: LowOrderVel(K_glb, KT_glb, v, d_v); break;
         case 1: HighOrderTargetSchemeVel(K_glb, KT_glb, M_glb, v, d_v); break;
         case 2: MCLVel(K_glb, KT_glb, M_glb, v, d_v); break;
         default: MFEM_ABORT("Unknown scheme for velocity remap!");
      }
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

void AdvectorOper::LowOrderVel(const SparseMatrix &K_glb, const SparseMatrix &KT_glb, Vector &v, Vector &d_v) const
{
   GroupCommunicator &gcomm = lummpedMr_H1.ParFESpace()->GroupComm();
   //Array<double> lumpedmassmatrix_array(lumpedMr_H1_vec.GetData(), lumpedMr_H1_vec.Size());
   //gcomm.Reduce<double>(lumpedmassmatrix_array, GroupCommunicator::Sum);
   //gcomm.Bcast(lumpedmassmatrix_array);

   ParFiniteElementSpace &pfes_H1_s = *Kr_H1.ParFESpace();
   const int dim     = pfes_H1_s.GetMesh()->Dimension();
   const int dofs_h1 = pfes_H1_s.GetVSize();
   const int dofs_h1_glb = pfes_H1_s.GlobalTrueVSize();

   d_v = 0.0;
   Vector v_d;
   Array<double> rhs_array(dofs_h1);
   HypreParVector v_d_hpr(&pfes_H1_s);

   const auto I = K_glb.ReadI();
   const auto J = K_glb.ReadJ();
   const auto K = K_glb.ReadData();
   
   for(int d = 0; d < dim; d++)
   {
      v_d.MakeRef(v, d * dofs_h1, dofs_h1);
      for(int i = 0; i < dofs_h1; i++)
      {
         int i_td = pfes_H1_s.GetLocalTDofNumber(i);
         if(i_td != -1)
         {
            v_d_hpr(i_td) = v_d(i);
         }
      }
      Vector *v_d_glb = v_d_hpr.GlobalVector();
      MFEM_VERIFY(v_d_hpr.Size() == K_glb.Height(), "true dof local vector size weird");
      MFEM_VERIFY( v_d_glb->Size() == pfes_H1_s.GlobalTrueVSize(), "glb vector size weird");

      for(int i = 0; i < dofs_h1; i++)
      {  
         rhs_array[i] = 0.0;

         int i_td = pfes_H1_s.GetLocalTDofNumber(i);
         if(i_td == -1) {continue;}

         // check for essential true dof
         int index = v_ess_tdofs.Find(i_td + d * pfes_H1_s.TrueVSize());
         // if it is essential true dof, then skip it
         if(index != -1)
         {
            // if truedof set dv/dt to zero under the assumption, 
            // that the last timestep already satisfies the bc
            continue;
         }

         int i_gl = pfes_H1_s.GetGlobalTDofNumber(i);

         for(int k = I[i_td]; k < I[i_td+1]; k++)
         {  
            int j_gl = J[k];
            if( i_gl == j_gl ) {continue;}

            double dij = max( 0.0, max(K[k], KT_glb(i_td, j_gl)));
            rhs_array[i] += (dij + K[k]) * ( v_d_glb->Elem(j_gl) -  v_d_glb->Elem(i_gl) );
         }
         rhs_array[i] /= lumpedMr_H1_vec(i);


      }

      gcomm.Reduce<double>(rhs_array, GroupCommunicator::Sum);
      gcomm.Bcast(rhs_array);

      for(int i = 0; i < dofs_h1; i++)
      {  
         d_v(i + d * dofs_h1) = rhs_array[i]; 
      }
      delete v_d_glb;
   }
}

void AdvectorOper::HighOrderTargetSchemeVel(const SparseMatrix &K_glb, const SparseMatrix &KT_glb, const SparseMatrix &M_glb, Vector &v, Vector &d_v) const
{
   GroupCommunicator &gcomm = lummpedMr_H1.ParFESpace()->GroupComm();
   //Array<double> lumpedmassmatrix_array(lumpedMr_H1_vec.GetData(), lumpedMr_H1_vec.Size());
   //gcomm.Reduce<double>(lumpedmassmatrix_array, GroupCommunicator::Sum);
   //gcomm.Bcast(lumpedmassmatrix_array);

   ParFiniteElementSpace &pfes_H1_s = *Kr_H1.ParFESpace();
   const int dim     = pfes_H1_s.GetMesh()->Dimension();
   const int dofs_h1 = pfes_H1_s.GetVSize();
   const int dofs_h1_glb = pfes_H1_s.GlobalTrueVSize();

   d_v = 0.0;
   Vector v_d;
   Array<double> rhs_array(dofs_h1), udot_array(dofs_h1);
   HypreParVector v_d_hpr(&pfes_H1_s), vdot(&pfes_H1_s);

   const auto I = K_glb.ReadI();
   const auto J = K_glb.ReadJ();
   const auto K = K_glb.ReadData();
   const auto M = M_glb.ReadData();
   
   for(int d = 0; d < dim; d++)
   {
      v_d.MakeRef(v, d * dofs_h1, dofs_h1);
      for(int i = 0; i < dofs_h1; i++)
      {
         int i_td = pfes_H1_s.GetLocalTDofNumber(i);
         if(i_td != -1)
         {
            v_d_hpr(i_td) = v_d(i);
         }
      }
      Vector *v_d_glb = v_d_hpr.GlobalVector();
      MFEM_VERIFY(v_d_hpr.Size() == K_glb.Height(), "true dof local vector size weird");
      MFEM_VERIFY( v_d_glb->Size() == pfes_H1_s.GlobalTrueVSize(), "glb vector size weird");

      for(int i = 0; i < dofs_h1; i++)
      {  
         int i_td = pfes_H1_s.GetLocalTDofNumber(i);
         if(i_td == -1) {continue;}
         vdot(i_td) = 0.0;
         int i_gl = pfes_H1_s.GetGlobalTDofNumber(i);

         for(int k = I[i_td]; k < I[i_td+1]; k++)
         {  
            int j_gl = J[k];
            if( i_gl == j_gl ) {continue;}

            double dij = max( 0.0, max(K[k], KT_glb(i_td, j_gl)));
            vdot(i_td) += (dij + K[k]) * ( v_d_glb->Elem(j_gl) -  v_d_glb->Elem(i_gl) );
         }
         vdot(i_td) /= lumpedMr_H1_vec(i);
      }

      Vector *vdot_glb = vdot.GlobalVector();


      for(int i = 0; i < dofs_h1; i++)
      {  
         rhs_array[i] = 0.0;

         int i_td = pfes_H1_s.GetLocalTDofNumber(i);
         if(i_td == -1) {continue;}

         // check for essential true dof
         int index = v_ess_tdofs.Find(i_td + d * pfes_H1_s.TrueVSize());
         // if it is essential true dof, then skip it
         if(index != -1){continue;}

         for(int k = I[i_td]; k < I[i_td+1]; k++)
         {  
            int j_gl = J[k];
            int i_gl = pfes_H1_s.GetGlobalTDofNumber(i);
            if( i_gl == j_gl ) {continue;}

            //double dij = max( 0.0, max(K[k], KT_glb(i_td, j_gl)));
            rhs_array[i] += K[k] * ( v_d_glb->Elem(j_gl) -  v_d_glb->Elem(i_gl)) + M[k] * ( vdot_glb->Elem(i_gl) - vdot_glb->Elem(j_gl));
         }
         rhs_array[i] /= lumpedMr_H1_vec(i);
      }

      gcomm.Reduce<double>(rhs_array, GroupCommunicator::Sum);
      gcomm.Bcast(rhs_array);


      for(int i = 0; i < dofs_h1; i++)
      {  
         d_v(i + d * dofs_h1) = rhs_array[i]; 
      }
      delete v_d_glb;
      delete vdot_glb;
   }
}

void AdvectorOper::MCLVel(const SparseMatrix &K_glb, const SparseMatrix &KT_glb, const SparseMatrix &M_glb, Vector &v, Vector &d_v) const
{
   GroupCommunicator &gcomm = lummpedMr_H1.ParFESpace()->GroupComm();
   //Array<double> lumpedmassmatrix_array(lumpedMr_H1_vec.GetData(), lumpedMr_H1_vec.Size());
   //gcomm.Reduce<double>(lumpedmassmatrix_array, GroupCommunicator::Sum);
   //gcomm.Bcast(lumpedmassmatrix_array);

   ParFiniteElementSpace &pfes_H1_s = *Kr_H1.ParFESpace();
   const int dim     = pfes_H1_s.GetMesh()->Dimension();
   const int dofs_h1 = pfes_H1_s.GetVSize();
   const int dofs_h1_glb = pfes_H1_s.GlobalTrueVSize();

   d_v = 0.0;
   Vector v_d;
   Array<double> rhs_array(dofs_h1), udot_array(dofs_h1);
   HypreParVector v_d_hpr(&pfes_H1_s), vdot(&pfes_H1_s), v_min(&pfes_H1_s), v_max(&pfes_H1_s);
   double fij, fij_bound, fij_star, wij, wji;

   const auto I = K_glb.ReadI();
   const auto J = K_glb.ReadJ();
   const auto K = K_glb.ReadData();
   const auto M = M_glb.ReadData();
   
   for(int d = 0; d < dim; d++)
   {
      v_d.MakeRef(v, d * dofs_h1, dofs_h1);
      for(int i = 0; i < dofs_h1; i++)
      {
         int i_td = pfes_H1_s.GetLocalTDofNumber(i);
         if(i_td != -1)
         {
            v_d_hpr(i_td) = v_d(i);
         }
      }
      Vector *v_d_glb = v_d_hpr.GlobalVector();
      MFEM_VERIFY(v_d_hpr.Size() == K_glb.Height(), "true dof local vector size weird");
      MFEM_VERIFY( v_d_glb->Size() == pfes_H1_s.GlobalTrueVSize(), "glb vector size weird");

      //compute low order time derivatives and local min and max
      for(int i = 0; i < dofs_h1; i++)
      {  
         int i_td = pfes_H1_s.GetLocalTDofNumber(i);
         if(i_td == -1) {continue;}
         vdot(i_td) = 0.0;
         int i_gl = pfes_H1_s.GetGlobalTDofNumber(i);

         v_min(i_td) = v_d_glb->Elem(i_gl);
         v_max(i_td) = v_d_glb->Elem(i_gl);
         for(int k = I[i_td]; k < I[i_td+1]; k++)
         {  
            int j_gl = J[k];
            if( i_gl == j_gl ) {continue;}
            v_min(i_td) = min(v_min(i_td), v_d_glb->Elem(j_gl));
            v_max(i_td) = max(v_max(i_td), v_d_glb->Elem(j_gl));
            double dij = max( 0.0, max(K[k], KT_glb(i_td, j_gl)));
            vdot(i_td) += (dij + K[k]) * ( v_d_glb->Elem(j_gl) -  v_d_glb->Elem(i_gl) );
         }
         vdot(i_td) /= lumpedMr_H1_vec(i);
      }

      Vector *vdot_glb = vdot.GlobalVector();
      Vector *vmin_glb = v_min.GlobalVector();
      Vector *vmax_glb = v_max.GlobalVector();


      for(int i = 0; i < dofs_h1; i++)
      {  
         rhs_array[i] = 0.0;

         int i_td = pfes_H1_s.GetLocalTDofNumber(i);
         if(i_td == -1) {continue;}

         // check for essential true dof
         int index = v_ess_tdofs.Find(i_td + d * pfes_H1_s.TrueVSize());
         // if it is essential true dof, then skip it
         if(index != -1){continue;}

         for(int k = I[i_td]; k < I[i_td+1]; k++)
         {  
            int j_gl = J[k];
            int i_gl = pfes_H1_s.GetGlobalTDofNumber(i);
            if( i_gl == j_gl ) {continue;}

            double dij = max( 0.0, max(K[k], KT_glb(i_td, j_gl)));

            // compute target flux
            fij = M[k] * (vdot_glb->Elem(i_gl) - vdot_glb->Elem(j_gl)) + dij * (v_d_glb->Elem(i_gl) - v_d_glb->Elem(j_gl));
            
            //limit target flux to enforce local bounds for the bar states (note, that dij = dji)
            wij = dij * (v_d_glb->Elem(i_gl) + v_d_glb->Elem(j_gl))  + K[k] * (v_d_glb->Elem(j_gl) - v_d_glb->Elem(i_gl));
            wji = dij * (v_d_glb->Elem(i_gl) + v_d_glb->Elem(j_gl))  + KT_glb(i_td, j_gl) * (v_d_glb->Elem(i_gl) - v_d_glb->Elem(j_gl)); 

            if(fij > 0)
            {
               fij_bound = min(2.0 * dij * vmax_glb->Elem(i_gl) - wij, wji - 2.0 * dij * vmin_glb->Elem(j_gl));
               fij_star = min(fij, fij_bound);

               // to get rid of rounding errors wich influence the sign 
               //fij_star = max(0.0, fij_star);
            }
            else
            {
               fij_bound = max(2.0 * dij * vmin_glb->Elem(i_gl) - wij, wji - 2.0 * dij * vmax_glb->Elem(j_gl));
               fij_star = max(fij, fij_bound);

               // to get rid of rounding errors wich influence the sign
               //fij_star = min(0.0, fij_star);
            }

            rhs_array[i] += (dij + K[k]) * ( v_d_glb->Elem(j_gl) -  v_d_glb->Elem(i_gl)) + fij_star;
         }
         rhs_array[i] /= lumpedMr_H1_vec(i);
      }

      gcomm.Reduce<double>(rhs_array, GroupCommunicator::Sum);
      gcomm.Bcast(rhs_array);


      for(int i = 0; i < dofs_h1; i++)
      {  
         d_v(i + d * dofs_h1) = rhs_array[i]; 
      }

      delete v_d_glb;
      delete vdot_glb;
      delete vmin_glb;
      delete vmax_glb;
   }
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


void DivergenceIntegrator::AssembleElementMatrix2(
        const FiniteElement &trial_fe, const FiniteElement &test_fe,
        ElementTransformation &Trans,  DenseMatrix &elmat)
{
   dim = trial_fe.GetDim();
   int trial_dof = trial_fe.GetDof();
   int test_dof = test_fe.GetDof();
   Vector v(dim);
   Vector d_col;
 
   dshape.SetSize(trial_dof, dim);
   gshape.SetSize(trial_dof, dim);
   Jadj.SetSize(dim);
   shape.SetSize(test_dof);
   elmat.SetSize(trial_dof, dim * test_dof);

   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(trial_fe, test_fe,
                                                             Trans);
 
   elmat = 0.0;
   elmat_comp.SetSize(test_dof, trial_dof);
 
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trial_fe.CalcDShape(ip, dshape);
      test_fe.CalcShape(ip, shape);
 
      Trans.SetIntPoint(&ip);
      CalcAdjugate(Trans.Jacobian(), Jadj);
 
      Mult(dshape, Jadj, gshape);
      double rho = 1.0;
      if (Q)
      {
         rho = Q->Eval(Trans, ip); 
      }
      shape *= ip.weight;

      for(int d = 0; d < dim; d++)
      {
         for(int j = 0; j < test_dof; j++)
         {
            for(int k = 0; k < trial_dof; k++)
            {
               elmat(j, k + d * trial_dof) += shape(j) * gshape(k,d) * rho;
            }
         }
      }
   }
}


const IntegrationRule &DivergenceIntegrator::GetRule(const FiniteElement
                                                    &trial_fe,
                                                    const FiniteElement &test_fe,
                                                   ElementTransformation &Trans)
{
   int order = Trans.OrderGrad(&trial_fe) + test_fe.GetOrder() + Trans.OrderJ();
   return IntRules.Get(trial_fe.GetGeomType(), order);
}

} // namespace hydrodynamics

} // namespace mfem
