// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include "general/forall.hpp"
#include "laghos_solver.hpp"
#include "linalg/kernels.hpp"
#include <unordered_map>

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{

namespace hydrodynamics
{

void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    ParGridFunction &gf, const char *title,
                    int x, int y, int w, int h, bool vec, const char *keys_in)
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
         const char* keys = (gf.FESpace()->GetMesh()->Dimension() == 2)
                            ? "mAcRjl" : "mmaaAcl";

         sock << "window_title '" << title << "'\n"
              << "window_geometry "
              << x << " " << y << " " << w << " " << h << "\n"
              << "keys " << (keys_in ? keys_in : keys);
         if ( vec ) { sock << "vvv"; }
         sock << std::endl;
      }

      if (myid == 0)
      {
         connection_failed = !sock && !newly_opened;
      }
      MPI_Bcast(&connection_failed, 1, MPI_INT, 0, comm);
   }
   while (connection_failed);
}

void LengthScaleAndCompression(const DenseMatrix &sgrad_v,
                               ElementTransformation &T,
                               const DenseMatrix &Jac0inv, double h0,
                               double &h, double &mu)
{
   const int dim = sgrad_v.Height();

   double eig_val_data[3], eig_vec_data[9];
   if (dim == 1)
   {
      eig_val_data[0] = sgrad_v(0, 0);
      eig_vec_data[0] = 1.;
   }
   else { sgrad_v.CalcEigenvalues(eig_val_data, eig_vec_data); }

   DenseMatrix Jpi(dim);
   // Computes the initial->physical transformation Jacobian.
   Mult(T.Jacobian(), Jac0inv, Jpi);
   Vector compr_dir(eig_vec_data, dim), ph_dir(dim);
   Jpi.Mult(compr_dir, ph_dir);

   // Change of the initial mesh size in the compression direction.
   h = h0 * ph_dir.Norml2() / compr_dir.Norml2();
   // Measure of maximal compression.
   mu = eig_val_data[0];
}

LagrangianHydroOperator::LagrangianHydroOperator(const int size,
                                                 ParFiniteElementSpace &h1,
                                                 ParFiniteElementSpace &l2,
                                                 const Array<int> &ess_tdofs,
                                                 Coefficient &rho_mixed_coeff,
                                                 VectorCoefficient &dist_coeff,
                                                 const IntegrationRule &vol_ir,
                                                 const IntegrationRule &face_ir,
                                                 const int source,
                                                 const double cfl,
                                                 const bool visc,
                                                 const bool vort,
                                                 const double cgt,
                                                 const int cgiter,
                                                 double ftz,
                                                 SIOptions &si_opt,
                                                 MaterialData &m_data) :
   TimeDependentOperator(size),
   H1(h1), L2(l2),
   pmesh(H1.GetParMesh()),
   H1Vsize(H1.GetVSize()),
   L2Vsize(L2.GetVSize()),
   block_offsets(5),
   x_gf(&H1),
   ess_tdofs(ess_tdofs),
   dim(pmesh->Dimension()),
   NE(pmesh->GetNE()),
   l2dofs_cnt(L2.GetFE(0)->GetDof()),
   h1dofs_cnt(H1.GetFE(0)->GetDof()),
   source_type(source), cfl(cfl),
   use_viscosity(visc),
   use_vorticity(vort),
   cg_rel_tol(cgt), cg_max_iter(cgiter),ftz_tol(ftz),
   Mv(&H1), Mv_spmat_copy(),
   Me_1(l2dofs_cnt, l2dofs_cnt, NE), Me_2(l2dofs_cnt, l2dofs_cnt, NE),
   Me_1_inv(l2dofs_cnt, l2dofs_cnt, NE), Me_2_inv(l2dofs_cnt, l2dofs_cnt, NE),
   ir(vol_ir), ir_face(face_ir),
   Q1D(int(floor(0.7 + pow(ir.GetNPoints(), 1.0 / dim)))),
   qdata(dim, NE, ir.GetNPoints()),
   qdata_is_current(false),
   forcemat_is_assembled(false),
   Force_1(&H1, &L2), Force_2(&H1, &L2), FaceForce(&H1, &L2),
   FaceForceMomentum(&H1), FaceForceEnergy_1(&L2), FaceForceEnergy_2(&L2),
   one(L2Vsize),
   rhs(H1Vsize),
   si_options(si_opt), mat_data(m_data)
{
   block_offsets[0] = 0;
   block_offsets[1] = block_offsets[0] + H1Vsize;
   block_offsets[2] = block_offsets[1] + H1Vsize;
   block_offsets[3] = block_offsets[2] + L2Vsize;
   block_offsets[4] = block_offsets[3] + L2Vsize;
   one = 1.0;

   // Standard local assembly and inversion for energy mass matrices.
   // 'Me' is used in the computation of the internal energy
   // which is used twice: once at the start and once at the end of the run.
   AlphaRhoCoeff arho_1_coeff(mat_data.ind0_1, mat_data.rho0_1),
                 arho_2_coeff(mat_data.ind0_2, mat_data.rho0_2);
   MassIntegrator mi_1(arho_1_coeff, &ir), mi_2(arho_2_coeff, &ir);
   for (int e = 0; e < NE; e++)
   {
      const FiniteElement &fe = *L2.GetFE(e);
      ElementTransformation &Tr = *L2.GetElementTransformation(e);
      const int attr = pmesh->GetAttribute(e);

      // Material 1.
      if (attr == 10 || attr == 15)
      {
         mi_1.AssembleElementMatrix(fe, Tr, Me_1(e));
         DenseMatrixInverse inv(&Me_1(e));
         inv.Factor();
         inv.GetInverseMatrix(Me_1_inv(e));
      }
      else { Me_1(e) = 0.0; Me_1_inv(e) = 0.0; }

      // Material 2.
      if (attr == 15 || attr == 20)
      {
         mi_2.AssembleElementMatrix(fe, Tr, Me_2(e));
         DenseMatrixInverse inv(&Me_2(e));
         inv.Factor();
         inv.GetInverseMatrix(Me_2_inv(e));
      }
      else { Me_2(e) = 0.0; Me_2_inv(e) = 0.0; }
   }

   // Standard assembly for the velocity mass matrix.
   VectorMassIntegrator *vmi = new VectorMassIntegrator(rho_mixed_coeff, &ir);
   Mv.AddDomainIntegrator(vmi);
   Mv.Assemble();
   Mv_spmat_copy = Mv.SpMat();

   // Values of rho0DetJ0 and Jac0inv at all quadrature points.
   // Initial local mesh size (assumes all mesh elements are the same).
   int Ne, ne = NE;
   double Volume, vol = 0.0;

   const int NQ = ir.GetNPoints();
   Vector rho1_vals(NQ), rho2_vals(NQ);
   for (int e = 0; e < NE; e++)
   {
      mat_data.rho0_1.GetValues(e, ir, rho1_vals);
      mat_data.rho0_2.GetValues(e, ir, rho2_vals);
      ElementTransformation &Tr = *H1.GetElementTransformation(e);
      for (int q = 0; q < NQ; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         DenseMatrixInverse Jinv(Tr.Jacobian());
         Jinv.GetInverseMatrix(qdata.Jac0inv(e*NQ + q));
         qdata.rho0DetJ0w_1(e*NQ + q) = ir.IntPoint(q).weight * Tr.Weight() *
                                        mat_data.ind0_1.GetValue(Tr, ip) * rho1_vals(q);
         qdata.rho0DetJ0w_2(e*NQ + q) = ir.IntPoint(q).weight * Tr.Weight() *
                                        mat_data.ind0_2.GetValue(Tr, ip) * rho2_vals(q);
      }
   }
   for (int e = 0; e < NE; e++) { vol += pmesh->GetElementVolume(e); }

   // Fill the GridFunction masses mat_data.rhodetJind0 at the L2 nodes.
   mat_data.UpdateInitialMasses();

   MPI_Allreduce(&vol, &Volume, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
   MPI_Allreduce(&ne, &Ne, 1, MPI_INT, MPI_SUM, pmesh->GetComm());
   switch (pmesh->GetElementBaseGeometry(0))
   {
      case Geometry::SEGMENT: qdata.h0 = Volume / Ne; break;
      case Geometry::SQUARE: qdata.h0 = sqrt(Volume / Ne); break;
      case Geometry::TRIANGLE: qdata.h0 = sqrt(2.0 * Volume / Ne); break;
      case Geometry::CUBE: qdata.h0 = pow(Volume / Ne, 1./3.); break;
      case Geometry::TETRAHEDRON: qdata.h0 = pow(6.0 * Volume / Ne, 1./3.); break;
      default: MFEM_ABORT("Unknown zone type!");
   }
   qdata.h0 /= (double) H1.GetOrder(0);

   ForceIntegrator *fi_1, *fi_2;
   // The total stress is always taken pointwise, based on the LS value.
   fi_1 = new ForceIntegrator(qdata.stressJinvT_1, mat_data.ind0_1);
   fi_2 = new ForceIntegrator(qdata.stressJinvT_2, mat_data.ind0_2);
   fi_1->SetIntRule(&ir);
   fi_2->SetIntRule(&ir);
   Force_1.AddDomainIntegrator(fi_1);
   Force_2.AddDomainIntegrator(fi_2);
   // Make a dummy assembly to figure out the sparsity.
   Force_1.Assemble(0);
   Force_1.Finalize(0);
   Force_2.Assemble(0);
   Force_2.Finalize(0);

   //
   // Shifted interface setup.
   //

   // Interface forces.
   auto *ffi = new FaceForceIntegrator(mat_data, dist_coeff);
   ffi->SetIntRule(&ir_face);
   ffi->SetShiftType(si_options.v_shift_type);
   ffi->SetScale(si_options.v_shift_scale);
   //FaceForce.AddTraceFaceIntegrator(ffi);
   FaceForce.AddFaceIntegrator(ffi);

   auto *mfi = new MomentumInterfaceIntegrator(mat_data, dist_coeff);
   mfi->SetIntRule(&ir_face);
   mfi->num_taylor    = si_options.num_taylor;
   mfi->v_shift_type  = si_options.v_shift_type;
   mfi->v_shift_scale = si_options.v_shift_scale;
   FaceForceMomentum.AddInteriorFaceIntegrator(mfi);

   auto *efi_1 = new EnergyInterfaceIntegrator(1, mat_data, qdata, dist_coeff);
   efi_1->SetIntRule(&ir_face);
   efi_1->num_taylor      = si_options.num_taylor;
   efi_1->e_shift_type    = si_options.e_shift_type;
   efi_1->e_shift_scale   = si_options.e_shift_scale;
   efi_1->diffusion       = si_options.e_shift_diffusion;
   efi_1->problem_visc    = use_viscosity;
   efi_1->diffusion_scale = si_options.e_shift_diffusion_scale;
   FaceForceEnergy_1.AddInteriorFaceIntegrator(efi_1);

   auto *efi_2 = new EnergyInterfaceIntegrator(2, mat_data, qdata, dist_coeff);
   efi_2->SetIntRule(&ir_face);
   efi_2->num_taylor      = si_options.num_taylor;
   efi_2->e_shift_type    = si_options.e_shift_type;
   efi_2->e_shift_scale   = si_options.e_shift_scale;
   efi_2->diffusion       = si_options.e_shift_diffusion;
   efi_2->problem_visc    = use_viscosity;
   efi_2->diffusion_scale = si_options.e_shift_diffusion_scale;
   FaceForceEnergy_2.AddInteriorFaceIntegrator(efi_2);

//   if (si_options.v_shift_type > 0)
//   {
//      // Make a dummy assembly to figure out the new sparsity.
//      ParGridFunction &p_tmp_1 = mat_data.p_1->GetPressure(),
//                      &p_tmp_2 = mat_data.p_2->GetPressure();
//      p_tmp_1 = 1.0;
//      p_tmp_2 = 1.0;
//      UpdateAlpha(mat_data.level_set, mat_data.alpha_1, mat_data.alpha_2);
//      FaceForce.Assemble(0);
//      FaceForce.Finalize(0);
//   }
//   // Done after the dummy assembly to avoid extra calculations.
//   ffi->SetDiffusion(si_options.v_shift_diffusion,
//                     si_options.v_shift_diffusion_scale);
}

LagrangianHydroOperator::~LagrangianHydroOperator() { }

void LagrangianHydroOperator::Mult(const Vector &S, Vector &dS_dt) const
{
   // Make sure that the mesh positions correspond to the ones in S. This is
   // needed only because some mfem time integrators don't update the solution
   // vector at every intermediate stage (hence they don't change the mesh).
   UpdateMesh(S);
   // The monolithic BlockVector stores the unknown fields as follows:
   // (Position, Velocity, Specific Internal Energy).
   Vector* sptr = const_cast<Vector*>(&S);
   ParGridFunction v;
   const int VsizeH1 = H1.GetVSize();
   v.MakeRef(&H1, *sptr, VsizeH1);
   // Set dx_dt = v (explicit).
   ParGridFunction dx;
   dx.MakeRef(&H1, dS_dt, 0);
   dx = v;

   SolveVelocity(S, dS_dt);
   SolveEnergy(S, v, dS_dt);
   qdata_is_current = false;
}

void LagrangianHydroOperator::SolveVelocity(const Vector &S,
                                            Vector &dS_dt) const
{
   Vector* sptr = const_cast<Vector*>(&S);
   ParGridFunction v;
   const int VsizeH1 = H1.GetVSize();
   v.MakeRef(&H1, *sptr, VsizeH1);
   auto tfi_v = FaceForceMomentum.GetIFLFI();
   auto v_integ = dynamic_cast<MomentumInterfaceIntegrator *>((*tfi_v)[0]);
   v_integ->SetVelocity(v);

   UpdateQuadratureData(S);
   AssembleForceMatrix();
   // The monolithic BlockVector stores the unknown fields as follows:
   // (Position, Velocity, Specific Internal Energy).
   ParGridFunction dv;
   dv.MakeRef(&H1, dS_dt, H1Vsize);
   dv = 0.0;

   ParGridFunction accel_src_gf;
   if (source_type == 2)
   {
      accel_src_gf.SetSpace(&H1);
      RTCoefficient accel_coeff(dim);
      accel_src_gf.ProjectCoefficient(accel_coeff);
      accel_src_gf.Read();
   }

   // This Force object is l2_dofs x h1_dofs (transpose of the paper one).
   Force_1.MultTranspose(one, rhs);
   Force_2.AddMultTranspose(one, rhs);

//   if (si_options.v_shift_type > 0)
//   {
//      FaceForce.AddMultTranspose(one, rhs, 1.0);
//   }

   rhs.Neg();

   if (si_options.v_shift_type > 0)
   {
      pmesh->ExchangeFaceNbrNodes();
      FaceForceMomentum.Assemble();
      rhs += FaceForceMomentum;
   }

   if (source_type == 2)
   {
      Vector rhs_accel(rhs.Size());
      Mv_spmat_copy.Mult(accel_src_gf, rhs_accel);
      rhs += rhs_accel;
   }

   Vector X, B;
   HypreParMatrix A;
   Mv.FormLinearSystem(ess_tdofs, dv, rhs, A, X, B);

   CGSolver cg(H1.GetParMesh()->GetComm());
   HypreSmoother prec;
   prec.SetType(HypreSmoother::Jacobi, 1);
   cg.SetPreconditioner(prec);
   cg.SetOperator(A);
   cg.SetRelTol(cg_rel_tol);
   cg.SetAbsTol(0.0);
   cg.SetMaxIter(cg_max_iter);
   cg.SetPrintLevel(-1);

   cg.Mult(B, X);
   Mv.RecoverFEMSolution(X, rhs, dv);

   v_integ->UnsetVelocity();
}

void LagrangianHydroOperator::SolveEnergy(const Vector &S, const Vector &v,
                                          Vector &dS_dt) const
{
   ParGridFunction vel, energy_1, energy_2;
   Vector *vptr = const_cast<Vector*>(&v);
   Vector *sptr = const_cast<Vector*>(&S);
   vel.MakeRef(&H1, *vptr, 0);
   energy_1.MakeRef(&L2, *sptr, 2*H1.GetVSize());
   energy_2.MakeRef(&L2, *sptr, 2*H1.GetVSize() + L2.GetVSize());
   auto tfi_v = FaceForceMomentum.GetIFLFI();
   auto v_integ = dynamic_cast<MomentumInterfaceIntegrator *>((*tfi_v)[0]);
   auto tfi_e_1 = FaceForceEnergy_1.GetIFLFI();
   auto e_integ_1 = dynamic_cast<EnergyInterfaceIntegrator *>((*tfi_e_1)[0]);
   auto tfi_e_2 = FaceForceEnergy_2.GetIFLFI();
   auto e_integ_2 = dynamic_cast<EnergyInterfaceIntegrator *>((*tfi_e_2)[0]);
   vel.ExchangeFaceNbrData();
   v_integ->SetVelocity(vel);
   e_integ_1->SetVandE(&vel, &energy_1);
   e_integ_2->SetVandE(&vel, &energy_2);

   UpdateQuadratureData(S);
   AssembleForceMatrix();

   // The monolithic BlockVector stores the unknown fields as follows:
   // (Position, Velocity, Specific Internal Energy).
   ParGridFunction de_1, de_2;
   de_1.MakeRef(&L2, dS_dt, H1Vsize*2);
   de_2.MakeRef(&L2, dS_dt, H1Vsize*2 + L2Vsize);
   de_1 = 0.0; de_2 = 0.0;

   Array<int> l2dofs;
   Vector e_rhs_1(L2Vsize), e_rhs_2(L2Vsize);

   // This Force object is l2_dofs x h1_dofs (transpose of the paper one).
   Force_1.Mult(v, e_rhs_1);
   Force_2.Mult(v, e_rhs_2);

//   if (si_options.e_shift_type == 1) { FaceForce.AddMult(v, e_rhs_1, 1.0); }
   if (si_options.e_shift_type > 1)
   {
      pmesh->ExchangeFaceNbrNodes();
      FaceForceEnergy_1.Assemble();
      e_rhs_1 -= FaceForceEnergy_1;
      FaceForceEnergy_2.Assemble();
      e_rhs_2 -= FaceForceEnergy_2;
   }
//   double ff1 = FaceForceEnergy_1.Norml1(),
//          ff2 = FaceForceEnergy_2.Norml1();
//   MPI_Allreduce(MPI_IN_PLACE, &ff1, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
//   MPI_Allreduce(MPI_IN_PLACE, &ff2, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
//   if (H1.GetMyRank() == 0)
//   {
//      cout << "e rhs diff: " << scientific << ff1 << " " << ff2  << endl;
//   }

   // Solve for energy, assemble the energy source if such exists.
   if (source_type == 1) // 2D Taylor-Green.
   {
      LinearForm e_source_1(&L2), e_source_2(&L2);
      TaylorCoefficient coeff_1(mat_data.alpha_1), coeff_2(mat_data.alpha_2);
      e_source_1.AddDomainIntegrator(new DomainLFIntegrator(coeff_1, &ir));
      e_source_1.Assemble();
      e_source_2.AddDomainIntegrator(new DomainLFIntegrator(coeff_2, &ir));
      e_source_2.Assemble();

      e_rhs_1 += e_source_1;
      e_rhs_2 += e_source_2;
   }

   Vector loc_rhs(l2dofs_cnt), loc_de(l2dofs_cnt);
   for (int e = 0; e < NE; e++)
   {
      const int attr = pmesh->GetAttribute(e);
      L2.GetElementDofs(e, l2dofs);

      // Material 1.
      if (attr == 10 || attr == 15)
      {
         e_rhs_1.GetSubVector(l2dofs, loc_rhs);
         Me_1_inv(e).Mult(loc_rhs, loc_de);
         de_1.SetSubVector(l2dofs, loc_de);
      }

      // Material 2.
      if (attr == 15 || attr == 20)
      {
         e_rhs_2.GetSubVector(l2dofs, loc_rhs);
         Me_2_inv(e).Mult(loc_rhs, loc_de);
         de_2.SetSubVector(l2dofs, loc_de);
      }
   }

   v_integ->UnsetVelocity();
   e_integ_1->UnsetVandE();
   e_integ_2->UnsetVandE();
}

void LagrangianHydroOperator::UpdateMesh(const Vector &S) const
{
   Vector* sptr = const_cast<Vector*>(&S);
   x_gf.MakeRef(&H1, *sptr, 0);
   H1.GetParMesh()->NewNodes(x_gf, false);
}

void LagrangianHydroOperator::UpdateMassMatrices(Coefficient &rho_coeff)
{
   // Assumption is Mv was connected to the same Coefficient from the input.
   Mv.Update();
   Mv.BilinearForm::operator=(0.0);
   Mv.Assemble();
   Mv_spmat_copy = Mv.SpMat();

   MassIntegrator mi(rho_coeff, &ir);
   for (int e = 0; e < NE; e++)
   {
      const FiniteElement &fe = *L2.GetFE(e);
      ElementTransformation &Tr = *L2.GetElementTransformation(e);
      const int attr = pmesh->GetAttribute(e);

      // Material 1.
      if (attr == 10 || attr == 15)
      {
         mi.AssembleElementMatrix(fe, Tr, Me_1(e));
         DenseMatrixInverse inv(&Me_1(e));
         inv.Factor();
         inv.GetInverseMatrix(Me_1_inv(e));
      }
      else { Me_1(e) = 0.0; Me_1_inv(e) = 0.0; }

      // Material 2.
      if (attr == 15 || attr == 20)
      {
         mi.AssembleElementMatrix(fe, Tr, Me_2(e));
         DenseMatrixInverse inv(&Me_2(e));
         inv.Factor();
         inv.GetInverseMatrix(Me_2_inv(e));
      }
      else { Me_2(e) = 0.0; Me_2_inv(e) = 0.0; }
   }
}

double LagrangianHydroOperator::GetTimeStepEstimate(const Vector &S) const
{
   UpdateMesh(S);
   UpdateQuadratureData(S);
   double glob_dt_est;
   const MPI_Comm comm = H1.GetParMesh()->GetComm();
   MPI_Allreduce(&qdata.dt_est, &glob_dt_est, 1, MPI_DOUBLE, MPI_MIN, comm);
   return glob_dt_est;
}

void LagrangianHydroOperator::ResetTimeStepEstimate() const
{
   qdata.dt_est = std::numeric_limits<double>::infinity();
}

void LagrangianHydroOperator::ComputeDensity(int mat_id,
                                             ParGridFunction &ind0,
                                             ParGridFunction &rho) const
{
   DenseMatrix Mrho(l2dofs_cnt);
   Vector rhs_z(l2dofs_cnt), rho_z(l2dofs_cnt);
   Array<int> dofs(l2dofs_cnt);
   DenseMatrixInverse inv(&Mrho);

   Vector &rhoDetJ = (mat_id == 1) ? qdata.rho0DetJ0w_1 : qdata.rho0DetJ0w_2;

   MassIntegrator mi(&ir);
   DensityIntegrator di(mat_id, rhoDetJ, &ind0);
   di.SetIntRule(&ir);

   for (int e = 0; e < NE; e++)
   {
      const FiniteElement &fe = *L2.GetFE(e);
      ElementTransformation &eltr = *L2.GetElementTransformation(e);
      di.AssembleRHSElementVect(fe, eltr, rhs_z);
      mi.AssembleElementMatrix(fe, eltr, Mrho);
      inv.Factor();

      inv.Mult(rhs_z, rho_z);

      L2.GetElementDofs(e, dofs);
      rho.SetSubVector(dofs, rho_z);
   }
}

double LagrangianHydroOperator::Mass(int mat_id) const
{
   double mass = (mat_id == 1) ? qdata.rho0DetJ0w_1.Sum()
                               : qdata.rho0DetJ0w_2.Sum();
   MPI_Allreduce(MPI_IN_PLACE, &mass, 1, MPI_DOUBLE, MPI_SUM, H1.GetComm());
   return mass;
}

double LagrangianHydroOperator::InternalEnergy(const ParGridFunction &e_1,
                                               const ParGridFunction &e_2) const
{
   Vector one(l2dofs_cnt), loc_e_1(l2dofs_cnt), loc_e_2(l2dofs_cnt);
   one = 1.0;
   Array<int> l2dofs;
   double ie = 0.0;
   for (int k = 0; k < NE; k++)
   {
      L2.GetElementDofs(k, l2dofs);
      e_1.GetSubVector(l2dofs, loc_e_1);
      e_2.GetSubVector(l2dofs, loc_e_2);
      ie += Me_1(k).InnerProduct(loc_e_1, one) +
            Me_2(k).InnerProduct(loc_e_2, one);
   }
   MPI_Allreduce(MPI_IN_PLACE, &ie, 1, MPI_DOUBLE, MPI_SUM, H1.GetComm());

   return ie;
}

double LagrangianHydroOperator::KineticEnergy(const ParGridFunction &v) const
{
   double ke = 0.5 * Mv_spmat_copy.InnerProduct(v, v);

   MPI_Allreduce(MPI_IN_PLACE, &ke, 1, MPI_DOUBLE, MPI_SUM, H1.GetComm());
   return ke;
}

double LagrangianHydroOperator::Momentum(const ParGridFunction &v) const
{
   Vector one(Mv_spmat_copy.Height());
   one = 1.0;
   double momentum = Mv_spmat_copy.InnerProduct(one, v);

   MPI_Allreduce(MPI_IN_PLACE, &momentum, 1, MPI_DOUBLE, MPI_SUM, H1.GetComm());
   return momentum;
}

void LagrangianHydroOperator::PrintPressures(const ParGridFunction &e_1,
                                             const ParGridFunction &e_2,
                                             const ParGridFunction &v,
                                             string prefix,
                                             int problem)
{
   if (problem != 8 && problem != 9) { return; }
   MFEM_VERIFY(L2.GetNRanks() == 1, "Pressure output only in 1D serial");

   std::ofstream fstream_p_1,   fstream_p_2,
                 fstream_r_1,   fstream_r_2,
                 fstream_e_1,   fstream_e_2,
                 fstream_p_tot, fstream_e_tot, fstream_r_tot, fstream_v;
   fstream_p_1.open(prefix + "p_1.out");     fstream_p_1.precision(8);
   fstream_p_2.open(prefix + "p_2.out");     fstream_p_2.precision(8);
   fstream_r_1.open(prefix + "r_1.out");     fstream_p_1.precision(8);
   fstream_r_2.open(prefix + "r_2.out");     fstream_p_2.precision(8);
   fstream_e_1.open(prefix + "e_1.out");     fstream_p_1.precision(8);
   fstream_e_2.open(prefix + "e_2.out");     fstream_p_2.precision(8);
   fstream_p_tot.open(prefix + "p_tot.out"); fstream_p_tot.precision(8);
   fstream_e_tot.open(prefix + "e_tot.out"); fstream_e_tot.precision(8);
   fstream_r_tot.open(prefix + "rho_tot.out"); fstream_r_tot.precision(8);
   fstream_v.open(prefix + "v.out"); fstream_v.precision(8);

   const int nqp = ir.GetNPoints();
   Vector pos(dim);
   for (int e = 0; e < NE; e++)
   {
      const int attr = pmesh->GetAttribute(e);
      ElementTransformation &Tr = *L2.GetElementTransformation(e);

      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         Tr.Transform(ip, pos);
         double detJ = Tr.Weight();

         const double ls = mat_data.level_set.GetValue(Tr, ip),
                      ve = v.GetValue(Tr, ip);

         double p_tot = 0.0, rho_tot = 0.0, e_tot = 0.0;
         if (attr == 10 || attr == 15)
         {
            double a1  = mat_data.alpha_1.GetValue(Tr, ip);
            double i1  = mat_data.ind0_1.GetValue(Tr, ip);
            double rho = qdata.rho0DetJ0w_1(e*nqp + q) / i1 / detJ / ip.weight;
            double en  = e_1.GetValue(Tr, ip);
            double g   = mat_data.gamma_1;
            double p   = (fabs(g - 4.4) > 1e-8) ? (g - 1.0) * rho * en
                                              : (g - 1.0) * rho * en - g*6.0e8;
            if (ls <= 0.0)
            {
               fstream_p_1 << pos(0) << " " << p << "\n";
               fstream_p_1.flush();
               fstream_r_1 << pos(0) << " " << rho << "\n";
               fstream_r_1.flush();
               fstream_e_1 << pos(0) << " " << en << "\n";
               fstream_e_1.flush();
            }

            p_tot   += a1 * p;
            rho_tot += a1 * rho;
            e_tot   += a1 * en;
         }

         if (attr == 15 || attr == 20)
         {
            double a2  = mat_data.alpha_2.GetValue(Tr, ip);
            double i2  = mat_data.ind0_2.GetValue(Tr, ip);
            double rho = qdata.rho0DetJ0w_2(e*nqp + q) / i2 / detJ / ip.weight;
            double en  = e_2.GetValue(Tr, ip);
            double g   = mat_data.gamma_2;
            double p = (fabs(g - 4.4) > 1e-8) ? (g - 1.0) * rho * en
                                              : (g - 1.0) * rho * en - g*6.0e8;
            if (ls >= 0.0)
            {
               fstream_p_2 << pos(0) << " " << p << "\n";
               fstream_p_2.flush();
               fstream_r_2 << pos(0) << " " << rho << "\n";
               fstream_r_2.flush();
               fstream_e_2 << pos(0) << " " << en << "\n";
               fstream_e_2.flush();
            }

            p_tot   += a2 * p;
            rho_tot += a2 * rho;
            e_tot   += a2 * en;
         }

         fstream_p_tot << pos(0) << " " << p_tot << "\n";
         fstream_r_tot << pos(0) << " " << rho_tot << "\n";
         fstream_e_tot << pos(0) << " " << e_tot << "\n";
         fstream_v     << pos(0) << " " << ve << "\n";
         fstream_p_tot.flush();
         fstream_r_tot.flush();
         fstream_e_tot.flush();
         fstream_v.flush();
      }
   }
   fstream_p_1.close();
   fstream_p_2.close();
   fstream_r_1.close();
   fstream_r_2.close();
   fstream_e_1.close();
   fstream_e_2.close();
   fstream_p_tot.close();
   fstream_r_tot.close();
   fstream_e_tot.close();
   fstream_v.close();
}

// Smooth transition between 0 and 1 for x in [-eps, eps].
double smooth_step_01(double x, double eps)
{
   const double y = (x + eps) / (2.0 * eps);
   if (y < 0.0) { return 0.0; }
   if (y > 1.0) { return 1.0; }
   return (3.0 - 2.0 * y) * y * y;
}

void LagrangianHydroOperator::UpdateQuadratureData(const Vector &S) const
{
   if (qdata_is_current) { return; }

   qdata_is_current = true;
   forcemat_is_assembled = false;

   // This code is only for the 1D/FA mode
   const int nqp = ir.GetNPoints();
   ParGridFunction x, v, e_1, e_2;
   Vector* sptr = const_cast<Vector*>(&S);
   x.MakeRef(&H1, *sptr, 0);
   v.MakeRef(&H1, *sptr, H1.GetVSize());
   e_1.MakeRef(&L2, *sptr, 2*H1.GetVSize());
   e_2.MakeRef(&L2, *sptr, 2*H1.GetVSize() + L2.GetVSize());
   Vector e_vals;
   DenseMatrix Jpi(dim), sgrad_v(dim), Jinv(dim), stress(dim), stressJiT(dim);

   // Update the alphas.
   UpdateAlpha(mat_data.level_set, mat_data.alpha_1, mat_data.alpha_2);
   // Update the pressure values (used for the shifted interface method).
   mat_data.p_1->UpdatePressure(mat_data.ind0_1, e_1);
   mat_data.p_2->UpdatePressure(mat_data.ind0_2, e_2);
   if (si_options.v_shift_type > 0 || si_options.e_shift_type > 0)
   {
      // Needed for shifted face integrals in parallel.
      mat_data.alpha_1.ExchangeFaceNbrData();
      mat_data.alpha_2.ExchangeFaceNbrData();
      mat_data.ind0_1.ExchangeFaceNbrData();
      mat_data.ind0_2.ExchangeFaceNbrData();
      mat_data.p_1->ExchangeFaceNbrData();
      mat_data.p_2->ExchangeFaceNbrData();
      mat_data.rho0DetJ_1.ExchangeFaceNbrData();
      mat_data.rho0DetJ_2.ExchangeFaceNbrData();
   }

   // Jacobians of reference->physical transformations for all quad points.
   DenseTensor Jpr(dim, dim, nqp);

   for (int k = 1; k <= 2; k++)
   {
      Vector &r0DJ_k = (k == 1) ? qdata.rho0DetJ0w_1 : qdata.rho0DetJ0w_2;
      ParGridFunction &ind_k = (k == 1) ? mat_data.ind0_1 : mat_data.ind0_2;
      double gamma_k = (k == 1) ? mat_data.gamma_1 : mat_data.gamma_2;
      ParGridFunction &e_k     = (k == 1) ? e_1 : e_2;
      DenseTensor &stressJinvT_k = (k == 1) ? qdata.stressJinvT_1
                                            : qdata.stressJinvT_2;

      double min_detJ = std::numeric_limits<double>::infinity();
      for (int e = 0; e < NE; e++)
      {
         const int attr = pmesh->GetAttribute(e);
         if ((k == 1 && attr == 20) || (k == 2 && attr == 10))
         {
            for (int vd = 0 ; vd < dim; vd++)
            {
               for (int gd = 0; gd < dim; gd++)
               {
                  for (int q = 0; q < nqp; q++)
                  {
                     stressJinvT_k(vd)(e*nqp + q, gd) = 0.0;
                  }
               }
            }
            continue;
         }

         ElementTransformation *T = H1.GetElementTransformation(e);
         e_k.GetValues(e, ir, e_vals);

         for (int q = 0; q < nqp; q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            T->SetIntPoint(&ip);
            Jpr(q) = T->Jacobian();
            const double detJ = Jpr(q).Det();
            min_detJ = fmin(min_detJ, detJ);
            // Assuming piecewise constant gamma that moves with the mesh.
            double rho   = r0DJ_k(e*nqp + q) / detJ /
                           ip.weight / ind_k.GetValue(*T, ip);
            double energy  = fmax(0.0, e_vals(q));

            if (IsFinite(rho) == false)
            {
               cout << e << " " << q << endl;
               T->Jacobian().Print();
               cout << detJ << " " << rho << " "
                    << gamma_k << " " << min_detJ << std::endl;
               MFEM_ABORT("rho bad");
            }

            double temperature, p, sound_speed;
            // Special case - stiffened gas;
            // Assumes that gamma = 4.4 is used only in problem 9 [water-air] !!
            if (fabs(gamma_k - 4.4) < 1e-8)
            {
               p              = (gamma_k - 1.0) * rho * energy - gamma_k*6.0e8;
               temperature    = fmax(p / rho, 1e-3);
            }
            else
            {
               p           = (gamma_k - 1.0) * rho * energy;
               temperature = (gamma_k - 1.0) * energy; // T = p / rho
            }
            sound_speed = sqrt(gamma_k * temperature);

            // Note that the Jacobian was already computed above. We've chosen
            // not to store the Jacobians for all batched quadrature points.
            CalcInverse(Jpr(q), Jinv);
            stress = 0.0;
            for (int d = 0; d < dim; d++) { stress(d, d) = -p; }
            double visc_coeff = 0.0;
            if (use_viscosity)
            {
               // Compression-based length scale at the point. The first
               // eigenvector of the symmetric velocity gradient gives the
               // direction of maximal compression. This is used to define the
               // relative change of the initial length scale.
               v.GetVectorGradient(*T, sgrad_v);
               sgrad_v.Symmetrize();

               // Length scale in compression direction, and measure of
               // maximal compression / expansion.
               double h, mu;
               LengthScaleAndCompression(sgrad_v, *T,
                                         qdata.Jac0inv(e*nqp + q),
                                         qdata.h0, h, mu);

               double vort_coeff = 1.0;
               if (use_vorticity)
               {
                  const double grad_norm = sgrad_v.FNorm();
                  const double div_v = fabs(sgrad_v.Trace());
                  vort_coeff = (grad_norm > 0.0) ? div_v / grad_norm : 1.0;
               }

               visc_coeff = 2.0 * rho * h * h * fabs(mu);
               // The following represents a "smooth" version of the statement
               // "if (mu < 0) visc_coeff += 0.5 rho h sound_speed".  Note that
               // eps must be scaled appropriately if a different unit system is
               // being used.
               const double eps = 1e-12;
               visc_coeff += 0.5 * rho * h * sound_speed * vort_coeff *
                             (1.0 - smooth_step_01(mu - 2.0 * eps, eps));
               stress.Add(visc_coeff, sgrad_v);
            }
            // Time step estimate at the point. Here the more relevant length
            // scale is related to the actual mesh deformation; we use the min
            // singular value of the ref->physical Jacobian. In addition, the
            // time step estimate should be aware of the presence of shocks.
            const double h_min =
                  Jpr(q).CalcSingularvalue(dim-1) / (double) H1.GetOrder(0);
            const double inv_dt = sound_speed / h_min +
                                  2.5 * visc_coeff / rho / h_min / h_min;
            if (min_detJ < 0.0)
            {
               // This will force repetition of the step with smaller dt.
               qdata.dt_est = 0.0;
            }
            else
            {
               if (inv_dt > 0.0)
               {
                  qdata.dt_est = fmin(qdata.dt_est, cfl*(1.0/inv_dt));
               }
            }
            MultABt(stress, Jinv, stressJiT);
            stressJiT *= ir.IntPoint(q).weight * detJ;

            for (int vd = 0 ; vd < dim; vd++)
            {
               for (int gd = 0; gd < dim; gd++)
               {
                  stressJinvT_k(vd)(e*nqp + q, gd) = stressJiT(vd, gd);
               }
            }
         }
      }
   }
}

void LagrangianHydroOperator::AssembleForceMatrix() const
{
   Force_1 = 0.0;
   Force_1.Assemble();
   Force_2 = 0.0;
   Force_2.Assemble();
//   if (si_options.v_shift_type > 0)
//   {
//      FaceForce = 0.0;
//      FaceForce.Assemble();
//   }
//   FaceForce_v.Assemble();
   forcemat_is_assembled = true;
}

} // namespace hydrodynamics

void HydroODESolver::Init(TimeDependentOperator &tdop)
{
   ODESolver::Init(tdop);
   hydro_oper = dynamic_cast<hydrodynamics::LagrangianHydroOperator *>(f);
   MFEM_VERIFY(hydro_oper, "HydroSolvers expect LagrangianHydroOperator.");
}

void RK2AvgSolver::Init(TimeDependentOperator &tdop)
{
   HydroODESolver::Init(tdop);
   const Array<int> &block_offsets = hydro_oper->GetBlockOffsets();
   V.SetSize(block_offsets[1], mem_type);
   dS_dt.Update(block_offsets, mem_type);
   dS_dt = 0.0;
   S0.Update(block_offsets, mem_type);
}

void RK2AvgSolver::Step(Vector &S, double &t, double &dt)
{
   // The monolithic BlockVector stores the unknown fields as follows:
   // (Position, Velocity, Specific Internal Energy).
   S0.Vector::operator=(S);
   Vector &v0 = S0.GetBlock(1);
   Vector &dx_dt = dS_dt.GetBlock(0);
   Vector &dv_dt = dS_dt.GetBlock(1);

   // In each sub-step:
   // - Update the global state Vector S.
   // - Compute dv_dt using S.
   // - Update V using dv_dt.
   // - Compute de_dt and dx_dt using S and V.

   // -- 1.
   // S is S0.
   hydro_oper->UpdateMesh(S);
   hydro_oper->SolveVelocity(S, dS_dt);
   // V = v0 + 0.5 * dt * dv_dt;
   add(v0, 0.5 * dt, dv_dt, V);
   hydro_oper->SolveEnergy(S, V, dS_dt);
   // SHIFT - average the velocity.
   dx_dt = V;

   // -- 2.
   // S = S0 + 0.5 * dt * dS_dt;
   add(S0, 0.5 * dt, dS_dt, S);
   hydro_oper->ResetQuadratureData();
   hydro_oper->UpdateMesh(S);
   hydro_oper->SolveVelocity(S, dS_dt);
   // V = v0 + 0.5 * dt * dv_dt;
   add(v0, 0.5 * dt, dv_dt, V);
   hydro_oper->SolveEnergy(S, V, dS_dt);
   dx_dt = V;

   // -- 3.
   // S = S0 + dt * dS_dt.
   add(S0, dt, dS_dt, S);
   hydro_oper->ResetQuadratureData();
   t += dt;
}

} // namespace mfem

#endif // MFEM_USE_MPI
