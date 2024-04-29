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

namespace mfem
{

namespace hydrodynamics
{

void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    ParGridFunction &gf, const char *title,
                    int x, int y, int w, int h, bool ls)
{
   ParMesh &pmesh = *gf.ParFESpace()->GetParMesh();

   int myid;
   MPI_Comm_rank(pmesh.GetComm(), &myid);

   bool newly_opened = false;

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
           << x << " " << y << " " << w << " " << h << "\n";
      if (!ls) { sock << "keys " << keys; }
      if (ls)
      {
         sock << "keys " << "mmRj\n";
         sock << "levellines 0 0 1";
      }
      sock << std::endl;
   }

   if (myid == 0 && newly_opened == false && ls)
   {
      sock << "levellines 0 0 1";
   }
}

LagrangianHydroOperator::LagrangianHydroOperator(const int size,
                                                 ParFiniteElementSpace &h1,
                                                 ParFiniteElementSpace &l2,
                                                 const Array<int> &ess_tdofs,
                                                 Coefficient &rho0_coeff,
                                                 ParGridFunction &rho0_gf,
                                                 ParGridFunction &gamma_gf,
                                                 const int source,
                                                 const double cfl,
                                                 const bool visc,
                                                 const bool vort,
                                                 const double cgt,
                                                 const int cgiter,
                                                 double ftz,
                                                 const int oq,
                                                 MaterialData &m_data) :
   TimeDependentOperator(size),
   H1(h1), L2(l2), H1c(H1.GetParMesh(), H1.FEColl(), 1),
   pmesh(H1.GetParMesh()),
   H1Vsize(H1.GetVSize()),
   L2Vsize(L2.GetVSize()),
   block_offsets(4),
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
   gamma_gf(gamma_gf),
   Mv(&H1), Mv_spmat_copy(),
   Me_1(l2dofs_cnt, l2dofs_cnt, NE), Me_2(l2dofs_cnt, l2dofs_cnt, NE),
   Me_inv_1(l2dofs_cnt, l2dofs_cnt, NE), Me_inv_2(l2dofs_cnt, l2dofs_cnt, NE),
   ir(IntRules.Get(pmesh->GetElementBaseGeometry(0),
                   (oq > 0) ? oq : 3 * H1.GetOrder(0) + L2.GetOrder(0) - 1)),
   full_ir(nullptr),
   cut_ir_1(NE), cut_ir_2(NE),
   qdata(),
   qdata_is_current(false),
   Force_1(&L2, &H1), Force_2(&L2, &H1),
   X(H1c.GetTrueVSize()),
   B(H1c.GetTrueVSize()),
   one(L2Vsize),
   rhs(H1Vsize),
   e_rhs(L2Vsize),
   mat_data(m_data)
{
   block_offsets[0] = 0;
   block_offsets[1] = block_offsets[0] + H1Vsize;
   block_offsets[2] = block_offsets[1] + H1Vsize;
   block_offsets[3] = block_offsets[2] + L2Vsize;
   one = 1.0;

   // Create all cut integration rules.
   IsoparametricTransformation Tr;
   GridFunctionCoefficient ls_coeff(&mat_data.level_set);
   ProductCoefficient ls_coeff_mat1(-1.0, ls_coeff);
   ProductCoefficient ls_coeff_mat2( 1.0, ls_coeff);
   const int cut_ir_order = 3;
   MomentFittingIntRules mf_ir_1(cut_ir_order, ls_coeff_mat1, 2),
                         mf_ir_2(cut_ir_order, ls_coeff_mat2, 2);
   for (int e = 0; e < NE; e++)
   {
      const int attr = pmesh->GetAttribute(e);
      pmesh->GetElementTransformation(e, &Tr);
      cut_ir_1[e] = nullptr;
      cut_ir_2[e] = nullptr;
      if (attr == 10 || attr == 15)
      {
         cut_ir_1[e] = new IntegrationRule;
         mf_ir_1.GetVolumeIntegrationRule
             (Tr, *const_cast<IntegrationRule *>(cut_ir_1[e]));

         if (full_ir == nullptr && attr == 10)
         {
            full_ir = new IntegrationRule;
            mf_ir_1.GetVolumeIntegrationRule
                (Tr, *const_cast<IntegrationRule *>(full_ir));
         }
      }
      if (attr == 15 || attr == 20)
      {
         cut_ir_2[e] = new IntegrationRule;
         mf_ir_2.GetVolumeIntegrationRule
             (Tr, *const_cast<IntegrationRule *>(cut_ir_2[e]));
      }
   }

   // Standard local assembly and inversion for energy mass matrices.
   // 'Me' is used in the computation of the internal energy
   // which is used twice: once at the start and once at the end of the run.
   //MassIntegrator mi_1(rho0_coeff, &ir), mi_2(rho0_coeff, &ir);
   CutMassIntegrator cmi_1(rho0_coeff, cut_ir_1), cmi_2(rho0_coeff, cut_ir_2);
   for (int e = 0; e < NE; e++)
   {
      const FiniteElement &fe = *L2.GetFE(e);
      ElementTransformation &Tr = *L2.GetElementTransformation(e);
      const int attr = pmesh->GetAttribute(e);

      // Material 1.
      if (attr == 10 || attr == 15)
      {
         DenseMatrixInverse inv(&Me_1(e));
         //mi_1.AssembleElementMatrix(fe, Tr, Me_1(e));
         cmi_1.AssembleElementMatrix(fe, Tr, Me_1(e));
         inv.Factor();
         inv.GetInverseMatrix(Me_inv_1(e));
      }

      // Material 2.
      if (attr == 15 || attr == 20)
      {
         DenseMatrixInverse inv(&Me_2(e));
         //mi_2.AssembleElementMatrix(fe, Tr, Me_2(e));
         cmi_2.AssembleElementMatrix(fe, Tr, Me_2(e));
         inv.Factor();
         inv.GetInverseMatrix(Me_inv_2(e));
      }
   }
   // Standard assembly for the velocity mass matrix.
   //VectorMassIntegrator *vmi = new VectorMassIntegrator(rho0_coeff, &ir);
   auto cut_vmi_1 = new CutVectorMassIntegrator(rho0_coeff, cut_ir_1),
        cut_vmi_2 = new CutVectorMassIntegrator(rho0_coeff, cut_ir_2);
   Mv.AddDomainIntegrator(cut_vmi_1);
   Mv.AddDomainIntegrator(cut_vmi_2);
   //Mv.AddDomainIntegrator(vmi);
   Mv.Assemble();
   Mv_spmat_copy = Mv.SpMat();

   // Values of rho0DetJ0 and Jac0inv at all quadrature points.
   // Initial local mesh size (assumes all mesh elements are the same).
   double Volume, vol = 0.0;

   const int nqp = (cut_ir_1[0]) ? cut_ir_1[0]->GetNPoints()
                                 : cut_ir_2[0]->GetNPoints();
   qdata.SetSizes(dim, NE, nqp);
   for (int e = 0; e < NE; e++)
   {
      ElementTransformation &Tr = *H1.GetElementTransformation(e);
      const IntegrationRule *ir_e = (cut_ir_1[e]) ? cut_ir_1[e] : cut_ir_2[e];
      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir_e->IntPoint(q);
         Tr.SetIntPoint(&ip);
         DenseMatrixInverse Jinv(Tr.Jacobian());
         Jinv.GetInverseMatrix(qdata.Jac0inv(e*nqp + q));
      }
   }
   for (int e = 0; e < NE; e++)
   {
      ElementTransformation &Tr = *H1.GetElementTransformation(e);
      qdata.rho0DetJ0w_1[e] = 0.0;
      qdata.rho0DetJ0w_2[e] = 0.0;
      if (cut_ir_1[e])
      {
         Vector rho1_vals(nqp);
         mat_data.rho0_1.GetValues(e, *cut_ir_1[e], rho1_vals);
         for (int q = 0; q < nqp; q++)
         {
            const IntegrationPoint &ip_1 = cut_ir_1[e]->IntPoint(q);
            Tr.SetIntPoint(&ip_1);
            qdata.rho0DetJ0w_1[e](q) = Tr.Weight() * rho1_vals(q);
         }
      }
      if (cut_ir_2[e])
      {
         Vector rho2_vals(nqp);
         mat_data.rho0_2.GetValues(e, *cut_ir_2[e], rho2_vals);
         for (int q = 0; q < nqp; q++)
         {
            const IntegrationPoint &ip_2 = cut_ir_2[e]->IntPoint(q);
            Tr.SetIntPoint(&ip_2);
            qdata.rho0DetJ0w_2[e](q) = Tr.Weight() * rho2_vals(q);
         }
      }
   }
   for (int e = 0; e < NE; e++) { vol += pmesh->GetElementVolume(e); }

   int Ne;
   MPI_Allreduce(&vol, &Volume, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
   MPI_Allreduce(&NE, &Ne, 1, MPI_INT, MPI_SUM, pmesh->GetComm());
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
   fi_1 = new ForceIntegrator(qdata.stressJinvT_1);
   fi_2 = new ForceIntegrator(qdata.stressJinvT_2);
   const IntegrationRule *ir_0 = (cut_ir_1[0]) ? cut_ir_1[0] : cut_ir_2[0];
   fi_1->SetIntRule(ir_0);
   fi_2->SetIntRule(ir_0);
   Force_1.AddDomainIntegrator(fi_1);
   Force_2.AddDomainIntegrator(fi_2);
   // Make a dummy assembly to figure out the sparsity.
   Force_1.Assemble(0);
   Force_1.Finalize(0);
   Force_2.Assemble(0);
   Force_2.Finalize(0);
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
   }

   Vector rhs_1(H1Vsize), rhs_2(H1Vsize);
   Force_1.Mult(one, rhs);
   Force_2.AddMult(one, rhs);
   rhs.Neg();

   if (source_type == 2)
   {
      Vector rhs_accel(rhs.Size());
      Mv_spmat_copy.Mult(accel_src_gf, rhs_accel);
      rhs += rhs_accel;
   }

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
}

void LagrangianHydroOperator::SolveEnergy(const Vector &S, const Vector &v,
                                          Vector &dS_dt) const
{
   UpdateQuadratureData(S);
   AssembleForceMatrix();

   // The monolithic BlockVector stores the unknown fields as follows:
   // (Position, Velocity, Specific Internal Energy).
   ParGridFunction de_1, de_2;
   de_1.MakeRef(&L2, dS_dt, H1Vsize*2);
   de_2.MakeRef(&L2, dS_dt, H1Vsize*2 + L2Vsize);
   de_1 = 0.0; de_2 = 0.0;

   // Solve for energy, assemble the energy source if such exists.
   LinearForm *e_source = nullptr;
   if (source_type == 1) // 2D Taylor-Green.
   {
      e_source = new LinearForm(&L2);
      TaylorCoefficient coeff;
      DomainLFIntegrator *d = new DomainLFIntegrator(coeff);
      e_source->AddDomainIntegrator(d);
      e_source->Assemble();
   }

   Array<int> l2dofs;
   Vector e_rhs_1(L2Vsize), e_rhs_2(L2Vsize);

   Force_1.MultTranspose(v, e_rhs_1);
   Force_2.MultTranspose(v, e_rhs_2);

   if (e_source)
   {
      e_rhs_1 += *e_source;
      e_rhs_2 += *e_source;
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
         Me_inv_1(e).Mult(loc_rhs, loc_de);
         de_1.SetSubVector(l2dofs, loc_de);
      }

      // Material 2.
      if (attr == 15 || attr == 20)
      {
         e_rhs_2.GetSubVector(l2dofs, loc_rhs);
         Me_inv_2(e).Mult(loc_rhs, loc_de);
         de_2.SetSubVector(l2dofs, loc_de);
      }
   }
   delete e_source;
}

void LagrangianHydroOperator::UpdateMesh(const Vector &S) const
{
   Vector* sptr = const_cast<Vector*>(&S);
   x_gf.MakeRef(&H1, *sptr, 0);
   H1.GetParMesh()->NewNodes(x_gf, false);
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
                                             ParGridFunction &rho) const
{
   rho.SetSpace(&L2);
   DenseMatrix Mrho(l2dofs_cnt);
   Vector rhs(l2dofs_cnt), rho_z(l2dofs_cnt);
   Array<int> dofs(l2dofs_cnt);
   DenseMatrixInverse inv(&Mrho);
   MassIntegrator mi(full_ir);
   DensityIntegrator di(mat_id, qdata);
   di.SetIntRule(full_ir);
   for (int e = 0; e < NE; e++)
   {
      L2.GetElementDofs(e, dofs);
      if ((mat_id == 1 && pmesh->GetAttribute(e) == 20) ||
          (mat_id == 2 && pmesh->GetAttribute(e) == 10))
      {
         rho_z = 0.0;
         rho.SetSubVector(dofs, rho_z);
         continue;
      }

      const FiniteElement &fe = *L2.GetFE(e);
      ElementTransformation &eltr = *L2.GetElementTransformation(e);
      di.AssembleRHSElementVect(fe, eltr, rhs);
      mi.AssembleElementMatrix(fe, eltr, Mrho);
      std::cout << e << " " << Mrho.Det() << std::endl;
      Mrho.Print();
      rhs.Print();
      inv.Factor();
      inv.Mult(rhs, rho_z);
      rho.SetSubVector(dofs, rho_z);
   }
}

double LagrangianHydroOperator::InternalEnergy(const ParGridFunction &e_1,
                                               const ParGridFunction &e_2) const
{
   double glob_ie = 0.0;

   Vector one(l2dofs_cnt), loc_e_1(l2dofs_cnt), loc_e_2(l2dofs_cnt);
   one = 1.0;
   Array<int> l2dofs;
   double loc_ie = 0.0;
   for (int e = 0; e < NE; e++)
   {
      L2.GetElementDofs(e, l2dofs);
      e_1.GetSubVector(l2dofs, loc_e_1);
      e_2.GetSubVector(l2dofs, loc_e_2);
      loc_ie += Me_1(e).InnerProduct(loc_e_1, one) +
                Me_2(e).InnerProduct(loc_e_2, one);
   }
   MPI_Comm comm = H1.GetParMesh()->GetComm();
   MPI_Allreduce(&loc_ie, &glob_ie, 1, MPI_DOUBLE, MPI_SUM, comm);

   return glob_ie;
}

double LagrangianHydroOperator::KineticEnergy(const ParGridFunction &v) const
{
   double glob_ke = 0.0;
   // This should be turned into a kernel so that it could be displayed in pa
   double loc_ke = 0.5 * Mv_spmat_copy.InnerProduct(v, v);
   MPI_Allreduce(&loc_ke, &glob_ke, 1, MPI_DOUBLE, MPI_SUM,
                 H1.GetParMesh()->GetComm());
   return glob_ke;
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

   const int nqp = (cut_ir_1[0]) ? cut_ir_1[0]->GetNPoints()
                                 : cut_ir_2[0]->GetNPoints();
   ParGridFunction x, v, e_1, e_2;
   Vector* sptr = const_cast<Vector*>(&S);
   x.MakeRef(&H1, *sptr, 0);
   v.MakeRef(&H1, *sptr, H1.GetVSize());
   e_1.MakeRef(&L2, *sptr, 2*H1.GetVSize());
   e_2.MakeRef(&L2, *sptr, 2*H1.GetVSize() + L2.GetVSize());
   Vector e_vals;
   DenseMatrix Jpi(dim), sgrad_v(dim), Jinv(dim), stress(dim), stressJiT(dim);

   for (int k = 1; k <= 2; k++)
   {
      Array<const IntegrationRule *> &ir_k = (k == 1) ? cut_ir_1 : cut_ir_2;
      std::vector<Vector> &r0DJ_k = (k == 1) ? qdata.rho0DetJ0w_1
                                             : qdata.rho0DetJ0w_2;
      double gamma_k              = (k == 1) ? mat_data.gamma_1
                                             : mat_data.gamma_2;
      ParGridFunction &e_k        = (k == 1) ? e_1 : e_2;
      DenseTensor &stressJinvT_k  = (k == 1) ? qdata.stressJinvT_1
                                             : qdata.stressJinvT_2;
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
         e_k.GetValues(e, *ir_k[e], e_vals);
         for (int q = 0; q < nqp; q++)
         {
            const IntegrationPoint &ip = ir_k[e]->IntPoint(q);
            T->SetIntPoint(&ip);
            const DenseMatrix Jpr(T->Jacobian());
            const double detJ = Jpr.Det();

            // Assuming piecewise constant gamma that moves with the mesh.
            const double rho    = r0DJ_k[e](q) / detJ;
            const double energy = fmax(0.0, e_vals(q));
            double p            = (gamma_k - 1.0) * rho * energy;
            double sound_speed  = sqrt(gamma_k * (gamma_k - 1.0) * energy);

            CalcInverse(Jpr, Jinv);
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

               double vorticity_coeff = 1.0;
               if (use_vorticity)
               {
                  const double grad_norm = sgrad_v.FNorm();
                  const double div_v = fabs(sgrad_v.Trace());
                  vorticity_coeff = (grad_norm > 0.0) ? div_v / grad_norm : 1.0;
               }

               sgrad_v.Symmetrize();
               double eig_val_data[3], eig_vec_data[9];
               if (dim == 1)
               {
                  eig_val_data[0] = sgrad_v(0, 0);
                  eig_vec_data[0] = 1.;
               }
               else { sgrad_v.CalcEigenvalues(eig_val_data, eig_vec_data); }
               Vector compr_dir(eig_vec_data, dim);
               // Computes the initial->physical transformation Jacobian.
               mfem::Mult(Jpr, qdata.Jac0inv(e*nqp + q), Jpi);
               Vector ph_dir(dim); Jpi.Mult(compr_dir, ph_dir);
               // Change of the initial mesh size in the compression direction.
               const double h = qdata.h0 * ph_dir.Norml2() /
                                compr_dir.Norml2();
               // Measure of maximal compression.
               const double mu = eig_val_data[0];
               visc_coeff = 2.0 * rho * h * h * fabs(mu);
               // The following represents a "smooth" version of the statement
               // "if (mu < 0) visc_coeff += 0.5 rho h sound_speed".  Note that
               // eps must be scaled appropriately if a different unit system is
               // being used.
               const double eps = 1e-12;
               visc_coeff += 0.5 * rho * h * sound_speed * vorticity_coeff *
                             (1.0 - smooth_step_01(mu - 2.0 * eps, eps));
               stress.Add(visc_coeff, sgrad_v);
            }
            // Time step estimate at the point. Here the more relevant length
            // scale is related to the actual mesh deformation; we use the min
            // singular value of the ref->physical Jacobian. In addition, the
            // time step estimate should be aware of the presence of shocks.
            const double h_min =
                Jpr.CalcSingularvalue(dim-1) / (double) H1.GetOrder(0);
            const double inv_dt = sound_speed / h_min +
                                  2.5 * visc_coeff / rho / h_min / h_min;
            if (detJ < 0.0)
            {
               // This will force repetition of the step with smaller dt.
               qdata.dt_est = 0.0;
            }
            else
            {
               if (inv_dt>0.0)
               {
                  qdata.dt_est = fmin(qdata.dt_est, cfl*(1.0/inv_dt));
               }
            }
            // Quadrature data for partial assembly of the force operator.
            MultABt(stress, Jinv, stressJiT);
            stressJiT *= ir_k[e]->IntPoint(q).weight * detJ;
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
