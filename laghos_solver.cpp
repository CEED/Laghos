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
                    int x, int y, int w, int h, bool vec)
{
   gf.HostRead();
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
                            ? "mAcRjlPPPPPPPP" : "mmaaAcl";

         sock << "window_title '" << title << "'\n"
              << "window_geometry "
              << x << " " << y << " " << w << " " << h << "\n"
              << "keys " << keys;
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

static void Rho0DetJ0Vol(const int dim, const int NE,
                         const IntegrationRule &ir,
                         ParMesh *pmesh,
                         ParFiniteElementSpace &L2,
                         const ParGridFunction &rho0,
                         QuadratureData &qdata,
                         double &volume)
{
   const int NQ = ir.GetNPoints();
   const int Q1D = IntRules.Get(Geometry::SEGMENT,ir.GetOrder()).GetNPoints();
   const int flags = GeometricFactors::JACOBIANS|GeometricFactors::DETERMINANTS;
   const GeometricFactors *geom = pmesh->GetGeometricFactors(ir, flags);
   Vector rho0Q(NQ*NE);
   rho0Q.UseDevice(true);
   Vector j, detj;
   const QuadratureInterpolator *qi = L2.GetQuadratureInterpolator(ir);
   qi->Mult(rho0, QuadratureInterpolator::VALUES, rho0Q, j, detj);
   auto W = ir.GetWeights().Read();
   auto R = Reshape(rho0Q.Read(), NQ, NE);
   auto J = Reshape(geom->J.Read(), NQ, dim, dim, NE);
   auto detJ = Reshape(geom->detJ.Read(), NQ, NE);
   auto V = Reshape(qdata.rho0DetJ0w.Write(), NQ, NE);
   Memory<double> &Jinv_m = qdata.Jac0inv.GetMemory();
   const MemoryClass mc = Device::GetMemoryClass();
   const int Ji_total_size = qdata.Jac0inv.TotalSize();
   auto invJ = Reshape(Jinv_m.Write(mc, Ji_total_size), dim, dim, NQ, NE);
   Vector vol(NE*NQ), one(NE*NQ);
   auto A = Reshape(vol.Write(), NQ, NE);
   auto O = Reshape(one.Write(), NQ, NE);
   MFEM_ASSERT(dim==2 || dim==3, "");
   if (dim==2)
   {
      MFEM_FORALL_2D(e, NE, Q1D, Q1D, 1,
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               const int q = qx + qy * Q1D;
               const double J11 = J(q,0,0,e);
               const double J12 = J(q,1,0,e);
               const double J21 = J(q,0,1,e);
               const double J22 = J(q,1,1,e);
               const double det = detJ(q,e);
               V(q,e) =  W[q] * R(q,e) * det;
               const double r_idetJ = 1.0 / det;
               invJ(0,0,q,e) =  J22 * r_idetJ;
               invJ(1,0,q,e) = -J12 * r_idetJ;
               invJ(0,1,q,e) = -J21 * r_idetJ;
               invJ(1,1,q,e) =  J11 * r_idetJ;
               A(q,e) = W[q] * det;
               O(q,e) = 1.0;
            }
         }
      });
   }
   else
   {
      MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
      {
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  const int q = qx + (qy + qz * Q1D) * Q1D;
                  const double J11 = J(q,0,0,e), J12 = J(q,0,1,e), J13 = J(q,0,2,e);
                  const double J21 = J(q,1,0,e), J22 = J(q,1,1,e), J23 = J(q,1,2,e);
                  const double J31 = J(q,2,0,e), J32 = J(q,2,1,e), J33 = J(q,2,2,e);
                  const double det = detJ(q,e);
                  V(q,e) = W[q] * R(q,e) * det;
                  const double r_idetJ = 1.0 / det;
                  invJ(0,0,q,e) = r_idetJ * ((J22 * J33)-(J23 * J32));
                  invJ(1,0,q,e) = r_idetJ * ((J32 * J13)-(J33 * J12));
                  invJ(2,0,q,e) = r_idetJ * ((J12 * J23)-(J13 * J22));
                  invJ(0,1,q,e) = r_idetJ * ((J23 * J31)-(J21 * J33));
                  invJ(1,1,q,e) = r_idetJ * ((J33 * J11)-(J31 * J13));
                  invJ(2,1,q,e) = r_idetJ * ((J13 * J21)-(J11 * J23));
                  invJ(0,2,q,e) = r_idetJ * ((J21 * J32)-(J22 * J31));
                  invJ(1,2,q,e) = r_idetJ * ((J31 * J12)-(J32 * J11));
                  invJ(2,2,q,e) = r_idetJ * ((J11 * J22)-(J12 * J21));
                  A(q,e) = W[q] * det;
                  O(q,e) = 1.0;
               }
            }
         }
      });
   }
   qdata.rho0DetJ0w.HostRead();
   volume = vol * one;
}

LagrangianHydroOperator::LagrangianHydroOperator(const int size,
                                                 ParFiniteElementSpace &h1,
                                                 ParFiniteElementSpace &l2,
                                                 const Array<int> &ess_tdofs,
                                                 Coefficient &rho0_coeff,
                                                 ParGridFunction &rho0_gf,
                                                 Coefficient &gamma_coeff,
                                                 ParGridFunction &gamma_gf,
                                                 const int source,
                                                 const double cfl,
                                                 const bool visc,
                                                 const bool p_assembly,
                                                 const double cgt,
                                                 const int cgiter,
                                                 double ftz,
                                                 const int oq) :
   TimeDependentOperator(size),
   H1(h1), L2(l2), H1c(H1.GetParMesh(), H1.FEColl(), 1),
   pmesh(H1.GetParMesh()),
   H1Vsize(H1.GetVSize()),
   H1TVSize(H1.TrueVSize()),
   H1GTVSize(H1.GlobalTrueVSize()),
   L2Vsize(L2.GetVSize()),
   L2TVSize(L2.TrueVSize()),
   L2GTVSize(L2.GlobalTrueVSize()),
   block_offsets(4),
   x_gf(&H1),
   ess_tdofs(ess_tdofs),
   dim(pmesh->Dimension()),
   NE(pmesh->GetNE()),
   l2dofs_cnt(L2.GetFE(0)->GetDof()),
   h1dofs_cnt(H1.GetFE(0)->GetDof()),
   source_type(source), cfl(cfl),
   use_viscosity(visc),
   p_assembly(p_assembly),
   cg_rel_tol(cgt), cg_max_iter(cgiter),ftz_tol(ftz),
   gamma_coeff(gamma_coeff),
   gamma_gf(gamma_gf),
   Mv(&H1), Mv_spmat_copy(),
   Me(l2dofs_cnt, l2dofs_cnt, NE),
   Me_inv(l2dofs_cnt, l2dofs_cnt, NE),
   ir(IntRules.Get(pmesh->GetElementBaseGeometry(0),
                   (oq > 0) ? oq : 3 * H1.GetOrder(0) + L2.GetOrder(0) - 1)),
   Q1D(int(floor(0.7 + pow(ir.GetNPoints(), 1.0 / dim)))),
   qdata(dim, NE, ir.GetNPoints()),
   qdata_is_current(false),
   forcemat_is_assembled(false),
   Force(&L2, &H1),
   ForcePA(nullptr), VMassPA(nullptr), EMassPA(nullptr),
   VMassPA_Jprec(nullptr),
   CG_VMass(H1.GetParMesh()->GetComm()),
   CG_EMass(L2.GetParMesh()->GetComm()),
   timer(p_assembly ? L2TVSize : 1),
   qupdate(dim, NE, Q1D, visc, cfl, &timer, gamma_gf, ir, H1, L2),
   X(H1c.GetTrueVSize()),
   B(H1c.GetTrueVSize()),
   one(L2Vsize),
   rhs(H1Vsize),
   e_rhs(L2Vsize),
   rhs_c_gf(&H1c),
   dvc_gf(&H1c)
{
   block_offsets[0] = 0;
   block_offsets[1] = block_offsets[0] + H1Vsize;
   block_offsets[2] = block_offsets[1] + H1Vsize;
   block_offsets[3] = block_offsets[2] + L2Vsize;
   one.UseDevice(true);
   one = 1.0;

   if (p_assembly)
   {
      ForcePA = new ForcePAOperator(qdata, H1, L2, ir);
      VMassPA = new MassPAOperator(H1c, ir, rho0_coeff);
      EMassPA = new MassPAOperator(L2, ir, rho0_coeff);
      // Inside the above constructors for mass, there is reordering of the mesh
      // nodes which is performed on the host. Since the mesh nodes are a
      // subvector, so we need to sync with the rest of the base vector (which
      // is assumed to be in the memory space used by the mfem::Device).
      H1.GetParMesh()->GetNodes()->ReadWrite();
      // Attributes 1/2/3 correspond to fixed-x/y/z boundaries, i.e.,
      // we must enforce v_x/y/z = 0 for the velocity components.
      const int bdr_attr_max = H1.GetMesh()->bdr_attributes.Max();
      Array<int> ess_bdr(bdr_attr_max);
      for (int c = 0; c < dim; c++)
      {
         ess_bdr = 0;
         ess_bdr[c] = 1;
         H1c.GetEssentialTrueDofs(ess_bdr, c_tdofs[c]);
         c_tdofs[c].Read();
      }
      X.UseDevice(true);
      B.UseDevice(true);
      rhs.UseDevice(true);
      e_rhs.UseDevice(true);
   }
   else
   {
      // Standard local assembly and inversion for energy mass matrices.
      // 'Me' is used in the computation of the internal energy
      // which is used twice: once at the start and once at the end of the run.
      MassIntegrator mi(rho0_coeff, &ir);
      for (int e = 0; e < NE; e++)
      {
         DenseMatrixInverse inv(&Me(e));
         const FiniteElement &fe = *L2.GetFE(e);
         ElementTransformation &Tr = *L2.GetElementTransformation(e);
         mi.AssembleElementMatrix(fe, Tr, Me(e));
         inv.Factor();
         inv.GetInverseMatrix(Me_inv(e));
      }
      // Standard assembly for the velocity mass matrix.
      VectorMassIntegrator *vmi = new VectorMassIntegrator(rho0_coeff, &ir);
      Mv.AddDomainIntegrator(vmi);
      Mv.Assemble();
      Mv_spmat_copy = Mv.SpMat();
   }

   // Values of rho0DetJ0 and Jac0inv at all quadrature points.
   // Initial local mesh size (assumes all mesh elements are the same).
   int Ne, ne = NE;
   double Volume, vol = 0.0;
   if (dim > 1) { Rho0DetJ0Vol(dim, NE, ir, pmesh, L2, rho0_gf, qdata, vol); }
   else
   {
      const int NQ = ir.GetNPoints();
      Vector rho_vals(NQ);
      for (int e = 0; e < NE; e++)
      {
         rho0_gf.GetValues(e, ir, rho_vals);
         ElementTransformation &Tr = *H1.GetElementTransformation(e);
         for (int q = 0; q < NQ; q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            Tr.SetIntPoint(&ip);
            DenseMatrixInverse Jinv(Tr.Jacobian());
            Jinv.GetInverseMatrix(qdata.Jac0inv(e*NQ + q));
            const double rho0DetJ0 = Tr.Weight() * rho_vals(q);
            qdata.rho0DetJ0w(e*NQ + q) = rho0DetJ0 * ir.IntPoint(q).weight;
         }
      }
      for (int e = 0; e < NE; e++) { vol += pmesh->GetElementVolume(e); }
   }
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

   if (p_assembly)
   {
      // Setup the preconditioner of the velocity mass operator.
      // BC are handled by the VMassPA, so ess_tdofs here can be empty.
      Array<int> ess_tdofs;
      VMassPA_Jprec = new OperatorJacobiSmoother(VMassPA->GetBF(), ess_tdofs);
      CG_VMass.SetPreconditioner(*VMassPA_Jprec);

      CG_VMass.SetOperator(*VMassPA);
      CG_VMass.SetRelTol(cg_rel_tol);
      CG_VMass.SetAbsTol(0.0);
      CG_VMass.SetMaxIter(cg_max_iter);
      CG_VMass.SetPrintLevel(-1);

      CG_EMass.SetOperator(*EMassPA);
      CG_EMass.iterative_mode = false;
      CG_EMass.SetRelTol(cg_rel_tol);
      CG_EMass.SetAbsTol(0.0);
      CG_EMass.SetMaxIter(cg_max_iter);
      CG_EMass.SetPrintLevel(-1);
   }
   else
   {
      ForceIntegrator *fi = new ForceIntegrator(qdata);
      fi->SetIntRule(&ir);
      Force.AddDomainIntegrator(fi);
      // Make a dummy assembly to figure out the sparsity.
      Force.Assemble(0);
      Force.Finalize(0);
   }
}

LagrangianHydroOperator::~LagrangianHydroOperator()
{
   if (p_assembly)
   {
      delete EMassPA;
      delete VMassPA;
      delete VMassPA_Jprec;
      delete ForcePA;
   }
}

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
   if (p_assembly)
   {
      timer.sw_force.Start();
      ForcePA->Mult(one, rhs);
      timer.sw_force.Stop();
      rhs.Neg();

      // Partial assembly solve for each velocity component
      const int size = H1c.GetVSize();
      const Operator *Pconf = H1c.GetProlongationMatrix();
      for (int c = 0; c < dim; c++)
      {
         dvc_gf.MakeRef(&H1c, dS_dt, H1Vsize + c*size);
         rhs_c_gf.MakeRef(&H1c, rhs, c*size);
         if (Pconf) { Pconf->MultTranspose(rhs_c_gf, B); }
         else { B = rhs_c_gf; }
         H1c.GetRestrictionMatrix()->Mult(dvc_gf, X);
         VMassPA->SetEssentialTrueDofs(c_tdofs[c]);
         VMassPA->EliminateRHS(B);
         timer.sw_cgH1.Start();
         CG_VMass.Mult(B, X);
         timer.sw_cgH1.Stop();
         timer.H1iter += CG_VMass.GetNumIterations();
         if (Pconf) { Pconf->Mult(X, dvc_gf); }
         else { dvc_gf = X; }
         // We need to sync the subvector 'dvc_gf' with its base vector
         // because it may have been moved to a different memory space.
         dvc_gf.GetMemory().SyncAlias(dS_dt.GetMemory(), dvc_gf.Size());
      }
   }
   else
   {
      timer.sw_force.Start();
      Force.Mult(one, rhs);
      timer.sw_force.Stop();
      rhs.Neg();

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
      timer.sw_cgH1.Start();
      cg.Mult(B, X);
      timer.sw_cgH1.Stop();
      timer.H1iter += cg.GetNumIterations();
      Mv.RecoverFEMSolution(X, rhs, dv);
   }
}

void LagrangianHydroOperator::SolveEnergy(const Vector &S, const Vector &v,
                                          Vector &dS_dt) const
{
   UpdateQuadratureData(S);
   AssembleForceMatrix();

   // The monolithic BlockVector stores the unknown fields as follows:
   // (Position, Velocity, Specific Internal Energy).
   ParGridFunction de;
   de.MakeRef(&L2, dS_dt, H1Vsize*2);
   de = 0.0;

   // Solve for energy, assemble the energy source if such exists.
   LinearForm *e_source = nullptr;
   if (source_type == 1) // 2D Taylor-Green.
   {
      e_source = new LinearForm(&L2);
      TaylorCoefficient coeff;
      DomainLFIntegrator *d = new DomainLFIntegrator(coeff, &ir);
      e_source->AddDomainIntegrator(d);
      e_source->Assemble();
   }

   Array<int> l2dofs;
   if (p_assembly)
   {
      timer.sw_force.Start();
      ForcePA->MultTranspose(v, e_rhs);
      timer.sw_force.Stop();
      if (e_source) { e_rhs += *e_source; }
      timer.sw_cgL2.Start();
      CG_EMass.Mult(e_rhs, de);
      timer.sw_cgL2.Stop();
      const HYPRE_Int cg_num_iter = CG_EMass.GetNumIterations();
      timer.L2iter += (cg_num_iter==0) ? 1 : cg_num_iter;
      // Move the memory location of the subvector 'de' to the memory
      // location of the base vector 'dS_dt'.
      de.GetMemory().SyncAlias(dS_dt.GetMemory(), de.Size());
   }
   else // not p_assembly
   {
      timer.sw_force.Start();
      Force.MultTranspose(v, e_rhs);
      timer.sw_force.Stop();
      if (e_source) { e_rhs += *e_source; }
      Vector loc_rhs(l2dofs_cnt), loc_de(l2dofs_cnt);
      for (int e = 0; e < NE; e++)
      {
         L2.GetElementDofs(e, l2dofs);
         e_rhs.GetSubVector(l2dofs, loc_rhs);
         timer.sw_cgL2.Start();
         Me_inv(e).Mult(loc_rhs, loc_de);
         timer.sw_cgL2.Stop();
         timer.L2iter += 1;
         de.SetSubVector(l2dofs, loc_de);
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

void LagrangianHydroOperator::ComputeDensity(ParGridFunction &rho) const
{
   rho.SetSpace(&L2);
   DenseMatrix Mrho(l2dofs_cnt);
   Vector rhs(l2dofs_cnt), rho_z(l2dofs_cnt);
   Array<int> dofs(l2dofs_cnt);
   DenseMatrixInverse inv(&Mrho);
   MassIntegrator mi(&ir);
   DensityIntegrator di(qdata);
   di.SetIntRule(&ir);
   for (int e = 0; e < NE; e++)
   {
      const FiniteElement &fe = *L2.GetFE(e);
      ElementTransformation &eltr = *L2.GetElementTransformation(e);
      di.AssembleRHSElementVect(fe, eltr, rhs);
      mi.AssembleElementMatrix(fe, eltr, Mrho);
      inv.Factor();
      inv.Mult(rhs, rho_z);
      L2.GetElementDofs(e, dofs);
      rho.SetSubVector(dofs, rho_z);
   }
}

double LagrangianHydroOperator::InternalEnergy(const ParGridFunction &gf) const
{
   double glob_ie = 0.0;
   // This should be turned into a kernel so that it could be displayed in pa
   if (!p_assembly)
   {
      Vector one(l2dofs_cnt), loc_e(l2dofs_cnt);
      one = 1.0;
      Array<int> l2dofs;
      double loc_ie = 0.0;
      for (int e = 0; e < NE; e++)
      {
         L2.GetElementDofs(e, l2dofs);
         gf.GetSubVector(l2dofs, loc_e);
         loc_ie += Me(e).InnerProduct(loc_e, one);
      }
      MPI_Comm comm = H1.GetParMesh()->GetComm();
      MPI_Allreduce(&loc_ie, &glob_ie, 1, MPI_DOUBLE, MPI_SUM, comm);
   }
   return glob_ie;
}

double LagrangianHydroOperator::KineticEnergy(const ParGridFunction &v) const
{
   double glob_ke = 0.0;
   // This should be turned into a kernel so that it could be displayed in pa
   if (!p_assembly)
   {
      double loc_ke = 0.5 * Mv_spmat_copy.InnerProduct(v, v);
      MPI_Allreduce(&loc_ke, &glob_ke, 1, MPI_DOUBLE, MPI_SUM,
                    H1.GetParMesh()->GetComm());
   }
   return glob_ke;
}

void LagrangianHydroOperator::PrintTimingData(bool IamRoot, int steps,
                                              const bool fom) const
{
   const MPI_Comm com = H1.GetComm();
   double my_rt[5], T[5];
   my_rt[0] = timer.sw_cgH1.RealTime();
   my_rt[1] = timer.sw_cgL2.RealTime();
   my_rt[2] = timer.sw_force.RealTime();
   my_rt[3] = timer.sw_qdata.RealTime();
   my_rt[4] = my_rt[0] + my_rt[2] + my_rt[3];
   MPI_Reduce(my_rt, T, 5, MPI_DOUBLE, MPI_MAX, 0, com);

   HYPRE_Int mydata[3], alldata[3];
   mydata[0] = timer.L2dof * timer.L2iter;
   mydata[1] = timer.quad_tstep;
   mydata[2] = NE;
   MPI_Reduce(mydata, alldata, 3, HYPRE_MPI_INT, MPI_SUM, 0, com);

   if (IamRoot)
   {
      using namespace std;
      // FOM = (FOM1 * T1 + FOM2 * T2 + FOM3 * T3) / (T1 + T2 + T3)
      const HYPRE_Int H1iter = p_assembly ? (timer.H1iter/dim) : timer.H1iter;
      const double FOM1 = 1e-6 * H1GTVSize * H1iter / T[0];
      const double FOM2 = 1e-6 * steps * (H1GTVSize + L2GTVSize) / T[2];
      const double FOM3 = 1e-6 * alldata[1] * ir.GetNPoints() / T[3];
      const double FOM = (FOM1 * T[0] + FOM2 * T[2] + FOM3 * T[3]) / T[4];
      const double FOM0 = 1e-6 * steps * (H1GTVSize + L2GTVSize) / T[4];
      cout << endl;
      cout << "CG (H1) total time: " << T[0] << endl;
      cout << "CG (H1) rate (megadofs x cg_iterations / second): "
           << FOM1 << endl;
      cout << endl;
      cout << "CG (L2) total time: " << T[1] << endl;
      cout << "CG (L2) rate (megadofs x cg_iterations / second): "
           << 1e-6 * alldata[0] / T[1] << endl;
      cout << endl;
      cout << "Forces total time: " << T[2] << endl;
      cout << "Forces rate (megadofs x timesteps / second): "
           << FOM2 << endl;
      cout << endl;
      cout << "UpdateQuadData total time: " << T[3] << endl;
      cout << "UpdateQuadData rate (megaquads x timesteps / second): "
           << FOM3 << endl;
      cout << endl;
      cout << "Major kernels total time (seconds): " << T[4] << endl;
      cout << "Major kernels total rate (megadofs x time steps / second): "
           << FOM << endl;
      if (!fom) { return; }
      const int QPT = ir.GetNPoints();
      const HYPRE_Int GNZones = alldata[2];
      const long ndofs = 2*H1GTVSize + L2GTVSize + QPT*GNZones;
      cout << endl;
      cout << "| Ranks " << "| Zones   "
           << "| H1 dofs " << "| L2 dofs "
           << "| QP "      << "| N dofs   "
           << "| FOM0   "
           << "| FOM1   " << "| T1   "
           << "| FOM2   " << "| T2   "
           << "| FOM3   " << "| T3   "
           << "| FOM    " << "| TT   "
           << "|" << endl;
      cout << setprecision(3);
      cout << "| " << setw(6) << H1.GetNRanks()
           << "| " << setw(8) << GNZones
           << "| " << setw(8) << H1GTVSize
           << "| " << setw(8) << L2GTVSize
           << "| " << setw(3) << QPT
           << "| " << setw(9) << ndofs
           << "| " << setw(7) << FOM0
           << "| " << setw(7) << FOM1
           << "| " << setw(5) << T[0]
           << "| " << setw(7) << FOM2
           << "| " << setw(5) << T[2]
           << "| " << setw(7) << FOM3
           << "| " << setw(5) << T[3]
           << "| " << setw(7) << FOM
           << "| " << setw(5) << T[4]
           << "| " << endl;
   }
}

// Smooth transition between 0 and 1 for x in [-eps, eps].
MFEM_HOST_DEVICE inline double smooth_step_01(double x, double eps)
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

   if (dim > 1) { return qupdate.UpdateQuadratureData(S, qdata); }

   // This code is only for the 1D/FA mode
   timer.sw_qdata.Start();
   const int nqp = ir.GetNPoints();
   ParGridFunction x, v, e;
   Vector* sptr = const_cast<Vector*>(&S);
   x.MakeRef(&H1, *sptr, 0);
   v.MakeRef(&H1, *sptr, H1.GetVSize());
   e.MakeRef(&L2, *sptr, 2*H1.GetVSize());
   Vector e_vals, e_loc(l2dofs_cnt), vector_vals(h1dofs_cnt * dim);
   DenseMatrix Jpi(dim), sgrad_v(dim), Jinv(dim), stress(dim), stressJiT(dim),
               vecvalMat(vector_vals.GetData(), h1dofs_cnt, dim);
   DenseTensor grad_v_ref(dim, dim, nqp);
   Array<int> L2dofs, H1dofs;
   // Batched computations are needed, because hydrodynamic codes usually
   // involve expensive computations of material properties. Although this
   // miniapp uses simple EOS equations, we still want to represent the batched
   // cycle structure.
   int nzones_batch = 3;
   const int nbatches =  NE / nzones_batch + 1; // +1 for the remainder.
   int nqp_batch = nqp * nzones_batch;
   double *gamma_b = new double[nqp_batch],
   *rho_b = new double[nqp_batch],
   *e_b   = new double[nqp_batch],
   *p_b   = new double[nqp_batch],
   *cs_b  = new double[nqp_batch];
   // Jacobians of reference->physical transformations for all quadrature points
   // in the batch.
   DenseTensor *Jpr_b = new DenseTensor[nzones_batch];
   for (int b = 0; b < nbatches; b++)
   {
      int z_id = b * nzones_batch; // Global index over zones.
      // The last batch might not be full.
      if (z_id == NE) { break; }
      else if (z_id + nzones_batch > NE)
      {
         nzones_batch = NE - z_id;
         nqp_batch    = nqp * nzones_batch;
      }
      double min_detJ = std::numeric_limits<double>::infinity();
      for (int z = 0; z < nzones_batch; z++)
      {
         ElementTransformation *T = H1.GetElementTransformation(z_id);
         Jpr_b[z].SetSize(dim, dim, nqp);
         e.GetValues(z_id, ir, e_vals);
         for (int q = 0; q < nqp; q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            T->SetIntPoint(&ip);
            Jpr_b[z](q) = T->Jacobian();
            const double detJ = Jpr_b[z](q).Det();
            min_detJ = fmin(min_detJ, detJ);
            const int idx = z * nqp + q;
            gamma_b[idx] = gamma_coeff.Eval(*T, ip);
            rho_b[idx] = qdata.rho0DetJ0w(z_id*nqp + q) / detJ / ip.weight;
            e_b[idx] = fmax(0.0, e_vals(q));
         }
         ++z_id;
      }
      // Batched computation of material properties.
      ComputeMaterialProperties(nqp_batch, gamma_b, rho_b, e_b, p_b, cs_b);
      z_id -= nzones_batch;
      for (int z = 0; z < nzones_batch; z++)
      {
         ElementTransformation *T = H1.GetElementTransformation(z_id);
         for (int q = 0; q < nqp; q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            T->SetIntPoint(&ip);
            // Note that the Jacobian was already computed above. We've chosen
            // not to store the Jacobians for all batched quadrature points.
            const DenseMatrix &Jpr = Jpr_b[z](q);
            CalcInverse(Jpr, Jinv);
            const double detJ = Jpr.Det(), rho = rho_b[z*nqp + q],
                         p = p_b[z*nqp + q], sound_speed = cs_b[z*nqp + q];
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
               double eig_val_data[3], eig_vec_data[9];
               eig_val_data[0] = sgrad_v(0, 0);
               eig_vec_data[0] = 1.;
               Vector compr_dir(eig_vec_data, dim);
               // Computes the initial->physical transformation Jacobian.
               mfem::Mult(Jpr, qdata.Jac0inv(z_id*nqp + q), Jpi);
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
               visc_coeff += 0.5 * rho * h * sound_speed *
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
            if (min_detJ < 0.0)
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
            stressJiT *= ir.IntPoint(q).weight * detJ;
            for (int vd = 0 ; vd < dim; vd++)
            {
               for (int gd = 0; gd < dim; gd++)
               {
                  qdata.stressJinvT(vd)(z_id*nqp + q, gd) =
                     stressJiT(vd, gd);
               }
            }
         }
         ++z_id;
      }
   }
   delete [] gamma_b;
   delete [] rho_b;
   delete [] e_b;
   delete [] p_b;
   delete [] cs_b;
   delete [] Jpr_b;
   timer.sw_qdata.Stop();
   timer.quad_tstep += NE;
}

template<int DIM> MFEM_HOST_DEVICE static inline
void QUpdateBody(const int NE, const int e,
                 const int NQ, const int q,
                 const bool use_viscosity,
                 const double h0,
                 const double h1order,
                 const double cfl,
                 const double infinity,
                 double* __restrict__ Jinv,
                 double* __restrict__ stress,
                 double* __restrict__ sgrad_v,
                 double* __restrict__ eig_val_data,
                 double* __restrict__ eig_vec_data,
                 double* __restrict__ compr_dir,
                 double* __restrict__ Jpi,
                 double* __restrict__ ph_dir,
                 double* __restrict__ stressJiT,
                 const double* __restrict__ d_gamma,
                 const double* __restrict__ d_weights,
                 const double* __restrict__ d_Jacobians,
                 const double* __restrict__ d_rho0DetJ0w,
                 const double* __restrict__ d_e_quads,
                 const double* __restrict__ d_grad_v_ext,
                 const double* __restrict__ d_Jac0inv,
                 double *d_dt_est,
                 double *d_stressJinvT)
{
   constexpr int DIM2 = DIM*DIM;
   double min_detJ = infinity;

   const int eq = e * NQ + q;
   const double gamma = d_gamma[e];
   const double weight =  d_weights[q];
   const double inv_weight = 1. / weight;
   const double *J = d_Jacobians + DIM2*(NQ*e + q);
   const double detJ = kernels::Det<DIM>(J);
   min_detJ = fmin(min_detJ, detJ);
   kernels::CalcInverse<DIM>(J, Jinv);
   const double R = inv_weight * d_rho0DetJ0w[eq] / detJ;
   const double E = fmax(0.0, d_e_quads[eq]);
   const double P = (gamma - 1.0) * R * E;
   const double S = sqrt(gamma * (gamma - 1.0) * E);
   for (int k = 0; k < DIM2; k++) { stress[k] = 0.0; }
   for (int d = 0; d < DIM; d++) { stress[d*DIM+d] = -P; }
   double visc_coeff = 0.0;
   if (use_viscosity)
   {
      // Compression-based length scale at the point. The first
      // eigenvector of the symmetric velocity gradient gives the
      // direction of maximal compression. This is used to define the
      // relative change of the initial length scale.
      const double *dV = d_grad_v_ext + DIM2*(NQ*e + q);
      kernels::Mult(DIM, DIM, DIM, dV, Jinv, sgrad_v);
      kernels::Symmetrize(DIM, sgrad_v);
      if (DIM == 1)
      {
         eig_val_data[0] = sgrad_v[0];
         eig_vec_data[0] = 1.;
      }
      else
      {
         kernels::CalcEigenvalues<DIM>(sgrad_v, eig_val_data, eig_vec_data);
      }
      for (int k=0; k<DIM; k++) { compr_dir[k] = eig_vec_data[k]; }
      // Computes the initial->physical transformation Jacobian.
      kernels::Mult(DIM, DIM, DIM, J, d_Jac0inv + eq*DIM*DIM, Jpi);
      kernels::Mult(DIM, DIM, Jpi, compr_dir, ph_dir);
      // Change of the initial mesh size in the compression direction.
      const double ph_dir_nl2 = kernels::Norml2(DIM, ph_dir);
      const double compr_dir_nl2 = kernels::Norml2(DIM, compr_dir);
      const double H = h0 * ph_dir_nl2 / compr_dir_nl2;
      // Measure of maximal compression.
      const double mu = eig_val_data[0];
      visc_coeff = 2.0 * R * H * H * fabs(mu);
      // The following represents a "smooth" version of the statement
      // "if (mu < 0) visc_coeff += 0.5 rho h sound_speed".  Note that
      // eps must be scaled appropriately if a different unit system is
      // being used.
      const double eps = 1e-12;
      visc_coeff += 0.5 * R * H  * S * (1.0 - smooth_step_01(mu-2.0*eps, eps));
      kernels::Add(DIM, DIM, visc_coeff, stress, sgrad_v, stress);
   }
   // Time step estimate at the point. Here the more relevant length
   // scale is related to the actual mesh deformation; we use the min
   // singular value of the ref->physical Jacobian. In addition, the
   // time step estimate should be aware of the presence of shocks.
   const double sv = kernels::CalcSingularvalue<DIM>(J, DIM - 1);
   const double h_min = sv / h1order;
   const double ih_min = 1. / h_min;
   const double irho_ih_min_sq = ih_min * ih_min / R ;
   const double idt = S * ih_min + 2.5 * visc_coeff * irho_ih_min_sq;
   if (min_detJ < 0.0)
   {
      // This will force repetition of the step with smaller dt.
      d_dt_est[eq] = 0.0;
   }
   else
   {
      if (idt > 0.0)
      {
         const double cfl_inv_dt = cfl / idt;
         d_dt_est[eq] = fmin(d_dt_est[eq], cfl_inv_dt);
      }
   }
   // Quadrature data for partial assembly of the force operator.
   kernels::MultABt(DIM, DIM, DIM, stress, Jinv, stressJiT);
   for (int k = 0; k < DIM2; k++) { stressJiT[k] *= weight * detJ; }
   for (int vd = 0 ; vd < DIM; vd++)
   {
      for (int gd = 0; gd < DIM; gd++)
      {
         const int offset = eq + NQ*NE*(gd + vd*DIM);
         d_stressJinvT[offset] = stressJiT[vd + gd*DIM];
      }
   }
}

template<int DIM, int Q1D> static inline
void QKernel(const int NE, const int NQ,
             const bool use_viscosity,
             const double h0,
             const double h1order,
             const double cfl,
             const double infinity,
             const ParGridFunction &gamma_gf,
             const Array<double> &weights,
             const Vector &Jacobians,
             const Vector &rho0DetJ0w,
             const Vector &e_quads,
             const Vector &grad_v_ext,
             const DenseTensor &Jac0inv,
             Vector &dt_est,
             DenseTensor &stressJinvT)
{
   constexpr int DIM2 = DIM*DIM;
   auto d_gamma = gamma_gf.Read();
   auto d_weights = weights.Read();
   auto d_Jacobians = Jacobians.Read();
   auto d_rho0DetJ0w = rho0DetJ0w.Read();
   auto d_e_quads = e_quads.Read();
   auto d_grad_v_ext = grad_v_ext.Read();
   auto d_Jac0inv = Read(Jac0inv.GetMemory(), Jac0inv.TotalSize());
   auto d_dt_est = dt_est.ReadWrite();
   auto d_stressJinvT = Write(stressJinvT.GetMemory(), stressJinvT.TotalSize());
   if (DIM == 2)
   {
      MFEM_FORALL_2D(e, NE, Q1D, Q1D, 1,
      {
         double Jinv[DIM2];
         double stress[DIM2];
         double sgrad_v[DIM2];
         double eig_val_data[3];
         double eig_vec_data[9];
         double compr_dir[DIM];
         double Jpi[DIM2];
         double ph_dir[DIM];
         double stressJiT[DIM2];
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               QUpdateBody<DIM>(NE, e, NQ, qx + qy * Q1D,
               use_viscosity, h0, h1order, cfl, infinity,
               Jinv, stress, sgrad_v, eig_val_data, eig_vec_data,
               compr_dir, Jpi, ph_dir, stressJiT,
               d_gamma, d_weights, d_Jacobians, d_rho0DetJ0w,
               d_e_quads, d_grad_v_ext, d_Jac0inv,
               d_dt_est, d_stressJinvT);
            }
         }
         MFEM_SYNC_THREAD;
      });
   }
   if (DIM == 3)
   {
      MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
      {
         double Jinv[DIM2];
         double stress[DIM2];
         double sgrad_v[DIM2];
         double eig_val_data[3];
         double eig_vec_data[9];
         double compr_dir[DIM];
         double Jpi[DIM2];
         double ph_dir[DIM];
         double stressJiT[DIM2];
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qz,z,Q1D)
               {
                  QUpdateBody<DIM>(NE, e, NQ, qx + Q1D * (qy + qz * Q1D),
                  use_viscosity, h0, h1order, cfl, infinity,
                  Jinv, stress, sgrad_v, eig_val_data, eig_vec_data,
                  compr_dir, Jpi, ph_dir, stressJiT,
                  d_gamma, d_weights, d_Jacobians, d_rho0DetJ0w,
                  d_e_quads, d_grad_v_ext, d_Jac0inv,
                  d_dt_est, d_stressJinvT);
               }
            }
         }
         MFEM_SYNC_THREAD;
      });
   }
}

void QUpdate::UpdateQuadratureData(const Vector &S, QuadratureData &qdata)
{
   timer->sw_qdata.Start();
   Vector* S_p = const_cast<Vector*>(&S);
   const int H1_size = H1.GetVSize();
   const double h1order = (double) H1.GetOrder(0);
   const double infinity = std::numeric_limits<double>::infinity();
   ParGridFunction x, v, e;
   x.MakeRef(&H1,*S_p, 0);
   H1R->Mult(x, e_vec);
   q1->SetOutputLayout(QVectorLayout::byVDIM);
   q1->Derivatives(e_vec, q_dx);
   v.MakeRef(&H1,*S_p, H1_size);
   H1R->Mult(v, e_vec);
   q1->Derivatives(e_vec, q_dv);
   e.MakeRef(&L2, *S_p, 2*H1_size);
   q2->SetOutputLayout(QVectorLayout::byVDIM);
   q2->Values(e, q_e);
   q_dt_est = qdata.dt_est;
   const int id = (dim << 4) | Q1D;
   typedef void (*fQKernel)(const int NE, const int NQ,
                            const bool use_viscosity,
                            const double h0, const double h1order,
                            const double cfl, const double infinity,
                            const ParGridFunction &gamma_gf,
                            const Array<double> &weights,
                            const Vector &Jacobians, const Vector &rho0DetJ0w,
                            const Vector &e_quads, const Vector &grad_v_ext,
                            const DenseTensor &Jac0inv,
                            Vector &dt_est, DenseTensor &stressJinvT);
   static std::unordered_map<int, fQKernel> qupdate =
   {
      {0x24,&QKernel<2,4>}, {0x26,&QKernel<2,6>}, {0x28,&QKernel<2,8>},
      {0x34,&QKernel<3,4>}, {0x36,&QKernel<3,6>}, {0x38,&QKernel<3,8>}
   };
   if (!qupdate[id])
   {
      mfem::out << "Unknown kernel 0x" << std::hex << id << std::endl;
      MFEM_ABORT("Unknown kernel");
   }
   qupdate[id](NE, NQ, use_viscosity, qdata.h0, h1order, cfl, infinity,
               gamma_gf, ir.GetWeights(), q_dx,
               qdata.rho0DetJ0w, q_e, q_dv,
               qdata.Jac0inv, q_dt_est, qdata.stressJinvT);
   qdata.dt_est = q_dt_est.Min();
   timer->sw_qdata.Stop();
   timer->quad_tstep += NE;
}

void LagrangianHydroOperator::AssembleForceMatrix() const
{
   if (forcemat_is_assembled || p_assembly) { return; }
   Force = 0.0;
   timer.sw_force.Start();
   Force.Assemble();
   timer.sw_force.Stop();
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
   V.UseDevice(true);
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
