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

#include "general/dbg.hpp"
#include "general/nvvp.hpp"
#include "laghos_solver.hpp"
#include "linalg/eigen.hpp"
#include <unordered_map>

#ifdef MFEM_USE_MPI

using namespace std;

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
         sock << "window_title '" << title << "'\n"
              << "window_geometry "
              << x << " " << y << " " << w << " " << h << "\n"
              << "keys mmaaAcl";
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

// *****************************************************************************
void ComputeRho0DetJ0AndVolume(const int dim,
                               const int NE,
                               const IntegrationRule &ir,
                               ParMesh *mesh,
                               ParFiniteElementSpace &l2_fes,
                               ParGridFunction &rho0,
                               QuadratureData &quad_data,
                               double &loc_area)
{
   const int NQ = ir.GetNPoints();
   const int Q1D = IntRules.Get(Geometry::SEGMENT,ir.GetOrder()).GetNPoints();
   // Get J & detJ
   const int flags = GeometricFactors::JACOBIANS|GeometricFactors::DETERMINANTS;
   const GeometricFactors *geom = mesh->GetGeometricFactors(ir, flags);
   // rho0: E (no R) => Q
   Vector rho0Q(NQ*NE);
   rho0Q.UseDevice(true);
   Vector j, detj;
   const QuadratureInterpolator *qi = l2_fes.GetQuadratureInterpolator(ir);
   qi->Mult(rho0, QuadratureInterpolator::VALUES, rho0Q, j, detj);
   // R/W
   auto W = ir.GetWeights().Read();
   auto R = Reshape(rho0Q.Read(), NQ, NE);
   auto J = Reshape(geom->J.Read(), NQ, dim, dim, NE);
   auto detJ = Reshape(geom->detJ.Read(), NQ, NE);
   auto V = Reshape(quad_data.rho0DetJ0w.Write(), NQ, NE);
   Memory<double> &Jinv_m = quad_data.Jac0inv.GetMemory();
   auto invJ = Reshape(Jinv_m.Write(Device::GetMemoryClass(),
                                    quad_data.Jac0inv.TotalSize()),
                       dim, dim, NQ, NE);
   Vector area(NE*NQ), one(NE*NQ);
   auto A = Reshape(area.Write(), NQ, NE);
   auto O = Reshape(one.Write(), NQ, NE);
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
   quad_data.rho0DetJ0w.HostRead();
   loc_area = area * one;
}

LagrangianHydroOperator::LagrangianHydroOperator(Coefficient &rho_coeff,
                                                 const int size,
                                                 ParFiniteElementSpace &h1_fes,
                                                 ParFiniteElementSpace &l2_fes,
                                                 const Array<int> &essential_tdofs,
                                                 ParGridFunction &rho0,
                                                 const int source_type_,
                                                 const double cfl_,
                                                 Coefficient *material_,
                                                 const bool visc,
                                                 const bool pa,
                                                 const double cgt,
                                                 const int cgiter,
                                                 const int order_q,
                                                 const bool qupt,
                                                 const double gm,
                                                 const bool ok,
                                                 int h1_basis_type) :
   TimeDependentOperator(size),
   H1FESpace(h1_fes), L2FESpace(l2_fes),
   H1compFESpace(h1_fes.GetParMesh(), h1_fes.FEColl(), 1),
   H1Vsize(H1FESpace.GetVSize()),
   H1TVSize(H1FESpace.TrueVSize()),
   H1GTVSize(H1FESpace.GlobalTrueVSize()),
   H1compTVSize(H1compFESpace.GetTrueVSize()),
   L2Vsize(L2FESpace.GetVSize()),
   L2TVSize(L2FESpace.TrueVSize()),
   L2GTVSize(L2FESpace.GlobalTrueVSize()),
   block_offsets(4),
   x_gf(&H1FESpace),
   ess_tdofs(essential_tdofs),
   dim(h1_fes.GetMesh()->Dimension()),
   nzones(h1_fes.GetMesh()->GetNE()),
   l2dofs_cnt(l2_fes.GetFE(0)->GetDof()),
   h1dofs_cnt(h1_fes.GetFE(0)->GetDof()),
   source_type(source_type_), cfl(cfl_),
   use_viscosity(visc), p_assembly(pa),
   okina(ok),
   cg_rel_tol(cgt), cg_max_iter(cgiter),
   material_pcf(material_),
   Mv(&h1_fes), Mv_spmat_copy(),
   Me(l2dofs_cnt, l2dofs_cnt, nzones),
   Me_inv(l2dofs_cnt, l2dofs_cnt, nzones),
   integ_rule(IntRules.Get(h1_fes.GetMesh()->GetElementBaseGeometry(0),
                           (order_q>0)? order_q :
                           3*h1_fes.GetOrder(0) + l2_fes.GetOrder(0) - 1)),
   quad_data(dim, nzones, integ_rule.GetNPoints()),
   quad_data_is_current(false), forcemat_is_assembled(false),
   tensors1D(H1FESpace.GetFE(0)->GetOrder(), L2FESpace.GetFE(0)->GetOrder(),
             int(floor(0.7 + pow(integ_rule.GetNPoints(), 1.0 / dim))),
             h1_basis_type == BasisType::Positive),
   evaluator(H1FESpace, &tensors1D),
   Force(&l2_fes, &h1_fes),
   VMassPA_prec(okina?H1compFESpace:H1FESpace),
   locEMassPA(quad_data, l2_fes, &tensors1D),
   CG_VMass(H1FESpace.GetParMesh()->GetComm()),
   CG_EMass(L2FESpace.GetParMesh()->GetComm()),
   locCG(),
   // Energy solver can be global, local or just a direct inverse
   timer(okina? L2TVSize: p_assembly? l2dofs_cnt: 1),
   // QUpdate bool and inputs
   qupdate(qupt),
   gamma(gm),
   Q(dim, nzones, use_viscosity, cfl, gamma,
     &timer, integ_rule, H1FESpace, L2FESpace),
   X(H1compFESpace.GetTrueVSize()),
   B(H1compFESpace.GetTrueVSize()),
   one(L2Vsize),
   rhs(H1Vsize),
   e_rhs(L2Vsize),
   rhs_c_gf(&H1compFESpace),
   dvc_gf(&H1compFESpace)
{
   push();
   block_offsets[0] = 0;
   block_offsets[1] = block_offsets[0] + H1Vsize;
   block_offsets[2] = block_offsets[1] + H1Vsize;
   block_offsets[3] = block_offsets[2] + L2Vsize;

   one.UseDevice(true);
   one = 1.0;

   if (not okina)
   {
      ForcePA = new ForcePAOperator(quad_data, h1_fes,l2_fes, &tensors1D);
      VMassPA = new MassPAOperator(quad_data, H1FESpace, &tensors1D);
      EMassPA = new MassPAOperator(quad_data, L2FESpace, &tensors1D);
   }
   else
   {
      push("ForcePA",Silver);
      ForcePA = new OkinaForcePAOperator(quad_data, h1_fes,l2_fes, integ_rule);
      pop();
      push("VMassPA",Silver);
      VMassPA = new OkinaMassPAOperator(rho_coeff, quad_data, H1compFESpace,
                                        integ_rule, &tensors1D);
      pop();
      push("EMassPA",Silver);
      EMassPA = new OkinaMassPAOperator(rho_coeff, quad_data, L2FESpace,
                                        integ_rule, &tensors1D);
      pop();
      // Inside the above constructors for mass, there is reordering of the mesh
      // nodes which is performed on the host. Since the mesh nodes are a
      // subvector, so we need to sync with the rest of the base vector (which
      // is assumed to be in the memory space used by the mfem::Device).
      H1FESpace.GetParMesh()->GetNodes()->ReadWrite();
      // FIXME: do the above with a method in Memory that syncs aliases with
      // their base Memory. How do we get the base here?

      // Attributes 1/2/3 correspond to fixed-x/y/z boundaries, i.e.,
      // we must enforce v_x/y/z = 0 for the velocity components.
      const int bdr_attr_max = H1FESpace.GetMesh()->bdr_attributes.Max();
      Array<int> ess_bdr(bdr_attr_max);
      for (int c = 0; c < dim; c++)
      {
         ess_bdr = 0; ess_bdr[c] = 1;
         H1compFESpace.GetEssentialTrueDofs(ess_bdr, c_tdofs[c]);
         c_tdofs[c].Read();
      }
      X.UseDevice(true);
      B.UseDevice(true);
      rhs.UseDevice(true);
      e_rhs.UseDevice(true);
   }

   GridFunctionCoefficient rho_coeff_gf(&rho0);

   // Standard local assembly and inversion for energy mass matrices.
   if (!p_assembly)
   {
      // 'Me' is used in the computation of the internal energy
      // which is used twice: once at the start and once at the end of the run.
      MassIntegrator mi(rho_coeff_gf, &integ_rule);
      for (int i = 0; i < nzones; i++)
      {
         DenseMatrixInverse inv(&Me(i));
         mi.AssembleElementMatrix(*l2_fes.GetFE(i),
                                  *l2_fes.GetElementTransformation(i), Me(i));
         inv.Factor();
         inv.GetInverseMatrix(Me_inv(i));
      }
   }

   // Standard assembly for the velocity mass matrix.
   if (!p_assembly)
   {
      push("Me_inv",LightSkyBlue);
      VectorMassIntegrator *vmi = new VectorMassIntegrator(rho_coeff_gf, &integ_rule);
      Mv.AddDomainIntegrator(vmi);
      Mv.Assemble();
      Mv_spmat_copy = Mv.SpMat();
      pop();
   }

   push("rho0DetJ0 / Jac0inv / Volume",MistyRose);
   // Values of rho0DetJ0 and Jac0inv at all quadrature points.
   // Initial local mesh size (assumes all mesh elements are of the same type).
   double loc_area = 0.0, glob_area;
   int loc_z_cnt = nzones, glob_z_cnt;
   ParMesh *pm = H1FESpace.GetParMesh();
   if (!p_assembly)
   {
      const int nqp = integ_rule.GetNPoints();
      Vector rho_vals(nqp);
      for (int i = 0; i < nzones; i++)
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
            quad_data.rho0DetJ0w(i*nqp + q) = rho0DetJ0 *
                                              integ_rule.IntPoint(q).weight;
         }
      }
      for (int i = 0; i < nzones; i++) { loc_area += pm->GetElementVolume(i); }
   }
   else
   {
      ComputeRho0DetJ0AndVolume(dim, nzones, integ_rule,
                                H1FESpace.GetParMesh(),
                                l2_fes,
                                rho0,
                                quad_data,
                                loc_area);
   }
   pop();

   push("MPI_Allreduce",Orange);
   MPI_Allreduce(&loc_area, &glob_area, 1, MPI_DOUBLE, MPI_SUM, pm->GetComm());
   dbg("glob_area=%.15e",glob_area);
   MPI_Allreduce(&loc_z_cnt, &glob_z_cnt, 1, MPI_INT, MPI_SUM, pm->GetComm());
   pop();
   switch (pm->GetElementBaseGeometry(0))
   {
      case Geometry::SEGMENT:
         quad_data.h0 = glob_area / glob_z_cnt; break;
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
   pop();

   if (p_assembly)
   {
      // Setup the preconditioner of the velocity mass operator.
      Vector d;
      (dim == 2) ? VMassPA->ComputeDiagonal2D(d) : VMassPA->ComputeDiagonal3D(d);
      VMassPA_prec.SetDiagonal(d);
   }
   else
   {
      ForceIntegrator *fi = new ForceIntegrator(quad_data);
      fi->SetIntRule(&integ_rule);
      Force.AddDomainIntegrator(fi);
      // Make a dummy assembly to figure out the sparsity.
      Force.Assemble(0);
      Force.Finalize(0);
   }

   if (p_assembly)
   {
      if (not okina)
      {
         locCG.SetOperator(locEMassPA);
         locCG.iterative_mode = false;
         locCG.SetRelTol(1e-8);
         locCG.SetAbsTol(1e-8 * numeric_limits<double>::epsilon());
         locCG.SetMaxIter(200);
         locCG.SetPrintLevel(-1);
      }
      else
      {
         CG_VMass.SetPreconditioner(VMassPA_prec);
         CG_VMass.SetOperator(*VMassPA);
         CG_VMass.SetRelTol(cg_rel_tol);
         CG_VMass.SetAbsTol(0.0);
         CG_VMass.SetMaxIter(cg_max_iter);
         CG_VMass.SetPrintLevel(0);

         CG_EMass.SetOperator(*EMassPA);
         CG_EMass.iterative_mode = false;
         CG_EMass.SetRelTol(1e-8);
         CG_EMass.SetAbsTol(1e-8 * numeric_limits<double>::epsilon());
         CG_EMass.SetMaxIter(200);
         CG_EMass.SetPrintLevel(-1);
      }
   }
   pop();
}

LagrangianHydroOperator::~LagrangianHydroOperator()
{
   delete EMassPA;
   delete VMassPA;
   delete ForcePA;
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
   const int VsizeH1 = H1FESpace.GetVSize();
   v.MakeRef(&H1FESpace, *sptr, VsizeH1);

   // Set dx_dt = v (explicit).
   ParGridFunction dx;
   dx.MakeRef(&H1FESpace, dS_dt, 0);
   dx = v;

   SolveVelocity(S, dS_dt);
   SolveEnergy(S, v, dS_dt);

   quad_data_is_current = false;
}

void LagrangianHydroOperator::SolveVelocity(const Vector &S,
                                            Vector &dS_dt) const
{
   UpdateQuadratureData(S);

   AssembleForceMatrix();

   // The monolithic BlockVector stores the unknown fields as follows:
   // (Position, Velocity, Specific Internal Energy).
   ParGridFunction dv;
   dv.MakeRef(&H1FESpace, dS_dt, H1Vsize);
   dv = 0.0;

   if (p_assembly)
   {
      timer.sw_force.Start();
      ForcePA->Mult(one, rhs);
      timer.sw_force.Stop();
      rhs.Neg();

      if (not okina)
      {
         Operator *cVMassPA;
         VMassPA->FormLinearSystem(ess_tdofs, dv, rhs, cVMassPA, X, B);
         CGSolver cg(H1FESpace.GetParMesh()->GetComm());
         cg.SetPreconditioner(VMassPA_prec);
         cg.SetOperator(*cVMassPA);
         cg.SetRelTol(cg_rel_tol);
         cg.SetAbsTol(0.0);
         cg.SetMaxIter(cg_max_iter);
         cg.SetPrintLevel(-1);
         timer.sw_cgH1.Start();
         cg.Mult(B, X);
         timer.sw_cgH1.Stop();
         timer.H1iter += cg.GetNumIterations();
         VMassPA->RecoverFEMSolution(X, rhs, dv);
         delete cVMassPA;
      }
      else // okina
      {
         // Partial assembly solve for each velocity component
         const int size = H1compFESpace.GetVSize();
         const Operator *Pconf = H1compFESpace.GetProlongationMatrix();
         OkinaMassPAOperator *kVMassPA = static_cast<OkinaMassPAOperator*>(VMassPA);
         for (int c = 0; c < dim; c++)
         {
            dvc_gf.MakeRef(&H1compFESpace, dS_dt, H1Vsize + c*size);
            rhs_c_gf.MakeRef(&H1compFESpace, rhs, c*size);
            if (Pconf) { Pconf->MultTranspose(rhs_c_gf, B); }
            else { B = rhs_c_gf; }
            H1compFESpace.GetRestrictionMatrix()->Mult(dvc_gf, X);
            kVMassPA->SetEssentialTrueDofs(c_tdofs[c]);
            kVMassPA->EliminateRHS(B);
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
      } // okina
   }
   else
   {
      timer.sw_force.Start();
      Force.Mult(one, rhs);
      timer.sw_force.Stop();
      rhs.Neg();

      HypreParMatrix A;
      Mv.FormLinearSystem(ess_tdofs, dv, rhs, A, X, B);
      CGSolver cg(H1FESpace.GetParMesh()->GetComm());
      HypreSmoother prec;
      prec.SetType(HypreSmoother::Jacobi, 1);
      cg.SetPreconditioner(prec);
      cg.SetOperator(A);
      cg.SetRelTol(cg_rel_tol); cg.SetAbsTol(0.0);
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
   de.MakeRef(&L2FESpace, dS_dt, H1Vsize*2);
   de = 0.0;

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
   Array<int> l2dofs;
   if (p_assembly)
   {
      timer.sw_force.Start();
      ForcePA->MultTranspose(v, e_rhs);
      timer.sw_force.Stop();

      if (e_source) { e_rhs += *e_source; }
      if (not okina)
      {
         Vector loc_rhs(l2dofs_cnt), loc_de(l2dofs_cnt);
         for (int z = 0; z < nzones; z++)
         {
            L2FESpace.GetElementDofs(z, l2dofs);
            e_rhs.GetSubVector(l2dofs, loc_rhs);
            locEMassPA.SetZoneId(z);
            timer.sw_cgL2.Start();
            locCG.Mult(loc_rhs, loc_de);
            timer.sw_cgL2.Stop();
            const HYPRE_Int cg_num_iter = locCG.GetNumIterations();
            timer.L2iter += (cg_num_iter==0) ? 1 : cg_num_iter;
            de.SetSubVector(l2dofs, loc_de);
         }
      }
      else // okina
      {
         timer.sw_cgL2.Start();
         CG_EMass.Mult(e_rhs, de);
         timer.sw_cgL2.Stop();
         const HYPRE_Int cg_num_iter = CG_EMass.GetNumIterations();
         timer.L2iter += (cg_num_iter==0) ? 1 : cg_num_iter;
         // Move the memory location of the subvector 'de' to the memory
         // location of the base vector 'dS_dt'.
         de.GetMemory().SyncAlias(dS_dt.GetMemory(), de.Size());
      }
   }
   else // not p_assembly
   {
      timer.sw_force.Start();
      Force.MultTranspose(v, e_rhs);
      timer.sw_force.Stop();
      if (e_source) { e_rhs += *e_source; }
      Vector loc_rhs(l2dofs_cnt), loc_de(l2dofs_cnt);
      for (int z = 0; z < nzones; z++)
      {
         L2FESpace.GetElementDofs(z, l2dofs);
         e_rhs.GetSubVector(l2dofs, loc_rhs);
         timer.sw_cgL2.Start();
         Me_inv(z).Mult(loc_rhs, loc_de);
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
   x_gf.MakeRef(&H1FESpace, *sptr, 0);
   H1FESpace.GetParMesh()->NewNodes(x_gf, false);
}

double LagrangianHydroOperator::GetTimeStepEstimate(const Vector &S) const
{
   UpdateMesh(S);
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

void LagrangianHydroOperator::ComputeDensity(ParGridFunction &rho) const
{
   rho.SetSpace(&L2FESpace);

   DenseMatrix Mrho(l2dofs_cnt);
   Vector rhs(l2dofs_cnt), rho_z(l2dofs_cnt);
   Array<int> dofs(l2dofs_cnt);
   DenseMatrixInverse inv(&Mrho);
   MassIntegrator mi(&integ_rule);
   DensityIntegrator di(quad_data);
   di.SetIntRule(&integ_rule);
   for (int i = 0; i < nzones; i++)
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

double LagrangianHydroOperator::InternalEnergy(const ParGridFunction &e) const
{
   double glob_ie = 0.0;
   // This should be turned into a kernel so that it could be displayed in pa
   if (!p_assembly)
   {
      Vector one(l2dofs_cnt), loc_e(l2dofs_cnt);
      one = 1.0;
      Array<int> l2dofs;
      double loc_ie = 0.0;
      for (int z = 0; z < nzones; z++)
      {
         L2FESpace.GetElementDofs(z, l2dofs);
         e.GetSubVector(l2dofs, loc_e);
         loc_ie += Me(z).InnerProduct(loc_e, one);
      }

      MPI_Allreduce(&loc_ie, &glob_ie, 1, MPI_DOUBLE, MPI_SUM,
                    H1FESpace.GetParMesh()->GetComm());
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
                    H1FESpace.GetParMesh()->GetComm());
   }
   return glob_ke;
}

void LagrangianHydroOperator::PrintTimingData(bool IamRoot, int steps,
                                              const bool fom) const
{
   const MPI_Comm com = H1FESpace.GetComm();
   double my_rt[5], rt_max[5];
   my_rt[0] = timer.sw_cgH1.RealTime();
   my_rt[1] = timer.sw_cgL2.RealTime();
   my_rt[2] = timer.sw_force.RealTime();
   my_rt[3] = timer.sw_qdata.RealTime();
   my_rt[4] = my_rt[0] + my_rt[2] + my_rt[3];
   MPI_Reduce(my_rt, rt_max, 5, MPI_DOUBLE, MPI_MAX, 0, com);

   HYPRE_Int mydata[3], alldata[3];
   mydata[0] = timer.L2dof * timer.L2iter;
   mydata[1] = timer.quad_tstep;
   mydata[2] = nzones;
   MPI_Reduce(mydata, alldata, 3, HYPRE_MPI_INT, MPI_SUM, 0, com);

   if (IamRoot)
   {
      using namespace std;
      // FOM = (FOM1 * time1 + FOM2 * time2 + FOM3 * time3) / (time1 + time2 + time3)
      const HYPRE_Int H1iter = okina ? (timer.H1iter/dim) : timer.H1iter;
      const double FOM1 = 1e-6 * H1GTVSize * H1iter / rt_max[0];
      const double FOM2 = 1e-6 * steps * (H1GTVSize + L2GTVSize) / rt_max[2];
      const double FOM3 = 1e-6 * alldata[1] * integ_rule.GetNPoints() / rt_max[3];
      const double FOM = (FOM1 * rt_max[0] + FOM2 * rt_max[2] + FOM3 *rt_max[3]) /
                         rt_max[4];
      cout << endl;
      cout << "CG (H1) total time: " << rt_max[0] << endl;
      cout << "CG (H1) rate (megadofs x cg_iterations / second): "
           << FOM1 << endl;
      cout << endl;
      cout << "CG (L2) total time: " << rt_max[1] << endl;
      cout << "CG (L2) rate (megadofs x cg_iterations / second): "
           << 1e-6 * alldata[0] / rt_max[1] << endl;
      cout << endl;
      cout << "Forces total time: " << rt_max[2] << endl;
      cout << "Forces rate (megadofs x timesteps / second): "
           << FOM2 << endl;
      cout << endl;
      cout << "UpdateQuadData total time: " << rt_max[3] << endl;
      cout << "UpdateQuadData rate (megaquads x timesteps / second): "
           << FOM3 << endl;
      cout << endl;
      cout << "Major kernels total time (seconds): " << rt_max[4] << endl;
      cout << "Major kernels total rate (megadofs x time steps / second): "
           << FOM << endl;
      if (!fom) { return; }
      const int QPT = integ_rule.GetNPoints();
      const HYPRE_Int GNZones = alldata[2];
      const long ndofs = 2*H1GTVSize + L2GTVSize + QPT*GNZones;
      cout << endl;
      cout << "| Ranks " << "| Zones   "
           << "| H1 dofs " << "| L2 dofs "
           << "| QP "      << "| N dofs   "
           << "| FOM1    " << "| T1    "
           << "| FOM2   " << "| T2   "
           << "| FOM3   " << "| T3   "
           << "| FOM    " << "| TT   |"<< endl;
      cout << setprecision(3);
      cout << "| " << setw(6) << H1FESpace.GetNRanks()
           << "| " << setw(8) << GNZones
           << "| " << setw(8) << H1GTVSize
           << "| " << setw(8) << L2GTVSize
           << "| " << setw(3) << QPT
           << "| " << setw(9) << ndofs
           << "| " << setw(8) << FOM1
           << "| " << setw(6) << rt_max[0]
           << "| " << setw(7) << FOM2
           << "| " << setw(5) << rt_max[2]
           << "| " << setw(7) << FOM3
           << "| " << setw(5) << rt_max[3]
           << "| " << setw(7) << FOM
           << "| " << setw(5) << rt_max[4]
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
   if (qupdate)
   {
      return Q.UpdateQuadratureData(S,
                                    quad_data_is_current,
                                    quad_data,
                                    &tensors1D);
   }

   if (quad_data_is_current) { return; }
   timer.sw_qdata.Start();

   const int nqp = integ_rule.GetNPoints();

   ParGridFunction x, v, e;
   Vector* sptr = const_cast<Vector*>(&S);
   x.MakeRef(&H1FESpace, *sptr, 0);
   v.MakeRef(&H1FESpace, *sptr, H1FESpace.GetVSize());
   e.MakeRef(&L2FESpace, *sptr, 2*H1FESpace.GetVSize());
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
   const int nbatches =  nzones / nzones_batch + 1; // +1 for the remainder.
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
      if (z_id == nzones) { break; }
      else if (z_id + nzones_batch > nzones)
      {
         nzones_batch = nzones - z_id;
         nqp_batch    = nqp * nzones_batch;
      }

      double min_detJ = numeric_limits<double>::infinity();
      for (int z = 0; z < nzones_batch; z++)
      {
         ElementTransformation *T = H1FESpace.GetElementTransformation(z_id);
         Jpr_b[z].SetSize(dim, dim, nqp);

         if (p_assembly)
         {
            // Energy values at quadrature point.
            L2FESpace.GetElementDofs(z_id, L2dofs);
            e.GetSubVector(L2dofs, e_loc);
            evaluator.GetL2Values(e_loc, e_vals);

            // All reference->physical Jacobians at the quadrature points.
            H1FESpace.GetElementVDofs(z_id, H1dofs);
            x.GetSubVector(H1dofs, vector_vals);
            evaluator.GetVectorGrad(vecvalMat, Jpr_b[z]);
         }
         else { e.GetValues(z_id, integ_rule, e_vals); }
         for (int q = 0; q < nqp; q++)
         {
            const IntegrationPoint &ip = integ_rule.IntPoint(q);
            T->SetIntPoint(&ip);
            if (!p_assembly) { Jpr_b[z](q) = T->Jacobian(); }
            const double detJ = Jpr_b[z](q).Det();
            min_detJ = min(min_detJ, detJ);

            const int idx = z * nqp + q;
            if (material_pcf == NULL) { gamma_b[idx] = 5./3.; } // Ideal gas.
            else { gamma_b[idx] = material_pcf->Eval(*T, ip); }
            rho_b[idx] = quad_data.rho0DetJ0w(z_id*nqp + q) / detJ / ip.weight;
            e_b[idx]   = fmax(0.0, e_vals(q));
         }
         ++z_id;
      }

      // Batched computation of material properties.
      ComputeMaterialProperties(nqp_batch, gamma_b, rho_b, e_b, p_b, cs_b);

      z_id -= nzones_batch;
      for (int z = 0; z < nzones_batch; z++)
      {
         ElementTransformation *T = H1FESpace.GetElementTransformation(z_id);
         if (p_assembly)
         {
            // All reference->physical Jacobians at the quadrature points.
            H1FESpace.GetElementVDofs(z_id, H1dofs);
            v.GetSubVector(H1dofs, vector_vals);
            evaluator.GetVectorGrad(vecvalMat, grad_v_ref);
         }
         for (int q = 0; q < nqp; q++)
         {
            const IntegrationPoint &ip = integ_rule.IntPoint(q);
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
               if (p_assembly)
               {
                  mfem::Mult(grad_v_ref(q), Jinv, sgrad_v);
               }
               else
               {
                  v.GetVectorGradient(*T, sgrad_v);
               }
               sgrad_v.Symmetrize();
               double eig_val_data[3], eig_vec_data[9];
               if (dim==1)
               {
                  eig_val_data[0] = sgrad_v(0, 0);
                  eig_vec_data[0] = 1.;
               }
               else { sgrad_v.CalcEigenvalues(eig_val_data, eig_vec_data); }
               Vector compr_dir(eig_vec_data, dim);
               // Computes the initial->physical transformation Jacobian.
               mfem::Mult(Jpr, quad_data.Jac0inv(z_id*nqp + q), Jpi);
               Vector ph_dir(dim); Jpi.Mult(compr_dir, ph_dir);
               // Change of the initial mesh size in the compression direction.
               const double h = quad_data.h0 * ph_dir.Norml2() /
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
               Jpr.CalcSingularvalue(dim-1) / (double) H1FESpace.GetOrder(0);
            const double inv_dt = sound_speed / h_min +
                                  2.5 * visc_coeff / rho / h_min / h_min;
            if (min_detJ < 0.0)
            {
               // This will force repetition of the step with smaller dt.
               quad_data.dt_est = 0.0;
            }
            else
            {
               if (inv_dt>0.0)
               {
                  quad_data.dt_est = min(quad_data.dt_est, cfl*(1.0/inv_dt));
               }
            }
            // Quadrature data for partial assembly of the force operator.
            MultABt(stress, Jinv, stressJiT);
            stressJiT *= integ_rule.IntPoint(q).weight * detJ;
            for (int vd = 0 ; vd < dim; vd++)
            {
               for (int gd = 0; gd < dim; gd++)
               {
                  quad_data.stressJinvT(vd)(z_id*nqp + q, gd) =
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
   quad_data_is_current = true;
   forcemat_is_assembled = false;

   timer.sw_qdata.Stop();
   timer.quad_tstep += nzones;
}

// *****************************************************************************
template<int dim> MFEM_HOST_DEVICE static inline
void QBody(const int nzones, const int z,
           const int nqp, const int q,
           const double gamma,
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
           const double* __restrict__ d_weights,
           const double* __restrict__ d_Jacobians,
           const double* __restrict__ d_rho0DetJ0w,
           const double* __restrict__ d_e_quads,
           const double* __restrict__ d_grad_v_ext,
           const double* __restrict__ d_Jac0inv,
           double *d_dt_est,
           double *d_stressJinvT)
{
   constexpr int dim2 = dim*dim;
   double min_detJ = infinity;

   const int zq = z * nqp + q;
   const double weight =  d_weights[q];
   const double inv_weight = 1. / weight;
   const double *J = d_Jacobians + dim2*(nqp*z + q);
   const double detJ = mfem::det<dim>(J);
   min_detJ = std::fmin(min_detJ,detJ);
   mfem::calcInverse<dim>(J,Jinv);
   // *****************************************************************
   const double rho = inv_weight * d_rho0DetJ0w[zq] / detJ;
   const double e   = std::fmax(0.0, d_e_quads[zq]);
   const double p   = (gamma - 1.0) * rho * e;
   const double sound_speed = std::sqrt(gamma * (gamma-1.0) * e);
   // *****************************************************************
   for (int k = 0; k < dim2; k+=1) { stress[k] = 0.0; }
   for (int d = 0; d < dim; d++) { stress[d*dim+d] = -p; }
   // *****************************************************************
   double visc_coeff = 0.0;
   if (use_viscosity)
   {
      // Compression-based length scale at the point. The first
      // eigenvector of the symmetric velocity gradient gives the
      // direction of maximal compression. This is used to define the
      // relative change of the initial length scale.
      const double *dV = d_grad_v_ext + dim2*(nqp*z + q);
      mfem::mult(dim, dim, dim, dV, Jinv, sgrad_v);
      DenseMatrix::symmetrize(dim,sgrad_v);
      if (dim==1)
      {
         eig_val_data[0] = sgrad_v[0];
         eig_vec_data[0] = 1.;
      }
      else
      {
         mfem::calcEigenvalues<dim>(sgrad_v, eig_val_data, eig_vec_data);
      }
      for (int k=0; k<dim; k+=1) { compr_dir[k]=eig_vec_data[k]; }
      // Computes the initial->physical transformation Jacobian.
      mfem::mult(dim,dim,dim, J, d_Jac0inv+zq*dim*dim, Jpi);
      mfem::multV(dim, dim, Jpi, compr_dir, ph_dir);
      // Change of the initial mesh size in the compression direction.
      const double ph_dir_nl2 = Vector::norml2(dim,ph_dir);
      const double compr_dir_nl2 = Vector::norml2(dim, compr_dir);
      const double h = h0 * ph_dir_nl2 / compr_dir_nl2;
      // Measure of maximal compression.
      const double mu = eig_val_data[0];
      visc_coeff = 2.0 * rho * h * h * std::fabs(mu);
      // The following represents a "smooth" version of the statement
      // "if (mu < 0) visc_coeff += 0.5 rho h sound_speed".  Note that
      // eps must be scaled appropriately if a different unit system is
      // being used.
      const double eps = 1e-12;
      visc_coeff += 0.5 * rho * h * sound_speed *
                    (1.0 - smooth_step_01(mu - 2.0 * eps, eps));
      //add(dim, dim, visc_coeff, sgrad_v, stress);
      mfem::add(dim, dim, visc_coeff, stress, sgrad_v, stress);
   }
   // Time step estimate at the point. Here the more relevant length
   // scale is related to the actual mesh deformation; we use the min
   // singular value of the ref->physical Jacobian. In addition, the
   // time step estimate should be aware of the presence of shocks.
   const double sv = mfem::calcSingularvalue<dim>(J);
   const double h_min = sv / h1order;
   const double inv_h_min = 1. / h_min;
   const double inv_rho_inv_h_min_sq = inv_h_min * inv_h_min / rho ;
   const double inv_dt = sound_speed * inv_h_min
                         + 2.5 * visc_coeff * inv_rho_inv_h_min_sq;
   if (min_detJ < 0.0)
   {
      // This will force repetition of the step with smaller dt.
      d_dt_est[zq] = 0.0;
   }
   else
   {
      if (inv_dt>0.0)
      {
         const double cfl_inv_dt = cfl / inv_dt;
         d_dt_est[zq] = std::fmin(d_dt_est[zq], cfl_inv_dt);
      }
   }
   // Quadrature data for partial assembly of the force operator.
   mfem::multABt(dim, dim, dim, stress, Jinv, stressJiT);
   for (int k=0; k<dim2; k+=1) { stressJiT[k] *= weight * detJ; }
   for (int vd = 0 ; vd < dim; vd++)
   {
      for (int gd = 0; gd < dim; gd++)
      {
         const int offset = zq + nqp*nzones*(gd+vd*dim);
         d_stressJinvT[offset] = stressJiT[vd+gd*dim];
      }
   }
}

// *****************************************************************************
template<int dim, int Q1D> static inline
void QKernel(const int nzones,
             const int nqp,
             const int nqp1D,
             const double gamma,
             const bool use_viscosity,
             const double h0,
             const double h1order,
             const double cfl,
             const double infinity,
             const Array<double> &weights,
             const Vector &Jacobians,
             const Vector &rho0DetJ0w,
             const Vector &e_quads,
             const Vector &grad_v_ext,
             const DenseTensor &Jac0inv,
             Vector &dt_est,
             DenseTensor &stressJinvT)
{
   constexpr int dim2 = dim*dim;
   auto d_weights = weights.Read();
   auto d_Jacobians = Jacobians.Read();
   auto d_rho0DetJ0w = rho0DetJ0w.Read();
   auto d_e_quads = e_quads.Read();
   auto d_grad_v_ext = grad_v_ext.Read();
   auto d_Jac0inv = Read(Jac0inv.GetMemory(), Jac0inv.TotalSize());
   auto d_dt_est = dt_est.ReadWrite();
   auto d_stressJinvT = Write(stressJinvT.GetMemory(),
                              stressJinvT.TotalSize());
   if (dim==2)
   {
      MFEM_FORALL_2D(z, nzones, Q1D, Q1D, 1,
      {
         double Jinv[dim2];
         double stress[dim2];
         double sgrad_v[dim2];
         double eig_val_data[3];
         double eig_vec_data[9];
         double compr_dir[dim];
         double Jpi[dim2];
         double ph_dir[dim];
         double stressJiT[dim2];
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               QBody<dim>(nzones, z, nqp, qx + qy * Q1D,
               gamma, use_viscosity, h0, h1order, cfl, infinity,
               Jinv,stress,sgrad_v,eig_val_data,eig_vec_data,
               compr_dir,Jpi,ph_dir,stressJiT,
               d_weights, d_Jacobians, d_rho0DetJ0w,
               d_e_quads, d_grad_v_ext, d_Jac0inv,
               d_dt_est, d_stressJinvT);
            }
         }
         MFEM_SYNC_THREAD;
      });
   }
   if (dim==3)
   {
      MFEM_FORALL_3D(z, nzones, Q1D, Q1D, Q1D,
      {
         double Jinv[dim2];
         double stress[dim2];
         double sgrad_v[dim2];
         double eig_val_data[3];
         double eig_vec_data[9];
         double compr_dir[dim];
         double Jpi[dim2];
         double ph_dir[dim];
         double stressJiT[dim2];
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qz,z,Q1D)
               {
                  QBody<dim>(nzones, z, nqp, qx + Q1D * (qy + qz * Q1D),
                  gamma, use_viscosity, h0, h1order, cfl, infinity,
                  Jinv,stress,sgrad_v,eig_val_data,eig_vec_data,
                  compr_dir,Jpi,ph_dir,stressJiT,
                  d_weights, d_Jacobians, d_rho0DetJ0w,
                  d_e_quads, d_grad_v_ext, d_Jac0inv,
                  d_dt_est, d_stressJinvT);
               }
            }
         }
         MFEM_SYNC_THREAD;
      });
   }
}

void QUpdate::UpdateQuadratureData(const Vector &S,
                                   bool &quad_data_is_current,
                                   QuadratureData &quad_data,
                                   const Tensors1D *tensors1D)
{
   if (quad_data_is_current) { return; }
   timer->sw_qdata.Start();
   Vector* S_p = const_cast<Vector*>(&S);
   const int H1_size = H1.GetVSize();
   const int nqp1D = tensors1D->LQshape1D.Width();
   const double h1order = (double) H1.GetOrder(0);
   const double infinity = std::numeric_limits<double>::infinity();
   ParGridFunction d_x, d_v, d_e;
   d_x.MakeRef(&H1,*S_p, 0);
   H1ER->Mult(d_x, d_h1_v_local_in);
   q1->Derivatives(d_h1_v_local_in, d_h1_grad_x_data);
   d_v.MakeRef(&H1,*S_p, H1_size);
   H1ER->Mult(d_v, d_h1_v_local_in);
   q1->Derivatives(d_h1_v_local_in, d_h1_grad_v_data);
   d_e.MakeRef(&L2, *S_p, 2*H1_size);
   q2->Values(d_e, d_l2_e_quads_data);
   d_dt_est = quad_data.dt_est;
   const int id = (dim<<4) | nqp1D;
   typedef void (*fQKernel)(const int NE, const int NQ, const int Q1D,
                            const double gamma, const bool use_viscosity,
                            const double h0, const double h1order,
                            const double cfl, const double infinity,
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
   qupdate[id](NE, NQ, nqp1D, gamma, use_viscosity, quad_data.h0,
               h1order, cfl, infinity, ir.GetWeights(), d_h1_grad_x_data,
               quad_data.rho0DetJ0w, d_l2_e_quads_data, d_h1_grad_v_data,
               quad_data.Jac0inv, d_dt_est, quad_data.stressJinvT);
   quad_data.dt_est = d_dt_est.Min();
   quad_data_is_current = true;
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

} // namespace mfem

#endif // MFEM_USE_MPI
