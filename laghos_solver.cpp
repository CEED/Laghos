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
                            ? "mAcRjl" : "mmaaAcl";

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
                                                 const int oq) :
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
   Me(l2dofs_cnt, l2dofs_cnt, NE),
   Me_inv(l2dofs_cnt, l2dofs_cnt, NE),
   ir(IntRules.Get(pmesh->GetElementBaseGeometry(0),
                   (oq > 0) ? oq : 3 * H1.GetOrder(0) + L2.GetOrder(0) - 1)),
   Q1D(int(floor(0.7 + pow(ir.GetNPoints(), 1.0 / dim)))),
   qdata(dim, NE, ir.GetNPoints()),
   qdata_is_current(false),
   forcemat_is_assembled(false),
   Force(&L2, &H1),
   X(H1c.GetTrueVSize()),
   B(H1c.GetTrueVSize()),
   one(L2Vsize),
   rhs(H1Vsize),
   e_rhs(L2Vsize)
{
   block_offsets[0] = 0;
   block_offsets[1] = block_offsets[0] + H1Vsize;
   block_offsets[2] = block_offsets[1] + H1Vsize;
   block_offsets[3] = block_offsets[2] + L2Vsize;
   one = 1.0;

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

   // Values of rho0DetJ0 and Jac0inv at all quadrature points.
   // Initial local mesh size (assumes all mesh elements are the same).
   int Ne, ne = NE;
   double Volume, vol = 0.0;

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

   ForceIntegrator *fi = new ForceIntegrator(qdata);
   fi->SetIntRule(&ir);
   Force.AddDomainIntegrator(fi);
   // Make a dummy assembly to figure out the sparsity.
   Force.Assemble(0);
   Force.Finalize(0);
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
      accel_src_gf.Read();
   }

   Force.Mult(one, rhs);
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
   Force.MultTranspose(v, e_rhs);
   if (e_source) { e_rhs += *e_source; }
   Vector loc_rhs(l2dofs_cnt), loc_de(l2dofs_cnt);
   for (int e = 0; e < NE; e++)
   {
      L2.GetElementDofs(e, l2dofs);
      e_rhs.GetSubVector(l2dofs, loc_rhs);
      Me_inv(e).Mult(loc_rhs, loc_de);
      de.SetSubVector(l2dofs, loc_de);
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

double ComputeVolumeIntegral(const int DIM, const int NE,const int NQ,
                             const int Q1D,const int VDIM,const double ln_norm,
                             const mfem::Vector& mass, const mfem::Vector& f)
{

   auto f_vals = mfem::Reshape(f.Read(),VDIM,NQ, NE);
   mfem::Vector integrand(NE*NQ);
   auto I = Reshape(integrand.Write(), NQ, NE);

   if (DIM == 1)
   {
      for (int e=0; e < NE; ++e)
      {
         for (int q = 0; q < NQ; ++q)
         {
            double vmag = 0;
            for (int k = 0; k < VDIM; k++)
            {
               vmag += pow(f_vals(k,q,e),ln_norm);
            }
            I(q,e) = vmag;
         }
      }
   }
   else if (DIM == 2)
   {
      MFEM_FORALL_2D(e, NE, Q1D, Q1D, 1,
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               const int q = qx + qy * Q1D;
               double vmag = 0;
               for (int k = 0; k < VDIM; k++)
               {
                  vmag += pow(f_vals(k,q,e),ln_norm);
               }
               I(q,e) = vmag;
            }
         }
      });
   }
   else if (DIM == 3)
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
                  double vmag = 0;
                  for (int k = 0; k < VDIM; k++)
                  {
                     vmag += pow(f_vals(k,q,e),ln_norm);
                  }
                  I(q,e) = vmag;
               }
            }
         }
      });

   }
   const double integral = integrand * mass;
   return integral;

}
double LagrangianHydroOperator::InternalEnergy(const ParGridFunction &gf) const
{
   double glob_ie = 0.0;

   // get the restriction and interpolator objects
   const QuadratureInterpolator* l2_interpolator = L2.GetQuadratureInterpolator(
                                                      ir);
   l2_interpolator->SetOutputLayout(QVectorLayout::byVDIM);
   auto L2r = L2.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   const int NQ = ir.GetNPoints();
   const int ND = L2.GetFE(0)->GetDof();
   Vector e_vector(NE*ND), eintQ(NE*NQ);

   // Get internal energy at the quadrature points
   L2r->Mult(gf, e_vector);
   l2_interpolator->Values(e_vector, eintQ);

   double internal_energy = ComputeVolumeIntegral(dim,NE,NQ,Q1D,1,1.0,
                                                  qdata.rho0DetJ0w,eintQ);

   MPI_Allreduce(&internal_energy, &glob_ie, 1, MPI_DOUBLE, MPI_SUM,
                 L2.GetParMesh()->GetComm());

   return glob_ie;
}

double LagrangianHydroOperator::KineticEnergy(const ParGridFunction &v) const
{
   double glob_ke = 0.0;

   // get the restriction and interpolator objects
   const QuadratureInterpolator* h1_interpolator = H1.GetQuadratureInterpolator(
                                                      ir);
   h1_interpolator->SetOutputLayout(QVectorLayout::byVDIM);
   auto H1r = H1.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   const int NQ = ir.GetNPoints();
   const int ND = H1.GetFE(0)->GetDof();
   Vector e_vector(dim*NE*ND), ekinQ(dim*NE*NQ);

   // Get internal energy at the quadrature points
   H1r->Mult(v, e_vector);
   h1_interpolator->Values(e_vector, ekinQ);

   // Get the IE, initial weighted mass

   double kinetic_energy = ComputeVolumeIntegral(dim,NE,NQ,Q1D,dim,2.0,
                                                 qdata.rho0DetJ0w,ekinQ);

   MPI_Allreduce(&kinetic_energy, &glob_ke, 1, MPI_DOUBLE, MPI_SUM,
                 H1.GetParMesh()->GetComm());

   return 0.5*glob_ke;
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
   ParGridFunction x, v, e;
   Vector* sptr = const_cast<Vector*>(&S);
   x.MakeRef(&H1, *sptr, 0);
   v.MakeRef(&H1, *sptr, H1.GetVSize());
   e.MakeRef(&L2, *sptr, 2*H1.GetVSize());
   Vector e_vals;
   DenseMatrix Jpi(dim), sgrad_v(dim), Jinv(dim), stress(dim), stressJiT(dim);

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
            // Assuming piecewise constant gamma that moves with the mesh.
            gamma_b[idx] = gamma_gf(z_id);
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

               double vorticity_coeff = 1.0;
               if (use_vorticity)
               {
                  const double grad_norm = sgrad_v.FNorm();
                  const double div_v = fabs(sgrad_v.Trace());
                  vorticity_coeff = (grad_norm > 0.0) ? div_v / grad_norm : 1.0;
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
}

void LagrangianHydroOperator::AssembleForceMatrix() const
{
   Force = 0.0;
   Force.Assemble();
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
