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

#include "laghos_solver.hpp"

using namespace std;

namespace mfem
{

namespace hydrodynamics
{

void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    GridFunction &gf, const char *title,
                    int x, int y, int w, int h, bool vec)
{
   Mesh &mesh = *gf.FESpace()->GetMesh();
   bool newly_opened = false;
   int connection_failed;

   do
   {
      if (!sock.is_open() || !sock)
      {
         sock.open(vishost, visport);
         sock.precision(8);
         newly_opened = true;
      }
      sock << "solution\n";

      mesh.Print(sock);
      gf.Save(sock);

      if (newly_opened)
      {
         sock << "window_title '" << title << "'\n"
              << "window_geometry "
              << x << " " << y << " " << w << " " << h << "\n"
              << "keys maaAc";
         if ( vec ) { sock << "vvv"; }
         sock << endl;
      }

      connection_failed = !sock && !newly_opened;
   }
   while (connection_failed);
}

LagrangianHydroOperator::LagrangianHydroOperator(int size,
                                                 FiniteElementSpace &h1_fes,
                                                 FiniteElementSpace &l2_fes,
                                                 Array<int> &essential_tdofs,
                                                 GridFunction &rho0,
                                                 int source_type_, double cfl_,
                                                 Coefficient *material_,
                                                 bool visc, bool pa,
                                                 double cgt, int cgiter,
                                                 int h1_basis_type)
   : TimeDependentOperator(size),
     H1FESpace(h1_fes), L2FESpace(l2_fes),
     ess_tdofs(essential_tdofs),
     dim(h1_fes.GetMesh()->Dimension()),
     nzones(h1_fes.GetMesh()->GetNE()),
     l2dofs_cnt(l2_fes.GetFE(0)->GetDof()),
     h1dofs_cnt(h1_fes.GetFE(0)->GetDof()),
     source_type(source_type_), cfl(cfl_),
     use_viscosity(visc), p_assembly(pa), cg_rel_tol(cgt), cg_max_iter(cgiter),
     material_pcf(material_),
     Mv(&h1_fes), Mv_spmat_copy(),
     Me(l2dofs_cnt, l2dofs_cnt, nzones), Me_inv(l2dofs_cnt, l2dofs_cnt, nzones),
     integ_rule(IntRules.Get(h1_fes.GetMesh()->GetElementBaseGeometry(0),
                             3*h1_fes.GetOrder(0) + l2_fes.GetOrder(0) - 1)),
     quad_data(dim, nzones, integ_rule.GetNPoints()),
     quad_data_is_current(false), forcemat_is_assembled(false),
     tensors1D(H1FESpace.GetFE(0)->GetOrder(), L2FESpace.GetFE(0)->GetOrder(),
               int(floor(0.7 + pow(integ_rule.GetNPoints(), 1.0 / dim))),
               h1_basis_type == BasisType::Positive),
     evaluator(H1FESpace, &tensors1D),
     Force(&l2_fes, &h1_fes), ForcePA(&quad_data, h1_fes, l2_fes, &tensors1D),
     VMassPA(&quad_data, H1FESpace, &tensors1D), VMassPA_prec(H1FESpace),
     locEMassPA(&quad_data, l2_fes, &tensors1D),
     locCG(), timer()
{
   GridFunctionCoefficient rho_coeff(&rho0);

   // Standard local assembly and inversion for energy mass matrices.
   MassIntegrator mi(rho_coeff, &integ_rule);
   for (int i = 0; i < nzones; i++)
   {
      DenseMatrixInverse inv(&Me(i));
      mi.AssembleElementMatrix(*l2_fes.GetFE(i),
                               *l2_fes.GetElementTransformation(i), Me(i));
      inv.Factor();
      inv.GetInverseMatrix(Me_inv(i));
   }

   // Standard assembly for the velocity mass matrix.
   VectorMassIntegrator *vmi = new VectorMassIntegrator(rho_coeff, &integ_rule);
   Mv.AddDomainIntegrator(vmi);
   Mv.Assemble();
   Mv_spmat_copy = Mv.SpMat();

   // Values of rho0DetJ0 and Jac0inv at all quadrature points.
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

   // Initial local mesh size (assumes all mesh elements are of the same type).
   double area = 0.0;
   Mesh *m = H1FESpace.GetMesh();
   for (int i = 0; i < nzones; i++) { area += m->GetElementVolume(i); }
   switch (m->GetElementBaseGeometry(0))
   {
      case Geometry::SEGMENT:
         quad_data.h0 = area / nzones; break;
      case Geometry::SQUARE:
         quad_data.h0 = sqrt(area / nzones); break;
      case Geometry::TRIANGLE:
         quad_data.h0 = sqrt(2.0 * area / nzones); break;
      case Geometry::CUBE:
         quad_data.h0 = pow(area / nzones, 1.0/3.0); break;
      case Geometry::TETRAHEDRON:
         quad_data.h0 = pow(6.0 * area / nzones, 1.0/3.0); break;
      default: MFEM_ABORT("Unknown zone type!");
   }
   quad_data.h0 /= (double) H1FESpace.GetOrder(0);

   ForceIntegrator *fi = new ForceIntegrator(quad_data);
   fi->SetIntRule(&integ_rule);
   Force.AddDomainIntegrator(fi);
   // Make a dummy assembly to figure out the sparsity.
   Force.Assemble(0);
   Force.Finalize(0);

   if (p_assembly)
   {
      // Setup the preconditioner of the velocity mass operator.
      Vector d;
      (dim == 2) ? VMassPA.ComputeDiagonal2D(d) : VMassPA.ComputeDiagonal3D(d);
      VMassPA_prec.SetDiagonal(d);
   }

   locCG.SetOperator(locEMassPA);
   locCG.iterative_mode = false;
   locCG.SetRelTol(1e-8);
   locCG.SetAbsTol(1e-8 * numeric_limits<double>::epsilon());
   locCG.SetMaxIter(200);
   locCG.SetPrintLevel(0);
}

void LagrangianHydroOperator::Mult(const Vector &S, Vector &dS_dt) const
{
   // Make sure that the mesh positions correspond to the ones in S. This is
   // needed only because some mfem time integrators don't update the solution
   // vector at every intermediate stage (hence they don't change the mesh).
   UpdateMesh(S);

   // The monolithic BlockVector stores the unknown fields as follows:
   // (Position, Velocity, Specific Internal Energy).
   Vector* sptr = (Vector*) &S;
   GridFunction v;
   const int VsizeH1 = H1FESpace.GetVSize();
   v.MakeRef(&H1FESpace, *sptr, VsizeH1);

   // Set dx_dt = v (explicit).
   GridFunction dx;
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

   const int VsizeL2 = L2FESpace.GetVSize();
   const int VsizeH1 = H1FESpace.GetVSize();

   // The monolithic BlockVector stores the unknown fields as follows:
   // (Position, Velocity, Specific Internal Energy).
   GridFunction dv;
   dv.MakeRef(&H1FESpace, dS_dt, VsizeH1);
   dv = 0.0;

   Vector one(VsizeL2), rhs(VsizeH1), B, X; one = 1.0;
   if (p_assembly)
   {
      timer.sw_force.Start();
      ForcePA.Mult(one, rhs);
      timer.sw_force.Stop();
      rhs.Neg();

      Operator *cVMassPA;
      VMassPA.FormLinearSystem(ess_tdofs, dv, rhs, cVMassPA, X, B);
      CGSolver cg;
      cg.SetPreconditioner(VMassPA_prec);
      cg.SetOperator(*cVMassPA);
      cg.SetRelTol(cg_rel_tol); cg.SetAbsTol(0.0);
      cg.SetMaxIter(cg_max_iter);
      cg.SetPrintLevel(0);
      timer.sw_cgH1.Start();
      cg.Mult(B, X);
      timer.sw_cgH1.Stop();
      timer.H1cg_iter += cg.GetNumIterations();
      VMassPA.RecoverFEMSolution(X, rhs, dv);
      delete cVMassPA;
   }
   else
   {
      timer.sw_force.Start();
      Force.Mult(one, rhs);
      timer.sw_force.Stop();
      rhs.Neg();

      SparseMatrix A;
      Mv.FormLinearSystem(ess_tdofs, dv, rhs, A, X, B);
      CGSolver cg;
      DSmoother prec(0);
      cg.SetPreconditioner(prec);
      cg.SetOperator(A);
      cg.SetRelTol(cg_rel_tol); cg.SetAbsTol(0.0);
      cg.SetMaxIter(cg_max_iter);
      cg.SetPrintLevel(0);
      timer.sw_cgH1.Start();
      cg.Mult(B, X);
      timer.sw_cgH1.Stop();
      timer.H1cg_iter += cg.GetNumIterations();
      Mv.RecoverFEMSolution(X, rhs, dv);
   }
}

void LagrangianHydroOperator::SolveEnergy(const Vector &S, const Vector &v,
                                          Vector &dS_dt) const
{
   UpdateQuadratureData(S);
   AssembleForceMatrix();

   const int VsizeL2 = L2FESpace.GetVSize();
   const int VsizeH1 = H1FESpace.GetVSize();

   // The monolithic BlockVector stores the unknown fields as follows:
   // (Position, Velocity, Specific Internal Energy).
   GridFunction de;
   de.MakeRef(&L2FESpace, dS_dt, VsizeH1*2);
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
   Vector e_rhs(VsizeL2), loc_rhs(l2dofs_cnt), loc_de(l2dofs_cnt);
   if (p_assembly)
   {
      timer.sw_force.Start();
      ForcePA.MultTranspose(v, e_rhs);
      timer.sw_force.Stop();

      if (e_source) { e_rhs += *e_source; }
      for (int z = 0; z < nzones; z++)
      {
         L2FESpace.GetElementDofs(z, l2dofs);
         e_rhs.GetSubVector(l2dofs, loc_rhs);
         locEMassPA.SetZoneId(z);
         timer.sw_cgL2.Start();
         locCG.Mult(loc_rhs, loc_de);
         timer.sw_cgL2.Stop();
         timer.L2dof_iter += locCG.GetNumIterations() * l2dofs_cnt;
         de.SetSubVector(l2dofs, loc_de);
      }
   }
   else
   {
      timer.sw_force.Start();
      Force.MultTranspose(v, e_rhs);
      timer.sw_force.Stop();
      if (e_source) { e_rhs += *e_source; }
      for (int z = 0; z < nzones; z++)
      {
         L2FESpace.GetElementDofs(z, l2dofs);
         e_rhs.GetSubVector(l2dofs, loc_rhs);
         timer.sw_cgL2.Start();
         Me_inv(z).Mult(loc_rhs, loc_de);
         timer.sw_cgL2.Stop();
         timer.L2dof_iter += l2dofs_cnt;
         de.SetSubVector(l2dofs, loc_de);
      }
   }
   delete e_source;
}

void LagrangianHydroOperator::UpdateMesh(const Vector &S) const
{
   Vector* sptr = (Vector*) &S;
   x_gf.MakeRef(&H1FESpace, *sptr, 0);
   H1FESpace.GetMesh()->NewNodes(x_gf, false);
}

double LagrangianHydroOperator::GetTimeStepEstimate(const Vector &S) const
{
   UpdateMesh(S);
   UpdateQuadratureData(S);

   return quad_data.dt_est;
}

void LagrangianHydroOperator::ResetTimeStepEstimate() const
{
   quad_data.dt_est = numeric_limits<double>::infinity();
}

void LagrangianHydroOperator::ComputeDensity(GridFunction &rho)
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

double LagrangianHydroOperator::InternalEnergy(const GridFunction &e) const
{
   Vector one(l2dofs_cnt), loc_e(l2dofs_cnt);
   one = 1.0;
   Array<int> l2dofs;

   double ie = 0.0;
   for (int z = 0; z < nzones; z++)
   {
      L2FESpace.GetElementDofs(z, l2dofs);
      e.GetSubVector(l2dofs, loc_e);
      ie += Me(z).InnerProduct(loc_e, one);
   }

   return ie;
}

double LagrangianHydroOperator::KineticEnergy(const GridFunction &v) const
{
   return 0.5 * Mv_spmat_copy.InnerProduct(v, v);
}

void LagrangianHydroOperator::PrintTimingData(int steps) const
{
   double runtime[5];
   runtime[0] = timer.sw_cgH1.RealTime();
   runtime[1] = timer.sw_cgL2.RealTime();
   runtime[2] = timer.sw_force.RealTime();
   runtime[3] = timer.sw_qdata.RealTime();
   runtime[4] = runtime[0] + runtime[2] + runtime[3];

   int data[2];
   data[0] = timer.L2dof_iter;
   data[1] = timer.quad_tstep;

   const int H1size = H1FESpace.GetVSize(),
             L2size = L2FESpace.GetVSize();
   using namespace std;
   cout << endl;
   cout << "CG (H1) total time: " << runtime[0] << endl;
   cout << "CG (H1) rate (megadofs x cg_iterations / second): "
        << 1e-6 * H1size * timer.H1cg_iter / runtime[0] << endl;
   cout << endl;
   cout << "CG (L2) total time: " << runtime[1] << endl;
   cout << "CG (L2) rate (megadofs x cg_iterations / second): "
        << 1e-6 * data[0] / runtime[1] << endl;
   cout << endl;
   // The Force operator is applied twice per time step, on the H1 and the L2
   // vectors, respectively.
   cout << "Forces total time: " << runtime[2] << endl;
   cout << "Forces rate (megadofs x timesteps / second): "
        << 1e-6 * steps * (H1size + L2size) / runtime[2] << endl;
   cout << endl;
   cout << "UpdateQuadData total time: " << runtime[3] << endl;
   cout << "UpdateQuadData rate (megaquads x timesteps / second): "
        << 1e-6 * data[1] * integ_rule.GetNPoints() / runtime[3] << endl;
   cout << endl;
   cout << "Major kernels total time (seconds): " << runtime[4] << endl;
   cout << "Major kernels total rate (megadofs x time steps / second): "
        << 1e-6 * steps * (H1size + L2size) / runtime[4] << endl;
}

// Smooth transition between 0 and 1 for x in [-eps, eps].
inline double smooth_step_01(double x, double eps)
{
   const double y = (x + eps) / (2.0 * eps);
   if (y < 0.0) { return 0.0; }
   if (y > 1.0) { return 1.0; }
   return (3.0 - 2.0 * y) * y * y;
}

void LagrangianHydroOperator::UpdateQuadratureData(const Vector &S) const
{
   if (quad_data_is_current) { return; }
   timer.sw_qdata.Start();

   const int nqp = integ_rule.GetNPoints();

   GridFunction x, v, e;
   Vector* sptr = (Vector*) &S;
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
   DenseTensor *Jpr_b = new DenseTensor[nqp_batch];
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
            e_b[idx]   = max(0.0, e_vals(q));
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

               // Note that the (mu < 0.0) check introduces discontinuous
               // behavior. This can lead to bigger differences in results when
               // there are round-offs around the zero for the min eigenvalue.
               // We've observed differences between Linux and Mac for some 3D
               // calculations.
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
               quad_data.dt_est = min(quad_data.dt_est, cfl * (1.0 / inv_dt) );
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
