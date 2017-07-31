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

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{

namespace miniapps
{

void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    ParGridFunction &gf, const char *title,
                    int x, int y, int w, int h, bool vec)
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
         sock << "window_title '" << title << "'\n"
              << "window_geometry "
              << x << " " << y << " " << w << " " << h << "\n"
              << "keys maaAc";
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

} // namespace miniapps

namespace hydrodynamics
{
LagrangianHydroOperator::LagrangianHydroOperator(Problem problem_,
                                                 OccaFiniteElementSpace &o_H1FESpace_,
                                                 OccaFiniteElementSpace &o_L2FESpace_,
                                                 Array<int> &ess_tdofs_,
                                                 ParGridFunction &rho0,
                                                 double cfl_,
                                                 double gamma_,
                                                 bool use_viscosity_)
: TimeDependentOperator(o_L2FESpace_.GetVSize() + 2*o_H1FESpace_.GetVSize()),
     problem(problem_),
     o_H1FESpace(o_H1FESpace_),
     o_L2FESpace(o_L2FESpace_),
     o_H1compFESpace(o_H1FESpace.GetMesh(),
                     o_H1FESpace.FEColl(),
                     1),
     H1FESpace(*((ParFiniteElementSpace*) o_H1FESpace_.GetFESpace())),
     L2FESpace(*((ParFiniteElementSpace*) o_L2FESpace_.GetFESpace())),
     H1compFESpace(H1FESpace.GetParMesh(),
                   H1FESpace.FEColl(),
                   1),
     ess_tdofs(ess_tdofs_),
     dim(H1FESpace.GetMesh()->Dimension()),
     zones_cnt(H1FESpace.GetMesh()->GetNE()),
     l2dofs_cnt(L2FESpace.GetFE(0)->GetDof()),
     h1dofs_cnt(H1FESpace.GetFE(0)->GetDof()),
     cfl(cfl_),
     gamma(gamma_),
     use_viscosity(use_viscosity_),
     Mv(&H1FESpace),
     Me_inv(l2dofs_cnt, l2dofs_cnt, zones_cnt),
     integ_rule(IntRules.Get(H1FESpace.GetMesh()->GetElementBaseGeometry(),
                             3*H1FESpace.GetOrder(0) + L2FESpace.GetOrder(0) - 1)),
     quad_data(dim, zones_cnt, integ_rule.GetNPoints()),
     quad_data_is_current(false),
     Force(&quad_data, o_H1FESpace, o_L2FESpace)
{
   GridFunctionCoefficient rho_coeff(&rho0);

   // Standard local assembly and inversion for energy mass matrices.
   DenseMatrix Me(l2dofs_cnt);
   DenseMatrixInverse inv(&Me);
   MassIntegrator mi(rho_coeff, &integ_rule);
   for (int i = 0; i < nzones; i++)
   {
      mi.AssembleElementMatrix(*L2FESpace.GetFE(i),
                               *L2FESpace.GetElementTransformation(i), Me);
      inv.Factor();
      inv.GetInverseMatrix(Me_inv(i));
   }

   // Standard assembly for the velocity mass matrix.
   VectorMassIntegrator *vmi = new VectorMassIntegrator(rho_coeff, &integ_rule);
   Mv.AddDomainIntegrator(vmi);
   Mv.Assemble();

   // Values of rho0DetJ0 and Jac0inv at all quadrature points.
   const int nqp = integ_rule.GetNPoints();
   Vector rho_vals(nqp);
   for (int i = 0; i < nzones; i++)
   {
      rho0.GetValues(i, integ_rule, rho_vals);
      ElementTransformation *T = H1FESpace.GetElementTransformation(i);
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

   // Initial local mesh size (assumes similar cells).
   double loc_area = 0.0, glob_area;
   int glob_z_cnt;
   ParMesh *pm = H1FESpace.GetParMesh();
   for (int i = 0; i < nzones; i++) { loc_area += pm->GetElementVolume(i); }
   MPI_Allreduce(&loc_area, &glob_area, 1, MPI_DOUBLE, MPI_SUM, pm->GetComm());
   MPI_Allreduce(&nzones, &glob_z_cnt, 1, MPI_INT, MPI_SUM, pm->GetComm());
   switch (pm->GetElementBaseGeometry(0))
   {
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

   // Make a dummy assembly to figure out the sparsity.

   tensors1D = new Tensors1D(H1FESpace.GetFE(0)->GetOrder(),
                             L2FESpace.GetFE(0)->GetOrder(),
                             int(floor(0.7 + pow(nqp, 1.0 / dim))));

   cg_print_level = 0;
   cg_max_iters   = 200;
   cg_rel_tol     = 1e-16;
   cg_abs_tol     = 0;
}

void LagrangianHydroOperator::Mult(const OccaVector &S, OccaVector &dS_dt) const {
  Vector h_S = S;
  Vector h_dS_dt = dS_dt;
  Mult(h_S, h_dS_dt);
  dS_dt = h_dS_dt;
}

void LagrangianHydroOperator::Mult(const Vector &S, Vector &dS_dt) const
{
   dS_dt = 0.0;

   // Make sure that the mesh positions correspond to the ones in S. This is
   // needed only because some mfem time integrators don't update the solution
   // vector at every intermediate stage (hence they don't change the mesh).
   const int Vsize_l2 = L2FESpace.GetVSize();
   const int Vsize_h1 = H1FESpace.GetVSize();

   // The monolithic BlockVector stores the unknown fields as follows:
   // - Position
   // - Velocity
   // - Specific Internal Energy
   ParGridFunction x, v, e;
   x.MakeRef(&H1FESpace, (Vector&) S, 0);
   v.MakeRef(&H1FESpace, (Vector&) S, Vsize_h1);
   e.MakeRef(&L2FESpace, (Vector&) S, Vsize_h1*2);

   ParGridFunction dx, dv, de;
   dx.MakeRef(&H1FESpace, dS_dt, 0);
   dv.MakeRef(&H1FESpace, dS_dt, Vsize_h1);
   de.MakeRef(&L2FESpace, dS_dt, Vsize_h1*2);

   o_H1FESpace.GetMesh()->NewNodes(x, false);
   UpdateQuadratureData(S);

   // Set dx_dt = v (explicit).
   dx = v;

   // Solve for velocity.
   OccaVector one(Vsize_l2);
   OccaVector rhs(Vsize_h1);
   one = 1.0;

   Force.Mult(one, rhs);
   rhs.Neg();

   OccaVector B(H1compFESpace.TrueVSize());
   OccaVector X(H1compFESpace.TrueVSize());

   // Partial assembly solve for each velocity component.
   OccaMassOperator VMass(&quad_data, o_H1compFESpace);
   const int size = H1compFESpace.GetVSize();
   for (int c = 0; c < dim; c++)
   {
      Vector dv_c(dv.GetData() + c*size, size);

      OccaVector o_rhs_c = rhs.GetRange(c*size, size);
      OccaVector o_dv_c(dv_c);

      // Attributes 1/2/3 correspond to fixed-x/y/z boundaries, i.e.,
      // we must enforce v_x/y/z = 0 for the velocity components.
      Array<int> ess_bdr(H1FESpace.GetParMesh()->bdr_attributes.Max());
      ess_bdr = 0;
      ess_bdr[c] = 1;

      o_dv_c = 0.0;
      o_H1compFESpace.GetProlongationOperator()->MultTranspose(o_rhs_c, B);
      o_H1compFESpace.GetRestrictionOperator()->Mult(o_dv_c, X);

      // True dofs as if there's only one component.
      Array<int> c_tdofs;
      o_H1compFESpace.GetFESpace()->GetEssentialTrueDofs(ess_bdr, c_tdofs);
      VMass.SetEssentialTrueDofs(c_tdofs);
      VMass.EliminateRHS(B);

      CG(H1FESpace.GetParMesh()->GetComm(),
         VMass, B, X,
         cg_print_level,
         cg_max_iters,
         cg_rel_tol,
         cg_abs_tol);

      o_H1compFESpace.GetProlongationOperator()->Mult(X, o_dv_c);
      dv_c = o_dv_c;
   }

   // Solve for energy, assemble the energy source if such exists.
   LinearForm *e_source = NULL;
   if ((problem == vortex) &&
       (dim == 2))
   {
      e_source = new LinearForm(&L2FESpace);
      TaylorCoefficient coeff;
      DomainLFIntegrator *d = new DomainLFIntegrator(coeff, &integ_rule);
      e_source->AddDomainIntegrator(d);
      e_source->Assemble();
   }

   OccaVector o_forceRHS(Vsize_l2);
   OccaVector o_v = v;
   Force.MultTranspose(o_v, o_forceRHS);

   if (e_source) {
     o_forceRHS += *e_source;
   }

   OccaMassOperator EMass(&quad_data, o_L2FESpace);
   OccaVector o_de = de;

   CG(L2FESpace.GetParMesh()->GetComm(),
      EMass, o_forceRHS, o_de,
      cg_print_level,
      cg_max_iters,
      cg_rel_tol,
      cg_abs_tol);
   de = o_de;

   delete e_source;

   quad_data_is_current = false;
}

double LagrangianHydroOperator::GetTimeStepEstimate(const Vector &S) const
{
   Vector* sptr = (Vector*) &S;
   ParGridFunction x;
   x.MakeRef(&H1FESpace, *sptr, 0);
   o_H1FESpace.GetMesh()->NewNodes(x, false);
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

void LagrangianHydroOperator::ComputeDensity(ParGridFunction &rho)
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

LagrangianHydroOperator::~LagrangianHydroOperator()
{
   delete tensors1D;
}

void LagrangianHydroOperator::UpdateQuadratureData(const Vector &S) const
{
   if (quad_data_is_current) { return; }

   const int nqp = integ_rule.GetNPoints();

   ParGridFunction e, v;
   Vector* sptr = (Vector*) &S;
   v.MakeRef(&H1FESpace, *sptr, H1FESpace.GetVSize());
   e.MakeRef(&L2FESpace, *sptr, 2*H1FESpace.GetVSize());
   Vector e_vals;
   DenseMatrix Jpi(dim), sgrad_v(dim), Jinv(dim), stress(dim), stressJiT(dim);
   DenseMatrix v_vals;

   // Batched computations are needed, because hydrodynamic codes usually
   // involve expensive computations of material properties. Although this
   // miniapp uses simple EOS equations, we still want to represent the batched
   // cycle structure.
   int nzones_batch = 3;
   const int nbatches =  nzones / nzones_batch + 1; // +1 for the remainder.
   int nqp_batch = nqp * nzones_batch;
   double *rho_b = new double[nqp_batch],
          *e_b   = new double[nqp_batch],
          *p_b   = new double[nqp_batch],
          *cs_b  = new double[nqp_batch];
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

      for (int z = 0; z < nzones_batch; z++)
      {
         ElementTransformation *T = H1FESpace.GetElementTransformation(z_id);
         e.GetValues(z_id, integ_rule, e_vals);
         for (int q = 0; q < nqp; q++)
         {
            const IntegrationPoint &ip = integ_rule.IntPoint(q);
            T->SetIntPoint(&ip);
            const double detJ = T->Weight();
            MFEM_VERIFY(detJ > 0.0, "Bad Jacobian determinant: " << detJ);

            const int idx = z * nqp + q;
            rho_b[idx] = quad_data.rho0DetJ0w(z_id*nqp + q) / detJ / ip.weight;
            e_b[idx]   = max(0.0, e_vals(q));
         }
         ++z_id;
      }

      // Batched computation of material properties.
      ComputeMaterialProperties(nqp_batch, rho_b, e_b, p_b, cs_b);

      z_id -= nzones_batch;
      for (int z = 0; z < nzones_batch; z++)
      {
         ElementTransformation *T = H1FESpace.GetElementTransformation(z_id);
         v.GetVectorValues(*T, integ_rule, v_vals);
         for (int q = 0; q < nqp; q++)
         {
            const IntegrationPoint &ip = integ_rule.IntPoint(q);
            T->SetIntPoint(&ip);
            // Note that the Jacobian was already computed above. We've chosen
            // not to store the Jacobians for all batched quadrature points.
            const DenseMatrix &Jpr = T->Jacobian();
            const double detJ = T->Weight(), rho = rho_b[z*nqp + q],
                         p = p_b[z*nqp + q], sound_speed = cs_b[z*nqp + q];

            stress = 0.0;
            for (int d = 0; d < dim; d++) { stress(d, d) = -p; }

            // Length scale at the point. The first eigenvector of the symmetric
            // velocity gradient gives the direction of maximal compression.
            // This is used to define the relative change of the initial length
            // scale.
            v.GetVectorGradient(*T, sgrad_v);
            sgrad_v.Symmetrize();
            double eig_val_data[3], eig_vec_data[9];
            sgrad_v.CalcEigenvalues(eig_val_data, eig_vec_data);
            Vector compr_dir(eig_vec_data, dim);
            // Computes the initial->physical transformation Jacobian.
            mfem::Mult(Jpr, quad_data.Jac0inv(z_id*nqp + q), Jpi);
            Vector ph_dir(dim); Jpi.Mult(compr_dir, ph_dir);
            // Change of the initial mesh size in the compression direction.
            const double h = quad_data.h0 * ph_dir.Norml2() /
                             compr_dir.Norml2();

            // Time step estimate at the point.
            quad_data.dt_est = min(quad_data.dt_est, cfl * h / sound_speed);

            if (use_viscosity)
            {
               // Measure of maximal compression.
               const double mu = eig_val_data[0];
               double visc_coeff = 2.0 * rho * h * h * fabs(mu);
               if (mu < 0.0) { visc_coeff += 0.5 * rho * h * sound_speed; }
               stress.Add(visc_coeff, sgrad_v);
            }

            // Quadrature data for partial assembly of the force operator.
            CalcInverse(Jpr, Jinv);
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
   delete rho_b;
   delete e_b;
   delete p_b;
   delete cs_b;
   quad_data_is_current = true;
}

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI
