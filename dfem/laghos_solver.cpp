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

namespace mfem::hydrodynamics
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

void LagrangianHydroOperator::LineSearch::ProcessNewState(
   const Vector &x) const
{
   if (hydro.options.taylor_green) { return; }

   s_new.SetSize(x.Size());
   s_new = x;
   s_new *= hydro.residual->dt;
   s_new += hydro.residual->x;

   real_t min_detJ = hydro.ComputeMinDet(s_new);
   if (min_detJ <= 0.0)
   {
      out << "ProcessNewState negative detJ = " << min_detJ << "\n";
   }
}

real_t LagrangianHydroOperator::LineSearch::ComputeScalingFactor(
   const Vector &x, const Vector &b) const
{
   if (hydro.options.taylor_green) { return 1.0; }

   const bool have_b = (b.Size() == Height());
   real_t lambda = 1.0;  // Initial step length
   x_new.SetSize(x.Size());
   r_new.SetSize(x.Size());
   s_new.SetSize(x.Size());

   const real_t initial_norm = Norm(r);  // Current residual norm

   grad->Mult(c, r_new);
   const real_t grad_norm = Norm(r_new);     // Gradient norm

   for (int i = 0; i < max_line_iter; i++)
   {
      // Try step: x_new = x - lambda * c
      add(x, -lambda, c, x_new);

      // Before calling Mult on the operator we have to check mesh
      // validity by computing the minimum determinant of the mesh
      // Jacobian.
      s_new = x_new;
      s_new *= hydro.residual->dt;
      s_new += hydro.residual->x;
      real_t min_detJ = hydro.ComputeMinDet(s_new);
      if (min_detJ <= 0.0)
      {
         out << "linesearch found negative detJ = " << min_detJ << "\n";
         lambda *= beta;
         continue;
      }

      // Evaluate residual at new point
      oper->Mult(x_new, r_new);
      if (have_b) { subtract(r_new, b, r_new); }

      const real_t new_norm = Norm(r_new);

      if (new_norm <= initial_norm - alpha * lambda * grad_norm)
      {
         // Found acceptable step
         return lambda;
      }

      // Backtrack
      lambda *= beta;
   }

   if (Mpi::Root()) { out << ">>> linesearch didn't converge\n"; }
   return 0.0;
}

// x, v, e blocks of a state vector, aliasing its memory
static std::array<Vector, 3> XVE(const Vector &s, const int h1, const int l2)
{
   auto &b = const_cast<Vector&>(s);
   return {Vector(b, 0, h1), Vector(b, h1, h1), Vector(b, 2*h1, l2)};
}

LagrangianHydroOperator::LagrangianHydroOperator(
   ParFiniteElementSpace &H1,
   ParFiniteElementSpace &L2,
   Array<int> &ess_tdof,
   const IntegrationRule &ir,
   FunctionCoefficient &rho0_coeff,
   ParGridFunction &x0_gf,
   ParGridFunction &rho0_gf,
   ParGridFunction &material_gf,
   std::shared_ptr<DifferentiableOperator> update_qdata,
   std::shared_ptr<DifferentiableOperator> momentum_mf,
   std::shared_ptr<DifferentiableOperator> momentum_pa,
   std::shared_ptr<DifferentiableOperator> energy_conservation_mf,
   std::shared_ptr<DifferentiableOperator> energy_conservation_pa,
   std::shared_ptr<DifferentiableOperator> total_internal_energy_mf,
   std::shared_ptr<DifferentiableOperator> total_kinetic_energy_mf,
   std::shared_ptr<DifferentiableOperator> density_mf,
   std::shared_ptr<DifferentiableOperator> taylor_source_mf,
   std::shared_ptr<QuadratureData> qdata,
   const Options &options) :
   TimeDependentOperator(2*H1.GetVSize()+L2.GetVSize()),
   H1(H1),
   L2(L2),
   H1c(H1.GetParMesh(), H1.FEColl(), 1),
   ess_tdof(ess_tdof),
   ir(ir),
   x0(x0_gf),
   rho0(rho0_gf),
   material(material_gf),
   update_qdata(update_qdata),
   momentum_mf(momentum_mf),
   momentum_pa(momentum_pa),
   energy_mf(energy_conservation_mf),
   energy_pa(energy_conservation_pa),
   internal_energy_mf(total_internal_energy_mf),
   kinetic_energy_mf(total_kinetic_energy_mf),
   density_mf(density_mf),
   taylor_source_mf(taylor_source_mf),
   taylor_source(options.taylor_green && H1.GetMesh()->Dimension() == 2),
   qdata(qdata),
   mesh_nodes(&H1),
   rhsvc(&H1c),
   dvc(&H1c),
   Mv_blf(&H1),
   Me_blf(&L2),
   rho0_coeff(rho0_coeff),
   RHSv(H1.GetTrueVSize()),
   rhsv(H1.GetVSize()),
   X(2*H1.GetTrueVSize()+L2.GetTrueVSize()),
   Xvc(H1c.GetTrueVSize()),
   K(2*H1.GetTrueVSize()+L2.GetTrueVSize()),
   B(H1c.GetTrueVSize()),
   RHSe(L2.GetTrueVSize()),
   rhse(L2.GetVSize()),
   cg_v(H1.GetComm()), cg_e(L2.GetComm()),
   source_x(taylor_source ? H1.GetTrueVSize() : 0),
   source_e(taylor_source ? L2.GetTrueVSize() : 0),
   energy_state(L2.GetTrueVSize(), Device::GetDeviceMemoryType()),
   energy_values(L2.GetTrueVSize(), Device::GetDeviceMemoryType()),
   density_rhs(&L2), density_rho0(L2.GetTrueVSize()),
   density_rhs_t(L2.GetTrueVSize()),
   density_rhs_e(L2.GetTypicalFE()->GetDof()),
   density_values(L2.GetTypicalFE()->GetDof()),
   density_mass(L2.GetTypicalFE()->GetDof()),
   density_inverse(&density_mass),
   density_dofs(L2.GetTypicalFE()->GetDof()), density_integrator(&ir),
   options(options), timing(options.timings)
{
   Mv = new MassPAOperator(H1c, ir, rho0_coeff);
   Me = new MassPAOperator(L2, ir, rho0_coeff);

   Array<int> empty_tdofs;
   Mv_Jprec = new OperatorJacobiSmoother(Mv->GetBF(), empty_tdofs);
   cg_v.SetPreconditioner(*Mv_Jprec);
   cg_v.SetOperator(*Mv);
   cg_e.SetOperator(*Me);
   for (auto *cg : {&cg_v, &cg_e})
   {
      cg->iterative_mode = false;
      cg->SetRelTol(options.cg_relative_tolerance);
      cg->SetAbsTol(0.0);
      cg->SetMaxIter(300);
      cg->SetPrintLevel(-1);
   }

   // Inside the above constructors for mass, there is reordering of the mesh
   // nodes which is performed on the host. Since the mesh nodes are a
   // subvector, so we need to sync with the rest of the base vector (which
   // is assumed to be in the memory space used by the mfem::Device).
   H1.GetParMesh()->GetNodes()->ReadWrite();
   // Attributes 1/2/3 correspond to fixed-x/y/z boundaries, i.e.,
   // we must enforce v_x/y/z = 0 for the velocity components.
   const int bdr_attr_max = H1.GetMesh()->bdr_attributes.Max();
   Array<int> ess_bdr(bdr_attr_max);
   for (int c = 0; c < H1.GetMesh()->Dimension(); c++)
   {
      ess_bdr = 0;
      ess_bdr[c] = 1;
      H1c.GetEssentialTrueDofs(ess_bdr, c_tdofs[c]);
      c_tdofs[c].Read();
   }

   Mv_blf.AddDomainIntegrator(new VectorMassIntegrator(rho0_coeff, &ir));
   Mv_blf.Assemble();
   Mv_blf.FormSystemMatrix(mfem::Array<int>(), Mv_mat);

   Me_blf.AddDomainIntegrator(new MassIntegrator(rho0_coeff, &ir));
   Me_blf.Assemble();
   Me_blf.FormSystemMatrix(mfem::Array<int>(), Me_mat);

   residual.reset(new Residual(*this, X,
                               options.fd_gradient,
                               options.dump_jacobians));

   preconditioner.reset(new Preconditioner(*this));

   auto gmres = new GMRESSolver(MPI_COMM_WORLD);
   gmres->SetKDim(500);
   gmres->SetMaxIter(options.krylov_maximum_iterations);
   gmres->SetRelTol(1e-12);
   gmres->SetAbsTol(0.0);
   gmres->SetPrintLevel(IterativeSolver::PrintLevel().None());
   gmres->SetPreconditioner(*preconditioner);
   krylov.reset(gmres);

   newton.reset(new LineSearch(MPI_COMM_WORLD, *this));
   newton->SetPrintLevel(IterativeSolver::PrintLevel().Summary());
   newton->SetOperator(*residual);
   newton->SetSolver(*krylov);
   newton->SetMaxIter(options.nonlinear_maximum_iterations);
   newton->SetRelTol(options.nonlinear_relative_tolerance);
   newton->SetAbsTol(0.0);
}

LagrangianHydroOperator::~LagrangianHydroOperator()
{
   delete Mv;
   delete Mv_Jprec;
   delete Me;
}

void LagrangianHydroOperator::Mult(const Vector &S, Vector &dSdt) const
{
   // Make sure that the mesh positions correspond to the ones in S. This is
   // needed only because some mfem time integrators don't update the solution
   // vector at every intermediate stage (hence they don't change the mesh).
   UpdateMesh(S);
   UpdateQuadratureData(S);

   const int H1vsize = H1.GetVSize();
   auto [x, v, e] = XVE(S, H1vsize, L2.GetVSize());
   auto [dx, dv, de] = XVE(dSdt, H1vsize, L2.GetVSize());

   // solve position
   dx = v;
   dx.GetMemory().SyncAlias(dSdt.GetMemory(), dx.Size());

   // solve velocity
   {
      dv = 0.0;

      ApplyMomentum(RHSv);
      RHSv.Neg();
      H1.GetRestrictionMatrix()->MultTranspose(RHSv, rhsv);

      // solve for each velocity component
      const int size = H1c.GetVSize();
      const Operator *Pconf = H1c.GetProlongationMatrix();
      for (int c = 0; c < H1.GetMesh()->Dimension(); c++)
      {
         dvc.MakeRef(&H1c, dSdt, H1vsize + c*size);
         rhsvc.MakeRef(&H1c, rhsv, c*size);
         if (Pconf) { Pconf->MultTranspose(rhsvc, B); }
         else { B = rhsvc; }

         Xvc = 0.0;
         Mv->SetEssentialTrueDofs(c_tdofs[c]);
         Mv->EliminateRHS(B);
         {
            HydroTimings::Scope timer(timing, HydroTimings::CG_H1);
            cg_v.Mult(B, Xvc);
         }
         if (timing.enabled)
         {
            const int iterations = cg_v.GetNumIterations();
            timing.entries[HydroTimings::CG_H1].iterations += iterations;
            timing.h1_work += HYPRE_BigInt(H1c.GetTrueVSize()) * iterations;
         }
         if (Pconf)
         {
            Pconf->Mult(Xvc, dvc);
         }
         else
         {
            dvc = Xvc;
         }
         dvc.GetMemory().SyncAlias(dSdt.GetMemory(), dvc.Size());
      }
   }

   // solve energy
   {
      de = 0.0;

      ApplyEnergy(v, RHSe);
      L2.GetRestrictionMatrix()->MultTranspose(RHSe, rhse);

      if (taylor_source)
      {
         H1.GetRestrictionMatrix()->Mult(x, source_x);
         source_vectors.input.MakeRef(source_x);
         source_vectors.output.MakeRef(source_e);
         taylor_source_mf->Mult(source_vectors.input, source_vectors.output);
         rhse += source_e;
      }

      {
         HydroTimings::Scope timer(timing, HydroTimings::CG_L2);
         cg_e.Mult(rhse, de);
      }
      if (timing.enabled)
      {
         const int iterations = cg_e.GetNumIterations();
         timing.entries[HydroTimings::CG_L2].iterations += iterations;
         timing.l2_work += HYPRE_BigInt(L2.GetTrueVSize()) * iterations;
      }
      de.GetMemory().SyncAlias(dSdt.GetMemory(), de.Size());
   }

   qdata_is_current = false;
}

void LagrangianHydroOperator::ApplyMomentum(Vector &force) const
{
   HydroTimings::Scope timer(timing, HydroTimings::FORCE);
   if (timing.enabled) { timing.force_work += H1.GetTrueVSize(); }
   momentum_vectors.input.MakeRef(qdata->stressp);
   momentum_vectors.output.MakeRef(force);
   momentum_pa->Mult(momentum_vectors.input, momentum_vectors.output);
}

void LagrangianHydroOperator::ApplyEnergy(const Vector &velocity,
                                          Vector &force) const
{
   HydroTimings::Scope timer(timing, HydroTimings::FORCE);
   if (timing.enabled) { timing.force_work += L2.GetTrueVSize(); }
   energy_vectors.input.MakeRef(velocity, qdata->stressp);
   energy_vectors.output.MakeRef(force);
   energy_pa->Mult(energy_vectors.input, energy_vectors.output);
}

void LagrangianHydroOperator::UpdateMesh(const Vector &S) const
{
   Vector* sptr = const_cast<Vector*>(&S);
   mesh_nodes.MakeRef(&H1, *sptr, 0);
   H1.GetParMesh()->NewNodes(mesh_nodes, false);
}

real_t LagrangianHydroOperator::GetTimeStepEstimate(const Vector &S)
{
   MFEM_VERIFY(qdata, "quadrature data (qdata) is null");
   UpdateMesh(S);
   real_t dt_est_local = LocalTimeStepEstimate(S);
   if (stage_dt_estimate) { dt_est_local = fmin(dt_est_local, dt_est_stages); }
   real_t dt_est_global;
   HydroTimings::Scope timer(timing, HydroTimings::COMMUNICATION);
   MPI_Allreduce(&dt_est_local, &dt_est_global, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MIN, L2.GetComm());
   return dt_est_global;
}

void LagrangianHydroOperator::ResetTimeStepEstimate() const
{
   dt_est_stages = std::numeric_limits<real_t>::infinity();
}

void LagrangianHydroOperator::ComputeDensity(ParGridFunction &rho) const
{
   rho.SetSpace(&L2);
   L2.GetProlongationMatrix()->MultTranspose(rho0, density_rho0);
   diagnostic_vectors.input.MakeRef(density_rho0, x0);
   diagnostic_vectors.output.MakeRef(density_rhs_t);
   density_mf->Mult(diagnostic_vectors.input, diagnostic_vectors.output);
   L2.GetProlongationMatrix()->Mult(density_rhs_t, density_rhs);
   for (int e = 0; e < L2.GetParMesh()->GetNE(); e++)
   {
      const FiniteElement &fe = *L2.GetFE(e);
      ElementTransformation &eltr = *L2.GetElementTransformation(e);
      L2.GetElementDofs(e, density_dofs);
      density_integrator.AssembleElementMatrix(fe, eltr, density_mass);
      density_inverse.Factor();
      density_rhs.GetElementDofValues(e, density_rhs_e);
      density_inverse.Mult(density_rhs_e, density_values);
      rho.SetSubVector(density_dofs, density_values);
   }
}

real_t LagrangianHydroOperator::InternalEnergy(ParGridFunction &e) const
{
   L2.GetRestrictionMatrix()->Mult(e, energy_state);
   diagnostic_vectors.input.MakeRef(energy_state, rho0, x0);
   diagnostic_vectors.output.MakeRef(energy_values);
   internal_energy_mf->Mult(diagnostic_vectors.input,
                            diagnostic_vectors.output);
   const real_t ie_local = energy_values.Sum();
   real_t ie_global = 0.0;
   MPI_Allreduce(&ie_local, &ie_global, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_SUM, L2.GetParMesh()->GetComm());
   return ie_global;
}

real_t LagrangianHydroOperator::KineticEnergy(ParGridFunction &v) const
{
   diagnostic_vectors.input.MakeRef(v, rho0, x0);
   diagnostic_vectors.output.MakeRef(energy_values);
   kinetic_energy_mf->Mult(diagnostic_vectors.input,
                           diagnostic_vectors.output);
   const real_t ke_local = energy_values.Sum();
   real_t ke_global = 0.0;
   MPI_Allreduce(&ke_local, &ke_global, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_SUM, H1.GetParMesh()->GetComm());
   return ke_global;
}

void LagrangianHydroOperator::UpdateQuadratureData(const Vector &S) const
{
   if (qdata_is_current) { return; }
   HydroTimings::Scope timer(timing, HydroTimings::QDATA);
   if (timing.enabled)
   {
      timing.quad_work += HYPRE_BigInt(L2.GetParMesh()->GetNE()) *
                          ir.GetNPoints();
   }
   qdata_is_current = true;
   auto [x, v, e] = XVE(S, H1.GetVSize(), L2.GetVSize());
   qdata_vectors.input.MakeRef(v, rho0, x0, x, material, e);
   qdata_vectors.output.MakeRef(qdata->stressp, qdata->dt_est);
   update_qdata->Mult(qdata_vectors.input, qdata_vectors.output);
   // the estimate is the min over all the updates
   if (stage_dt_estimate)
   {
      dt_est_stages = fmin(dt_est_stages, qdata->dt_est.Min());
   }
}

// DerivativeOperator::Assemble returns a new matrix, owned by the caller
static std::unique_ptr<HypreParMatrix> AssembleMatrix(DerivativeOperator &d)
{
   HypreParMatrix *A = nullptr;
   d.Assemble(A);
   return std::unique_ptr<HypreParMatrix>(A);
}

void LagrangianHydroOperator::Jacobian::Mult(const Vector &u, Vector &y) const
{
   HydroTimings::Scope timer(hydro.timing, HydroTimings::JACOBIAN_ACTION);
   w = u;
   auto [wx, wv, we] = XVE(w, H1tsize, L2tsize);
   auto [zx, zv, ze] = XVE(z, H1tsize, L2tsize);
   zv_output.MakeRef(zv);
   ze_output.MakeRef(ze);
   auto [yx, yv, ye] = XVE(y, H1tsize, L2tsize);

   // position
   yx = wv;
   yx *= -h;
   yx += wx;

   // velocity
   dRvdx->Mult(wx, zv_output);
   zv *= h;
   yv = zv;
   dRvdv->Mult(wv, zv_output);
   zv *= h;
   yv += zv;
   Vector wvc, yvc;
   for (int c = 0; c < hydro.H1.GetMesh()->Dimension(); c++)
   {
      wvc.MakeRef(wv, c*hydro.H1c.GetTrueVSize(), hydro.H1c.GetTrueVSize());
      yvc.MakeRef(yv, c*hydro.H1c.GetTrueVSize(), hydro.H1c.GetTrueVSize());
      hydro.Mv->FullAddMult(wvc, yvc);
      yvc.SyncAliasMemory(yv);
   }
   yv.SyncAliasMemory(y);

   dRvde->Mult(we, zv_output);
   zv *= h;
   yv += zv;
   yv.SetSubVector(hydro.ess_tdof, 0.0);

   // energy
   //                          [ wx ]
   // [ dRe/dx dRe/dv dRe/de ] [ wv ]
   //                          [ we ]
   //

   dRedx->Mult(wx, ze_output);
   if (dTaylorSourcedx)
   {
      source_output.MakeRef(source);
      dTaylorSourcedx->Mult(wx, source_output);
      ze += source;
   }
   ze *= -h;
   ye = ze;

   dRedv->Mult(wv, ze_output);
   ze *= -h;
   ye += ze;

   dRede->Mult(we, ze_output);
   ze *= -h;
   hydro.Me->FullAddMult(we, ze);

   ye += ze;

   yx.SyncAliasMemory(y);
   yv.SyncAliasMemory(y);
   ye.SyncAliasMemory(y);
}

MemoryClass LagrangianHydroOperator::Jacobian::GetMemoryClass() const
{
   return Device::GetDeviceMemoryClass();
}

void LagrangianHydroOperator::Preconditioner::SetOperator(const Operator &op)
{
   if (!rebuild) { return; }
   HydroTimings::Scope timer(hydro.timing, HydroTimings::AMG_SETUP);

   jacobian = dynamic_cast<const Jacobian*>(&op);
   MFEM_VERIFY(jacobian != nullptr, "Preconditioner error.");
   MFEM_VERIFY(hydro.options.preconditioner_type ==
               PRECONDITIONER_TYPE::BLOCK_DIAGONAL_AMG,
               "Unknown preconditioner type.");

   const real_t h = jacobian->GetH();

   // dRvdv = (Mv + h dF/dv)
   auto dRvdv_ptr = AssembleMatrix(*jacobian->GetdRvdv());
   HypreParMatrix &dRvdv_mat = *dRvdv_ptr;
   HypreParMatrix *Mv_hdRvdv_mat = Add(1.0, hydro.Mv_mat, h, dRvdv_mat);
   auto tmp2 = Mv_hdRvdv_mat->EliminateRowsCols(hydro.ess_tdof);
   delete tmp2;

   // dRede = Me - h * dF^T/de
   auto dRede_ptr = AssembleMatrix(*jacobian->GetdRede());
   HypreParMatrix &dRede_mat = *dRede_ptr;
   HypreParMatrix *Me_hdRede_mat = Add(1.0, hydro.Me_mat, -h, dRede_mat);

   vv_mat.reset(Mv_hdRvdv_mat);
   amg_v.reset(new HypreBoomerAMG(*vv_mat));
   amg_v->SetPrintLevel(0);

   ee_mat.reset(Me_hdRede_mat);
   amg_e.reset(new HypreBoomerAMG(*ee_mat));
   amg_e->SetPrintLevel(0);

   rebuild = false;
   setup_pending = true;
}

void LagrangianHydroOperator::Preconditioner::Mult(const Vector &x,
                                                   Vector &y) const
{
   w = x;
   const int H1tsize = hydro.H1.GetTrueVSize();
   const int L2tsize = hydro.L2.GetTrueVSize();
   auto [wx, wv, we] = XVE(w, H1tsize, L2tsize);
   auto [yx, yv, ye] = XVE(y, H1tsize, L2tsize);

   // Include hypre's deferred hierarchy construction in setup time
   if (setup_pending)
   {
      HydroTimings::Scope setup(hydro.timing, HydroTimings::AMG_SETUP,
                                false);
      amg_v->Setup(wv, yv);
      amg_e->Setup(we, ye);
      setup_pending = false;
   }
   HydroTimings::Scope timer(hydro.timing, HydroTimings::AMG_APPLY);
   yx = wx;
   amg_v->Mult(wv, yv);
   amg_e->Mult(we, ye);

   yx.SyncAliasMemory(y);
   yv.SyncAliasMemory(y);
   ye.SyncAliasMemory(y);
}

void LagrangianHydroOperator::Residual::Mult(const Vector &k, Vector &R) const
{
   u = k;
   u *= dt;
   u += x;

   auto [kx, kv, ke] = XVE(k, H1tsize, L2tsize);
   auto [ux, uv, ue] = XVE(u, H1tsize, L2tsize);
   auto [Rx, Rv, Re] = XVE(R, H1tsize, L2tsize);
   auto [ux_l, uv_l, ue_l] = XVE(u_l, H1vsize, L2vsize);

   hydro.H1.GetProlongationMatrix()->Mult(ux, ux_l);
   hydro.H1.GetProlongationMatrix()->Mult(uv, uv_l);
   hydro.L2.GetProlongationMatrix()->Mult(ue, ue_l);
   ux_l.SyncAliasMemory(u_l);
   uv_l.SyncAliasMemory(u_l);
   ue_l.SyncAliasMemory(u_l);

   hydro.UpdateMesh(ux_l);
   hydro.mesh_nodes.SyncMemory(ux_l);

   Rx = kx;
   Rx -= uv;

   hydro.qdata_is_current = false;
   hydro.UpdateQuadratureData(u_l);

   hydro.ApplyMomentum(Rv);

   Vector kvc, Rvc;
   for (int c = 0; c < hydro.H1.GetMesh()->Dimension(); c++)
   {
      kvc.MakeRef(kv, c*hydro.H1c.GetTrueVSize(), hydro.H1c.GetTrueVSize());
      Rvc.MakeRef(Rv, c*hydro.H1c.GetTrueVSize(), hydro.H1c.GetTrueVSize());
      hydro.Mv->FullAddMult(kvc, Rvc);
      Rvc.SyncAliasMemory(Rv);
   }
   Rv.SyncAliasMemory(R);

   Rv.SetSubVector(hydro.ess_tdof, 0.0);

   hydro.ApplyEnergy(uv_l, Re);

   Re.Neg();

   if (hydro.taylor_source)
   {
      hydro.source_vectors.input.MakeRef(ux);
      hydro.source_vectors.output.MakeRef(e_source_t);
      hydro.taylor_source_mf->Mult(hydro.source_vectors.input,
                                  hydro.source_vectors.output);
      Re -= e_source_t;
   }

   hydro.Me->FullAddMult(ke, Re);

   Rx.SyncAliasMemory(R);
   Rv.SyncAliasMemory(R);
   Re.SyncAliasMemory(R);
}

Operator& LagrangianHydroOperator::Residual::GetGradient(const Vector &k) const
{
   HydroTimings::Scope timer(hydro.timing, HydroTimings::JACOBIAN_SETUP);
   u = k;
   u *= dt;
   u += x;

   auto [ux, uv, ue] = XVE(u, H1tsize, L2tsize);

   if (fd_gradient)
   {
      fd_jacobian = std::make_shared<FDJacobian>(*this, k, 1e-8);
      return *fd_jacobian;
   }
   else
   {
      // The derivatives take their state as T-vectors, in the inputs order
      hydro.L2.GetRestrictionMatrix()->Mult(hydro.rho0, rho0_t);
      hydro.H1.GetRestrictionMatrix()->Mult(hydro.x0, x0_t);
      hydro.material.ParFESpace()->GetRestrictionMatrix()->Mult(hydro.material,
                                                                material_t);
      state.MakeRef(uv, rho0_t, x0_t, ux, material_t, ue);

      auto dRvdx = hydro.momentum_mf->GetDerivative(COORDINATES, state);
      auto dRvdv = hydro.momentum_mf->GetDerivative(VELOCITY, state);
      auto dRvde = hydro.momentum_mf->GetDerivative(INTERNAL_ENERGY, state);
      auto dRedx = hydro.energy_mf->GetDerivative(COORDINATES, state);
      auto dRedv = hydro.energy_mf->GetDerivative(VELOCITY, state);
      auto dRede =
         hydro.energy_mf->GetDerivative(INTERNAL_ENERGY, state);
      std::shared_ptr<DerivativeOperator> dTaylorSourcedx;
      if (hydro.taylor_source)
      {
         hydro.source_vectors.input.MakeRef(ux);
         dTaylorSourcedx = hydro.taylor_source_mf->GetDerivative(
                             COORDINATES, hydro.source_vectors.input);
      }

      jacobian = std::make_shared<Jacobian>(
                    hydro, dRvdx, dRvdv, dRvde, dRedx, dRedv, dRede, dTaylorSourcedx, dt);

      if (dump_jacobians > 0)
      {
         auto ess_tdof_backup(hydro.ess_tdof);
         hydro.ess_tdof.SetSize(0);

         out << "\ndumping jacobians\n";
         std::ofstream jvpmat("jvpmat.m");
         jacobian->PrintMatlab(jvpmat);
         jvpmat.close();

         fd_jacobian = std::make_shared<FDJacobian>(*this, k, 1e-8);
         std::ofstream fdjacmat("fdjacmat.m");
         fd_jacobian->PrintMatlab(fdjacmat);
         fdjacmat.close();

         if (dump_jacobians == 1) { std::exit(0); }
         hydro.ess_tdof = ess_tdof_backup;
      }

      return *jacobian;
   }
}

void LagrangianHydroOperator::ImplicitSolve(const real_t dt, const Vector &x,
                                            Vector &k)
{
   const int H1tsize = H1.GetTrueVSize(), L2tsize = L2.GetTrueVSize();
   auto [xx, xv, xe] = XVE(x, H1.GetVSize(), L2.GetVSize());
   auto [Xx, Xv, Xe] = XVE(X, H1tsize, L2tsize);

   H1.GetRestrictionMatrix()->Mult(xx, Xx);
   H1.GetRestrictionMatrix()->Mult(xv, Xv);
   L2.GetRestrictionMatrix()->Mult(xe, Xe);

   Xx.SyncAliasMemory(X);
   Xv.SyncAliasMemory(X);
   Xe.SyncAliasMemory(X);

   residual->SetTimeStep(dt);

   Vector zero;
   K = X;
   if (current_dt != dt || lag >= options.preconditioner_lag)
   {
      lag = 0;
      current_dt = dt;
      preconditioner->SetRebuildFlag(true);
   }
   lag++;

   {
      HydroTimings::Scope timer(timing, HydroTimings::NEWTON);
      newton->Mult(zero, K);
      if (timing.enabled)
      {
         timing.entries[HydroTimings::NEWTON].iterations +=
            newton->GetNumIterations();
      }
   }

   // If some SDIRK stage didn't converge, keep it at false.
   // (these are reset after every time step).
   if (newton_converged) { newton_converged = newton->GetConverged(); }
   newton_max_iter = std::max(newton_max_iter, newton->GetNumIterations());

   auto [Kx, Kv, Ke] = XVE(K, H1tsize, L2tsize);
   auto [kx, kv, ke] = XVE(k, H1.GetVSize(), L2.GetVSize());
   H1.GetProlongationMatrix()->Mult(Kx, kx);
   H1.GetProlongationMatrix()->Mult(Kv, kv);
   L2.GetProlongationMatrix()->Mult(Ke, ke);
}

void LagrangianHydroOperator::ResetQuadratureData() const
{
   qdata_is_current = false;
}

void LagrangianHydroOperator::ResetNewtonInfo()
{
   newton_converged = true;
   newton_max_iter = -1;
}

void LagrangianHydroOperator::PrintTimingData(bool fom) const
{
   if (!timing.enabled) { return; }
   using T = HydroTimings;
   std::array<double, T::COUNT + 1> local_time{}, time{};
   std::array<HYPRE_BigInt, T::COUNT> local_calls{}, calls{};
   std::array<HYPRE_BigInt, T::COUNT> local_iterations{}, iterations{};
   for (int i = 0; i < T::COUNT; i++)
   {
      local_time[i] = timing.entries[i].watch.RealTime();
      local_calls[i] = timing.entries[i].calls;
      local_iterations[i] = timing.entries[i].iterations;
   }
   // The stock major-kernel comparison excludes the L2 solve
   local_time[T::COUNT] = local_time[T::CG_H1] + local_time[T::FORCE] +
                          local_time[T::QDATA];
   const MPI_Comm comm = H1.GetComm();
   MPI_Reduce(local_time.data(), time.data(), T::COUNT + 1, MPI_DOUBLE,
              MPI_MAX, 0, comm);
   MPI_Reduce(local_calls.data(), calls.data(), T::COUNT,
              HYPRE_MPI_BIG_INT, MPI_MAX, 0, comm);
   MPI_Reduce(local_iterations.data(), iterations.data(), T::COUNT,
              HYPRE_MPI_BIG_INT, MPI_MAX, 0, comm);
   HYPRE_BigInt local_work[] = {timing.h1_work, timing.l2_work,
                              timing.force_work, timing.quad_work,
                              L2.GetParMesh()->GetNE()};
   HYPRE_BigInt work[5]{};
   MPI_Reduce(local_work, work, 5, HYPRE_MPI_BIG_INT, MPI_SUM, 0, comm);
   const auto h1_dofs = H1.GlobalTrueVSize();
   const auto l2_dofs = L2.GlobalTrueVSize();
   int rank;
   MPI_Comm_rank(comm, &rank);
   if (rank != 0) { return; }

   const auto rate = [](double work, double seconds)
   {
      return seconds > 0.0 ? 1e-6 * work / seconds : 0.0;
   };
   const double total = time[T::COUNT];
   const double fom0 = rate(work[2], total);
   const double fom1 = rate(work[0], time[T::CG_H1]);
   const double fom2 = rate(work[2], time[T::FORCE]);
   const double fom3 = rate(work[3], time[T::QDATA]);
   const double weighted = rate(double(work[0]) + work[2] + work[3], total);
   const auto flags = out.flags();
   const auto precision = out.precision();
   out << std::scientific << std::setprecision(6) << "\n";
   const auto section = [&](const char *name, T::Section id)
   {
      out << name << " total time: " << time[id]
          << " (calls: " << calls[id];
      if (id == T::CG_H1 || id == T::CG_L2 || id == T::NEWTON)
      {
         out << ", iterations: " << iterations[id];
      }
      out << ")\n";
   };
   section("CG (H1)", T::CG_H1);
   out << "CG (H1) rate (megadofs x cg_iterations / second): "
       << fom1 << "\n";
   section("CG (L2)", T::CG_L2);
   out << "CG (L2) rate (megadofs x cg_iterations / second): "
       << rate(work[1], time[T::CG_L2]) << "\n";
   section("Forces", T::FORCE);
   out << "Forces rate (megadofs x applications / second): "
       << fom2 << "\n";
   section("UpdateQuadData", T::QDATA);
   out << "UpdateQuadData rate (megaquads x updates / second): "
       << fom3 << "\n";
   out << "Major kernels total time (stock subset, excludes L2): "
       << total << "\n"
       << "Major kernels weighted rate (stock FOM): " << weighted << "\n";
   section("Time integration wall", T::SOLVE);
   out << "Time integration rate (mega force dofs / second): "
       << rate(work[2], time[T::SOLVE]) << "\n";
   section("Time-step MPI reduction", T::COMMUNICATION);
   if (calls[T::NEWTON])
   {
      out << "Implicit regions overlap Newton and integration wall time\n";
      section("Jacobian setup", T::JACOBIAN_SETUP);
      section("Jacobian action", T::JACOBIAN_ACTION);
      section("AMG block setup", T::AMG_SETUP);
      section("AMG block application", T::AMG_APPLY);
      section("Newton solve", T::NEWTON);
   }
   out << "Times, calls and iterations are MPI maxima; "
       << "work is globally summed\n";
   if (fom)
   {
      const auto ndofs = 2*h1_dofs + l2_dofs + ir.GetNPoints()*work[4];
      out << "\n| Ranks | Zones | H1 dofs | L2 dofs | QP | N dofs"
          << " | FOM0 | FOM1 | T1 | FOM2 | T2 | FOM3 | T3 | FOM | TT |\n"
          << "| " << H1.GetNRanks() << " | " << work[4]
          << " | " << h1_dofs << " | " << l2_dofs
          << " | " << ir.GetNPoints() << " | " << ndofs
          << " | " << fom0 << " | " << fom1 << " | " << time[T::CG_H1]
          << " | " << fom2 << " | " << time[T::FORCE]
          << " | " << fom3 << " | " << time[T::QDATA]
          << " | " << weighted << " | " << total << " |\n";
   }
   out.flags(flags);
   out.precision(precision);
}

void LagrangianHydroOperator::DensityScatter()
{
   Mesh current = H1.GetParMesh()->GetSerialMesh(0);
   GridFunction r0 = rho0.GetSerialGridFunction(0, current);
   GridFunction x0_s = x0.GetSerialGridFunction(0, current);

   if (Mpi::WorldRank() != 0) { return; }

   // det(J0) is the initial element, not 1/NE of a uniform mesh.
   Mesh initial(current);
   const bool nodes = initial.GetNodes() &&
                      initial.GetNodes()->Size() == x0_s.Size();
   MFEM_VERIFY(nodes, "initial positions do not match the gathered mesh");
   *initial.GetNodes() = x0_s;

   std::ofstream fstream_rho("rho.out");
   fstream_rho.precision(8);

   const int nqp = ir.GetNPoints();
   const int dim = current.Dimension();
   Vector pos(dim);
   for (int e = 0; e < current.GetNE(); e++)
   {
      ElementTransformation &Tr = *current.GetElementTransformation(e);
      ElementTransformation &Tr0 = *initial.GetElementTransformation(e);
      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         Tr0.SetIntPoint(&ip);
         Tr.Transform(ip, pos);

         const double detJ = Tr.Weight();
         const double detJ0 = Tr0.Weight();
         // rho0 * detJ0 / detJ.
         const double rho = r0.GetValue(Tr, ip) * detJ0 / detJ;
         double r2 = 0.0;
         for (int d = 0; d < dim; d++) { r2 += pos(d) * pos(d); }
         fstream_rho << std::sqrt(r2) << " " << rho << "\n";
      }
   }
   fstream_rho.close();
}

real_t LagrangianHydroOperator::ComputeMinDet(const Vector &S) const
{
   auto sptr = const_cast<Vector*>(&S);
   Vector x, x_loc(H1.GetVSize());
   x.MakeRef(*sptr, 0, H1.GetTrueVSize());
   H1.GetProlongationMatrix()->Mult(x, x_loc);
   ParGridFunction xgf;
   xgf.MakeRef(&H1, x_loc, 0);
   auto g = GeometricFactors(xgf, ir,
                             GeometricFactors::JACOBIANS |
                             GeometricFactors::DETERMINANTS);
   auto min = g.detJ.Min();
   MPI_Allreduce(MPI_IN_PLACE, &min, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MIN, L2.GetComm());
   return min;
}

real_t LagrangianHydroOperator::LocalTimeStepEstimate(const Vector &S) const
{
   // Update at S, the current quadrature data can be of another state
   qdata_is_current = false;
   UpdateQuadratureData(S);
   return qdata->dt_est.Min();
}

} // namespace mfem::hydrodynamics
