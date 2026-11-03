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

extern int problem;

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
   if (problem == 0) { return; }

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
   if (problem == 0) { return 1.0; }

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
   std::shared_ptr<DifferentiableOperator> dtest_mf,
   std::shared_ptr<DifferentiableOperator> momentum_mf,
   std::shared_ptr<DifferentiableOperator> momentum_pa,
   std::shared_ptr<DifferentiableOperator> energy_conservation_mf,
   std::shared_ptr<DifferentiableOperator> energy_conservation_pa,
   std::shared_ptr<DifferentiableOperator> total_internal_energy_mf,
   std::shared_ptr<DifferentiableOperator> total_kinetic_energy_mf,
   std::shared_ptr<DifferentiableOperator> density_mf,
   std::shared_ptr<DifferentiableOperator> taylor_source_mf,
   std::shared_ptr<QuadratureData> qdata,
   const Options &options,
   Vector& external_data) :
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
   dtest_mf(dtest_mf),
   momentum_mf(momentum_mf),
   momentum_pa(momentum_pa),
   energy_conservation_mf(energy_conservation_mf),
   energy_conservation_pa(energy_conservation_pa),
   total_internal_energy_mf(total_internal_energy_mf),
   total_kinetic_energy_mf(total_kinetic_energy_mf),
   density_mf(density_mf),
   taylor_source_mf(taylor_source_mf),
   taylor_source(problem == 0 && H1.GetMesh()->Dimension() == 2),
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
   options(options),
   external_data(external_data)
{
   Mv = new MassPAOperator(H1c, ir, rho0_coeff);
   Array<int> empty_tdofs;
   Mv_Jprec = new OperatorJacobiSmoother(Mv->GetBF(), empty_tdofs);

   Me = new MassPAOperator(L2, ir, rho0_coeff);

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
   UpdateMesh(S);
   UpdateQuadratureData(S);

   const int H1vsize = H1.GetVSize();
   auto [x, v, e] = XVE(S, H1vsize, L2.GetVSize());
   auto [dx, dv, de] = XVE(dSdt, H1vsize, L2.GetVSize());

   // solve position
   dx = v;

   // solve velocity
   {
      dv = 0.0;

      MultiVector Y{RHSv};
      momentum_pa->Mult(MultiVector{qdata->stressp}, Y);
      RHSv.Neg();
      H1.GetRestrictionMatrix()->MultTranspose(RHSv, rhsv);

      // solve for each velocity component
      const int size = H1c.GetVSize();
      const Operator *Pconf = H1c.GetProlongationMatrix();
      for (int c = 0; c < H1.GetMesh()->Dimension(); c++)
      {
         dvc.MakeRef(&H1c, dSdt, H1vsize + c*size);
         rhsvc.MakeRef(&H1c, rhsv, c*size);
         if (Pconf)
         {
            Pconf->MultTranspose(rhsvc, B);
         }
         else
         {
            B = rhsvc;
         }

         CGSolver cg(H1c.GetParMesh()->GetComm());
         cg.SetPreconditioner(*Mv_Jprec);
         cg.SetOperator(*Mv);
         cg.SetRelTol(options.cg_relative_tolerance);
         cg.SetAbsTol(0.0);
         cg.SetMaxIter(300);
         cg.SetPrintLevel(-1);

         H1c.GetRestrictionMatrix()->Mult(dvc, Xvc);
         Mv->SetEssentialTrueDofs(c_tdofs[c]);
         Mv->EliminateRHS(B);
         cg.Mult(B, Xvc);
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

      MultiVector Y{RHSe};
      energy_conservation_pa->Mult(MultiVector{v, qdata->stressp}, Y);
      L2.GetRestrictionMatrix()->MultTranspose(RHSe, rhse);

      if (taylor_source)
      {
         Vector xt(H1.GetTrueVSize()), es(L2.GetTrueVSize());
         H1.GetRestrictionMatrix()->Mult(x, xt);
         MultiVector ES{es};
         taylor_source_mf->Mult(MultiVector{xt}, ES);
         rhse += es;
      }

      CGSolver cg(L2.GetParMesh()->GetComm());
      cg.SetOperator(*Me);
      cg.iterative_mode = false;
      cg.SetRelTol(options.cg_relative_tolerance);
      cg.SetAbsTol(0.0);
      cg.SetMaxIter(300);
      cg.SetPrintLevel(-1);
      cg.Mult(rhse, de);
      de.GetMemory().SyncAlias(dSdt.GetMemory(), de.Size());
   }

   qdata_is_current = false;
}

void LagrangianHydroOperator::UpdateMesh(const Vector &S) const
{
   Vector* sptr = const_cast<Vector*>(&S);
   mesh_nodes.MakeRef(&H1, *sptr, 0);
   H1.GetParMesh()->NewNodes(mesh_nodes, false);
}

real_t LagrangianHydroOperator::GetTimeStepEstimate(const Vector &S)
{
   MFEM_VERIFY(dtest_mf, "dt estimate operator (dtest_mf) is null");
   MFEM_VERIFY(qdata, "quadrature data (qdata) is null");

   UpdateMesh(S);
   UpdateQuadratureData(S);

   real_t dt_est_local = LocalTimeStepEstimate(S);
   if (stage_dt_estimate) { dt_est_local = fmin(dt_est_local, dt_est_stages); }

   real_t dt_est_global;
   MPI_Allreduce(&dt_est_local, &dt_est_global, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MIN,
                 L2.GetComm());

   return dt_est_global;
}

void LagrangianHydroOperator::ResetTimeStepEstimate() const
{
   dt_est_stages = std::numeric_limits<real_t>::infinity();
}

void LagrangianHydroOperator::ComputeDensity(ParGridFunction &rho) const
{
   rho.SetSpace(&L2);

   ParGridFunction rhs_l(&L2);

   Vector rho0_t(L2.GetTrueVSize()),
          rho_t(L2.GetTrueVSize()),
          rhs(L2.GetTrueVSize());

   const int l2dofs_cnt = L2.GetFE(0)->GetDof();
   DenseMatrix Mrho(l2dofs_cnt);
   DenseMatrixInverse inv(&Mrho);
   Vector rhs_e(l2dofs_cnt), rho_z(l2dofs_cnt);
   Array<int> dofs(l2dofs_cnt);
   MassIntegrator mi(&ir);

   L2.GetProlongationMatrix()->MultTranspose(rho0, rho0_t);
   MultiVector Y{rhs};
   density_mf->Mult(MultiVector{rho0_t, x0}, Y);
   L2.GetProlongationMatrix()->Mult(rhs, rhs_l);

   for (int e = 0; e < L2.GetParMesh()->GetNE(); e++)
   {
      const FiniteElement &fe = *L2.GetFE(e);
      ElementTransformation &eltr = *L2.GetElementTransformation(e);
      L2.GetElementDofs(e, dofs);
      mi.AssembleElementMatrix(fe, eltr, Mrho);
      inv.Factor();
      rhs_l.GetElementDofValues(e, rhs_e);
      inv.Mult(rhs_e, rho_z);
      rho.SetSubVector(dofs, rho_z);
   }
}

real_t LagrangianHydroOperator::InternalEnergy(ParGridFunction &e) const
{
   const auto mt = Device::GetDeviceMemoryType();
   Vector E(L2.GetTrueVSize(), mt), Y(L2.GetTrueVSize(), mt);
   L2.GetRestrictionMatrix()->Mult(e, E);
   MultiVector MY{Y};
   total_internal_energy_mf->Mult(MultiVector{E, rho0, x0}, MY);
   const real_t ie_local = Y.Sum();
   real_t ie_global = 0.0;
   MPI_Allreduce(&ie_local, &ie_global, 1, MPI_DOUBLE, MPI_SUM,
                 L2.GetParMesh()->GetComm());
   return ie_global;
}

real_t LagrangianHydroOperator::KineticEnergy(ParGridFunction &v) const
{
   const auto mt = Device::GetDeviceMemoryType();
   Vector Y(L2.GetTrueVSize(), mt);
   MultiVector MY{Y};
   total_kinetic_energy_mf->Mult(MultiVector{v, rho0, x0}, MY);
   const real_t ke_local = Y.Sum();
   real_t ke_global = 0.0;
   MPI_Allreduce(&ke_local, &ke_global, 1, MPI_DOUBLE, MPI_SUM,
                 H1.GetParMesh()->GetComm());
   return ke_global;
}


void LagrangianHydroOperator::UpdateQuadratureData(const Vector &S) const
{
   if (qdata_is_current) { return; }
   qdata_is_current = true;

   auto [x, v, e] = XVE(S, H1.GetVSize(), L2.GetVSize());
   MultiVector Y{qdata->stressp};
   update_qdata->Mult(MultiVector{v, rho0, x0, x, material, e}, Y);

   // As stock Laghos: the estimate is the min over all the updates
   if (stage_dt_estimate)
   {
      dt_est_stages = fmin(dt_est_stages, LocalTimeStepEstimate(S));
   }
}

// DerivativeOperator::Assemble returns a new matrix, owned by the caller
static std::unique_ptr<HypreParMatrix> AssembleMatrix(DerivativeOperator &d)
{
   HypreParMatrix *A = nullptr;
   d.Assemble(A);
   return std::unique_ptr<HypreParMatrix>(A);
}

void LagrangianHydroOperator::Jacobian::Mult(
   const Vector &u, Vector &y) const
{
   w = u;
   auto [wx, wv, we] = XVE(w, H1tsize, L2tsize);
   auto [zx, zv, ze] = XVE(z, H1tsize, L2tsize);
   MultiVector Zv{zv}, Ze{ze};
   auto [yx, yv, ye] = XVE(y, H1tsize, L2tsize);

   // position
   yx = wv;
   yx *= -h;
   yx += wx;

   // velocity
   dRvdx->Mult(wx, Zv);
   zv *= h;
   yv = zv;
   dRvdv->Mult(wv, Zv);
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

   dRvde->Mult(we, Zv);
   zv *= h;
   yv += zv;
   yv.SetSubVector(hydro.ess_tdof, 0.0);

   // energy
   //                          [ wx ]
   // [ dRe/dx dRe/dv dRe/de ] [ wv ]
   //                          [ we ]
   //

   dRedx->Mult(wx, Ze);
   if (dTaylorSourcedx)
   {
      Vector ts(L2tsize);
      MultiVector Ts{ts};
      dTaylorSourcedx->Mult(wx, Ts);
      ze += ts;
   }
   ze *= -h;
   ye = ze;

   dRedv->Mult(wv, Ze);
   ze *= -h;
   ye += ze;

   dRede->Mult(we, Ze);
   ze *= -h;
   hydro.Me->FullAddMult(we, ze);

   ye += ze;

   yx.SyncAliasMemory(y);
   yv.SyncAliasMemory(y);
   ye.SyncAliasMemory(y);
}

MemoryClass
LagrangianHydroOperator::Jacobian::GetMemoryClass() const
{
   return Device::GetDeviceMemoryClass();
}

void LagrangianHydroOperator::Preconditioner::SetOperator(const Operator &op)
{
   if (!rebuild) { return; }

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
}

void LagrangianHydroOperator::Preconditioner::Mult(const Vector &x,
                                                   Vector &y) const
{
   w = x;
   const int H1tsize = hydro.H1.GetTrueVSize();
   const int L2tsize = hydro.L2.GetTrueVSize();
   auto [wx, wv, we] = XVE(w, H1tsize, L2tsize);
   auto [yx, yv, ye] = XVE(y, H1tsize, L2tsize);

   yx = wx;
   amg_v->Mult(wv, yv);
   amg_e->Mult(we, ye);

   yx.SyncAliasMemory(y);
   yv.SyncAliasMemory(y);
   ye.SyncAliasMemory(y);
}

void LagrangianHydroOperator::Residual::Mult(
   const Vector &k, Vector &R) const
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

   MultiVector RV{Rv};
   hydro.momentum_pa->Mult(MultiVector{hydro.qdata->stressp}, RV);

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

   MultiVector RE{Re};
   hydro.energy_conservation_pa->Mult(MultiVector{uv_l, hydro.qdata->stressp},
                                      RE);

   Re.Neg();

   if (hydro.taylor_source)
   {
      MultiVector ES{e_source_t};
      hydro.taylor_source_mf->Mult(MultiVector{ux}, ES);
      Re -= e_source_t;
   }

   hydro.Me->FullAddMult(ke, Re);

   Rx.SyncAliasMemory(R);
   Rv.SyncAliasMemory(R);
   Re.SyncAliasMemory(R);
}

Operator& LagrangianHydroOperator::Residual::GetGradient(
   const Vector &k) const
{
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
      Vector rho0_t(L2tsize), x0_t(H1tsize),
             material_t(hydro.material.ParFESpace()->GetTrueVSize());
      hydro.L2.GetRestrictionMatrix()->Mult(hydro.rho0, rho0_t);
      hydro.H1.GetRestrictionMatrix()->Mult(hydro.x0, x0_t);
      hydro.material.ParFESpace()->GetRestrictionMatrix()->Mult(hydro.material,
                                                                material_t);
      const MultiVector state{uv, rho0_t, x0_t, ux, material_t, ue};

      auto dRvdx = hydro.momentum_mf->GetDerivative(COORDINATES, state);
      auto dRvdv = hydro.momentum_mf->GetDerivative(VELOCITY, state);
      auto dRvde = hydro.momentum_mf->GetDerivative(SPECIFIC_INTERNAL_ENERGY, state);
      auto dRedx = hydro.energy_conservation_mf->GetDerivative(COORDINATES, state);
      auto dRedv = hydro.energy_conservation_mf->GetDerivative(VELOCITY, state);
      auto dRede =
         hydro.energy_conservation_mf->GetDerivative(SPECIFIC_INTERNAL_ENERGY, state);
      std::shared_ptr<DerivativeOperator> dTaylorSourcedx;
      if (hydro.taylor_source)
      {
         dTaylorSourcedx =
            hydro.taylor_source_mf->GetDerivative(COORDINATES, MultiVector{ux});
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

   newton->Mult(zero, K);

   // If some SDIRK stage didn't converge, keep it at false.
   // (these are reset after every time step).
   if (newton_converged)
   {
      newton_converged = newton->GetConverged();
   }
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

void LagrangianHydroOperator::DensityScatter()
{
   Mesh m0 = H1.GetParMesh()->GetSerialMesh(0);
   GridFunction r0 = rho0.GetSerialGridFunction(0, m0);

   if (Mpi::WorldRank() != 0) { return; }

   std::ofstream fstream_rho;
   fstream_rho.open("rho.out");
   fstream_rho.precision(8);

   const int nqp = ir.GetNPoints();
   Vector pos(2);
   const int NE = m0.GetNE();
   for (int e = 0; e < NE; e++)
   {
      ElementTransformation &Tr = *m0.GetElementTransformation(e);

      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         Tr.Transform(ip, pos);

         double detJ = Tr.Weight();
         // rho0 * detJ0 / detJ.
         double rho = r0.GetValue(Tr, ip)  * (1.0 / NE) / detJ;
         double r = sqrt(pos(0) * pos(0) + pos(1) * pos(1));

         fstream_rho << r << " " << rho << "\n";
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
   auto [x, v, e] = XVE(S, H1.GetVSize(), L2.GetVSize());
   MultiVector Y{qdata->dt_est};
   dtest_mf->Mult(MultiVector{v, rho0, x0, x, material, e}, Y);
   return qdata->dt_est.Min();
}

} // namespace mfem::hydrodynamics
