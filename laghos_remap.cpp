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
#include "laghos_remap_fct.hpp"
#include "laghos_remap_sync.hpp"

using namespace std;
namespace mfem
{

namespace ale
{
#ifdef MFEM_USE_GSLIB
void InterpolationRemap::Remap(const ParGridFunction &source,
                               const ParGridFunction &x_new,
                               ParGridFunction &interpolated)
{
   if (source.ParFESpace()->GetMyRank() == 0)
   {
      std::cout << "--- Interpolation remap of velocity." << std::endl;
   }

   ParMesh &pmesh_src = *source.ParFESpace()->GetParMesh();
   ParFiniteElementSpace &pfes_tgt = *interpolated.ParFESpace();

   const int dim = pmesh_src.Dimension();
   MFEM_VERIFY(dim > 1, "Interpolation remap works only in 2D and 3D.");

   const int NE = pmesh_src.GetNE();
   const int nsp = interpolated.ParFESpace()->GetFE(0)->GetNodes().GetNPoints();
   const int ncomp = source.VectorDim();

   // Generate list of points where the grid function will be evaluated.
   Vector vxyz = x_new;

   // vxyz.SetSize(nsp * NE * dim);
   // for (int e = 0; e < NE; e++)
   // {
   //    const IntegrationRule &ir = pfes_tgt.GetFE(e)->GetNodes();

   //    // Transformation of the element with the new coordinates.
   //    IsoparametricTransformation Tr;
   //    pmesh_src.GetElementTransformation(e, x_new, &Tr);

   //    // Node positions of the interpolated f-n (new element coordinates).
   //    DenseMatrix pos_target_nodes;
   //    Tr.Transform(ir, pos_target_nodes);
   //    Vector rowx(vxyz.GetData() + e*nsp, nsp),
   //           rowy(vxyz.GetData() + e*nsp + NE*nsp, nsp), rowz;
   //    if (dim == 3)
   //    {
   //       rowz.SetDataAndSize(vxyz.GetData() + e*nsp + 2*NE*nsp, nsp);
   //    }
   //    pos_target_nodes.GetRow(0, rowx);
   //    pos_target_nodes.GetRow(1, rowy);
   //    if (dim == 3) { pos_target_nodes.GetRow(2, rowz); }
   // }

   const int nodes_cnt = vxyz.Size() / dim;

   // Evaluate source grid function.
   Vector interp_vals(ncomp * nodes_cnt);
   FindPointsGSLIB finder(pfes_tgt.GetComm());
   finder.Setup(pmesh_src, 1.0);
   finder.SetDistanceToleranceForPointsFoundOnBoundary(0.05);
   finder.SetDefaultInterpolationValue(-1.0);
   finder.FindPoints(vxyz);
   const Array<unsigned int> &codes = finder.GetCode();
   int cnt = 0;
   for (int i = 0; i < nodes_cnt; i++)
   {
      if (codes[i] == 2)
      {
         cnt++;
         cout << vxyz(i) << " " << vxyz(i + nodes_cnt) << endl;
      }
   }
   std::cout << cnt << std::endl;
   MFEM_VERIFY(cnt == 0, "Points not found");
   finder.Interpolate(source, interp_vals);

   interpolated = interp_vals;
}

#endif

RemapAdvector::RemapAdvector(const ParMesh &m, int order_v, int order_e,
                             double cfl, RemapScheme remap_, RemapVelocity remap_v_,
                             bool remap_v_stable_, const Array<int> &ess_tdofs)
    : pmesh(m, true), dim(pmesh.Dimension()),
    fec_L2(order_e, pmesh.Dimension(), BasisType::Positive),
    fec_H1(order_v, pmesh.Dimension(), BasisType::Positive),
    fec_H1Lag(order_v, pmesh.Dimension()),
    pfes_L2(&pmesh, &fec_L2, 1),
    pfes_H1(&pmesh, &fec_H1, pmesh.Dimension()),
    pfes_H1_s(&pmesh, (remap_v_stable_)?(&fec_H1):(&fec_H1Lag)),
    pfes_H1Lag(&pmesh, &fec_H1Lag, pmesh.Dimension()),
    v_ess_tdofs(ess_tdofs),
    remap_scheme(remap_),
    remap_v(remap_v_), remap_v_stable(remap_v_stable_),
    cfl_factor(cfl),
    offsets(), S(), v(), rho(), e(), x0()
{
   const int vsize_H1 = pfes_H1.GetVSize(), vsize_L2 = pfes_L2.GetVSize();

   // Arrangement: velocity (dim), density (1), energy (1).
   offsets.SetSize(NVars+1);
   offsets = 0;
   offsets[Velocity+1] = vsize_H1;
   offsets[Density+1] = vsize_L2;
   offsets[Energy+1] = vsize_L2;
   offsets.PartialSum();
   S.Update(offsets);

   if (remap_v_stable)
   {
      v.MakeRef(&pfes_H1, S, offsets[Velocity]);
   }
   else
   {
      v.MakeRef(&pfes_H1Lag, S, offsets[Velocity]);
   }
   rho.MakeRef(&pfes_L2, S, offsets[Density]);
   e.MakeRef(&pfes_L2, S, offsets[Energy]);

   switch (remap_scheme)
   {
   case RemapScheme::Nonconservative:
      ode_solver = make_unique<RK3SSPSolver>();
      break;
   case RemapScheme::GeomConsistent:
      ode_solver_gc = make_unique<geom_consistent_solvers::ForwardEulerSolver>();
      break;
   }
}

void RemapAdvector::InitFromLagr(const Vector &nodes0,
                                 const ParGridFunction &vel,
                                 const IntegrationRule &rho_ir,
                                 const Vector &rhoDetJw,
                                 const ParGridFunction &lagr_eps)
{
   // Save the integration rule
   ir_rho = &rho_ir;

   // Original positions of the local mesh.
   x0 = nodes0;
   GridFunction *x = pmesh.GetNodes();
   *x = x0;

   // Velocity
   SolutionTransfer_H1 transfer_h1(v_ess_tdofs, pfes_H1_s, rho_ir);
   switch (remap_scheme)
   {
   case RemapScheme::Nonconservative:
      if (remap_v_stable)
      {
         transfer_h1.TransferVelocity_Lagr2Remap(vel, v);
      }
      else { v = vel; }
      break;
   case RemapScheme::GeomConsistent:
      transfer_h1.TransferMomentumJac_Lagr2Remap(rhoDetJw, vel, v);
      break;
   }
   

   // Thermodynamic quantities
   SolutionTransfer_L2 transfer_l2(pfes_L2, rho_ir);

   switch (remap_scheme)
   {
   case RemapScheme::Nonconservative:
      transfer_l2.TransferDensity_Lagr2Remap(rhoDetJw, rho);
      e  = lagr_eps;
      break;
   case RemapScheme::GeomConsistent:
      detJ.SetSpace(rho.ParFESpace()); detJ = 0.;
      transfer_l2.TransferJac_Larg2Remap(detJ);
      transfer_l2.TransferDensityJac_Lagr2Remap(rhoDetJw, detJ, rho);
      transfer_l2.TransferEnergyJac_Lagr2Remap(rhoDetJw, rho, lagr_eps, e);
      break;
   }
}

void RemapAdvector::ComputeAtNewPosition(const Vector &new_nodes,
                                         const Array<int> &ess_tdofs,
                                         const Array<int> &ess_vdofs)
{
   const int vsize_H1 = pfes_H1.GetVSize();

   // This will be used to move the positions.
   GridFunction *x = pmesh.GetNodes();
   *x = x0;

   // Velocity of the positions.
   ParGridFunction u(&pfes_H1Lag);
   subtract(new_nodes, x0, u);

   ParFiniteElementSpace *pfes_H1_v;
   AdvectorOper *oper;

   if (remap_v_stable)
   {
      // Bernstein scalar space.
      // Scalar space only used when velocity remap with limiter
      pfes_H1_v = &pfes_H1;
   }
   else
   {
      // Define scalar FE spaces for the solution, and the advection operator.
      pfes_H1_v = &pfes_H1Lag;
   }

   ODESolver *ode;
   switch (remap_scheme)
   {
   case RemapScheme::Nonconservative:
   {
      oper = new AdvectorNonconservativeOper(
         x0, ess_tdofs, ess_vdofs, u, rho, *ir_rho,
         *pfes_H1_v, pfes_H1_s, pfes_L2,
         remap_v, remap_v_stable);
      ode_solver->Init(*oper);
      ode = ode_solver.get();
      break;
   }
   case RemapScheme::GeomConsistent:
   {
      auto *op = new AdvectorGeomConsOper(
         x0, ess_tdofs, ess_vdofs, u, rho, *ir_rho,
         *pfes_H1_v, pfes_H1_s, pfes_L2,
         remap_v, remap_v_stable);
      ode_solver_gc->Init(*op);
      oper = op;
      ode = ode_solver_gc.get();
      break;
   }
   }

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
   MPI_Allreduce(&v_loc, &u_max, 1, MPI_DOUBLE, MPI_MAX, pfes_H1Lag.GetComm());
   MPI_Allreduce(&h_loc, &h_min, 1, MPI_DOUBLE, MPI_MIN, pfes_H1Lag.GetComm());

   if (u_max == 0.0) { return; } // No need to change the fields.

   u_max = std::sqrt(u_max);
   double dt = cfl_factor * h_min / u_max;

   char vishost[] = "localhost";
   int  visport   = 19916;
   int Wx = 0, Wy = 0; // window position
   int Ww = 350, Wh = 350; // window size
   int offx = Ww+10; // window offsets x
   int offy = Ww+100; // window offsets y
   Wx += offx;
   Wy += offy;

   double t = 0.0;
   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= 1.0)
      {
         dt = 1.0 - t;
         last_step = true;
      }

      const real_t mass = oper->Mass(rho, t);
      const real_t momentum = oper->Momentum(v, t);
      const real_t energy = oper->InternalEnergy(e, t);

      if (pmesh.GetMyRank() == 0)
      {
         cout << ". " << ti
              << " mass: " << mass
              << " momentum: " << momentum
              << " energy: " << energy
              << std::endl;
      }

      oper->SetDt(dt);
      ode->Step(S, t, dt);

      hydrodynamics::VisualizeField(vis_rho, vishost, visport, rho,
                                    "Remapped Density", Wx, Wy, Ww, Wh);
      Wx += offx;
      if (remap_v != RemapVelocity::None)
      {
         hydrodynamics::VisualizeField(vis_v, vishost, visport, v,
                                       "Remapped Velocity", Wx, Wy, Ww, Wh);
         Wx += offx;
      }
      hydrodynamics::VisualizeField(vis_e, vishost, visport, e,
                                    "Remapped Energy", Wx, Wy, Ww, Wh);
   }

   delete oper;
}

void RemapAdvector::TransferToLagr(ParGridFunction &rho0_gf,
                                   ParGridFunction &vel,
                                   const IntegrationRule &ir_rho,
                                   Vector &rhoDetJw,
                                   const IntegrationRule &ir_rho_b,
                                   Vector &rhoDetJ_be,
                                   ParGridFunction &lagr_eps)
{
   // Thermodynamic quantities
   SolutionTransfer_L2 transfer_l2(pfes_L2, ir_rho);

   // Density
   switch (remap_scheme)
   {
   case RemapScheme::Nonconservative:
      // This is used to update the mass matrices.
      rho0_gf = rho;
      // Just copy energy.
      lagr_eps = e;
      break;
   case RemapScheme::GeomConsistent:
      transfer_l2.TransferJac_Larg2Remap(detJ);
      transfer_l2.TransferDensityJac_Remap2Lagr(detJ, rho, rho0_gf);
      break;
   }

   // Update the quadrature of the Lagrangian invariant

   const int NE  = pmesh.GetNE();
   const int nqp = ir_rho.GetNPoints();

   Vector rho_vals(nqp);
   for (int k = 0; k < NE; k++)
   {
      // Must use the space of the results.
      ElementTransformation &T = *pmesh.GetElementTransformation(k);
      rho0_gf.GetValues(T, ir_rho, rho_vals);
      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir_rho.IntPoint(q);
         T.SetIntPoint(&ip);
         rhoDetJw(k*nqp + q) = rho_vals(q) * T.Weight() * ip.weight;
      }
   }

   // const int NBE = pmesh_lagr.GetNBE();
   // for (int be = 0; be < NBE; be++)
   // {
   //    int b_nqp = ir_rho_b.GetNPoints();
   //    auto b_face_tr = pmesh_lagr.GetBdrFaceTransformations(be);
   //    if (b_face_tr == nullptr) { continue; }
   //    for (int q = 0; q < b_nqp; q++)
   //    {
   //       const IntegrationPoint &ip_f = ir_rho_b.IntPoint(q);
   //       b_face_tr->SetAllIntPoints(&ip_f);
   //       ElementTransformation &tr_el = b_face_tr->GetElement1Transformation();
   //       double detJ = tr_el.Weight();
   //       MFEM_VERIFY(detJ > 0, "Negative detJ at a face! " << detJ);
   //       rhoDetJ_be(be * b_nqp + q) = detJ * rho0_gf.GetValue(tr_el);
   //    }
   // }

   // Velocity
   SolutionTransfer_H1 transfer_h1(v_ess_tdofs, pfes_H1_s, ir_rho);
   switch (remap_scheme)
   {
   case RemapScheme::Nonconservative:
      if (remap_v_stable)
      {
         transfer_h1.TransferVelocity_Remap2Lagr(v, vel);
      }
      else
      {
         // just copy velocity otherwise
         vel = v;
      }
      break;
   case RemapScheme::GeomConsistent:
      transfer_h1.TransferMomentumJac_Remap2Lagr(rhoDetJw, v, vel);
      break;
   }

   // Energy
   switch (remap_scheme)
   {
   case RemapScheme::Nonconservative:
      // Just copy energy.
      lagr_eps = e;
      break;
   case RemapScheme::GeomConsistent:
      transfer_l2.TransferEnergyJac_Remap2Lagr(rhoDetJw, rho, e, lagr_eps);
      break;
   }
}

AdvectorOper::AdvectorOper(const Vector &x_start,
                           ParGridFunction &mesh_vel,
                           ParFiniteElementSpace &pfes_H1,
                           ParFiniteElementSpace &pfes_L2)
: x0(x_start), x_now(*pfes_H1.GetMesh()->GetNodes()),
  u(mesh_vel)
{
   // Construct offsets
   const int vdofs_h1 = pfes_H1.GetVSize();
   const int vdofs_l2 = pfes_L2.GetVSize();
   offsets.SetSize(RemapAdvector::NVars+1);
   offsets = 0;
   offsets[RemapAdvector::Velocity+1] = vdofs_h1; // velocity
   offsets[RemapAdvector::Density+1] = vdofs_l2; // density
   offsets[RemapAdvector::Energy+1] = vdofs_l2; // energy
   offsets.PartialSum();

   width = height = offsets.Last();
}

void AdvectorOper::SetDt(real_t delta_t)
{
   if (op_th) { op_th->SetDt(delta_t); }
}

void AdvectorOper::SetTime(real_t t)
{
   TimeDependentOperator::SetTime(t);
   if (op_v) { op_v->SetTime(t); }
   if (op_th) { op_th->SetTime(t); }
}

real_t AdvectorOper::Momentum(ParGridFunction &v, real_t t)
{
   add(x0, t, u, x_now);

   if (op_v) { return op_v->Momentum(v); }
   return 0.;
}

real_t AdvectorOper::Mass(ParGridFunction &rho, real_t t)
{
   add(x0, t, u, x_now);
   if (op_th) { return op_th->Mass(rho); }
   return 0.;
}

real_t AdvectorOper::InternalEnergy(ParGridFunction &e, real_t t)
{
   add(x0, t, u, x_now);
   if (op_th) { return op_th->InternalEnergy(e); }
   return 0.;
}

AdvectorNonconservativeOper::AdvectorNonconservativeOper(
   const Vector &x_start,
   const Array<int> &v_ess_td,
   const Array<int> &v_ess_vd,
   ParGridFunction &mesh_vel,
   ParGridFunction &rho,
   const IntegrationRule &ir_rho,
   ParFiniteElementSpace &pfes_H1,
   ParFiniteElementSpace &pfes_H1_s,
   ParFiniteElementSpace &pfes_L2,
   RemapAdvector::RemapVelocity remap_v,
   bool remap_v_s)
  : AdvectorOper(x_start, mesh_vel, pfes_H1, pfes_L2),
    u_coeff(&u),
    rho_coeff(&rho)
{
   // Velocity advector
   if (remap_v != RemapAdvector::RemapVelocity::None)
      op_v = make_unique<AdvectorVelocityNonconservativeOper>(
         v_ess_td, v_ess_vd, rho_coeff, u_coeff,
         pfes_H1, pfes_H1_s, remap_v, remap_v_s);

   // In parallel, the assembly of Kr_L2 needs to see values from MPI-neighbors.
   // That is, the rho_coeff must be evaluated in MPI-neighbor zones.
   rho.ExchangeFaceNbrData();

   op_th = make_unique<AdvectorThermoNonconservativeOper>(
      rho_coeff, u_coeff, pfes_L2);   
}

void AdvectorNonconservativeOper::Mult(const Vector &U, Vector &dU) const
{
   // Move the mesh.
   const double t = GetTime();
   add(x0, t, u, x_now);

   dU = 0.0;

   // Block view
   const BlockVector bU(const_cast<Vector&>(U), offsets);
   BlockVector bdU(dU, offsets);

   // Velocity remap.
   if (op_v)
   {
      const Vector &v = bU.GetBlock(RemapAdvector::Velocity);
      Vector &d_v = bdU.GetBlock(RemapAdvector::Velocity);
      op_v->Mult(v, d_v);
   }

   // In parallel, rho_coeff must be evaluated in MPI-neighbor zones.
   auto rho_gf_const = dynamic_cast<const ParGridFunction *>
                       (rho_coeff.GetGridFunction());
   auto rho_pgf = const_cast<ParGridFunction *>(rho_gf_const);
   rho_pgf->ExchangeFaceNbrData();

   // Thermodynamic remap.
   if (op_th)
   {
      // Here we assume the thermodynamic state is in one piece
      const Vector th(const_cast<Vector&>(U), offsets[RemapAdvector::Density], op_th->Width());
      Vector dth(dU, offsets[RemapAdvector::Density], op_th->Width());
      op_th->Mult(th, dth);
   }
}

AdvectorGeomConsOper::AdvectorGeomConsOper(
   const Vector &x_start,
   const Array<int> &v_ess_td,
   const Array<int> &v_ess_vd,
   ParGridFunction &mesh_vel,
   ParGridFunction &rho,
   const IntegrationRule &ir_rho_,
   ParFiniteElementSpace &pfes_H1,
   ParFiniteElementSpace &pfes_H1_s,
   ParFiniteElementSpace &pfes_L2,
   RemapAdvector::RemapVelocity remap_v,
   bool remap_v_s)
  : AdvectorOper(x_start, mesh_vel, pfes_H1, pfes_L2),
    u_coeff(&u),
    rho_coeff(&rho),
    ir_rho(ir_rho_),
    fec_f(pfes_L2.FEColl()->GetOrder(), pfes_L2.GetParMesh()->Dimension()),
    fec_a(pfes_L2.FEColl()->GetOrder()+1, pfes_L2.GetParMesh()->Dimension()),
    pfes_f(pfes_L2.GetParMesh(), &fec_f),
    pfes_a(pfes_L2.GetParMesh(), &fec_a),
    DD(&pfes_f), CC(&pfes_a)
{
   // Velocity advector
   if (remap_v != RemapAdvector::RemapVelocity::None)
      op_v = make_unique<AdvectorVelocityGeomConsOper>(
         ir_rho, v_ess_td, v_ess_vd, pfes_H1, pfes_H1_s, remap_v, remap_v_s);

   // In parallel, the assembly of Kr_L2 needs to see values from MPI-neighbors.
   // That is, the rho_coeff must be evaluated in MPI-neighbor zones.
   rho.ExchangeFaceNbrData();

   op_th = make_unique<AdvectorThermoGeomConsOper>(ir_rho, pfes_L2);

   // Reference space divdiv form
   DD.AddDomainIntegrator(new DivRDivRIntegrator());
   DD.Assemble();
   DD.Finalize();
   DD.ParallelAssembleInternalMatrix();

   const ParMesh *pmesh = pfes_f.GetParMesh();
   ess_bdr.SetSize(pmesh->bdr_attributes.Size() > 0 ? pmesh->bdr_attributes.Max() : 1);
   ess_bdr = -1;
   pfes_f.GetEssentialTrueDofs(ess_bdr, ess_tdofs_f);
   DD.ParallelEliminateTDofs(ess_tdofs_f);

   // Solenoidal potential matrix
   CC.AddDomainIntegrator(new DiffusionIntegrator());
   CC.Assemble(0);
   CC.Finalize(0);
}

void AdvectorGeomConsOper::MultConserv(const ParGridFunction &flux, const Vector &U, Vector &dU) const
{
   // Move the mesh.
   const double t = GetTime();
   add(x0, t, u, x_now);

   dU = 0.0;

   // Block view
   const BlockVector bU(const_cast<Vector&>(U), offsets);
   BlockVector bdU(dU, offsets);

   // Velocity remap.
   if (op_v)
   {
      const Vector &v = bU.GetBlock(RemapAdvector::Velocity);
      Vector &d_v = bdU.GetBlock(RemapAdvector::Velocity);
      auto *gcop_v = static_cast<AdvectorVelocityGeomConsOper*>(op_v.get());
      gcop_v->MultConserv(flux, v, d_v);
   }

   // In parallel, rho_coeff must be evaluated in MPI-neighbor zones.
   auto rho_gf_const = dynamic_cast<const ParGridFunction *>
                       (rho_coeff.GetGridFunction());
   auto rho_pgf = const_cast<ParGridFunction *>(rho_gf_const);
   rho_pgf->ExchangeFaceNbrData();

   // Thermodynamic remap.
   if (op_th)
   {
      // Here we assume the thermodynamic state is in one piece
      const Vector th(const_cast<Vector&>(U), offsets[RemapAdvector::Density], op_th->Width());
      Vector dth(dU, offsets[RemapAdvector::Density], op_th->Width());
      auto *gcop_th = static_cast<AdvectorThermoGeomConsOper*>(op_th.get());
      gcop_th->MultConserv(flux, th, dth);
   }
}

void AdvectorGeomConsOper::LimitUpdate(real_t dt, const Vector &U, Vector &dU)
{
   // Move the mesh.
   const double t = GetTime();
   add(x0, t+dt, u, x_now);

   // Thermodynamic limiting.
   if (op_th)
   {
      // Here we assume the thermodynamic state is in one piece
      const Vector th(const_cast<Vector&>(U), offsets[RemapAdvector::Density], op_th->Width());
      Vector dth(dU, offsets[RemapAdvector::Density], op_th->Width());
      auto *gcop_th = static_cast<AdvectorThermoGeomConsOper*>(op_th.get());
      gcop_th->LimitUpdate(dt, th, dth);
   }
}

void AdvectorVelocityOper::LowOrderVel(const SparseMatrix &K_glb, const SparseMatrix &KT_glb, const Vector &v, Vector &d_v) const
{
   GroupCommunicator &gcomm = lummpedMr_H1.ParFESpace()->GroupComm();

   ParFiniteElementSpace &pfes_H1_s = *Kr_H1.ParFESpace();
   const int dim     = pfes_H1_s.GetMesh()->Dimension();
   const int dofs_h1 = pfes_H1_s.GetVSize();

   d_v = 0.0;
   Array<double> rhs_array(dofs_h1);
   HypreParVector v_d_hpr(&pfes_H1_s);

   const auto I = K_glb.ReadI();
   const auto J = K_glb.ReadJ();
   const auto K = K_glb.ReadData();
   const auto KT = KT_glb.ReadData();

   for(int d = 0; d < dim; d++)
   {
      const Vector v_d(const_cast<Vector&>(v), d * dofs_h1, dofs_h1);
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

            double dij = max( 0.0, max(-K[k], -KT[k]));
            //dij = max( abs(K[k]), abs (KT[k]));
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


void AdvectorVelocityOper::HighOrderTargetSchemeVel(const SparseMatrix &K_glb, const SparseMatrix &KT_glb, const SparseMatrix &M_glb, const Vector &v, Vector &d_v) const
{
   GroupCommunicator &gcomm = lummpedMr_H1.ParFESpace()->GroupComm();
   //Array<double> lumpedmassmatrix_array(lumpedMr_H1_vec.GetData(), lumpedMr_H1_vec.Size());
   //gcomm.Reduce<double>(lumpedmassmatrix_array, GroupCommunicator::Sum);
   //gcomm.Bcast(lumpedmassmatrix_array);

   ParFiniteElementSpace &pfes_H1_s = *Kr_H1.ParFESpace();
   const int dim     = pfes_H1_s.GetMesh()->Dimension();
   const int dofs_h1 = pfes_H1_s.GetVSize();

   d_v = 0.0;
   Array<double> rhs_array(dofs_h1), udot_array(dofs_h1);
   HypreParVector v_d_hpr(&pfes_H1_s), vdot(&pfes_H1_s);

   const auto I = K_glb.ReadI();
   const auto J = K_glb.ReadJ();
   const auto K = K_glb.ReadData();
   const auto KT = KT_glb.ReadData();
   const auto M = M_glb.ReadData();

   for(int d = 0; d < dim; d++)
   {
      const Vector v_d(const_cast<Vector&>(v), d * dofs_h1, dofs_h1);
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

            double dij = max( 0.0, max(-K[k], -KT[k]));
            //dij = max( abs(K[k]), abs(KT[k]));
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

void AdvectorVelocityOper::MCLVel(const SparseMatrix &K_glb, const SparseMatrix &KT_glb, const SparseMatrix &M_glb, const Vector &v, Vector &d_v) const
{
   GroupCommunicator &gcomm = lummpedMr_H1.ParFESpace()->GroupComm();
   //Array<double> lumpedmassmatrix_array(lumpedMr_H1_vec.GetData(), lumpedMr_H1_vec.Size());
   //gcomm.Reduce<double>(lumpedmassmatrix_array, GroupCommunicator::Sum);
   //gcomm.Bcast(lumpedmassmatrix_array);

   ParFiniteElementSpace &pfes_H1_s = *Kr_H1.ParFESpace();
   const int dim     = pfes_H1_s.GetMesh()->Dimension();
   const int dofs_h1 = pfes_H1_s.GetVSize();

   d_v = 0.0;
   Array<double> rhs_array(dofs_h1), udot_array(dofs_h1);
   HypreParVector v_d_hpr(&pfes_H1_s), vdot(&pfes_H1_s), v_min(&pfes_H1_s), v_max(&pfes_H1_s);
   double fij, fij_bound, fij_star, wij, wji;

   const auto I = K_glb.ReadI();
   const auto J = K_glb.ReadJ();
   const auto K = K_glb.ReadData();
   const auto KT = KT_glb.ReadData();
   const auto M = M_glb.ReadData();

   for(int d = 0; d < dim; d++)
   {
      const Vector v_d(const_cast<Vector&>(v), d * dofs_h1, dofs_h1);
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
            if( i_gl == j_gl) {continue;}// || is_global_ess_dof[j_gl + d * pfes_H1_s.GlobalTrueVSize()] )
            v_min(i_td) = min(v_min(i_td), v_d_glb->Elem(j_gl));
            v_max(i_td) = max(v_max(i_td), v_d_glb->Elem(j_gl));
            double kij = -K[k];
            double kji = -KT[k];// * (!is_global_ess_dof[j_gl + d * pfes_H1_s.GlobalTrueVSize()]);
            //double dij = max(max(0.0, kij), kji);
            double dij = max( abs(kij), abs(kji));
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
            if( i_gl == j_gl)// || is_global_ess_dof[j_gl + d * pfes_H1_s.GlobalTrueVSize()])
            {continue;}

            double kij = -K[k];
            double kji = - KT[k];// * (!is_global_ess_dof[j_gl + d * pfes_H1_s.GlobalTrueVSize()]);

            //double dij = max(max(0.0,kji),kij);
            //dij = max( abs(K[k]), abs(KT[k]));
            double dij = max( abs(kij), abs(kji));
            fij = M[k] * (vdot_glb->Elem(i_gl) - vdot_glb->Elem(j_gl)) + dij * (v_d_glb->Elem(i_gl) - v_d_glb->Elem(j_gl));

            //limit target flux to enforce local bounds for the bar states (note, that dij = dji)
            wij = dij * (v_d_glb->Elem(i_gl) + v_d_glb->Elem(j_gl))  + K[k] * (v_d_glb->Elem(j_gl) - v_d_glb->Elem(i_gl));
            wji = dij * (v_d_glb->Elem(i_gl) + v_d_glb->Elem(j_gl))  + KT[k]  * (v_d_glb->Elem(i_gl) - v_d_glb->Elem(j_gl));

            //KT_glb(i_td, j_gl)
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

void AdvectorVelocityNonconservativeOper::ClipAndScale(
   const ParFiniteElementSpace &pfes, const Vector &v, Vector &d_v) const
{
   d_v = 0.0;
   auto conv_int = new ConvectionIntegrator(rho_u_coeff);
   auto mass_int = new MassIntegrator(rho_coeff);

   // scalar finite element space
   const int nEl = pfes.GetNE();
   const int dim = pfes.GetMesh()->Dimension();
   const int nDofs = pfes.GetVSize();
   Array<int> dofs;
   Vector ve;

   Array<double> v_max, v_min;
   ComputeVelocityMinMax(v, v_min, v_max);
   Vector vdot(v.Size());
   ComputeTimeDerivatives(v, conv_int, pfes, vdot);

   for(int e = 0; e < nEl; e++)
   {
      auto element = pfes.GetFE(e);
      auto eltrans = pfes.GetElementTransformation(e);
      DenseMatrix Ke, Me;
      conv_int->AssembleElementMatrix (*element, *eltrans, Ke);
      mass_int->AssembleElementMatrix (*element, *eltrans, Me);

      pfes.GetElementDofs(e, dofs);
      Vector re(dofs.Size()), vdote(dofs.Size()), fe(dofs.Size()),
          gamma_e(dofs.Size()), fe_star(dofs.Size());

      Vector me(Me.Height());
      lumpedMr_H1_vec.GetSubVector(dofs, me);

      MFEM_VERIFY(Me.Height()==dofs.Size(), "element dof sizes weird1");
      MFEM_VERIFY(Me.Width()==dofs.Size(), "element dof sizes weird2");

      for(int d = 0; d < dim; d++)
      {
         Array<int> dofs_d = dofs;
         for(int i = 0; i < dofs.Size(); i++)
         {
            dofs_d[i] = dofs[i] + d * nDofs;
         }
         v.GetSubVector(dofs_d, ve);
         vdot.GetSubVector(dofs_d, vdote);

         Ke.Mult(ve, re);

         for(int i = 0; i < dofs.Size(); i++ )
         {
            for(int j = 0; j < dofs.Size(); j++)
            {
               if(j >= i) { continue;}
               double dije = max(max(-Ke(i,j), -Ke(j,i)), 0.0);
               //dije = max(abs(Ke(i,j)), abs (Ke(j,i)));
               double diffusion = dije * (ve(j) - ve(i));

               re(i) += diffusion;
               re(j) -= diffusion;
            }
         }

         fe = 0.0;
         gamma_e = 0.0;
         // compute raw antidiffusive fluxes
         for(int i = 0; i < dofs.Size(); i++)
         {
            for(int j = 0; j < dofs.Size(); j++)
            {
               if(j >= i) {continue;}
               double dije = max(max(-Ke(i,j), -Ke(j,i)), 0.0);
               //dije = max(abs(Ke(i,j)), abs (Ke(j,i)));
               double fije = dije * (ve(i) - ve(j)) + Me(i,j) * (vdote(i) - vdote(j));
               fe(i) += fije;
               fe(j) -= fije;

               gamma_e(i) += dije;
               gamma_e(j) += dije;
            }
         }
         MFEM_VERIFY(abs(fe.Sum()) < 1e-15, "raw antidiff fluxes.." );

         gamma_e *= 2.0;

         double P_plus = 0.0;
         double P_minus = 0.0;
         fe_star = 0.0;
         // clip
         for(int i = 0; i < dofs.Size(); i++)
         {
            double fie_max = gamma_e(i) * (v_max[dofs_d[i]] - ve(i));
            double fie_min = gamma_e(i) * (v_min[dofs_d[i]] - ve(i));

            fe_star(i) = min(max(fie_min, fe(i)), fie_max);

            P_plus += max(fe_star(i), 0.0);
            P_minus += min(fe_star(i), 0.0);
         }
         const double P = P_minus + P_plus;

         //scale
         for(int i = 0; i < dofs.Size(); i++)
         {
            if(fe_star(i) > 1e-15 && P > 1e-15)
            {
               fe_star(i) *= - P_minus / P_plus;
            }
            else if(fe_star(i) < -1e-15 && P < -1e-15)
            {
               fe_star(i) *= - P_plus / P_minus;
            }
         }
         //MFEM_VERIFY(abs(fe_star.Sum()) < 1e-14, "Scale?" );

         for(int i = 0; i < dofs.Size(); i++)
         {
            d_v(dofs_d[i]) += re(i) + fe_star(i);

            // HO stabilized sltn.
            //d_v(dofs_d[i]) += re(i) + fe(i);

            // LO stabilized sltn.
            //d_v(dofs_d[i]) += re(i);
         }
      }
   }

   GroupCommunicator &gcomm = Mr_H1.ParFESpace()->GroupComm();
   Array<double> dv_array(d_v.GetData(), d_v.Size());
   gcomm.Reduce<double>(dv_array, GroupCommunicator::Sum);
   gcomm.Bcast(dv_array);

   Vector dv_comp;
   for(int d = 0; d < dim; d++)
   {
      dv_comp.MakeRef(d_v, d * nDofs, nDofs);
      dv_comp /= lumpedMr_H1_vec;
   }
   d_v.SetSubVector(v_ess_vdofs, 0.0);

   delete conv_int;
   delete mass_int;
}

void AdvectorVelocityOper::ComputeTimeDerivatives(const Vector &v, ConvectionIntegrator* conv_int, const ParFiniteElementSpace &pfes, Vector &vdot) const
{
   vdot = 0.0;

   // scalar finite element space
   const int nEl = pfes.GetNE();
   const int dim = pfes.GetMesh()->Dimension();
   const int nDofs = pfes.GetVSize();
   Array<int> dofs;
   Vector ve;

   for(int e = 0; e < nEl; e++)
   {
      auto element = pfes.GetFE(e);
      auto eltrans = pfes.GetElementTransformation(e);
      DenseMatrix Ke;
      conv_int->AssembleElementMatrix (*element, *eltrans, Ke);

      pfes.GetElementDofs(e, dofs);
      Vector vdote(dofs.Size());

      for(int d = 0; d < dim; d++)
      {
         Array<int> dofs_d = dofs;
         for(int i = 0; i < dofs.Size(); i++)
         {
            dofs_d[i] = dofs[i] + d * nDofs;
         }
         v.GetSubVector(dofs_d, ve);

         Ke.Mult(ve, vdote);

         for(int i = 0; i < dofs.Size(); i++ )
         {
            for(int j = 0; j < dofs.Size(); j++)
            {
               if(j >= i) { continue;}
               double dije = max(max(-Ke(i,j), -Ke(j,i)), 0.0);
               //dije = max(abs(Ke(i,j)), abs (Ke(j,i)));
               double diffusion = dije * (ve(j) - ve(i));

               vdote(i) += diffusion;
               vdote(j) -= diffusion;
            }
         }

         for(int i = 0; i < dofs.Size(); i++)
         {
            vdot(dofs_d[i]) += vdote(i);
         }
      }
   }

   GroupCommunicator &gcomm = Mr_H1.ParFESpace()->GroupComm();
   Array<double> vdot_array(vdot.GetData(), vdot.Size());
   gcomm.Reduce<double>(vdot_array, GroupCommunicator::Sum);
   gcomm.Bcast(vdot_array);

   Vector vdot_comp;
   for(int d = 0; d < dim; d++)
   {
      vdot_comp.MakeRef(vdot, d * nDofs, nDofs);
      vdot_comp /= lumpedMr_H1_vec;
   }
}

void AdvectorVelocityOper::ComputeVelocityMinMax(const Vector &v, Array<double> &v_min, Array<double> &v_max) const
{
   v_min.SetSize(v.Size());
   v_max.SetSize(v.Size());

   auto I = Mr_H1_s.SpMat().GetI();
   auto J = Mr_H1_s.SpMat().GetJ();
   int nDofs = Mr_H1_s.Height();
   int dim = Mr_H1_s.ParFESpace()->GetMesh()->Dimension();

   Vector v_comp(nDofs);
   for(int d = 0; d < dim; d++)
   {
      v_comp.SetDataAndSize(v.GetData() + d * nDofs, nDofs);

      for(int i = 0; i < nDofs; i++)
      {
         //cout << i << endl;
         //cout << "here" << endl;
         v_max[i + d * nDofs] = v_comp(i);
         v_min[i + d * nDofs] = v_comp(i);
         for(int k = I[i]; k < I[i+1]; k++)
         {
            int j = J[k];

            v_max[i + d * nDofs] = max(v_max[i + d * nDofs], v_comp(j));
            v_min[i + d * nDofs] = min(v_min[i + d * nDofs], v_comp(j));

         }
      }
   }

   GroupCommunicator &gcomm = Mr_H1.ParFESpace()->GroupComm();
   gcomm.Reduce<double>(v_max, GroupCommunicator::Max);
   gcomm.Bcast(v_max);
   gcomm.Reduce<double>(v_min, GroupCommunicator::Min);
   gcomm.Bcast(v_min);

}

void AdvectorThermoOper::ComputeElementsMinMax(
   const Vector &u, Vector &u_min, Vector &u_max,
   const Array<bool> *active_el, const Array<bool> *active_dof) const
{
   const int NE = pfes_L2.GetNE(), ndof = pfes_L2.GetFE(0)->GetDof();
   int dof_id;
   u.HostRead(); u_min.HostReadWrite(); u_max.HostReadWrite();
   for (int k = 0; k < NE; k++)
   {
      u_min(k) = numeric_limits<double>::infinity();
      u_max(k) = -numeric_limits<double>::infinity();

      // Inactive elements don't affect the bounds.
      if (active_el && (*active_el)[k] == false) { continue; }

      for (int i = 0; i < ndof; i++)
      {
         dof_id = k*ndof + i;
         // Inactive dofs don't affect the bounds.
         if (active_dof && (*active_dof)[dof_id] == false) { continue; }

         u_min(k) = min(u_min(k), u(dof_id));
         u_max(k) = max(u_max(k), u(dof_id));
      }
   }
}

void AdvectorThermoOper::ComputeSparsityBounds(
   const ParFiniteElementSpace &pfes, const Vector &el_min, const Vector &el_max,
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

AdvectorVelocityOper::AdvectorVelocityOper(
   const Array<int> &v_ess_td, const Array<int> &v_ess_vd,
   ParFiniteElementSpace &pfes_H1_, ParFiniteElementSpace &pfes_H1_s_,
   RemapAdvector::RemapVelocity scheme_, bool remap_v_s)
:   remap_v(scheme_),
    remap_v_stable(remap_v_s),
    v_ess_tdofs(v_ess_td),
    v_ess_vdofs(v_ess_vd),
    pfes_H1(pfes_H1_), pfes_H1_s(pfes_H1_s_),
    Mr_H1(&pfes_H1), Mr_H1_s(&pfes_H1_s), Kr_H1(&pfes_H1_s), KrT_H1(&pfes_H1_s),
    lummpedMr_H1(&pfes_H1_s)
{
}

AdvectorVelocityNonconservativeOper::AdvectorVelocityNonconservativeOper(
   const Array<int> &v_ess_td, const Array<int> &v_ess_vd, Coefficient &rho_coeff_,
   VectorCoefficient &u_coeff_, ParFiniteElementSpace &pfes_H1, ParFiniteElementSpace &pfes_H1_s,
   RemapAdvector::RemapVelocity scheme_, bool remap_v_s)
:   AdvectorVelocityOper(v_ess_td, v_ess_vd, pfes_H1, pfes_H1_s, scheme_, remap_v_s),
    rho_coeff(rho_coeff_), u_coeff(u_coeff_),
    rho_u_coeff(rho_coeff, u_coeff)
{
   // no need for Vector Massmatrix in stablised velocity remap
   // MCL only uses the first component of this, but unstable remap needs vector mass matrix
   if (remap_v_stable)
   {
      Mr_H1_s.AddDomainIntegrator(new MassIntegrator(rho_coeff));
      Mr_H1_s.Assemble(0);
      Mr_H1_s.Finalize(0);

      // lumped Massmatrix only needed for limiter
      //lumped_mass_int = new LumpedIntegrator(new MassIntegrator(rho_coeff));
      lummpedMr_H1.AddDomainIntegrator( new LumpedIntegrator(new MassIntegrator(rho_coeff)));
      lummpedMr_H1.Assemble(0);
      lummpedMr_H1.Finalize(0);

      // to get the transposed entries, which we don't have acces for when j is a tdof an another core.
      // only needed for limiter
      KrT_H1.AddDomainIntegrator(new TransposeIntegrator(new ConvectionIntegrator(rho_u_coeff)));
      KrT_H1.Assemble(0);
      KrT_H1.Finalize(0);
   }
   else
   {
      Mr_H1.AddDomainIntegrator(new VectorMassIntegrator(rho_coeff));
      Mr_H1.Assemble(0);
      Mr_H1.Finalize(0);
   }

   // discrete convection operator
   // NOTE: since the velocity is convected with the negative mesh velocity
   // we technically are assembling - K
   Kr_H1.AddDomainIntegrator(new ConvectionIntegrator(rho_u_coeff));
   Kr_H1.Assemble(0);
   Kr_H1.Finalize(0);
}

void AdvectorVelocityNonconservativeOper::Mult(const Vector &v, Vector &d_v) const
{
   d_v = 0.0;

   if (remap_v == RemapAdvector::RemapVelocity::None) { return; }

   const int dim     = pfes_H1_s.GetMesh()->Dimension();
   const int dofs_h1 = pfes_H1_s.GetVSize();

   if (remap_v_stable == false)
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

      // Velocity remap.
      Mr_H1.BilinearForm::operator=(0.0);
      Mr_H1.Assemble();
      Kr_H1.BilinearForm::operator=(0.0);
      Kr_H1.Assemble();
      HypreParMatrix *A = Mr_H1.ParallelAssemble();
      lin_solver.SetOperator(*A);
      Vector rhs_v(dofs_h1*dim);
      for (int d = 0; d < dim; d++)
      {
         const Vector v_comp(const_cast<Vector&>(v), d * dofs_h1, dofs_h1);
         Vector rhs_v_comp(rhs_v, d * dofs_h1, dofs_h1);
         Kr_H1.Mult(v_comp, rhs_v_comp);
      }
      const Operator *P_v = pfes_H1.GetProlongationMatrix();
      Vector RHS_V(P_v->Width()), X_V(P_v->Width());
      P_v->MultTranspose(rhs_v, RHS_V);
      X_V = 0.0;
      OperatorHandle M_elim;
      //M_elim.EliminateRowsCols(Mass_oper, v_ess_tdofs);
      //Mass_oper.EliminateBC(M_elim, v_ess_tdofs, X_V, RHS_V);
      lin_solver.Mult(RHS_V, X_V);
      P_v->Mult(X_V, d_v);
   }
   else
   {
      Mr_H1_s.BilinearForm::operator=(0.0);
      Mr_H1_s.Assemble();
      Kr_H1.BilinearForm::operator=(0.0);
      Kr_H1.Assemble();
      KrT_H1.BilinearForm::operator=(0.0);
      KrT_H1.Assemble();

      lummpedMr_H1.BilinearForm::operator=(0.0);
      lummpedMr_H1.Assemble();
      lummpedMr_H1.SpMat().GetDiag(lumpedMr_H1_vec);

      // Sum up to get global entries of the lumped mass matrix
      GroupCommunicator &gcomm = lummpedMr_H1.ParFESpace()->GroupComm();
      Array<double> lumpedmassmatrix_array(lumpedMr_H1_vec.GetData(), lumpedMr_H1_vec.Size());
      gcomm.Reduce<double>(lumpedmassmatrix_array, GroupCommunicator::Sum);
      gcomm.Bcast(lumpedmassmatrix_array);
      for(int i = 0; i < lumpedMr_H1_vec.Size(); i++)
      {
         MFEM_VERIFY(lumpedMr_H1_vec(i) > 1e-12, "lumped mass matrix entry negative or zero!");
      }

      // Get the global stencil of the local truedofs
      HypreParMatrix *M_hpm = Mr_H1_s.ParallelAssemble();
      HypreParMatrix *K_hpm = Kr_H1.ParallelAssemble();
      HypreParMatrix *KT_hpm = KrT_H1.ParallelAssemble();
      SparseMatrix M_glb, K_glb, KT_glb;
      M_hpm->MergeDiagAndOffd(M_glb);
      K_hpm->MergeDiagAndOffd(K_glb);
      KT_hpm->MergeDiagAndOffd(KT_glb);

      switch(remap_v)
      {
      case RemapAdvector::RemapVelocity::LowOrder:
         LowOrderVel(K_glb, KT_glb, v, d_v);
         break;
      case RemapAdvector::RemapVelocity::HighOrderTarget:
         HighOrderTargetSchemeVel(K_glb, KT_glb, M_glb, v, d_v);
         break;
      case RemapAdvector::RemapVelocity::MCL:
         MCLVel(K_glb, KT_glb, M_glb, v, d_v);
         break;
      case RemapAdvector::RemapVelocity::ClipAndScale:
         ClipAndScale(pfes_H1_s, v, d_v);
         break;
      default: MFEM_ABORT("Unknown scheme for velocity remap!");
      }
   }
}

real_t AdvectorVelocityNonconservativeOper::Momentum(ParGridFunction &v) const
{
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

void AdvectorVelocityGeomConsOper::RefConvectionIntegrator::
AssembleElementVector(const FiniteElement &fe, ElementTransformation &Tr,
   const Vector &elfun, Vector &elvec)
{
   const int ndof = fe.GetDof();
   const int ndim = fe.GetDim();

   shape.SetSize(ndof);
   dshape.SetSize(ndof, ndim);
   vdshape.SetSize(ndof);
   v.SetSize(ndim);
   vxt.SetSize(ndim);

   elvec.SetSize(ndof * ndim);
   elvec = 0.;

   Vector elvec_v, elfun_v;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      const int order = fe.GetOrder() + Tr.OrderGrad(&fe) + Tr.Order();
      ir = &IntRules.Get(fe.GetGeomType(), order);
   }
   const int nqp = ir->GetNPoints();

   for(int q = 0; q < nqp; q++)
   {
      const IntegrationPoint &ip = ir->IntPoint(q);
      Tr.SetIntPoint(&ip);
      f.GetVectorValue(Tr, ip, v);

      fe.CalcShape(ip, shape);
      fe.CalcDShape(ip, dshape);

      Tr.AdjugateJacobian().Mult(v, vxt);
      dshape.Mult(vxt, vdshape);

      const real_t w = -ip.weight;

      for (int v = 0; v < ndim; v++)
      {
         elvec_v.MakeRef(elvec, v*ndof, ndof);
         elfun_v.MakeRef(const_cast<Vector&>(elfun), v*ndof, ndof);
         elvec_v.Add(w * (elfun_v * shape), vdshape);
      }
   }
}

AdvectorVelocityGeomConsOper::AdvectorVelocityGeomConsOper(
   const IntegrationRule &ir_rho_, const Array<int> &v_ess_td, const Array<int> &v_ess_vd,
   ParFiniteElementSpace &pfes_H1, ParFiniteElementSpace &pfes_H1_s,
   RemapAdvector::RemapVelocity scheme, bool remap_v_s)
: AdvectorVelocityOper(v_ess_td, v_ess_vd, pfes_H1, pfes_H1_s, scheme, remap_v_s),
  ir_rho(ir_rho_), MJ(&pfes_H1), detJ(&pfes_H1_s)
{
   // Solution transfer
   trans = make_unique<SolutionTransfer_H1>(v_ess_tdofs, pfes_H1_s, ir_rho);

   // Interpolation matrix
   auto *mi = new SolutionTransfer_H1::RefMassIntegrator(&ir_rho);
   mi->SetVDim(pfes_H1.GetVDim());
   MJ.AddDomainIntegrator(mi);
   MJ.Assemble();
   MJ.Finalize();
   MJ.ParallelAssembleInternalMatrix();
   MJ.ParallelEliminateTDofs(v_ess_tdofs);
}

void AdvectorVelocityGeomConsOper::MultConserv(const ParGridFunction &flux, const Vector &U, Vector &dU) const
{
   ParMesh &pmesh = *pfes_H1.GetParMesh();
   const int NE = pmesh.GetNE();
   const int vdim = pfes_H1.GetVDim();

   // Current Jacobians
   trans->TransferJac_Larg2Remap(detJ);

   // RHS
   Vector bdU(pfes_H1.GetVSize());
   bdU = 0.;

   RefConvectionIntegrator Ki(flux);
   Vector x_z, dbx_z, detJ_z;
   Array<int> dofs, vdofs;

   for(int k = 0; k < NE; k++)
   {
      pfes_H1_s.GetElementVDofs(k, dofs);
      const int ndof = dofs.Size();
      vdofs = dofs;
      pfes_H1.DofsToVDofs(vdofs);

      detJ.GetSubVector(dofs, detJ_z);
      U.GetSubVector(vdofs, x_z);

      // Reduced the quantity to the non-conservative form
      for(int v = 0; v < vdim; v++)
         for(int i = 0; i < ndof; i++)
            x_z(i + v*ndof) /= detJ_z(i);

      Ki.AssembleElementVector(*pfes_H1.GetFE(k),
                               *pfes_H1.GetElementTransformation(k),
                               x_z, dbx_z);

      bdU.AddElementVector(vdofs, dbx_z);
   }

   // Invert by mass matrix
   HypreSmoother prec;
   prec.SetType(HypreSmoother::Jacobi, 1);

   CGSolver lin_solver(pfes_H1.GetComm());
   lin_solver.SetRelTol(1e-8);
   lin_solver.SetAbsTol(0.0);
   lin_solver.SetMaxIter(100);
   lin_solver.SetPrintLevel(0);
   lin_solver.SetPreconditioner(prec);
   lin_solver.SetOperator(*MJ.ParallelAssembleInternalMatrix());

   const int ntdof = pfes_H1_s.GetTrueVSize();

   Vector X(ntdof * vdim), RHS(ntdof * vdim);
   pfes_H1.GetProlongationMatrix()->MultTranspose(bdU, RHS);

   X = 0.;
   MJ.ParallelEliminateTDofsInRHS(v_ess_tdofs, X, RHS);
      
   lin_solver.Mult(RHS, X);
      
   pfes_H1.GetProlongationMatrix()->Mult(X, dU);
}

void AdvectorVelocityGeomConsOper::LimitUpdate(real_t dt, const Vector &U, Vector &dU)
{
}

real_t AdvectorVelocityGeomConsOper::Momentum(ParGridFunction &rhouJ) const
{
   HypreParMatrix &MJ_m = *MJ.ParallelAssembleInternalMatrix();
   Vector b(MJ_m.Height());
   rhouJ.SetTrueVector();
   MJ_m.Mult(rhouJ.GetTrueVector(), b);
   real_t mom = b.Sum();
   MPI_Allreduce(MPI_IN_PLACE, &mom, 1, MFEM_MPI_REAL_T, MPI_SUM, pfes_H1.GetComm());
   return mom;
}

AdvectorThermoOper::AdvectorThermoOper(ParFiniteElementSpace &pfes_L2_)
: pfes_L2(pfes_L2_)
{
   // Construct offsets
   const int vdofs_l2 = pfes_L2.GetVSize();
   offsets.SetSize(NVars+1);
   offsets = 0;
   offsets[Density+1] = vdofs_l2; // density
   offsets[Energy+1] = vdofs_l2; // energy
   offsets.PartialSum();

   width = height = offsets.Last();
}

AdvectorThermoNonconservativeOper::AdvectorThermoNonconservativeOper(
   Coefficient &rho_coeff_, VectorCoefficient &u_coeff_, ParFiniteElementSpace &pfes_L2)
   :  AdvectorThermoOper(pfes_L2),
      rho_coeff(rho_coeff_), u_coeff(u_coeff_),
      rho_u_coeff(rho_coeff, u_coeff),
      M_L2(&pfes_L2), M_L2_Lump(&pfes_L2), K_L2(&pfes_L2),
      Mr_L2(&pfes_L2),  Mr_L2_Lump(&pfes_L2), Kr_L2(&pfes_L2)
{
   // Assemble mass matrices
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
   auto dgt_ir_1 = new DGTraceIntegrator(rho_coeff, u_coeff, -1.0, -0.5);
   auto dgt_br_1 = new DGTraceIntegrator(rho_coeff, u_coeff, -1.0, -0.5);
   Kr_L2.AddInteriorFaceIntegrator(new TransposeIntegrator(dgt_ir_1));
   Kr_L2.AddBdrFaceIntegrator(new TransposeIntegrator(dgt_br_1));
   Kr_L2.KeepNbrBlock(true);
   Kr_L2.Assemble(0);
   Kr_L2.Finalize(0);
}

void AdvectorThermoNonconservativeOper::Mult(const Vector &U, Vector &dU) const
{
   ParFiniteElementSpace &pfes_L2 = *M_L2.ParFESpace();
   const int NE      = pfes_L2.GetNE();
   const int size_L2 = pfes_L2.GetVSize();

   dU = 0.0;

   // Block view
   const BlockVector bU(const_cast<Vector&>(U), offsets);
   BlockVector bdU(dU, offsets);

   // Density remap.
   Vector el_min(NE), el_max(NE);
   K_L2.BilinearForm::operator=(0.0);
   K_L2.Assemble();
   M_L2.BilinearForm::operator=(0.0);
   M_L2.Assemble();
   M_L2_Lump.BilinearForm::operator=(0.0);
   M_L2_Lump.Assemble();
   Vector d_rho_HO(size_L2), d_rho_LO(size_L2);
   Vector lumpedM; M_L2_Lump.SpMat().GetDiag(lumpedM);
   DiscreteUpwindLOSolver lo_solver(pfes_L2, K_L2.SpMat(), lumpedM);
   LocalInverseHOSolver ho_solver(M_L2, K_L2);
   Vector rho_min(size_L2), rho_max(size_L2);
   FluxBasedFCT fct_solver(pfes_L2, dt,
                           K_L2.SpMat(), lo_solver.GetKmap(), M_L2.SpMat());
   const Vector &rho = bU.GetBlock(Density);
   Vector &d_rho = bdU.GetBlock(Density);
   lo_solver.CalcLOSolution(rho, d_rho_LO);
   ho_solver.CalcHOSolution(rho, d_rho_HO);
   const ParGridFunction rho_gf(const_cast<ParFiniteElementSpace*>(&pfes_L2),
                                const_cast<Vector&>(rho));
   const_cast<ParGridFunction&>(rho_gf).ExchangeFaceNbrData();
   ComputeElementsMinMax(rho_gf, el_min, el_max);
   ComputeSparsityBounds(pfes_L2, el_min, el_max, rho_min, rho_max);
   fct_solver.CalcFCTSolution(rho_gf, lumpedM, d_rho_HO, d_rho_LO,
                              rho_min, rho_max, d_rho);

   // Energy remap.
   Kr_L2.BilinearForm::operator=(0.0);
   Kr_L2.Assemble();
   Mr_L2.BilinearForm::operator=(0.0);
   Mr_L2.Assemble();
   Mr_L2_Lump.BilinearForm::operator=(0.0);
   Mr_L2_Lump.Assemble();
   Vector d_e_HO(size_L2), d_e_LO(size_L2), Me_lumped;
   Vector e_min(size_L2), e_max(size_L2);
   Mr_L2_Lump.SpMat().GetDiag(Me_lumped);
   DiscreteUpwindLOSolver lo_e_solver(pfes_L2, Kr_L2.SpMat(), Me_lumped);
   LocalInverseHOSolver ho_e_solver(Mr_L2, Kr_L2);
   FluxBasedFCT fct_e_solver(pfes_L2, dt, Kr_L2.SpMat(),
                             lo_e_solver.GetKmap(), Mr_L2.SpMat());
   const Vector &e = bU.GetBlock(Energy);
   Vector &d_e = bdU.GetBlock(Energy);
   lo_e_solver.CalcLOSolution(e, d_e_LO);
   ho_e_solver.CalcHOSolution(e, d_e_HO);
   const ParGridFunction e_gf(const_cast<ParFiniteElementSpace*>(&pfes_L2),
                              const_cast<Vector&>(e));
   const_cast<ParGridFunction&>(e_gf).ExchangeFaceNbrData();
   ComputeElementsMinMax(e_gf, el_min, el_max);
   ComputeSparsityBounds(pfes_L2, el_min, el_max, e_min, e_max);
   fct_e_solver.CalcFCTSolution(e_gf, Me_lumped, d_e_HO, d_e_LO,
                                e_min, e_max, d_e);
}

real_t AdvectorThermoNonconservativeOper::Mass(ParGridFunction &rho) const
{
   M_L2.BilinearForm::operator=(0.0);
   M_L2.Assemble();

   Vector one(M_L2.SpMat().Height());
   one = 1.0;
   const real_t loc_rho = M_L2.InnerProduct(one, rho);

   real_t glob_rho;
   MPI_Allreduce(&loc_rho, &glob_rho, 1, MFEM_MPI_REAL_T, MPI_SUM,
                 Mr_L2.ParFESpace()->GetComm());
   return glob_rho;
}

real_t AdvectorThermoNonconservativeOper::InternalEnergy(ParGridFunction &e) const
{
   Mr_L2.BilinearForm::operator=(0.0);
   Mr_L2.Assemble();

   Vector one(Mr_L2.SpMat().Height());
   one = 1.0;
   const real_t loc_e = Mr_L2.InnerProduct(one, e);

   real_t glob_e;
   MPI_Allreduce(&loc_e, &glob_e, 1, MFEM_MPI_REAL_T, MPI_SUM,
                 Mr_L2.ParFESpace()->GetComm());
   return glob_e;
}


void AdvectorGeomConsOper::DivRDivRIntegrator::
AssembleElementMatrix(const FiniteElement &el, ElementTransformation &Trans,
   DenseMatrix &elmat)
{
   int dof = el.GetDof();
   double c;

#ifdef MFEM_THREAD_SAFE
   Vector divshape(dof);
#else
   divshape.SetSize(dof);
#endif
   elmat.SetSize(dof);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = 2 * el.GetOrder() - 2; // <--- OK for RTk
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   elmat = 0.0;

   for (int i = 0; i < ir -> GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      el.CalcDivShape (ip, divshape);

      Trans.SetIntPoint (&ip);
      c = ip.weight;

      // elmat += c * divshape * divshape ^ t
      AddMult_a_VVt (c, divshape, elmat);
   }
}

void AdvectorThermoGeomConsOper::RefConvectionIntegrator::
AssembleElementMatrix(const FiniteElement &fe, ElementTransformation &Tr,
   DenseMatrix &elmat)
{
   const int ndof = fe.GetDof();
   const int ndim = fe.GetDim();

   shape.SetSize(ndof);
   dshape.SetSize(ndof, ndim);
   vdshape.SetSize(ndof);
   v.SetSize(ndim);
   vxt.SetSize(ndim);

   elmat.SetSize(ndof);
   elmat = 0.;

   const int nqp = IntRule->GetNPoints();
   for(int q = 0; q < nqp; q++)
   {
      const IntegrationPoint &ip = IntRule->IntPoint(q);
      Tr.SetIntPoint(&ip);
      f.GetVectorValue(Tr, ip, v);

      fe.CalcShape(ip, shape);
      fe.CalcDShape(ip, dshape);

      Tr.AdjugateJacobian().Mult(v, vxt);
      dshape.Mult(vxt, vdshape);

      double w = -ip.weight;

      AddMult_a_VWt(w, vdshape, shape, elmat);
   }
}

void AdvectorThermoGeomConsOper::RefFaceConvectionIntegrator::
AssembleFaceMatrix(const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   const FiniteElementSpace *fes_face = f.FESpace();
   const FiniteElement &fe_face = *fes_face->GetFaceElement(Trans.Face->ElementNo);

   int ndof1, ndof2, ndof_face;
   double un, a, b, w;

   ndof1 = el1.GetDof();
   ndof2 = (Trans.Elem2No >= 0)?(el2.GetDof()):(0);
   ndof_face = fe_face.GetDof();
   
   shape1.SetSize(ndof1);
   shape2.SetSize(ndof2);
   shape_face.SetSize(ndof_face);
   elmat.SetSize(ndof1 + ndof2);
   elmat = 0.0;

   //fluxes
   fes_face->GetFaceVDofs(Trans.Face->ElementNo, vdofs_face);
   f.GetSubVector(vdofs_face, f_f);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      if (Trans.Elem2No >= 0)
         order = (min(Trans.Elem1->OrderW(), Trans.Elem2->OrderW()) +
                2 * max(el1.GetOrder(), el2.GetOrder()));
      else
      {
         order = Trans.Elem1->OrderW() + 2 * el1.GetOrder();
      }
      if (el1.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

      el1.CalcShape(eip1, shape1);
      if(ndof2)
         el2.CalcShape(eip2, shape2);

      fe_face.CalcShape(ip, shape_face);
      un = f_f * shape_face;

      a = 0.5 * un;
      b = - 0.5 * fabs(un);
      // note: if |alpha/2|==|beta| then |a|==|b|, i.e. (a==b) or (a==-b)
      //       and therefore two blocks in the element matrix contribution
      //       (from the current quadrature point) are 0

      w = ip.weight * (a + b);
      if (w != 0.0)
      {
         for (int i = 0; i < ndof1; i++)
            for (int j = 0; j < ndof1; j++)
            {
               elmat(i, j) += w * shape1(i) * shape1(j);
            }
      }

      if (ndof2)
      {
         if (w != 0.0)
            for (int i = 0; i < ndof2; i++)
               for (int j = 0; j < ndof1; j++)
               {
                  elmat(ndof1 + i, j) -= w * shape2(i) * shape1(j);
               }

         w = ip.weight * (b - a);
         if (w != 0.0)
         {
            for (int i = 0; i < ndof2; i++)
               for (int j = 0; j < ndof2; j++)
               {
                  elmat(ndof1 + i, ndof1 + j) += w * shape2(i) * shape2(j);
               }

            for (int i = 0; i < ndof1; i++)
               for (int j = 0; j < ndof2; j++)
               {
                  elmat(i, ndof1 + j) -= w * shape1(i) * shape2(j);
               }
         }
      }
   }
}

AdvectorThermoGeomConsOper::AdvectorThermoGeomConsOper(
   const IntegrationRule &ir_rho_, ParFiniteElementSpace &pfes_L2_)
   : AdvectorThermoOper(pfes_L2_), ir_rho(ir_rho_),
   pfes_vL2(pfes_L2.GetParMesh(), pfes_L2.FEColl(), NVars), detJ(&pfes_L2),
   MJ(pfes_L2.GetVSize()), KJ(pfes_L2.GetVSize() + pfes_L2.GetFaceNbrVSize())
{
   // Solution transfer
   trans = make_unique<SolutionTransfer_L2>(pfes_L2, ir_rho);

   // Lumped mass matrix
   MJ_lumped.SetSize(pfes_L2.GetVSize());
   const int NE = pfes_L2.GetNE();
   Vector mJ_k;
   Array<int> dofs;
   for (int k = 0; k < NE; k++)
   {
      const FiniteElement &fe = *pfes_L2.GetFE(k);
      Geometry::Type g = fe.GetGeomType();
      const DenseMatrix &MJ_k = trans->GetRefMassMatrix(g);
      MJ_k.GetRowSums(mJ_k);

      pfes_L2.GetElementDofs(k, dofs);
      MJ.AddSubMatrix(dofs, dofs, MJ_k);
      MJ_lumped.SetSubVector(dofs, mJ_k);
   }
   MJ.Finalize();

   // Propagator matrix - interior faces
   const ParMesh &pmesh = *pfes_L2.GetParMesh();
   const int nfaces = pmesh.GetNumFaces();
   DenseMatrix K_f;
   Array<int> dofs2;
   for (int f = 0; f < nfaces; f++)
   {
      int el1, el2;
      pmesh.GetFaceElements(f, &el1, &el2);
      if (el2 < 0) { continue; }
      pfes_L2.GetElementDofs(el1, dofs);
      pfes_L2.GetElementDofs(el2, dofs2);
      dofs.Append(dofs2);
      K_f.SetSize(dofs.Size());
      K_f = 0.; // dummy
      KJ.AddSubMatrix(dofs, dofs, K_f, 0);
   }

   // Propagator matrix - shared faces
   const int nshared = pmesh.GetNSharedFaces();
   const int ndofs_L2 = pfes_L2.GetVSize();
   for (int sf = 0; sf < nshared; sf++)
   {
      int el1, el2;
      const int f = pmesh.GetSharedFace(sf);
      pmesh.GetFaceElements(f, &el1, &el2);
      pfes_L2.GetElementDofs(el1, dofs);
      pfes_L2.GetFaceNbrElementVDofs(-1-el2, dofs2);
      for (int i = 0; i < dofs2.Size(); i++)
      {
         dofs2[i] += ndofs_L2;
      }
      dofs.Append(dofs2);
      K_f.SetSize(dofs.Size());
      K_f = 0.; // dummy
      KJ.AddSubMatrix(dofs, dofs, K_f, 0);
   }
   KJ.Finalize(0);
}

void AdvectorGeomConsOper::ImplicitSolveFluxRHS(Vector &rhs) const
{
   const int NE = pfes_f.GetNE();
   const int nqp = ir_rho.GetNPoints();

   Vector shape, rhs_z;
   Array<int> vdofs;

   for(int k = 0; k < NE; k++)
   {
      const FiniteElement *fe = pfes_f.GetFE(k);
      ElementTransformation *Tr = pfes_f.GetElementTransformation(k);
      pfes_f.GetElementVDofs(k, vdofs);

      rhs_z.SetSize(vdofs.Size());
      shape.SetSize(vdofs.Size());

      rhs_z = 0.;

      for(int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir_rho.IntPoint(q);
         Tr->SetIntPoint(&ip);
         const real_t w = Tr->Weight() * ip.weight;
         
         fe->CalcDivShape(ip, shape);
         rhs_z.Add(w, shape);
      }

      rhs.AddElementVector(vdofs, rhs_z);
   }
}

void AdvectorGeomConsOper::ImplicitSolveSolenoidalRHS(const Vector &f, Vector &rhs) const
{
   const int NE = pfes_a.GetNE();
   const int nqp = ir_rho.GetNPoints();
   const int dim = pfes_a.GetParMesh()->Dimension();
   const int sdim = pfes_a.GetParMesh()->SpaceDimension();

   DenseMatrix dshape, dshapext, vshape;
   Vector f_z, rhs_z, dshape_d;
   Array<int> dofs_h1, vdofs_rt;

   for(int k = 0; k < NE; k++)
   {
      const FiniteElement *fe_h1 = pfes_a.GetFE(k);
      const FiniteElement *fe_rt = pfes_f.GetFE(k);
      ElementTransformation *Tr = pfes_a.GetElementTransformation(k);
      pfes_a.GetElementDofs(k, dofs_h1);
      pfes_f.GetElementVDofs(k, vdofs_rt);
      f.GetSubVector(vdofs_rt, f_z);
      
      const int ndof_h1 = fe_h1->GetDof();
      const int ndof_rt = fe_rt->GetDof();
      rhs_z.SetSize(ndof_h1);
      dshape.SetSize(ndof_h1, dim);
      dshapext.SetSize(ndof_h1, sdim);
      vshape.SetSize(ndof_rt, sdim);

      rhs_z = 0.;

      for(int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir_rho.IntPoint(q);
         Tr->SetIntPoint(&ip);
         fe_h1->CalcDShape(ip, dshape);
         mfem::Mult(dshape, Tr->AdjugateJacobian(), dshapext);

         Vector f_q(sdim);
         fe_rt->CalcVShape(*Tr, vshape);
         vshape.MultTranspose(f_z, f_q);

         const real_t w = ip.weight;
         
         // +f_x dy
         dshape.GetColumnReference(1, dshape_d);
         rhs_z.Add(+w * f_q(0), dshape_d);
         
         // -f_y dx
         dshape.GetColumnReference(0, dshape_d);
         rhs_z.Add(-w * f_q(1), dshape_d);
      }

      rhs.AddElementVector(dofs_h1, rhs_z);
   }
}

void AdvectorGeomConsOper::ImplicitSolveFlux(real_t dt, ParGridFunction &flux)
{
   // move the mesh to the initial position
   add(x0, t, u, x_now);

   // setup the flux
   if (!flux.FESpace()) { flux.SetSpace(&pfes_f); }

   // mass vector
   Vector rhs(pfes_f.GetVSize());
   rhs = 0.;
   
   ImplicitSolveFluxRHS(rhs);
   rhs.Neg();

   // move to new coordinates
   add(x0, t+dt, u, x_now);

   // add new mass vector
   ImplicitSolveFluxRHS(rhs);

   // mass derivative
   rhs *= 1./dt;

   // project velocity at half-time

   add(x0, t + 0.5 * dt, u, x_now);
   VectorGridFunctionCoefficient u_coeff(&u);
   flux.ProjectCoefficient(u_coeff);
   Vector flux_v(flux);
   add(x0, t, u, x_now);

   // set up solution and rhs true vectors 

   Vector X(pfes_f.GetTrueVSize()), RHS(pfes_f.GetTrueVSize());

   flux.ParallelProject(X);
   pfes_f.GetProlongationMatrix()->MultTranspose(rhs, RHS);
   DD.ParallelEliminateTDofsInRHS(ess_tdofs_f, X, RHS);

   // set up solver and solve for the flux

   const HypreParMatrix &DD_m = *DD.ParallelAssembleInternalMatrix();
   
   HypreBoomerAMG prec(DD_m);
   prec.SetPrintLevel(0);

   CGSolver solver(pfes_f.GetComm());
   solver.SetRelTol(1e-6);
   const real_t norm_rhs = InnerProduct(pfes_f.GetComm(), RHS, RHS);
   solver.SetAbsTol(norm_rhs * 1e-6);
   solver.SetMaxIter(1000);
   solver.SetPrintLevel(3);
   solver.SetPreconditioner(prec);
   solver.SetOperator(DD_m);
   solver.iterative_mode = true;

   solver.Mult(RHS, X);

   flux.Distribute(X);

   // solenoidal rhs

   Vector rhs_a(pfes_a.GetVSize());
   rhs_a = 0.;

   flux_v -= flux;

   ImplicitSolveSolenoidalRHS(flux_v, rhs_a);

   // assemble the system for Hemholtz decomposition

   ParGridFunction a(&pfes_a);
   a = 0.;

   HypreParVector X_a(&pfes_a), RHS_a(&pfes_a);

   a.ParallelProject(X_a);
   pfes_a.GetProlongationMatrix()->MultTranspose(rhs_a, RHS_a);

   CC.Update();
   CC.Assemble();
   CC.Finalize();
   CC.ParallelAssembleInternalMatrix();
   CC.ParallelEliminateEssentialBC(ess_bdr, X_a, RHS_a);

   // solve for the solenoidal potential

   const HypreParMatrix &CC_m = *CC.ParallelAssembleInternalMatrix();

   HypreBoomerAMG prec_a(CC_m);
   prec_a.SetPrintLevel(0);

   HyprePCG solver_a(pfes_a.GetComm());
   solver_a.SetTol(1e-6);
   solver_a.SetAbsTol(0.);
   solver_a.SetMaxIter(1000);
   solver_a.SetPrintLevel(3);
   solver_a.SetPreconditioner(prec_a);
   solver_a.SetOperator(CC_m);

   solver_a.Mult(RHS_a, X_a);

   a.Distribute(X_a);

   // apply the solenoidal correction

   const int NE = pfes_a.GetNE();
   CurlInterpolator ci;
   DenseMatrix curl;
   Vector f_k, a_k;
   Array<int> vdofs;

   for (int k = 0; k < NE; k++)
   {
      const FiniteElement &fe_f = *pfes_f.GetFE(k);
      const FiniteElement &fe_a = *pfes_a.GetFE(k);
      ElementTransformation &Tr = *pfes_f.GetElementTransformation(k);

      ci.AssembleElementMatrix2(fe_a, fe_f, Tr, curl);

      a.GetElementDofValues(k, a_k);
      pfes_f.GetElementVDofs(k, vdofs);
      f_k.SetSize(vdofs.Size());

      curl.Mult(a_k, f_k);
      
      flux.AddElementVector(vdofs, f_k);
   }
}

void AdvectorThermoGeomConsOper::MultConserv(const ParGridFunction &flux, const Vector &U, Vector &dU) const
{
   ParMesh &pmesh = *pfes_L2.GetParMesh();
   const int NE = pmesh.GetNE();

   // Grid function view
   ParGridFunction U_gf(const_cast<ParFiniteElementSpace*>(&pfes_vL2), const_cast<Vector&>(U));
   ParGridFunction dU_gf(const_cast<ParFiniteElementSpace*>(&pfes_vL2), dU);
   U_gf.ExchangeFaceNbrData();

   // Current Jacobians
   trans->TransferJac_Larg2Remap(detJ);
   detJ.ExchangeFaceNbrData();

   // Propagator

   KJ = 0.;
   RefConvectionIntegrator Ki(flux, &ir_rho);
   RefFaceConvectionIntegrator Kfi(flux);

   // Propagator - domain
   DenseMatrix K_k;
   Vector x_k, dbx_k, detJ_k;
   Array<int> dofs, vdofs;

   for(int k = 0; k < NE; k++)
   {
      pfes_L2.GetElementDofs(k, dofs);
      const int ndof = dofs.Size();
      vdofs = dofs;
      pfes_vL2.DofsToVDofs(vdofs);
      
      // Assemble the face matrix
      Ki.AssembleElementMatrix(*pfes_L2.GetFE(k),
                               *pfes_L2.GetElementTransformation(k),
                               K_k);
                               
      // Reduced the quantity to the non-conservative form
      detJ.GetSubVector(dofs, detJ_k);
      K_k.InvRightScaling(detJ_k);

      // Store the matrix
      KJ.AddSubMatrix(dofs, dofs, K_k);

      // Multiply the values
      U_gf.GetSubVector(vdofs, x_k);
      dbx_k.SetSize(x_k.Size());

      for(int v = 0; v < NVars; v++)
      {
         Vector x_kv(x_k, v*ndof, ndof);
         Vector dbx_kv(dbx_k, v*ndof, ndof);
         K_k.Mult(x_kv, dbx_kv);
      }

      dU_gf.SetSubVector(vdofs, dbx_k);
   }

   // Propagator - faces
   const int nfaces = pmesh.GetNumFaces();
   DenseMatrix K_f;
   Array<int> dofs2;

   for(int f = 0; f < nfaces; f++)
   {
      FaceElementTransformations *ftr = pmesh.GetInteriorFaceTransformations(f);
      if (!ftr) { continue; }

      const FiniteElement *fe1, *fe2;
      fe1 = pfes_L2.GetFE(ftr->Elem1No);
      pfes_L2.GetElementDofs(ftr->Elem1No, dofs);
      if(ftr->Elem2No >= 0)
      {
         fe2 = pfes_L2.GetFE(ftr->Elem2No);
         pfes_L2.GetElementDofs(ftr->Elem2No, dofs2);
         dofs.Append(dofs2);
      }
      else
      {
         fe2 = fe1;
      }
      const int ndof = dofs.Size();
      vdofs = dofs;
      pfes_vL2.DofsToVDofs(vdofs);
      
      // Assemble the face matrix
      Kfi.AssembleFaceMatrix(*fe1, *fe2, *ftr, K_f);
      
      // Reduced the quantity to the non-conservative form
      detJ.GetSubVector(dofs, detJ_k);
      K_f.InvRightScaling(detJ_k);

      // Store the matrix
      KJ.AddSubMatrix(dofs, dofs, K_f);

      U_gf.GetSubVector(vdofs, x_k);
      dbx_k.SetSize(x_k.Size());

      // Multiply the values
      for(int v = 0; v < NVars; v++)
      {
         Vector x_kv(x_k, v*ndof, ndof);
         Vector dbx_kv(dbx_k, v*ndof, ndof);
         K_f.Mult(x_kv, dbx_kv);
      }

      dU_gf.AddElementVector(vdofs, dbx_k);
   }

   // Propagator - shared faces
   const int nshared = pmesh.GetNSharedFaces();
   DenseMatrix K_floc, K_fnbr;
   Vector x_nbr, detJ_loc, detJ_nbr;
   Array<int> dofs_nbr, vdofs_nbr;

   for(int sf = 0; sf < nshared; sf++)
   {
      FaceElementTransformations *ftr = pmesh.GetSharedFaceTransformations(sf);
      const FiniteElement *fe1, *fe2;

      fe1 = pfes_L2.GetFE(ftr->Elem1No);
      fe2 = pfes_L2.GetFaceNbrFE(ftr->Elem2No - NE);
      pfes_L2.GetElementDofs(ftr->Elem1No, dofs);
      pfes_L2.GetFaceNbrElementVDofs(ftr->Elem2No - NE, dofs_nbr);
      const int ndof = dofs.Size();
      const int ndof_nbr = dofs_nbr.Size();
      vdofs = dofs;
      pfes_vL2.DofsToVDofs(vdofs);
      pfes_vL2.GetFaceNbrElementVDofs(ftr->Elem2No - NE, vdofs_nbr);

      // Assemble the face matrix
      Kfi.AssembleFaceMatrix(*fe1, *fe2, *ftr, K_f);
      
      // Reduced the quantity to the non-conservative form
      detJ_k.SetSize(ndof + ndof_nbr);
      detJ_loc.MakeRef(detJ_k, 0, ndof);
      detJ.GetSubVector(dofs, detJ_loc);
      detJ_nbr.MakeRef(detJ_k, ndof, ndof_nbr);
      detJ.FaceNbrData().GetSubVector(dofs_nbr, detJ_nbr);

      K_f.InvRightScaling(detJ_k);
      
      // Store the matrix
      for (int i = 0; i < dofs_nbr.Size(); i++)
         dofs_nbr[i] += pfes_L2.GetVSize();
      dofs.Append(dofs_nbr);

      KJ.AddSubMatrix(dofs, dofs, K_f);

      // Multiply the values
      K_floc.CopyMN(K_f, ndof, ndof, 0, 0);
      K_fnbr.CopyMN(K_f, ndof, ndof_nbr, 0, ndof);

      U_gf.GetSubVector(vdofs, x_k);
      U_gf.FaceNbrData().GetSubVector(vdofs_nbr, x_nbr);
      dbx_k.SetSize(x_k.Size());

      for(int v = 0; v < NVars; v++)
      {
         Vector x_kv(x_k, v*ndof, ndof);
         Vector x_nbrv(x_nbr, v*ndof_nbr, ndof_nbr);
         Vector dbx_kv(dbx_k, v*ndof, ndof);
         K_floc.Mult(x_kv, dbx_kv);
         K_fnbr.AddMult(x_nbrv, dbx_kv);
      }

      dU_gf.AddElementVector(vdofs, dbx_k);
   }

   // Inverse mass matrix
   for(int k = 0; k < NE; k++)
   {
      pfes_L2.GetElementDofs(k, dofs);
      const int ndof = dofs.Size();
      vdofs = dofs;
      pfes_vL2.DofsToVDofs(vdofs);

      auto g = pfes_L2.GetFE(k)->GetGeomType();
      const LUFactors lu(trans->GetRefMassInverse(g));

      dU_gf.GetSubVector(vdofs, dbx_k);

      lu.Solve(ndof, StateVars::NVars, dbx_k.GetData());

      dU_gf.SetSubVector(vdofs, dbx_k);
   }
}

void AdvectorThermoGeomConsOper::LimitUpdate(real_t dt, const Vector &U, Vector &dU)
{
   // Block view
   const BlockVector bU(const_cast<Vector&>(U), offsets);
   BlockVector bdU(dU, offsets);

   // LO solution
   DiscreteUpwindLOSolver solver_lo(pfes_L2, KJ, MJ_lumped);
   FluxBasedFCT fct(pfes_L2, dt, KJ, solver_lo.GetKmap(), MJ);

   const int NE = pfes_L2.GetNE();
   const int ndofs = pfes_L2.GetVSize();
   Vector el_min(NE), el_max(NE);
   Vector dof_min(ndofs), dof_max(ndofs);
   Vector dU_v_LO(ndofs), u_v(ndofs), U_vm1_new(ndofs);
   Array<bool> u_bool_el, u_bool_dofs, u_bool_el_new, u_bool_dofs_new;

   for (int v = 0; v < NVars; v++)
   {
      const Vector &U_vm1 = (v > 0)?(bU.GetBlock(v-1)):(detJ);
      const Vector &U_v = bU.GetBlock(v);

      // low-order solution

      solver_lo.CalcLOSolution(U_v, dU_v_LO);

      // compute ratio

      ComputeRatio(NE, U_v, U_vm1, u_v, u_bool_el, u_bool_dofs);

      // element min/max

      ComputeElementsMinMax(u_v, el_min, el_max, &u_bool_el, &u_bool_dofs);

      // dof min/max

      ComputeSparsityBounds(pfes_L2, el_min, el_max, dof_min, dof_max);

      // evole u and get the new active dofs
      if (v > 0)
      {
         const Vector &dU_vm1 = bdU.GetBlock(v-1);
         add(1.0, U_vm1, dt, dU_vm1, U_vm1_new);
         ComputeBoolIndicators(NE, U_vm1_new, u_bool_el_new, u_bool_dofs_new);
      }
      else
      {
         ParGridFunction U_vm1_new_gf(&pfes_L2, U_vm1_new);
         trans->TransferJac_Larg2Remap(U_vm1_new_gf);

         u_bool_el_new.SetSize(NE);
         u_bool_el_new = true;
         u_bool_dofs_new.SetSize(ndofs);
         u_bool_dofs_new = true;
      }

      // FCT

      const ParGridFunction U_v_gf(&pfes_L2, const_cast<Vector&>(U_v));
      const_cast<ParGridFunction&>(U_v_gf).ExchangeFaceNbrData();
      Vector &dU_v = bdU.GetBlock(v);

      fct.CalcFCTProduct(U_v_gf, MJ_lumped, dU_v, dU_v_LO,
         dof_min, dof_max, U_vm1_new, u_bool_el_new, u_bool_dofs_new, dU_v);
   }
}

real_t AdvectorThermoGeomConsOper::Mass(ParGridFunction &rhoJ) const
{
   real_t mass = 0.;

   const ParFiniteElementSpace &pfes = *rhoJ.ParFESpace();
   const ParMesh &pmesh = *pfes.GetParMesh();
   const int NE = pmesh.GetNE();
   Vector rhoJ_k, one_k;
   Array<int> vdofs;

   for (int k = 0; k < NE; k++)
   {
      pfes.GetElementVDofs(k, vdofs);
      rhoJ.GetSubVector(vdofs, rhoJ_k);

      if (one_k.Size() != vdofs.Size())
      {
         one_k.SetSize(vdofs.Size());
         one_k = 1.;
      }

      auto g = pfes.GetFE(k)->GetGeomType();
      mass += trans->GetRefMassMatrix(g).InnerProduct(rhoJ_k, one_k);
   }

   MPI_Allreduce(MPI_IN_PLACE, &mass, 1, MFEM_MPI_REAL_T, MPI_SUM, pmesh.GetComm());

   return mass;
}

real_t AdvectorThermoGeomConsOper::InternalEnergy(ParGridFunction &rhoeJ) const
{
   return Mass(rhoeJ);
}


void SolutionTransfer_L2::RefMassIntegrator::AssembleElementMatrix(
   const FiniteElement &fe, ElementTransformation &Trans, DenseMatrix &elmat)
{
   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      const int order = 2 * fe.GetOrder();
      ir = &IntRules.Get(fe.GetGeomType(), order);
   }
   const int nqp = ir->GetNPoints();

   const int ndof = fe.GetDof();
   Vector shape(ndof);
   elmat.SetSize(ndof * vdim);
   elmat = 0.;

   DenseMatrix elmat_d;
   if (vdim > 1)
   {
      elmat_d.SetSize(ndof);
      elmat_d = 0.;
   }
   else
   {
      elmat_d.MakeRef(elmat.GetMemory(), 0, ndof, ndof);
   }

   for (int q = 0; q < nqp; q++)
   {
      const IntegrationPoint &ip = ir->IntPoint(q);
      fe.CalcShape(ip, shape);
      AddMult_a_VVt(ip.weight, shape, elmat_d);
   }

   if (vdim > 1)
   {
      for (int v = 0; v < vdim; v++)
         elmat.SetSubMatrix(v*ndof, v*ndof, elmat_d);
   }
}

SolutionTransfer_L2::SolutionTransfer_L2(const ParFiniteElementSpace &pfes_L2, const IntegrationRule &ir)
: pmesh(*pfes_L2.GetParMesh()), fec0(0, pmesh.Dimension()), pfes0(const_cast<ParMesh*>(&pmesh), &fec0), ir_rho(ir)
{
   // Interpolation matrix (inverse)
   RefMassIntegrator mi(&ir_rho);
   Array<Geometry::Type> geoms;
   pmesh.GetGeometries(pmesh.Dimension(), geoms);
   const FiniteElementCollection *fec_L2 = pfes_L2.FEColl();
   IsoparametricTransformation Tr; // dummy
   for (Geometry::Type g : geoms)
   {
      const FiniteElement *fe = fec_L2->GetFE(g, fec_L2->GetOrder());
      const int ndof = fe->GetDof();
      mi.AssembleElementMatrix(*fe, Tr, MJ[g]);

      MJi[g] = MJ[g];
      MJi_piv[g].SetSize(ndof);
      LUFactors lu(MJi[g].GetData(), MJi_piv[g].GetData());
      lu.Factor(ndof);
   }
}

void SolutionTransfer_L2::ComputeMinMax(const Vector &lmins, const Vector &lmaxs, Vector &mins, Vector &maxs)
{
   ParMesh &pmesh = *pfes0.GetParMesh();
   const int NE = pmesh.GetNE();
   
   mins = lmins;
   maxs = lmaxs;

   // One-level face neighbors max / min.
   ParGridFunction min_pgf(&pfes0, const_cast<Vector&>(lmins));
   ParGridFunction max_pgf(&pfes0, const_cast<Vector&>(lmaxs));
   min_pgf.ExchangeFaceNbrData();
   max_pgf.ExchangeFaceNbrData();
   const Vector &gmins = min_pgf.FaceNbrData();
   const Vector &gmaxs = max_pgf.FaceNbrData();
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
            mins(k) = std::min(mins(k), lmins(face_nbr_el[n]));
            maxs(k) = std::max(maxs(k), lmaxs(face_nbr_el[n]));
         }
         else
         {
            // MPI face neighbor.
            mins(k) = std::min(mins(k), gmins(face_nbr_el[n] - NE));
            maxs(k) = std::max(maxs(k), gmaxs(face_nbr_el[n] - NE));
         }
      }
   }
}

void SolutionTransfer_L2::LimitFluxes(real_t y_avg, real_t y_min, real_t y_max, std::function<real_t(int)> &&w_z, DenseMatrix &F)
{
   const int dof_cnt = F.Width();
   Vector gp(dof_cnt), gm(dof_cnt);

   // Calculate incoming/outgoing fluxes
   gp = 0.0;
   gm = 0.0;
   for (int i = 1; i < dof_cnt; i++)
   {
      for (int j = 0; j < i; j++)
      {
         real_t fij = F(i, j);
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

   // Calculate Zalesak limiter
   for (int i = 0; i < dof_cnt; i++)
   {
      real_t rp = max(w_z(i) * (y_max - y_avg), 0.0);
      real_t rm = min(w_z(i) * (y_min - y_avg), 0.0);
      real_t sp = gp(i), sm = gm(i);

      gp(i) = (rp < sp) ? rp / sp : 1.0;
      gm(i) = (rm > sm) ? rm / sm : 1.0;
   }

   // Calculate local increments
   for (int i = 1; i < dof_cnt; i++)
   {
      for (int j = 0; j < i; j++)
      {
         real_t &fij = F(i, j), aij;

         if (fij >= 0.0)
         {
            aij = std::min(gp(i), gm(j));
         }
         else
         {
            aij = std::min(gm(i), gp(j));
         }

         fij *= aij;
      }
   }
}

void SolutionTransfer_L2::TransferL2Monotonous(
    std::function<void(int, DenseMatrix &, LUFactors &)> &&M, const Vector &lmins, const Vector &lmaxs,
    std::function<void(int, Vector &)> &&b, ParGridFunction &y)
{
   ParMesh &pmesh = *y.ParFESpace()->GetParMesh();
   const int NE = pmesh.GetNE();

   Vector mins, maxs;
   ComputeMinMax(lmins, lmaxs, mins, maxs);

   // HO solution - FCT_Project.
   const int dof_cnt = y.Size() / NE;
   DenseMatrix M_z(dof_cnt), F(dof_cnt);
   DenseMatrixInverse M_zi(&M_z);
   LUFactors M_zlu(nullptr, nullptr);
   Vector rhs(dof_cnt), y_HO(dof_cnt), y_z(dof_cnt), m_z(dof_cnt),
          beta(dof_cnt), z(dof_cnt);
   Array<int> dofs(dof_cnt);
   
   for (int k = 0; k < NE; k++)
   {
      // Get local rhs
      b(k, rhs);

      // Get local mass matrix
      M(k, M_z, M_zlu);

      // Construct contracted mass matrix
      M_z.GetRowSums(m_z);

      // Calculate high-order solution
      if (M_zlu.data)
      {
         y_HO = rhs;
         M_zlu.Solve(dof_cnt, 1, y_HO.GetData());
      }
      else
      {
         M_zi.Factor();
         M_zi.Mult(rhs, y_HO);
      }

      // Calculate the average
      const real_t y_avg = rhs.Sum() / m_z.Sum();

      beta = m_z;
      beta /= beta.Sum();

      // Calculate antisymmetric fluxes
      for (int i = 0; i < dof_cnt; i++) { z(i) = rhs(i) - m_z(i) * y_avg; }

      for (int i = 1; i < dof_cnt; i++)
      {
         for (int j = 0; j < i; j++)
         {
            F(i, j) = M_z(i, j) * (y_HO(i) - y_HO(j)) +
                      (beta(j) * z(i) - beta(i) * z(j));
         }
      }

      // Limit the fluxes
      LimitFluxes(y_avg, mins(k), maxs(k), [&](int i) { return m_z(i); }, F);

      // Calculate local increments
      y_z = y_avg;
      for (int i = 1; i < dof_cnt; i++)
      {
         for (int j = 0; j < i; j++)
         {
            real_t fij = F(i, j);
            y_z(i) += fij / m_z(i);
            y_z(j) -= fij / m_z(j);
         }
      }

      y.ParFESpace()->GetElementDofs(k, dofs);
      y.SetSubVector(dofs, y_z);
   }
}

void SolutionTransfer_L2::TransferXYL2Monotonous(
   std::function<void(int, DenseMatrix &, LUFactors &)> &&M, const Vector &lmins, const Vector &lmaxs,
   const ParGridFunction &x, std::function<void(int, Vector &)> &&b, ParGridFunction &y)
{
      ParMesh &pmesh = *y.ParFESpace()->GetParMesh();
   const int NE = pmesh.GetNE();

   Vector mins, maxs;
   ComputeMinMax(lmins, lmaxs, mins, maxs);

   // HO solution - FCT_Project.
   const int dof_cnt = y.Size() / NE;
   DenseMatrix M_z(dof_cnt), F(dof_cnt);
   DenseMatrixInverse M_zi(&M_z);
   LUFactors M_zlu(nullptr, nullptr);
   Vector x_z(dof_cnt), rhs(dof_cnt), xy_HO(dof_cnt), y_z(dof_cnt), m_z(dof_cnt),
          beta(dof_cnt), z(dof_cnt);
   Array<int> dofs(dof_cnt);

   for (int k = 0; k < NE; k++)
   {
      // Get local x
      y.ParFESpace()->GetElementDofs(k, dofs);
      x.GetSubVector(dofs, x_z);

      // Get local rhs
      b(k, rhs);

      // Get local mass matrix
      M(k, M_z, M_zlu);

      // Construct contracted mass matrix
      M_z.GetRowSums(m_z);

      // Calculate high-order solution
      if (M_zlu.data)
      {
         xy_HO = rhs;
         M_zlu.Solve(dof_cnt, 1, xy_HO.GetData());
      }
      else
      {
         M_zi.Factor();
         M_zi.Mult(rhs, xy_HO);
      }

      // Calculate the average
      const real_t mx_sum = m_z * x_z;
      const real_t y_avg = (mx_sum != 0.) ? (rhs.Sum() / mx_sum):(0.);

      beta = m_z;
      beta /= beta.Sum();

      // Calculate antisymmetric fluxes
      for (int i = 0; i < dof_cnt; i++) { z(i) = rhs(i) - m_z(i) * x_z(i) * y_avg; }

      for (int i = 1; i < dof_cnt; i++)
      {
         for (int j = 0; j < i; j++)
         {
            F(i, j) = M_z(i, j) * (xy_HO(i) - xy_HO(j)) +
                      (beta(j) * z(i) - beta(i) * z(j));
         }
      }

      // Limit the fluxes
      LimitFluxes(y_avg, mins(k), maxs(k), [&](int i) { return m_z(i) * x_z(i); }, F);

      // Calculate local increments
      y_z = y_avg;
      for (int i = 1; i < dof_cnt; i++)
      {
         for (int j = 0; j < i; j++)
         {
            real_t fij = F(i, j);
            y_z(i) += (x_z(i) != 0.) ? (fij / (m_z(i) * x_z(i))) : (0.);
            y_z(j) -= (x_z(j) != 0.) ? (fij / (m_z(j) * x_z(j))) : (0.);
         }
      }

      y.SetSubVector(dofs, y_z);
   }
}

void SolutionTransfer_L2::TransferDensity_Lagr2Remap(const Vector &rhoDetJw,
                                                  ParGridFunction &rho)
{
   const ParFiniteElementSpace &pfes = *rho.ParFESpace();
   const int NE = pfes.GetNE(), nqp = ir_rho.GetNPoints();
   Vector rho_min_loc(NE), rho_max_loc(NE);

   // Local max / min.
   for (int k = 0; k < NE; k++)
   {
      ElementTransformation &T = *pfes.GetElementTransformation(k);
      rho_min_loc(k) = +infinity();
      rho_max_loc(k) = -infinity();

      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir_rho.IntPoint(q);
         T.SetIntPoint(&ip);
         const real_t detJ = T.Jacobian().Det();
         const real_t rho = rhoDetJw(k * nqp + q) / detJ / ip.weight;

         rho_min_loc(k) = std::min(rho_min_loc(k), rho);
         rho_max_loc(k) = std::max(rho_max_loc(k), rho);
      }
   }

   // Mass matrix
   MassIntegrator mi(&ir_rho);
   auto M = [&pfes,&mi](int k, DenseMatrix &M_z, LUFactors &) {
      const FiniteElement &fe = *pfes.GetFE(k);
      ElementTransformation &T = *pfes.GetElementTransformation(k);
      mi.AssembleElementMatrix(fe, T, M_z);
   };

   // Righ hand side
   hydrodynamics::DensityIntegrator di(rhoDetJw);
   di.SetIntRule(&ir_rho);

   auto brho = [&pfes,&di](int k, Vector &rhs) {
      const FiniteElement &fe = *pfes.GetFE(k);
      ElementTransformation &T = *pfes.GetElementTransformation(k);
      di.AssembleRHSElementVect(fe, T, rhs);
   };

   TransferL2Monotonous(M, rho_min_loc, rho_max_loc, brho, rho);
}

void SolutionTransfer_L2::TransferJac_Larg2Remap(ParGridFunction &detJ)
{
   const ParFiniteElementSpace &pfes = *detJ.ParFESpace();
   const int NE = pfes.GetNE(), nqp = ir_rho.GetNPoints();
   Vector detJ_min_loc(NE), detJ_max_loc(NE);

   // Local max / min.
   for (int k = 0; k < NE; k++)
   {
      ElementTransformation &T = *pfes.GetElementTransformation(k);
      detJ_min_loc(k) = +infinity();
      detJ_max_loc(k) = 0.;

      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir_rho.IntPoint(q);
         T.SetIntPoint(&ip);
         const real_t detJ = T.Jacobian().Det();
         MFEM_ASSERT(detJ > 0., "Non-positive Jacobian!");
         detJ_min_loc(k) = std::min(detJ_min_loc(k), detJ);
         detJ_max_loc(k) = std::max(detJ_max_loc(k), detJ);
      }
   }

   // Interpolation matrix
   auto M = [&pfes, this]
   (int k, DenseMatrix &M_z, LUFactors &M_zlu) {
      const FiniteElement &fe = *pfes.GetFE(k);
      const Geometry::Type g = fe.GetGeomType();
      M_z = MJ[g];
      M_zlu.data = MJi[g].GetData();
      M_zlu.ipiv = MJi_piv[g].GetData();
   };

   // Right hand side
   ConstantCoefficient one;
   DomainLFIntegrator dlfi(one, &ir_rho);

   auto bdetJ = [&pfes,&dlfi](int k, Vector &rhs) {
      const FiniteElement &fe = *pfes.GetFE(k);
      ElementTransformation &T = *pfes.GetElementTransformation(k);
      dlfi.AssembleRHSElementVect(fe, T, rhs);
   };

   TransferL2Monotonous(M, detJ_min_loc, detJ_max_loc, bdetJ, detJ);
}

void SolutionTransfer_L2::TransferDensityJac_Lagr2Remap(
   const Vector &rhoDetJw, const ParGridFunction &detJ, ParGridFunction &rhoJ)
{
   const ParFiniteElementSpace &pfes = *rhoJ.ParFESpace();
   const int NE = pfes.GetNE(), nqp = ir_rho.GetNPoints();
   Vector rho_min_loc(NE), rho_max_loc(NE);

   // Local max / min.
   for (int k = 0; k < NE; k++)
   {
      ElementTransformation &T = *pfes.GetElementTransformation(k);
      rho_min_loc(k) = +infinity();
      rho_max_loc(k) = -infinity();

      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir_rho.IntPoint(q);
         T.SetIntPoint(&ip);
         const real_t detJ = T.Jacobian().Det();
         const real_t rho = rhoDetJw(k * nqp + q) / detJ / ip.weight;

         rho_min_loc(k) = std::min(rho_min_loc(k), rho);
         rho_max_loc(k) = std::max(rho_max_loc(k), rho);
      }
   }

   // Interpolation matrix
   auto M = [&pfes, this](int k, DenseMatrix &M_z, LUFactors &M_zlu) {
      const FiniteElement &fe = *pfes.GetFE(k);
      const Geometry::Type g = fe.GetGeomType();
      M_z = MJ[g];
      M_zlu.data = MJi[g].GetData();
      M_zlu.ipiv = MJi_piv[g].GetData();
   };

   // Righ hand side
   hydrodynamics::DensityIntegrator di(rhoDetJw);
   di.SetIntRule(&ir_rho);

   auto brho = [&pfes,&di](int k, Vector &rhs) {
      const FiniteElement &fe = *pfes.GetFE(k);
      ElementTransformation &T = *pfes.GetElementTransformation(k);
      di.AssembleRHSElementVect(fe, T, rhs);
   };

   TransferXYL2Monotonous(M, rho_min_loc, rho_max_loc, detJ, brho, rhoJ);

   // Jacobian product
   for (int i = 0; i < rhoJ.Size(); i++)
      rhoJ(i) *= detJ(i);
}

void SolutionTransfer_L2::TransferEnergyJac_Lagr2Remap(
   const Vector &rhoDetJw, const ParGridFunction &rhoJ, const ParGridFunction &eps,
   ParGridFunction &rhoeJ)
{
   const ParFiniteElementSpace &pfes = *rhoeJ.ParFESpace();
   const int NE = pfes.GetNE();
   Vector eps_min_loc(NE), eps_max_loc(NE);
   Vector eps_z;

   // Local max / min.
   for (int k = 0; k < NE; k++)
   {
      eps.GetElementDofValues(k, eps_z);
      eps_min_loc(k) = eps_z.Min();
      eps_max_loc(k) = eps_z.Max();
   }

   // Interpolation matrix
   auto M = [&pfes, this](int k, DenseMatrix &M_z, LUFactors &M_zlu) {
      const FiniteElement &fe = *pfes.GetFE(k);
      const Geometry::Type g = fe.GetGeomType();
      M_z = MJ[g];
      M_zlu.data = MJi[g].GetData();
      M_zlu.ipiv = MJi_piv[g].GetData();
   };

   // Righ hand side
   hydrodynamics::InternalEnergyIntegrator iei(rhoDetJw, eps);
   iei.SetIntRule(&ir_rho);

   auto beps = [&pfes,&iei](int k, Vector &rhs) {
      const FiniteElement &fe = *pfes.GetFE(k);
      ElementTransformation &T = *pfes.GetElementTransformation(k);
      iei.AssembleRHSElementVect(fe, T, rhs);
   };

   TransferXYL2Monotonous(M, eps_min_loc, eps_max_loc, rhoJ, beps, rhoeJ);

   // density and Jacobian product
   for (int i = 0; i < rhoJ.Size(); i++)
      rhoeJ(i) *= rhoJ(i);
}

void SolutionTransfer_L2::TransferDensityJac_Remap2Lagr(
   const ParGridFunction &detJ, const ParGridFunction &rhoJ, ParGridFunction &rho)
{
   const ParFiniteElementSpace &pfes = *rhoJ.ParFESpace();
   const int NE = pfes.GetNE();
   Vector rho_min_loc(NE), rho_max_loc(NE);
   Vector detJ_z, rhoJ_z;

   // Local max / min.
   for (int k = 0; k < NE; k++)
   {
      detJ.GetElementDofValues(k, detJ_z);
      rhoJ.GetElementDofValues(k, rhoJ_z);
      rho_min_loc(k) = +infinity();
      rho_max_loc(k) = -infinity();
      const int ndof = detJ_z.Size();

      for (int i = 0; i < ndof; i++)
      {
         const real_t rho = rhoJ_z(i) / detJ_z(i);

         rho_min_loc(k) = std::min(rho_min_loc(k), rho);
         rho_max_loc(k) = std::max(rho_max_loc(k), rho);
      }
   }

   // Mass matrix
   MassIntegrator mi(&ir_rho);
   auto M = [&pfes,&mi](int k, DenseMatrix &M_z, LUFactors &) {
      const FiniteElement &fe = *pfes.GetFE(k);
      ElementTransformation &T = *pfes.GetElementTransformation(k);
      mi.AssembleElementMatrix(fe, T, M_z);
   };

   // Right hand side
   auto brho = [&pfes,&rhoJ,this](int k, Vector &rhs) {
      const FiniteElement &fe = *pfes.GetFE(k);
      ElementTransformation &T = *pfes.GetElementTransformation(k);
      Vector shape(fe.GetDof()), rhoJ_z;
      rhoJ.GetElementDofValues(T.ElementNo, rhoJ_z);
      const int nqp = ir_rho.GetNPoints();
      rhs.SetSize(fe.GetDof());
      rhs = 0.;
      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir_rho.IntPoint(q);
         T.SetIntPoint(&ip);
         fe.CalcShape(ip, shape);
         const real_t rhoJ = rhoJ_z * shape;
         rhs.Add(ip.weight * rhoJ, shape);
      }
   };

   TransferL2Monotonous(M, rho_min_loc, rho_max_loc, brho, rho);
}

void SolutionTransfer_L2::TransferEnergyJac_Remap2Lagr(
   const Vector &rhoDetJw, const ParGridFunction &rhoJ, const ParGridFunction &rhoeJ, ParGridFunction &eps)
{
   const ParFiniteElementSpace &pfes = *rhoeJ.ParFESpace();
   const int NE = pfes.GetNE();
   Vector eps_min_loc(NE), eps_max_loc(NE);
   Vector rhoJ_z, rhoeJ_z;

   // Local max / min.
   for (int k = 0; k < NE; k++)
   {
      rhoJ.GetElementDofValues(k, rhoJ_z);
      rhoeJ.GetElementDofValues(k, rhoeJ_z);
      eps_min_loc(k) = +infinity();
      eps_max_loc(k) = -infinity();
      const int ndof = rhoJ_z.Size();

      for (int i = 0; i < ndof; i++)
      {
         const real_t eps = (rhoJ_z(i) != 0.) ? (rhoeJ_z(i) / rhoJ_z(i)) : (0.);

         eps_min_loc(k) = std::min(eps_min_loc(k), eps);
         eps_max_loc(k) = std::max(eps_max_loc(k), eps);
      }
   }

   // Energy mass matrix
   auto Me = [&pfes,&rhoDetJw,this](int k, DenseMatrix &M_z, LUFactors &) {
      const FiniteElement &fe = *pfes.GetFE(k);
      const int nqp = ir_rho.GetNPoints();
      Vector shape(fe.GetDof());
      M_z.SetSize(fe.GetDof());
      M_z = 0.;
      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir_rho.IntPoint(q);
         fe.CalcShape(ip, shape);
         AddMult_a_VVt(rhoDetJw(k*nqp + q), shape, M_z);
      }
   };

   // Right hand side
   auto beps = [&pfes,&rhoeJ,this](int k, Vector &rhs) {
      const FiniteElement &fe = *pfes.GetFE(k);
      ElementTransformation &T = *pfes.GetElementTransformation(k);
      Vector shape(fe.GetDof()), rhoeJ_z;
      rhoeJ.GetElementDofValues(T.ElementNo, rhoeJ_z);
      const int nqp = ir_rho.GetNPoints();
      rhs.SetSize(fe.GetDof());
      rhs = 0.;
      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir_rho.IntPoint(q);
         T.SetIntPoint(&ip);
         fe.CalcShape(ip, shape);
         const real_t rhoJ = rhoeJ_z * shape;
         rhs.Add(ip.weight * rhoJ, shape);
      }
   };

   TransferL2Monotonous(Me, eps_min_loc, eps_max_loc, beps, eps);
}


SolutionTransfer_H1::SolutionTransfer_H1(const Array<int> &v_ess_tdofs_, const ParFiniteElementSpace &pfes_H1_s, const IntegrationRule &ir)
: v_ess_tdofs(v_ess_tdofs_), ir_rho(ir)
{
   const ParMesh &pmesh = *pfes_H1_s.GetParMesh();

   // Interpolation matrix and lumped diagonal
   RefMassIntegrator mi(&ir_rho);
   Vector mJ_g[Geometry::NUM_GEOMETRIES];
   Array<Geometry::Type> geoms;
   pmesh.GetGeometries(pmesh.Dimension(), geoms);
   const FiniteElementCollection *fec_H1 = pfes_H1_s.FEColl();
   IsoparametricTransformation Tr; // dummy
   for (Geometry::Type g : geoms)
   {
      const FiniteElement *fe = fec_H1->GetFE(g, fec_H1->GetOrder());
      mi.AssembleElementMatrix(*fe, Tr, MJ[g]);
      MJ[g].GetRowSums(mJ_g[g]);
   }

   // Assemble the lumped mass matrix
   Vector mJ_loc(pfes_H1_s.GetVSize());
   Array<int> dofs;
   mJ_loc = 0.;
   const int NE = pmesh.GetNE();
   for (int k = 0; k < NE; k++)
   {
      pfes_H1_s.GetElementDofs(k, dofs);
      Geometry::Type g = pfes_H1_s.GetFE(k)->GetGeomType();
      mJ_loc.AddElementVector(dofs, mJ_g[g]);
   }
   mJ.SetSize(pfes_H1_s.GetTrueVSize());
   pfes_H1_s.GetProlongationMatrix()->MultTranspose(mJ_loc, mJ);
}

void SolutionTransfer_H1::TransferVelocity_Lagr2Remap(const ParGridFunction &vel_Lag, ParGridFunction &vel)
{
   const ParFiniteElementSpace &pfes_H1Lag = *vel_Lag.ParFESpace();
   ParFiniteElementSpace &pfes_H1 = *vel.ParFESpace();

   // project velocity field into Bernstein FE space via lumped L2 projection
   ParMixedBilinearForm M_mixed(const_cast<ParFiniteElementSpace*>(&pfes_H1Lag), &pfes_H1);
   M_mixed.AddDomainIntegrator(new VectorMassIntegrator());
   M_mixed.Assemble(0);
   M_mixed.Finalize(0);

   OperatorHandle M;
   M_mixed.FormRectangularSystemMatrix(v_ess_tdofs, v_ess_tdofs, M);

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

   const Operator *R_v = pfes_H1.GetRestrictionMatrix();
   Vector RHS_V(R_v->Height()), X_V(R_v->Height()), VEL(R_v->Height()), M_L(R_v->Height());
   R_v->Mult(vel_Lag, VEL);
   R_v->Mult(lumped_vec, M_L);
   M->Mult(VEL, RHS_V);
   RHS_V /= M_L;
   vel.Distribute(RHS_V);
}

void SolutionTransfer_H1::TransferVelocity_Remap2Lagr(const ParGridFunction &vel, ParGridFunction &vel_Lag)
{
   VectorGridFunctionCoefficient v_coeff(&vel);
   vel_Lag.ProjectCoefficient(v_coeff);

   // // project velocity field back to Lagrange FE space via lumped L2 projection
   // ParMixedBilinearForm M_mixed(&pfes_H1, &pfes_H1Lag);
   // M_mixed.AddDomainIntegrator(new VectorMassIntegrator());
   // M_mixed.Assemble(0);
   // M_mixed.Finalize(0);

   // OperatorHandle M;
   // M_mixed.FormRectangularSystemMatrix(v_ess_tdofs, v_ess_tdofs, M);

   // ParBilinearForm M_lumped(&pfes_H1Lag);
   // M_lumped.AddDomainIntegrator(new LumpedIntegrator(new VectorMassIntegrator()));
   // M_lumped.Assemble(0);
   // M_lumped.Finalize(0);

   // Vector lumped_vec(M_lumped.Height());
   // M_lumped.SpMat().GetDiag(lumped_vec);
   // GroupCommunicator &gcomm = pfes_H1Lag.GroupComm();
   // Array<double> lumpedmassmatrix_array(lumped_vec.GetData(), lumped_vec.Size());
   // gcomm.Reduce<double>(lumpedmassmatrix_array, GroupCommunicator::Sum);
   // gcomm.Bcast(lumpedmassmatrix_array);

   // const Operator *R_v = pfes_H1Lag.GetRestrictionMatrix();
   // Vector RHS_V(R_v->Height()), X_V(R_v->Height()), V(R_v->Height()), M_L(R_v->Height());
   // R_v->Mult(vel, V);
   // R_v->Mult(lumped_vec, M_L);
   // M->Mult(V, RHS_V);
   // RHS_V /= M_L;
   // vel_Lag.Distribute(RHS_V);
}

void SolutionTransfer_H1::TransferJac_Larg2Remap(ParGridFunction &detJ)
{
   ParFiniteElementSpace &pfes_H1_s = *detJ.ParFESpace();

   ConstantCoefficient one;
   ParLinearForm b(&pfes_H1_s);
   b.AddDomainIntegrator(new DomainLFIntegrator(one, &ir_rho));
   b.Assemble();

   Vector B(pfes_H1_s.GetTrueVSize());
   b.ParallelAssemble(B);
   Vector detJ_tv(pfes_H1_s.GetTrueVSize());
   for (int i = 0; i < detJ_tv.Size(); i++)
      detJ_tv(i) = B(i) / mJ(i);
   detJ.Distribute(detJ_tv);
}

void SolutionTransfer_H1::TransferMomentumJac_Lagr2Remap(
   const Vector &rhoDetJw, const ParGridFunction &vel, ParGridFunction &rhouJ)
{
   ParFiniteElementSpace &pfes_H1_Lag = *vel.ParFESpace();
   ParFiniteElementSpace &pfes_H1 = *rhouJ.ParFESpace();
   const int vdim = pfes_H1.GetVDim();
   Vector b(pfes_H1.GetVSize()); b = 0.;
   DenseMatrix vel_k, b_k;
   Vector shape_Lag, shape, vel_q(vdim);
   Array<int> vdofs_Lag, vdofs;
   const int NE = pfes_H1.GetNE();
   const int nqp = ir_rho.GetNPoints();
   for(int k = 0; k < NE; k++)
   {
      const FiniteElement &fe_Lag = *pfes_H1_Lag.GetFE(k);
      const FiniteElement &fe = *pfes_H1.GetFE(k);
      const int ndofs_Lag = fe_Lag.GetDof();
      const int ndofs = fe.GetDof();
      shape_Lag.SetSize(ndofs_Lag);
      shape.SetSize(ndofs);
      pfes_H1_Lag.GetElementVDofs(k, vdofs_Lag);
      pfes_H1.GetElementVDofs(k, vdofs);
      vel_k.SetSize(ndofs_Lag, vdim);
      vel.GetSubVector(vdofs_Lag, vel_k.GetData());
      b_k.SetSize(ndofs, vdim);
      b_k = 0.;
      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir_rho.IntPoint(q);
         fe_Lag.CalcShape(ip, shape_Lag);
         fe.CalcShape(ip, shape);
         vel_k.MultTranspose(shape_Lag, vel_q);
         const real_t w = rhoDetJw(k * nqp + q);
         for (int v = 0; v < vdim; v++)
         {
            Vector b_kv;
            b_k.GetColumnReference(v, b_kv);
            b_kv.Add(w * vel_q(v), shape);
         }
      }
      b.AddElementVector(vdofs, b_k.GetData());
   }

   Vector B(pfes_H1.GetTrueVSize());
   pfes_H1.GetProlongationMatrix()->MultTranspose(b, B);
   Vector rhouJ_tv(pfes_H1.GetTrueVSize());
   for (int i = 0; i < mJ.Size(); i++)
      for (int v = 0; v < vdim; v++)
         rhouJ_tv(i + v*mJ.Size()) = B(i + v*mJ.Size()) / mJ(i);
   rhouJ.Distribute(rhouJ_tv);
}

void SolutionTransfer_H1::TransferMomentumJac_Remap2Lagr(const Vector &rhoDetJw, const ParGridFunction &rhouJ, ParGridFunction &vel)
{
   ParFiniteElementSpace &pfes_H1_Lag = *vel.ParFESpace();
   ParFiniteElementSpace &pfes_H1 = *rhouJ.ParFESpace();
   const int vdim = pfes_H1.GetVDim();

#if 1
   // H1 lumped projection
   Vector m(pfes_H1_Lag.GetVSize()); m = 0.;
   Vector b(pfes_H1_Lag.GetVSize()); b = 0.;
   DenseMatrix rhouJ_k, m_k, b_k;
   Vector shape_Lag, shape, rhouJ_q(vdim);
   Array<int> vdofs_Lag, vdofs;
   const int NE = pfes_H1.GetNE();
   const int nqp = ir_rho.GetNPoints();
   for(int k = 0; k < NE; k++)
   {
      const FiniteElement &fe_Lag = *pfes_H1_Lag.GetFE(k);
      const FiniteElement &fe = *pfes_H1.GetFE(k);
      const int ndofs_Lag = fe_Lag.GetDof();
      const int ndofs = fe.GetDof();
      shape_Lag.SetSize(ndofs_Lag);
      shape.SetSize(ndofs);
      pfes_H1_Lag.GetElementVDofs(k, vdofs_Lag);
      pfes_H1.GetElementVDofs(k, vdofs);
      rhouJ_k.SetSize(ndofs, vdim);
      rhouJ.GetSubVector(vdofs, rhouJ_k.GetData());
      m_k.SetSize(ndofs_Lag, vdim);
      m_k = 0.;
      b_k.SetSize(ndofs_Lag, vdim);
      b_k = 0.;
      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir_rho.IntPoint(q);
         fe_Lag.CalcShape(ip, shape_Lag);
         fe.CalcShape(ip, shape);
         rhouJ_k.MultTranspose(shape, rhouJ_q);
         const real_t w = rhoDetJw(k * nqp + q);
         for (int v = 0; v < vdim; v++)
         {
            Vector m_kv, b_kv;
            m_k.GetColumnReference(v, m_kv);
            b_k.GetColumnReference(v, b_kv);
            m_kv.Add(w, shape_Lag);
            b_kv.Add(ip.weight * rhouJ_q(v), shape_Lag);
         }
      }
      m.AddElementVector(vdofs_Lag, m_k.GetData());
      b.AddElementVector(vdofs_Lag, b_k.GetData());
   }

   Vector M(pfes_H1_Lag.GetTrueVSize());
   pfes_H1_Lag.GetProlongationMatrix()->MultTranspose(m, M);
   Vector B(pfes_H1_Lag.GetTrueVSize());
   pfes_H1_Lag.GetProlongationMatrix()->MultTranspose(b, B);
   Vector vel_tv(pfes_H1_Lag.GetTrueVSize());
   for (int i = 0; i < M.Size(); i++)
      vel_tv(i) = B(i) / M(i);
   vel.Distribute(vel_tv);
#else
   // H1 projection
   ParFiniteElementSpace pfes_H1_Lag_s(pfes_H1_Lag.GetParMesh(), pfes_H1_Lag.FEColl());
   ParBilinearForm Mv(&pfes_H1_Lag_s);
   Vector b(pfes_H1_Lag.GetVSize()); b = 0.;
   DenseMatrix rhouJ_k, m_k, b_k;
   Vector shape_Lag, shape, rhouJ_q(vdim);
   Array<int> vdofs_Lag, vdofs;
   const int NE = pfes_H1.GetNE();
   const int nqp = ir_rho.GetNPoints();
   for(int k = 0; k < NE; k++)
   {
      const FiniteElement &fe_Lag = *pfes_H1_Lag.GetFE(k);
      const FiniteElement &fe = *pfes_H1.GetFE(k);
      const int ndofs_Lag = fe_Lag.GetDof();
      const int ndofs = fe.GetDof();
      shape_Lag.SetSize(ndofs_Lag);
      shape.SetSize(ndofs);
      pfes_H1_Lag.GetElementVDofs(k, vdofs_Lag);
      pfes_H1.GetElementVDofs(k, vdofs);
      rhouJ_k.SetSize(ndofs, vdim);
      rhouJ.GetSubVector(vdofs, rhouJ_k.GetData());
      m_k.SetSize(ndofs_Lag);
      m_k = 0.;
      b_k.SetSize(ndofs_Lag, vdim);
      b_k = 0.;
      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir_rho.IntPoint(q);
         fe_Lag.CalcShape(ip, shape_Lag);
         fe.CalcShape(ip, shape);
         rhouJ_k.MultTranspose(shape, rhouJ_q);

         // right hand side
         for (int v = 0; v < vdim; v++)
         {
            Vector b_kv;
            b_k.GetColumnReference(v, b_kv);
            b_kv.Add(ip.weight * rhouJ_q(v), shape_Lag);
         }
         const real_t w = rhoDetJw(k * nqp + q);
         AddMult_a_VVt(w, shape_Lag, m_k);
      }

      // mass matrix
      Mv.AssembleElementMatrix(k, m_k);
      b.AddElementVector(vdofs_Lag, b_k.GetData());
   }

   Mv.Assemble();
   Mv.Finalize();
   HypreParMatrix &Mv_m = *Mv.ParallelAssembleInternalMatrix();

   const int ntdofs = pfes_H1_Lag_s.GetTrueVSize();
   Vector B(ntdofs * vdim);
   pfes_H1_Lag.GetProlongationMatrix()->MultTranspose(b, B);
   Vector X(ntdofs * vdim);
   
   HypreSmoother prec;
   prec.SetType(HypreSmoother::Jacobi, 1);
   GMRESSolver lin_solver(pfes_H1_Lag_s.GetComm());
   lin_solver.SetPreconditioner(prec);
   lin_solver.SetRelTol(1e-10);
   lin_solver.SetAbsTol(0.0);
   lin_solver.SetMaxIter(100);
   lin_solver.SetPrintLevel(3);
   lin_solver.SetOperator(Mv_m);

   Vector X_v, B_v;

   for (int v = 0; v < vdim; v++)
   {
      B_v.MakeRef(B, v*ntdofs, ntdofs);
      X_v.MakeRef(X, v*ntdofs, ntdofs);
      lin_solver.Mult(B_v, X_v);
   }
   vel.Distribute(X);
#endif
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

} // namespace ale

} // namespace mfem
