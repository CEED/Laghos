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

namespace hydrodynamics
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
              << "keys maaAcl";
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

// ***************************************************************************
// * LagrangianHydroOperator
// ***************************************************************************
LagrangianHydroOperator::LagrangianHydroOperator(int size,
                                                 HipFiniteElementSpace &h1_fes,
                                                 HipFiniteElementSpace &l2_fes,
                                                 Array<int> &essential_tdofs,
                                                 HipGridFunction &rho0,
                                                 int source_type_, double cfl_,
                                                 Coefficient *material_,
                                                 bool visc, bool pa,
                                                 double cgt, int cgiter)
   : HipTimeDependentOperator(size),
     H1FESpace(h1_fes), L2FESpace(l2_fes),
     H1compFESpace(h1_fes.GetParMesh(), h1_fes.FEColl(),1),
     ess_tdofs(essential_tdofs),
     dim(h1_fes.GetMesh()->Dimension()),
     nzones(h1_fes.GetMesh()->GetNE()),
     l2dofs_cnt(l2_fes.GetFE(0)->GetDof()),
     h1dofs_cnt(h1_fes.GetFE(0)->GetDof()),
     source_type(source_type_), cfl(cfl_),
     use_viscosity(visc), p_assembly(pa), cg_rel_tol(cgt), cg_max_iter(cgiter),
     material_pcf(material_),
     integ_rule(IntRules.Get(h1_fes.GetMesh()->GetElementBaseGeometry(0),
                             3*h1_fes.GetOrder(0) + l2_fes.GetOrder(0) - 1)),
     quad_data(dim, nzones, integ_rule.GetNPoints()),
     quad_data_is_current(false),
     VMassPA(H1compFESpace, integ_rule, &quad_data),
     EMassPA(L2FESpace, integ_rule, &quad_data),
     VMassPA_prec(H1FESpace),
     ForcePA(H1FESpace, L2FESpace, integ_rule, &quad_data),
     CG_VMass(H1FESpace.GetParMesh()->GetComm()),
     CG_EMass(L2FESpace.GetParMesh()->GetComm()),
     timer(),
     v(),e(),
     rhs(H1FESpace.GetVSize()),
     B(H1compFESpace.GetTrueVSize()),X(H1compFESpace.GetTrueVSize()),
     one(L2FESpace.GetVSize(),1.0),
     e_rhs(L2FESpace.GetVSize()),
     rhs_c(H1compFESpace.GetVSize()),
     v_local(H1FESpace.GetVDim() * H1FESpace.GetLocalDofs()*nzones),
     e_quad()
{
   // Initial local mesh size (assumes similar cells).
   double loc_area = 0.0, glob_area;
   int loc_z_cnt = nzones, glob_z_cnt;
   ParMesh *pm = H1FESpace.GetParMesh();
   for (int i = 0; i < nzones; i++) { loc_area += pm->GetElementVolume(i); }
   MPI_Allreduce(&loc_area, &glob_area, 1, MPI_DOUBLE, MPI_SUM, pm->GetComm());
   MPI_Allreduce(&loc_z_cnt, &glob_z_cnt, 1, MPI_INT, MPI_SUM, pm->GetComm());
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

   quad_data.dqMaps = HipDofQuadMaps::Get(H1FESpace,integ_rule);
   quad_data.geom = HipGeometry::Get(H1FESpace,integ_rule);
   quad_data.Jac0inv = quad_data.geom->invJ;

   HipVector rhoValues; // used in rInitQuadratureData
   rho0.ToQuad(integ_rule, rhoValues);

   if (dim==1) { assert(false); }
   const int NUM_QUAD = integ_rule.GetNPoints();

   rInitQuadratureData(NUM_QUAD,
                       nzones,
                       rhoValues,
                       quad_data.geom->detJ,
                       quad_data.dqMaps->quadWeights,
                       quad_data.rho0DetJ0w);

   // Needs quad_data.rho0DetJ0w
   ForcePA.Setup();
   VMassPA.Setup();
   EMassPA.Setup();

   {
      // Setup the preconditioner of the velocity mass operator.
      //Vector d;
      //#warning ComputeDiagonal
      //(dim == 2) ? VMassPA.ComputeDiagonal2D(d) : VMassPA.ComputeDiagonal3D(d);
      //VMassPA_prec.SetDiagonal(d);
   }

   CG_VMass.SetOperator(VMassPA);
   CG_VMass.SetRelTol(cg_rel_tol);
   CG_VMass.SetAbsTol(0.0);
   CG_VMass.SetMaxIter(cg_max_iter);
   CG_VMass.SetPrintLevel(-1);

   CG_EMass.SetOperator(EMassPA);
   CG_EMass.iterative_mode = false;
   CG_EMass.SetRelTol(1e-8);
   CG_EMass.SetAbsTol(1e-8 * numeric_limits<double>::epsilon());
   CG_EMass.SetMaxIter(200);
   CG_EMass.SetPrintLevel(-1);
}

// *****************************************************************************
LagrangianHydroOperator::~LagrangianHydroOperator() {}

// *****************************************************************************
void LagrangianHydroOperator::Mult(const HipVector &S, HipVector &dS_dt) const
{
   dS_dt = 0.0;

   // Make sure that the mesh positions correspond to the ones in S. This is
   // needed only because some mfem time integrators don't update the solution
   // vector at every intermediate stage (hence they don't change the mesh).
   Vector h_x = HipVector(S.GetRange(0, H1FESpace.GetVSize()));
   ParGridFunction x(&H1FESpace, h_x.GetData());
   H1FESpace.GetParMesh()->NewNodes(x, false);

   UpdateQuadratureData(S);

   // The monolithic BlockVector stores the unknown fields as follows:
   // - Position
   // - Velocity
   // - Specific Internal Energy
   const int VsizeL2 = L2FESpace.GetVSize();
   const int VsizeH1 = H1FESpace.GetVSize();

   v = S.GetRange(VsizeH1, VsizeH1);
   e = S.GetRange(2*VsizeH1, VsizeL2);

   HipVector dx = dS_dt.GetRange(0, VsizeH1);
   HipVector dv = dS_dt.GetRange(VsizeH1, VsizeH1);
   HipVector de = dS_dt.GetRange(2*VsizeH1, VsizeL2);

   // Set dx_dt = v (explicit)
   dx = v;

   // Solve for velocity.
   timer.sw_force.Start();
   ForcePA.Mult(one, rhs);
   timer.sw_force.Stop();
   rhs.Neg();

   // Partial assembly solve for each velocity component.
   const int size = H1compFESpace.GetVSize();

   for (int c = 0; c < dim; c++)
   {
      rhs_c = rhs.GetRange(c*size, size);
      HipVector dv_c = dv.GetRange(c*size, size);
      Array<int> c_tdofs;
      Array<int> ess_bdr(H1FESpace.GetMesh()->bdr_attributes.Max());
      // Attributes 1/2/3 correspond to fixed-x/y/z boundaries, i.e.,
      // we must enforce v_x/y/z = 0 for the velocity components.
      ess_bdr = 0; ess_bdr[c] = 1;
      // Essential true dofs as if there's only one component.
      H1compFESpace.GetEssentialTrueDofs(ess_bdr, c_tdofs);

      dv_c = 0.0;

      H1compFESpace.GetProlongationOperator()->MultTranspose(rhs_c, B);
      H1compFESpace.GetRestrictionOperator()->Mult(dv_c, X);

      VMassPA.SetEssentialTrueDofs(c_tdofs);
      VMassPA.EliminateRHS(B);

      timer.sw_cgH1.Start();
      CG_VMass.Mult(B, X);
      timer.sw_cgH1.Stop();
      timer.H1cg_iter += CG_VMass.GetNumIterations();
      //printf("\n[H1cg_iter] %d",timer.H1cg_iter);
      H1compFESpace.GetProlongationOperator()->Mult(X, dv_c);
   }


   // Solve for energy, assemble the energy source if such exists.
   LinearForm *e_source = NULL;
   if (source_type == 1) // 2D Taylor-Green.
   {
      e_source = new LinearForm(&L2FESpace);
      assert(L2FESpace.FEColl());
      TaylorCoefficient coeff;
      DomainLFIntegrator *d = new DomainLFIntegrator(coeff, &integ_rule);
      e_source->AddDomainIntegrator(d);
      e_source->Assemble();
   }
   Array<int> l2dofs;
   {
      timer.sw_force.Start();
      ForcePA.MultTranspose(v, e_rhs);
      timer.sw_force.Stop();
   }

   if (e_source) { e_rhs += *e_source; }

   {
      timer.sw_cgL2.Start();
      CG_EMass.Mult(e_rhs, de);
      timer.sw_cgL2.Stop();
      timer.L2cg_iter += CG_EMass.GetNumIterations();
   }
   delete e_source;
   quad_data_is_current = false;
}

double LagrangianHydroOperator::GetTimeStepEstimate(const HipVector &S) const
{
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
   DensityIntegrator di(quad_data,integ_rule);
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

void LagrangianHydroOperator::PrintTimingData(bool IamRoot, int steps)
{
   double my_rt[5], rt_max[5];
   my_rt[0] = timer.sw_cgH1.RealTime();
   my_rt[1] = timer.sw_cgL2.RealTime();
   my_rt[2] = timer.sw_force.RealTime();
   my_rt[3] = timer.sw_qdata.RealTime();
   my_rt[4] = my_rt[0] + my_rt[2] + my_rt[3];
   MPI_Reduce(my_rt, rt_max, 5, MPI_DOUBLE, MPI_MAX, 0, H1FESpace.GetComm());

   HYPRE_Int mydata[2], alldata[2];
   mydata[0] = timer.L2cg_iter;
   mydata[1] = timer.quad_tstep;
   MPI_Reduce(mydata, alldata, 2, HYPRE_MPI_INT, MPI_SUM, 0, H1FESpace.GetComm());

   if (IamRoot)
   {
      const HYPRE_Int H1gsize = H1FESpace.GlobalTrueVSize(),
                      L2gsize = L2FESpace.GlobalTrueVSize();
      using namespace std;
      cout << endl;
      cout << "CG (H1) total time: " << rt_max[0] << endl;
      cout << "CG (H1) rate (megadofs="<<H1gsize<<" x cg_iterations="<<timer.H1cg_iter<<" / second): "
           << 1e-6 * H1gsize * timer.H1cg_iter / rt_max[0] << endl;
      cout << endl;
      cout << "CG (L2) total time: " << rt_max[1] << endl;
      cout << "CG (L2) rate (megadofs x cg_iterations / second): "
           << 1e-6 * L2gsize * timer.L2cg_iter/*alldata[0]*/ / rt_max[1] << endl;
      cout << endl;
      // The Force operator is applied twice per time step, on the H1 and the L2
      // vectors, respectively.
      cout << "Forces total time: " << rt_max[2] << endl;
      cout << "Forces rate (megadofs x timesteps / second): "
           << 1e-6 * steps * (H1gsize + L2gsize) / rt_max[2] << endl;
      cout << endl;
      cout << "UpdateQuadData total time: " << rt_max[3] << endl;
      cout << "UpdateQuadData rate (megaquads x timesteps / second): "
           << 1e-6 * alldata[1] * integ_rule.GetNPoints() / rt_max[3] << endl;
      cout << endl;
      cout << "Major kernels total time (seconds): " << rt_max[4] << endl;
      cout << "Major kernels total rate (megadofs x time steps / second): "
           << 1e-6 * H1gsize * steps / rt_max[4] << endl;
   }
}

// *****************************************************************************
void LagrangianHydroOperator::UpdateQuadratureData(const HipVector &S) const
{
   if (quad_data_is_current) { return; }

   timer.sw_qdata.Start();

   const int vSize = H1FESpace.GetVSize();
   const int eSize = L2FESpace.GetVSize();

   const HipVector x = S.GetRange(0, vSize);
   HipVector v = S.GetRange(vSize, vSize);
   HipGridFunction e(L2FESpace, S.GetRange(2*vSize, eSize));

   quad_data.geom = HipGeometry::Get(H1FESpace,integ_rule,x);
   H1FESpace.GlobalToLocal(v, v_local);
   e.ToQuad(integ_rule, e_quad);

   const int NUM_QUAD = integ_rule.GetNPoints();
   const IntegrationRule &ir1D = IntRules.Get(Geometry::SEGMENT,
                                              integ_rule.GetOrder());
   const int NUM_QUAD_1D  = ir1D.GetNPoints();
   const int NUM_DOFS_1D  = H1FESpace.GetFE(0)->GetOrder()+1;

   ElementTransformation *T = H1FESpace.GetElementTransformation(0);
   const IntegrationPoint &ip = integ_rule.IntPoint(0);
   const double gamma = material_pcf->Eval(*T, ip);
   if (rconfig::Get().Share())
      rUpdateQuadratureDataS(gamma,
                             quad_data.h0,
                             cfl,
                             use_viscosity,
                             dim,
                             NUM_QUAD,
                             NUM_QUAD_1D,
                             NUM_DOFS_1D,
                             nzones,
                             quad_data.dqMaps->dofToQuad,
                             quad_data.dqMaps->dofToQuadD,
                             quad_data.dqMaps->quadWeights,
                             v_local,
                             e_quad,
                             quad_data.rho0DetJ0w,
                             quad_data.Jac0inv,
                             quad_data.geom->J,
                             quad_data.geom->invJ,
                             quad_data.geom->detJ,
                             quad_data.stressJinvT,
                             quad_data.dtEst);
   else
      rUpdateQuadratureData(gamma,
                            quad_data.h0,
                            cfl,
                            use_viscosity,
                            dim,
                            NUM_QUAD,
                            NUM_QUAD_1D,
                            NUM_DOFS_1D,
                            nzones,
                            quad_data.dqMaps->dofToQuad,
                            quad_data.dqMaps->dofToQuadD,
                            quad_data.dqMaps->quadWeights,
                            v_local,
                            e_quad,
                            quad_data.rho0DetJ0w,
                            quad_data.Jac0inv,
                            quad_data.geom->J,
                            quad_data.geom->invJ,
                            quad_data.geom->detJ,
                            quad_data.stressJinvT,
                            quad_data.dtEst);

   quad_data.dt_est = quad_data.dtEst.Min();
   quad_data_is_current = true;
   timer.sw_qdata.Stop();
   timer.quad_tstep += nzones;
}

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI
