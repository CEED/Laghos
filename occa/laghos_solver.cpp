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

LagrangianHydroOperator::LagrangianHydroOperator(ProblemOption problem_,
                                                 OccaFiniteElementSpace &o_H1FESpace_,
                                                 OccaFiniteElementSpace &o_L2FESpace_,
                                                 Array<int> &essential_tdofs_,
                                                 OccaGridFunction &rho0,
                                                 double cfl_,
                                                 Coefficient *material_,
                                                 bool use_viscosity_,
                                                 double cg_tol_,
                                                 int cg_max_iter_,
                                                 occa::properties &props)
   : TimeDependentOperator(o_L2FESpace_.GetVSize() + 2*o_H1FESpace_.GetVSize()),
     problem(problem_),
     device(o_H1FESpace_.GetDevice()),
     o_H1FESpace(o_H1FESpace_),
     o_L2FESpace(o_L2FESpace_),
     o_H1compFESpace(o_H1FESpace.GetMesh(),
                     o_H1FESpace.FEColl(),
                     1),
     H1FESpace(*((ParFiniteElementSpace*) o_H1FESpace.GetFESpace())),
     L2FESpace(*((ParFiniteElementSpace*) o_L2FESpace.GetFESpace())),
     essential_tdofs(essential_tdofs_),
     dim(H1FESpace.GetMesh()->Dimension()),
     elements(H1FESpace.GetMesh()->GetNE()),
     l2dofs_cnt(L2FESpace.GetFE(0)->GetDof()),
     h1dofs_cnt(H1FESpace.GetFE(0)->GetDof()),
     cfl(cfl_),
     use_viscosity(use_viscosity_),
     Mv(&H1FESpace),
     Me_inv(l2dofs_cnt, l2dofs_cnt, elements),
     integ_rule(IntRules.Get(H1FESpace.GetMesh()->GetElementBaseGeometry(0),
                             3*H1FESpace.GetOrder(0) + L2FESpace.GetOrder(0) - 1)),
     cg_max_iter(cg_max_iter_),
     cg_rel_tol(cg_tol_),
     cg_abs_tol(0.0),
     cg_print_level(0),
     quad_data(dim, elements, integ_rule.GetNPoints()),
     quad_data_is_current(false),
     VMass(o_H1compFESpace, integ_rule, &quad_data),
     EMass(o_L2FESpace, integ_rule, &quad_data),
     Force(o_H1FESpace, o_L2FESpace, integ_rule, &quad_data),
     timer()
{

   Vector rho0_ = rho0;
   GridFunction rho0_gf(&L2FESpace, rho0_.GetData());
   GridFunctionCoefficient rho_coeff(&rho0_gf);

   // Standard local assembly and inversion for energy mass matrices.
   DenseMatrix Me(l2dofs_cnt);
   DenseMatrixInverse inv(&Me);
   MassIntegrator mi(rho_coeff, &integ_rule);
   for (int el = 0; el < elements; ++el)
   {
      mi.AssembleElementMatrix(*L2FESpace.GetFE(el),
                               *L2FESpace.GetElementTransformation(el), Me);
      inv.Factor();
      inv.GetInverseMatrix(Me_inv(el));
   }

   // Standard assembly for the velocity mass matrix.
   VectorMassIntegrator *vmi = new VectorMassIntegrator(rho_coeff, &integ_rule);
   Mv.AddDomainIntegrator(vmi);
   Mv.Assemble();

   // Initial local mesh size (assumes similar cells).
   double loc_area = 0.0, glob_area;
   int glob_z_cnt;
   ParMesh *pm = H1FESpace.GetParMesh();
   for (int el = 0; el < elements; ++el)
   {
      loc_area += pm->GetElementVolume(el);
   }
   MPI_Allreduce(&loc_area, &glob_area, 1, MPI_DOUBLE, MPI_SUM, pm->GetComm());
   MPI_Allreduce(&elements, &glob_z_cnt, 1, MPI_INT, MPI_SUM, pm->GetComm());
   switch (pm->GetElementBaseGeometry(0))
   {
      case Geometry::SQUARE:
         quad_data.h0 = sqrt(glob_area / glob_z_cnt);
         break;
      case Geometry::TRIANGLE:
         quad_data.h0 = sqrt(2.0 * glob_area / glob_z_cnt);
         break;
      case Geometry::CUBE:
         quad_data.h0 = pow(glob_area / glob_z_cnt, 1.0/3.0);
         break;
      case Geometry::TETRAHEDRON:
         quad_data.h0 = pow(6.0 * glob_area / glob_z_cnt, 1.0/3.0);
         break;
      default: MFEM_ABORT("Unknown zone type!");
   }
   quad_data.h0 /= (double) H1FESpace.GetOrder(0);

   // Setup OCCA QuadratureData
   quad_data.device = device;

   quad_data.dqMaps = OccaDofQuadMaps::Get(device,
                                           o_H1FESpace,
                                           integ_rule);
   quad_data.geom = OccaGeometry::Get(device,
                                      o_H1FESpace,
                                      integ_rule);

   quad_data.Jac0inv = quad_data.geom.invJ;

   OccaVector rhoValues;
   rho0.ToQuad(integ_rule, rhoValues);

   SetProperties(o_H1FESpace, integ_rule, quad_data.props);
   quad_data.props["defines/H0"]            = quad_data.h0;
   quad_data.props["defines/CFL"]           = cfl;
   quad_data.props["defines/USE_VISCOSITY"] = use_viscosity;
   quad_data.props += props;

   occa::kernel initKernel = device.buildKernel("occa://laghos/quadratureData.okl",
                                                "InitQuadratureData",
                                                quad_data.props);

   initKernel(elements,
              rhoValues,
              quad_data.geom.detJ,
              quad_data.dqMaps.quadWeights,
              quad_data.rho0DetJ0w);

   updateKernel = device.buildKernel("occa://laghos/quadratureData.okl",
                                     stringWithDim("UpdateQuadratureData", dim),
                                     quad_data.props);

   // Needs quad_data.rho0DetJ0w
   Force.Setup();
   VMass.Setup();
   EMass.Setup();
}

void LagrangianHydroOperator::Mult(const OccaVector &S,
                                   OccaVector &dS_dt) const
{
   dS_dt = 0.0;

   // Make sure that the mesh positions correspond to the ones in S. This is
   // needed only because some mfem time integrators don't update the solution
   // vector at every intermediate stage (hence they don't change the mesh).
   const int Vsize_h1 = H1FESpace.GetVSize();
   const int Vsize_l2 = L2FESpace.GetVSize();

   // The monolithic BlockVector stores the unknown fields as follows:
   // - Position
   // - Velocity
   // - Specific Internal Energy
   OccaVector x = S.GetRange(0         , Vsize_h1);
   OccaVector v = S.GetRange(Vsize_h1  , Vsize_h1);
   OccaVector e = S.GetRange(2*Vsize_h1, Vsize_l2);

   OccaVector dx = dS_dt.GetRange(0         , Vsize_h1);
   OccaVector dv = dS_dt.GetRange(Vsize_h1  , Vsize_h1);
   OccaVector de = dS_dt.GetRange(2*Vsize_h1, Vsize_l2);

   Vector h_x = x;
   ParGridFunction h_px(&H1FESpace, h_x.GetData());

   o_H1FESpace.GetMesh()->NewNodes(h_px, false);
   UpdateQuadratureData(S);

   // Set dx_dt = v (explicit).
   dx = v;

   // Solve for velocity.
   OccaVector one(Vsize_l2);
   OccaVector rhs(Vsize_h1);
   one = 1.0;

   timer.sw_force.Start();
   Force.Mult(one, rhs);
   timer.sw_force.Stop();
   rhs.Neg();

   OccaVector B(o_H1compFESpace.GetTrueVSize());
   OccaVector X(o_H1compFESpace.GetTrueVSize());

   // Partial assembly solve for each velocity component.
   dv = 0.0;

   for (int c = 0; c < dim; c++)
   {
      const int size = o_H1compFESpace.GetVSize();
      OccaVector rhs_c = rhs.GetRange(c*size, size);
      OccaVector dv_c  = dv.GetRange(c*size, size);

      // Attributes 1/2/3 correspond to fixed-x/y/z boundaries, i.e.,
      // we must enforce v_x/y/z = 0 for the velocity components.
      Array<int> ess_bdr(H1FESpace.GetParMesh()->bdr_attributes.Max());
      ess_bdr = 0;
      ess_bdr[c] = 1;

      o_H1compFESpace.GetProlongationOperator()->MultTranspose(rhs_c, B);
      o_H1compFESpace.GetRestrictionOperator()->Mult(dv_c, X);

      // True dofs as if there's only one component.
      Array<int> c_tdofs;
      o_H1compFESpace.GetFESpace()->GetEssentialTrueDofs(ess_bdr, c_tdofs);

      VMass.SetEssentialTrueDofs(c_tdofs);
      VMass.EliminateRHS(B);

      {
         OccaCGSolver cg(H1FESpace.GetParMesh()->GetComm());
         cg.SetOperator(VMass);
         cg.SetRelTol(cg_rel_tol);
         cg.SetAbsTol(cg_abs_tol);
         cg.SetMaxIter(cg_max_iter);
         cg.SetPrintLevel(0);
         timer.sw_cgH1.Start();

         cg.Mult(B, X);

         timer.sw_cgH1.Stop();
         timer.H1cg_iter += cg.GetNumIterations();
      }

      o_H1compFESpace.GetProlongationOperator()->Mult(X, dv_c);
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

   OccaVector forceRHS(Vsize_l2);
   timer.sw_force.Start();
   Force.MultTranspose(v, forceRHS);
   timer.sw_force.Stop();

   if (e_source)
   {
      forceRHS += *e_source;
   }

   {
      // TODO: Local element-wise CG
      OccaCGSolver cg(L2FESpace.GetParMesh()->GetComm());
      cg.SetOperator(EMass);
      cg.SetRelTol(sqrt(cg_rel_tol));
      cg.SetAbsTol(sqrt(cg_abs_tol));
      cg.SetMaxIter(cg_max_iter);
      cg.SetPrintLevel(cg_print_level);

      timer.sw_cgL2.Start();
      cg.Mult(forceRHS, de);
      timer.sw_cgL2.Stop();
      timer.L2dof_iter += cg.GetNumIterations() * l2dofs_cnt;
   }

   delete e_source;
   quad_data_is_current = false;
}

double LagrangianHydroOperator::GetTimeStepEstimate(const OccaVector &S) const
{
   OccaVector x = S.GetRange(0, H1FESpace.GetVSize());
   Vector h_x = x;
   ParGridFunction h_px(&H1FESpace, h_x.GetData());
   o_H1FESpace.GetMesh()->NewNodes(h_px, false);

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

   Vector rho0DetJ0w = quad_data.rho0DetJ0w;

   for (int el = 0; el < elements; ++el)
   {
      di.AssembleRHSElementVect(*L2FESpace.GetFE(el),
                                *L2FESpace.GetElementTransformation(el),
                                integ_rule,
                                rho0DetJ0w,
                                rhs);
      mi.AssembleElementMatrix(*L2FESpace.GetFE(el),
                               *L2FESpace.GetElementTransformation(el),
                               Mrho);
      inv.Factor();
      inv.Mult(rhs, rho_z);
      L2FESpace.GetElementDofs(el, dofs);
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
   mydata[0] = timer.L2dof_iter;
   mydata[1] = timer.quad_tstep;
   MPI_Reduce(mydata, alldata, 2, HYPRE_MPI_INT, MPI_SUM, 0,
              H1FESpace.GetComm());

   if (IamRoot)
   {
      const HYPRE_Int H1gsize = H1FESpace.GlobalTrueVSize(),
                      L2gsize = L2FESpace.GlobalTrueVSize();
      using namespace std;
      cout << endl;
      cout << "CG (H1) total time: " << rt_max[0] << endl;
      cout << "CG (H1) rate (megadofs x cg_iterations / second): "
           << 1e-6 * H1gsize * timer.H1cg_iter / rt_max[0] << endl;
      cout << endl;
      cout << "CG (L2) total time: " << rt_max[1] << endl;
      cout << "CG (L2) rate (megadofs x cg_iterations / second): "
           << 1e-6 * alldata[0] / rt_max[1] << endl;
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

LagrangianHydroOperator::~LagrangianHydroOperator() {}

void LagrangianHydroOperator::UpdateQuadratureData(const OccaVector &S) const
{
   if (quad_data_is_current)
   {
      return;
   }

   timer.sw_qdata.Start();
   quad_data_is_current = true;

   const int vSize = o_H1FESpace.GetVSize();
   const int eSize = o_L2FESpace.GetVSize();

   OccaGridFunction v(&o_H1FESpace, S.GetRange(vSize  , vSize));
   OccaGridFunction e(&o_L2FESpace, S.GetRange(2*vSize, eSize));

   quad_data.geom = OccaGeometry::Get(device,
                                      o_H1FESpace,
                                      integ_rule);

   OccaVector v2(device,
                 o_H1FESpace.GetVDim() * o_H1FESpace.GetLocalDofs() * elements);
   o_H1FESpace.GlobalToLocal(v, v2);

   OccaVector eValues;
   e.ToQuad(integ_rule, eValues);

   updateKernel(elements,
                quad_data.dqMaps.dofToQuad,
                quad_data.dqMaps.dofToQuadD,
                quad_data.dqMaps.quadWeights,
                v2,
                eValues,
                quad_data.rho0DetJ0w,
                quad_data.Jac0inv,
                quad_data.geom.J,
                quad_data.geom.invJ,
                quad_data.geom.detJ,
                quad_data.stressJinvT,
                quad_data.dtEst);

   quad_data.dt_est = quad_data.dtEst.Min();

   timer.sw_qdata.Stop();
   timer.quad_tstep += elements;
}
} // namespace hydrodynamics
} // namespace mfem

#endif // MFEM_USE_MPI
