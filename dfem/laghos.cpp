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
//
// 3point:
// explicit:
//   mpirun -np 8 laghos -p 3 -glvis -tf 5.0 -av -av-type 7 -ov 4 -oe 3 -rs 1 -cfl 0.5 -s 2 -vs 50
// implicit:
//   mpirun -np 8 laghos -p 3 -glvis -tf 5.0 -nmi 50 -pt 1 -kmi 20 -av -av-type 7 -ov 3 -oe 2 -rs 2 -cfl 32 -s 12 -vs 10
//
// Sedov:
// explicit:
//   mpirun -np 8 laghos -p 1 -glvis -tf 0.8 -av -av-type 7 -ov 2 -oe 1 -s 3 -cfl 0.5 -rs 3 -vs 50
// implicit:
//   mpirun -np 8 laghos -p 1 -glvis -tf 0.8 -nmi 50 -pt 1 -kmi 20 -av -av-type 7 -ov 2 -oe 1 -s 12 -cfl 8 -vs 10 -rs 3
//
// TG:
// explicit 2D, 3D:
//   mpirun -np 8 laghos -p 0 -glvis -tf 0.75 -ov 3 -oe 2 -rs 2 -cfl 0.5 -s 4 -vs 100
//   mpirun -np 8 laghos -m ./cube.mesh -p 0 -glvis -tf 0.75 -av -av-type 7 -ov 2 -oe 1 -rs 1 -cfl 0.5 -s 4 -vs 10
// implicit 2D, 3D:
//   mpirun -np 8 laghos -p 0 -glvis -tf 0.75 -nmi 50 -pt 1 -kmi 20 -ov 3 -oe 2 -rs 3 -cfl 32 -s 12 -vs 10
//   mpirun -np 8 laghos -m ./cube.mesh -p 0 -glvis -tf 0.75 -av -av-type 7 -nmi 50 -pt 1 -kmi 20 -ov 2 -oe 1 -rs 1 -cfl 32 -s 12 -vs 10


#include "laghos_solver.hpp"

#include "laghos_tests.hpp"

using namespace mfem;

int problem = 0;

int main(int argc, char *argv[])
{
   LaghosResults *results = laghos_results; // set by laghos_tests
   if (!Mpi::IsInitialized()) { Mpi::Init(); }
   Hypre::Init();

   out << std::setprecision(15);

   const char *device_config = "cpu";

   const char *mesh_file = "./rectangle01_quad.mesh";

   int dim = 2;
   int refinements = 0;
   int order_v = 2;
   int order_e = 1;
   int order_q = -1;
   real_t t_final = 0.0;
   real_t blast_position[] = {0.0, 0.0, 0.0};
   int ode_solver_type = 4;
   bool fd_gradient = false;
   bool use_viscosity = false;
   real_t cfl = 0.5;
   real_t nonlinear_relative_tolerance = 1e-5;
   int nonlinear_maximum_iterations = 10;
   int krylov_maximum_iterations = 10;
   int preconditioner_lag = 0;
   int vis_steps = 1;
   bool glvis = false;
   bool paraview = false;
   bool save_files = true;
   int viscosity_type = 2;
   real_t viscosity_q1 = 0.5;
   real_t viscosity_q2 = 1.0;
   bool stock_dt = false;
   int preconditioner_type =
      hydrodynamics::PRECONDITIONER_TYPE::BLOCK_DIAGONAL_AMG;
   int dump_jacobians = 0;
   int nretry = 100;
   OptionsParser args(argc, argv);
   args.AddOption(&dim, "-dim", "--dimension", "Dimension of the problem.");
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&refinements, "-rs", "--ref", "");
   args.AddOption(&order_v, "-ov", "--ov", "");
   args.AddOption(&order_e, "-oe", "--oe", "");
   args.AddOption(&order_q, "-oq", "--oq", "");
   args.AddOption(&t_final, "-tf", "--tf", "");
   args.AddOption(&problem, "-p", "--p", "");
   args.AddOption(&cfl, "-cfl", "--cfl", "");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&use_viscosity, "-av", "--av", "-no-av", "--no-av", "");
   args.AddOption(&fd_gradient, "-fd", "--fd", "-no-fd", "--no-fd", "");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6,\n\t"
                  "            7 - RK2Avg."
                  "            11 - Backward Euler"
                  "            12 - Implicit Midpoint"
                  "            13 - SDIRK33Solver");
   args.AddOption(&nonlinear_maximum_iterations, "-nmi", "--nmi",
                  "Maximum number of nonlinear iterations.");
   args.AddOption(&nonlinear_relative_tolerance, "-nrt", "--nrt",
                  "Nonlinear relative tolerance.");
   args.AddOption(&krylov_maximum_iterations, "-kmi", "--kmi",
                  "Maximum number of Krylov iterations.");
   args.AddOption(&preconditioner_lag, "-pl", "--pl",
                  "Number of nonlinear solves to wait before updating the preconditioner.");
   args.AddOption(&preconditioner_type, "-pt", "--pt",
                  "Preconditioner type: 0 - SuperLU_DIST, 1 - Block Diagonal AMG");
   args.AddOption(&vis_steps, "-vs", "--vis-steps",
                  "Number of visualization steps.");
   args.AddOption(&glvis, "-glvis", "--glvis", "-no-glvis", "--no-glvis", "");
   args.AddOption(&paraview, "-paraview", "--paraview",
                  "-no-paraview", "--no-paraview",
                  "Enable ParaView output (VTK files; can be expensive).");
   args.AddOption(&save_files, "-save", "--save-files",
                  "-no-save", "--no-save-files",
                  "Save the final mesh and fields (and rho.out) files.");
   args.AddOption(&viscosity_type, "-av-type", "--av-type",
                  "Artificial viscosity type: 1 stock Laghos tensor viscosity"
                  " (explicit only), 7 smooth viscosity.");
   args.AddOption(&viscosity_q1, "-av-q1", "--av-q1",
                  "Artificial viscosity (type 7) linear coefficient.");
   args.AddOption(&viscosity_q2, "-av-q2", "--av-q2",
                  "Artificial viscosity (type 7) quadratic coefficient.");
   args.AddOption(&stock_dt, "-stock-dt", "--stock-dt",
                  "-no-stock-dt", "--no-stock-dt",
                  "Time step estimate as the min over all the RK stages and "
                  "the new state, as stock Laghos (explicit only).");
   args.AddOption(&dump_jacobians, "-dump-jacobians", "--dump-jacobians", "");
   args.AddOption(&nretry, "-nretry", "--nretry", "");
   args.ParseCheck();

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   real_t blast_energy = (dim == 2) ? 0.25 : 0.125;

   Mesh serial_mesh = Mesh(mesh_file, true, true);

   if ((dim == 2) && (problem == 0 || problem == 1))
   {
      serial_mesh = Mesh(Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL,
                                               true));
      const int NBE = serial_mesh.GetNBE();
      for (int b = 0; b < NBE; b++)
      {
         Element *bel = serial_mesh.GetBdrElement(b);
         const int attr = (b < NBE/2) ? 2 : 1;
         bel->SetAttribute(attr);
      }
   }
   if (dim == 3 && (problem == 0 || problem == 1))
   {
      serial_mesh = Mesh(Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON,
                                               true));
      const int NBE = serial_mesh.GetNBE();
      for (int b = 0; b < NBE; b++)
      {
         Element *bel = serial_mesh.GetBdrElement(b);
         const int attr = (b < NBE/3) ? 3 : (b < 2*NBE/3) ? 1 : 2;
         bel->SetAttribute(attr);
      }
   }

   if (problem == 2)
   {
      serial_mesh = Mesh(Mesh::MakeCartesian1D(1));
      serial_mesh.GetBdrElement(0)->SetAttribute(1);
      serial_mesh.GetBdrElement(1)->SetAttribute(1);
   }

   for (int i = 0; i < refinements; i++)
   {
      serial_mesh.UniformRefinement();
   }

   // The mesh dimension can change based on selected problem setup (e.g. -p 2).
   dim = serial_mesh.Dimension();

   ParMesh mesh = ParMesh(MPI_COMM_WORLD, serial_mesh);

   // Define the parallel finite element spaces. We use:
   // - H1 (Gauss-Lobatto, continuous) for position and velocity.
   // - L2 (Bernstein, discontinuous) for specific internal energy.
   H1_FECollection H1FEC(order_v, dim);
   ParFiniteElementSpace H1FESpace(&mesh, &H1FEC, dim);
   L2_FECollection L2FEC(order_e, dim, BasisType::Positive);
   ParFiniteElementSpace L2FESpace(&mesh, &L2FEC);

   const auto global_ne = mesh.GetGlobalNE();
   const auto global_h1tsize = H1FESpace.GlobalTrueVSize();
   const auto global_l2tsize = L2FESpace.GlobalTrueVSize();

   if (Mpi::Root())
   {
      out << "num el: " << global_ne << "\n";
      out << "num kinematic dofs: " << global_h1tsize << "\n";
      out << "num thermodynamic dofs: " << global_l2tsize << "\n";
   }
   if (results)
   {
      results->ne = global_ne;
      results->h1_dofs = global_h1tsize;
      results->l2_dofs = global_l2tsize;
   }

   Array<int> ess_tdof, ess_vdofs;
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max()), dofs_marker, dofs_list;
      for (int d = 0; d < mesh.Dimension(); d++)
      {
         // Attributes 1/2/3 correspond to fixed-x/y/z boundaries,
         // i.e., we must enforce v_x/y/z = 0 for the velocity components.
         ess_bdr = 0; ess_bdr[d] = 1;
         H1FESpace.GetEssentialTrueDofs(ess_bdr, dofs_list, d);
         ess_tdof.Append(dofs_list);
         H1FESpace.GetEssentialVDofs(ess_bdr, dofs_marker, d);
         FiniteElementSpace::MarkerToList(dofs_marker, dofs_list);
         ess_vdofs.Append(dofs_list);
      }
   }
   ess_tdof.Sort();
   ess_tdof.Unique();

   // The monolithic BlockVector stores unknown fields as:
   // - 0 -> position
   // - 1 -> velocity
   // - 2 -> specific internal energy
   const int Vsize_l2 = L2FESpace.GetVSize();
   const int Vsize_h1 = H1FESpace.GetVSize();
   Array<int> offset(4);
   offset[0] = 0;
   offset[1] = offset[0] + Vsize_h1;
   offset[2] = offset[1] + Vsize_h1;
   offset[3] = offset[2] + Vsize_l2;
   BlockVector S(offset, Device::GetDeviceMemoryType());

   ParGridFunction x_gf, v_gf, e_gf;
   x_gf.MakeRef(&H1FESpace, S, offset[0]);
   v_gf.MakeRef(&H1FESpace, S, offset[1]);
   e_gf.MakeRef(&L2FESpace, S, offset[2]);

   mesh.SetNodalGridFunction(&x_gf);
   x_gf.SyncAliasMemory(S);

   ParGridFunction x0_gf = x_gf;

   auto v0 = [](const Vector &x, Vector &v)
   {
      switch (problem)
      {
         case 0:
            v(0) =  sin(M_PI*x(0)) * cos(M_PI*x(1));
            v(1) = -cos(M_PI*x(0)) * sin(M_PI*x(1));
            if (x.Size() == 3)
            {
               v(0) *= cos(M_PI*x(2));
               v(1) *= cos(M_PI*x(2));
               v(2) = 0.0;
            }
            break;
         case 1: v = 0.0; break;
         case 2: v = 0.0; break;
         case 3: v = 0.0; break;
         default: MFEM_ABORT("error");
      }
   };

   VectorFunctionCoefficient v_coeff(dim, v0);
   v_gf.ProjectCoefficient(v_coeff);
   for (int i = 0; i < ess_vdofs.Size(); i++)
   {
      v_gf(ess_vdofs[i]) = 0.0;
   }
   v_gf.SyncAliasMemory(S);

   auto rho0 = [&dim](const Vector &x)
   {
      switch (problem)
      {
         case 0: return 1.0;
         case 1: return 1.0;
         case 2: return (x(0) < 0.5) ? 1.0 : 0.1;
         case 3: return (dim == 2) ? x(0) > 1.0 && x(1) > 1.5 ? 0.125 : 1.0
                           : x(0) > 1.0 && ((x(1) < 1.5 && x(2) < 1.5) ||
                                            (x(1) > 1.5 && x(2) > 1.5)) ? 0.125 : 1.0;
         default: MFEM_ABORT("error");
      }
   };

   ParGridFunction rho0_gf(&L2FESpace);
   FunctionCoefficient rho0_coeff(rho0);
   L2_FECollection l2_fec(order_e, mesh.Dimension());
   ParFiniteElementSpace l2_fes(&mesh, &l2_fec);
   ParGridFunction l2_rho0_gf(&l2_fes), l2_e(&l2_fes);
   l2_rho0_gf.ProjectCoefficient(rho0_coeff);
   rho0_gf.ProjectGridFunction(l2_rho0_gf);

   auto gamma_func = [](const Vector &x)
   {
      switch (problem)
      {
         case 0: return 5.0 / 3.0;
         case 1: return 1.4;
         case 2: return 1.4;
         case 3: return (x(0) > 1.0 && x(1) <= 1.5) ? 1.4 : 1.5;
         default: MFEM_ABORT("error");
      }
   };

   auto e0 = [&rho0, &gamma_func](const Vector &x)
   {
      switch (problem)
      {
         case 0:
         {
            const real_t denom = 2.0 / 3.0;  // (5/3 - 1) * density.
            real_t val;
            if (x.Size() == 2)
            {
               val = 1.0 + (cos(2*M_PI*x(0)) + cos(2*M_PI*x(1))) / 4.0;
            }
            else
            {
               val = 100.0 + ((cos(2*M_PI*x(2)) + 2) *
                              (cos(2*M_PI*x(0)) + cos(2*M_PI*x(1))) - 2) / 16.0;
            }
            return val/denom;
         }
         case 1: return 0.0; // This case in initialized in main().
         case 2: return (x(0) < 0.5) ? 1.0 / rho0(x) / (gamma_func(x) - 1.0)
                           : 0.1 / rho0(x) / (gamma_func(x) - 1.0);
         case 3: return (x(0) > 1.0) ? 0.1 / rho0(x) / (gamma_func(x) - 1.0)
                           : 1.0 / rho0(x) / (gamma_func(x) - 1.0);
         default: MFEM_ABORT("error");
      }
   };

   if (problem == 1)
   {
      DeltaCoefficient e_coeff(blast_position[0], blast_position[1],
                               blast_position[2], blast_energy);
      l2_e.ProjectCoefficient(e_coeff);
   }
   else
   {
      FunctionCoefficient e_coeff(e0);
      l2_e.ProjectCoefficient(e_coeff);
   }

   e_gf.ProjectGridFunction(l2_e);
   e_gf.SyncAliasMemory(S);

   L2_FECollection material_fec(0, dim);
   ParFiniteElementSpace L2CFESpace(&mesh, &material_fec);
   ParGridFunction material_gf(&L2CFESpace);
   FunctionCoefficient material_coeff(gamma_func);
   material_gf.ProjectCoefficient(material_coeff);

   ParGridFunction rho_gf(&L2FESpace);

   IntegrationRule ir = IntRules.Get(mesh.GetElementBaseGeometry(0),
                                     3 * H1FESpace.GetOrder(0) + L2FESpace.GetOrder(0) - 1);

   if (Mpi::Root())
   {
      out << "num qp: " << ir.GetNPoints() << "\n";
   }

   // Create external data vector.
   Vector external_data(hydrodynamics::EXT_DATA_IDX::COUNT);
   external_data[hydrodynamics::EXT_DATA_IDX::CFL] = cfl;
   external_data[hydrodynamics::EXT_DATA_IDX::ORDER_VEL] = order_v;
   external_data[hydrodynamics::EXT_DATA_IDX::VISCOSITY_FLAG] = use_viscosity;
   external_data[hydrodynamics::EXT_DATA_IDX::VISCOSITY_TYPE] = viscosity_type;
   external_data[hydrodynamics::EXT_DATA_IDX::VISC_Q1] = viscosity_q1;
   external_data[hydrodynamics::EXT_DATA_IDX::VISC_Q2] = viscosity_q2;
   external_data[hydrodynamics::EXT_DATA_IDX::DT_ESTIMATE] =
      std::numeric_limits<real_t>::infinity();

   hydrodynamics::LagrangianHydroOperator *hydro = nullptr;
   switch (dim)
   {
      case 2:
      {
         hydro = hydrodynamics::CreateLagrangianHydroOperator<2>(
                    H1FESpace, L2FESpace, ess_tdof, rho0_coeff,
                    x0_gf, rho0_gf, material_gf, external_data, ir,
                    fd_gradient, dump_jacobians,
                    nonlinear_maximum_iterations, nonlinear_relative_tolerance,
                    krylov_maximum_iterations, preconditioner_lag,
                    (hydrodynamics::PRECONDITIONER_TYPE)preconditioner_type);
         break;
      }
      case 3:
      {
         hydro = hydrodynamics::CreateLagrangianHydroOperator<3>(
                    H1FESpace, L2FESpace, ess_tdof, rho0_coeff,
                    x0_gf, rho0_gf, material_gf, external_data, ir,
                    fd_gradient, dump_jacobians,
                    nonlinear_maximum_iterations, nonlinear_relative_tolerance,
                    krylov_maximum_iterations, preconditioner_lag,
                    (hydrodynamics::PRECONDITIONER_TYPE)preconditioner_type);
         break;
      }
      default: MFEM_ABORT("unsupported mesh dimension: " << dim);
   }
   hydro->stage_dt_estimate = stock_dt;

   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK2Solver(0.5); break;
      case 3: ode_solver = new RK3SSPSolver; break;
      case 4: ode_solver = new RK4Solver; break;
      case 6: ode_solver = new RK6Solver; break;
      case 11: ode_solver = new BackwardEulerSolver; break;
      case 12: ode_solver = new ImplicitMidpointSolver; break;
      case 13: ode_solver = new SDIRK33Solver; break;
      case 14: ode_solver = new SDIRK34Solver; break;
      default:
         out << "Unknown ODE solver type: " << ode_solver_type << '\n';
         return -1;
   }
   ode_solver->Init(*hydro);

   hydro->ComputeDensity(rho_gf);
   const real_t energy_init = hydro->InternalEnergy(e_gf) +
                              hydro->KineticEnergy(v_gf);

   socketstream vis_rho, vis_v, vis_e;
   char vishost[] = "localhost";
   int  visport   = 19916;
   if (glvis)
   {
      // Make sure all MPI ranks have sent their 'v' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(mesh.GetComm());
      vis_rho.precision(8);
      vis_v.precision(8);
      vis_e.precision(8);
      int Wx = 0, Wy = 0; // window position
      const int Ww = 350, Wh = 350; // window size
      int offx = Ww+10; // window offsets
      if (problem != 0 && problem != 4)
      {
         hydrodynamics::VisualizeField(vis_rho, vishost, visport, rho_gf,
                                       "Density", Wx, Wy, Ww, Wh);
      }
      Wx += offx;
      hydrodynamics::VisualizeField(vis_v, vishost, visport, v_gf,
                                    "Velocity", Wx, Wy, Ww, Wh);
      Wx += offx;
      hydrodynamics::VisualizeField(vis_e, vishost, visport, e_gf,
                                    "Specific Internal Energy", Wx, Wy, Ww, Wh);
   }

   if (Mpi::Root())
   {
      out << "energy initial: " << energy_init << "\n";
   }

   real_t t = 0.0;
   hydro->ResetTimeStepEstimate();
   real_t dt = hydro->GetTimeStepEstimate(S);

   if (Mpi::Root())
   {
      out << "time step estimate: " << dt << "\n";
   }
   if (results) { results->dt_init = dt; }

   real_t t_old;
   bool last_step = false;
   BlockVector S_old(S);

   std::unique_ptr<QuadratureSpace> qs;
   std::unique_ptr<QuadratureFunction> vqf;
   std::unique_ptr<QuadratureFunction> rqf;
   std::unique_ptr<QuadratureFunction> eqf;
   std::unique_ptr<ParaViewDataCollection> dc;

   auto paraview_save = [&](const int ti, const real_t time)
   {
      if (!dc) { return; }
      hydro->ComputeDensity(rho_gf);
      vqf->ProjectGridFunction(v_gf);
      rqf->ProjectGridFunction(rho_gf);
      eqf->ProjectGridFunction(e_gf);
      dc->SetCycle(ti);
      dc->SetTime(time);
      dc->Save();
   };

   if (paraview)
   {
      qs.reset(new QuadratureSpace(mesh, ir));
      vqf.reset(new QuadratureFunction(qs.get(), v_gf.VectorDim()));
      rqf.reset(new QuadratureFunction(qs.get(), rho_gf.VectorDim()));
      eqf.reset(new QuadratureFunction(qs.get(), e_gf.VectorDim()));

      dc.reset(new ParaViewDataCollection("dfem", &mesh));
      dc->SetLevelsOfDetail(order_v);
      dc->SetDataFormat(VTKFormat::BINARY);
      dc->SetHighOrderOutput(true);
      dc->RegisterField("velocity", &v_gf);
      dc->RegisterField("density", &rho_gf);
      dc->RegisterField("specific_internal_energy", &e_gf);
      dc->RegisterField("material", &material_gf);
      dc->RegisterQField("velocity_qf", vqf.get());
      dc->RegisterQField("density_qf", rqf.get());
      dc->RegisterQField("specific_internal_energy_qf", eqf.get());

      paraview_save(0, 0.0);
   }

   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final)
      {
         dt = t_final - t;
         last_step = true;
      }

      S_old = S;
      t_old = t;
      hydro->ResetTimeStepEstimate();

      hydro->ResetNewtonInfo();
      ode_solver->Step(S, t, dt);

      // Adaptive time step control.
      const real_t dt_est = hydro->GetTimeStepEstimate(S);

      if (ode_solver_type > 10) // implicit.
      {
         // Don't repeat for TGreen.
         if (problem != 0)
         {
            // Repeat.
            if (hydro->newton_converged == false || hydro->newton_max_iter >= 4)
            {
               dt *= 0.85;
               t = t_old;
               S = S_old;
               hydro->ResetQuadratureData();
               last_step = false;
               if (Mpi::Root())
               {
                  out << "Repeating step " << ti << " with dt: " << dt << "." << std::endl;
                  if (!hydro->newton_converged)
                  {
                     out << "Newton did not converge in "
                         << hydro->newton_max_iter << " iterations.\n";
                  }
                  else { out << "Estimated dt lower than taken dt.\n"; }
               }

               if (!hydro->newton_converged)
               {
                  if (nretry == 1)
                  {
                     hydro->residual->dump_jacobians = 1;
                  }
                  else if (nretry == 0)
                  {
                     if (Mpi::Root()) { out << "no retry planned, exit\n"; }
                     exit(1);
                  }
                  nretry--;
               }

               if (results) { results->repeats++; }
               ti--; continue;
            }
            else if (hydro->newton_max_iter <= 1)
            {
               dt *= 1.05;
               if (Mpi::Root()) { out << "Increasing dt: " << dt << std::endl; }
            }
         }
      }
      else if (dt_est < dt) // explicit.
      {
         // Repeat (solve again) with a decreased time step - decrease of the
         // time estimate suggests appearance of oscillations.
         dt *= 0.85;
         if (dt < std::numeric_limits<real_t>::epsilon())
         { MFEM_ABORT("The time step crashed!"); }
         t = t_old;
         S = S_old;
         hydro->ResetQuadratureData();
         last_step = false;
         if (Mpi::Root()) { out << "Repeating step " << ti << std::endl; }
         if (results) { results->repeats++; }
         ti--; continue;
      }
      else if (dt_est > 1.25 * dt) { dt *= 1.02; }

      x_gf.SyncAliasMemory(S);
      v_gf.SyncAliasMemory(S);
      e_gf.SyncAliasMemory(S);

      // Make sure that the mesh corresponds to the new solution state. This is
      // needed, because some time integrators use different S-type vectors
      // and the oper object might have redirected the mesh positions to those.
      mesh.NewNodes(x_gf, false);

      if (paraview && (ti % vis_steps == 0 || last_step))
      {
         paraview_save(ti, t);
      }

      if (results)
      {
         results->steps = ti;
         results->history.push_back({ti, t, dt});
      }

      if (ti % vis_steps == 0 || last_step)
      {
         if (Mpi::Root()) { out << "dt_est: " << dt_est << "\n"; }

         if (Mpi::Root())
         {
            out << "step " << std::setw(5) << ti
                << ",\tt = " << std::setw(5) << t
                << ",\tdt = " << std::setw(5) << dt;
            out << std::endl;
         }
      }

      if (glvis && (ti % vis_steps == 0 || last_step))
      {
         hydro->ComputeDensity(rho_gf);
         int Wx = 0, Wy = 0; // window position
         int Ww = 350, Wh = 350; // window size
         int offx = Ww+10; // window offsets
         if (problem != 0 && problem != 4)
         {
            hydrodynamics::VisualizeField(vis_rho, vishost, visport, rho_gf,
                                          "Density", Wx, Wy, Ww, Wh);
         }
         Wx += offx;
         hydrodynamics::VisualizeField(vis_v, vishost, visport,
                                       v_gf, "Velocity", Wx, Wy, Ww, Wh);
         Wx += offx;
         hydrodynamics::VisualizeField(vis_e, vishost, visport, e_gf,
                                       "Specific Internal Energy", Wx, Wy, Ww,Wh);
         Wx += offx;
      }
   }

   const real_t energy_final = hydro->InternalEnergy(e_gf)
                               + hydro->KineticEnergy(v_gf);

   if (Mpi::Root())
   {
      out << std::scientific << std::setprecision(2)
          << "Energy diff: " << fabs(energy_init - energy_final) << std::endl;
   }

   if (results)
   {
      Vector x_tdofs, v_tdofs, e_tdofs;
      x_gf.GetTrueDofs(x_tdofs);
      v_gf.GetTrueDofs(v_tdofs);
      e_gf.GetTrueDofs(e_tdofs);
      const MPI_Comm comm = mesh.GetComm();
      results->t = t;
      results->dt = dt;
      results->energy_init = energy_init;
      results->energy_final = energy_final;
      results->x_norm = std::sqrt(InnerProduct(comm, x_tdofs, x_tdofs));
      results->v_norm = std::sqrt(InnerProduct(comm, v_tdofs, v_tdofs));
      results->e_norm = std::sqrt(InnerProduct(comm, e_tdofs, e_tdofs));
   }

   if (problem == 0)
   {
      const real_t v_err_max = v_gf.ComputeMaxError(v_coeff);
      const real_t v_err_l1 = v_gf.ComputeL1Error(v_coeff);
      const real_t v_err_l2 = v_gf.ComputeL2Error(v_coeff);
      if (Mpi::Root())
      {
         out << "L_inf  error: " << v_err_max << std::endl
             << "L_1    error: " << v_err_l1 << std::endl
             << "L_2    error: " << v_err_l2 << std::endl;
      }
   }

   if (save_files)
   {
      if (problem == 1)
      {
         hydro->DensityScatter();
      }

      if (problem == 0 && dim == 3)
      {
         mesh.SaveAsOne("tgreen.mesh");
         v_gf.SaveAsOne("tgreen.gf");
      }
      if (problem == 1 && dim == 3)
      {
         mesh.SaveAsOne("sedov.mesh");
         rho_gf.SaveAsOne("sedov.gf");
      }
      if (problem == 3)
      {
         mesh.SaveAsOne("3point.mesh");
         rho_gf.SaveAsOne("3point.gf");
      }
   }

   delete hydro;

   return 0;
}
