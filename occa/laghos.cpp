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
//                     __                __
//                    / /   ____  ____  / /_  ____  _____
//                   / /   / __ `/ __ `/ __ \/ __ \/ ___/
//                  / /___/ /_/ / /_/ / / / / /_/ (__  )
//                 /_____/\__,_/\__, /_/ /_/\____/____/
//                             /____/
//
//             High-order Lagrangian Hydrodynamics Miniapp
//
// Laghos(LAGrangian High-Order Solver) is a miniapp that solves the
// time-dependent Euler equation of compressible gas dynamics in a moving
// Lagrangian frame using unstructured high-order finite element spatial
// discretization and explicit high-order time-stepping. Laghos is based on the
// numerical algorithm described in the following article:
//
//    V. Dobrev, Tz. Kolev and R. Rieben, "High-order curvilinear finite element
//    methods for Lagrangian hydrodynamics", SIAM Journal on Scientific
//    Computing, (34) 2012, pp. B606–B641, https://doi.org/10.1137/120864672.
//
// Sample runs:
//    mpirun -np 8 laghos -p 0 -m data/square01_quad.mesh -rs 3 -tf 0.75
//    mpirun -np 8 laghos -p 0 -m data/square01_tri.mesh  -rs 1 -tf 0.75
//    mpirun -np 8 laghos -p 0 -m data/cube01_hex.mesh    -rs 1 -tf 2.0
//    mpirun -np 8 laghos -p 1 -m data/square01_quad.mesh -rs 3 -tf 0.8
//    mpirun -np 8 laghos -p 1 -m data/square01_quad.mesh -rs 0 -tf 0.8 -ok 7 -ot 6
//    mpirun -np 8 laghos -p 1 -m data/cube01_hex.mesh    -rs 2 -tf 0.6
//    mpirun -np 8 laghos -p 2 -m data/segment01.mesh     -rs 5 -tf 0.2
//    mpirun -np 8 laghos -p 3 -m data/rectangle01_quad.mesh -rs 2 -tf 3.0
//    mpirun -np 8 laghos -p 3 -m data/box01_hex.mesh        -rs 1 -tf 3.0
//
// Test problems:
//    p = 0  --> Taylor-Green vortex (smooth problem).
//    p = 1  --> Sedov blast.
//    p = 2  --> 1D Sod shock tube.
//    p = 3  --> Triple point.


#include "laghos_solver.hpp"
#include <memory>
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;
using namespace mfem::hydrodynamics;

ProblemOption problem = vortex;

void display_banner(ostream & os);

int main(int argc, char *argv[])
{
   // Initialize MPI.
   MPI_Session mpi(argc, argv);
   int myid = mpi.WorldRank();

   mpiout.setup();

   // Print the banner.
   if (mpi.Root()) { display_banner(cout); }

   // Parse command-line options.
   const char *mesh_file = "data/square01_quad.mesh";
   int rs_levels = 0;
   int rp_levels = 0;
   int order_v = 2;
   int order_e = 1;
   ODESolverOption odeSolverType = RK4;
   double t_final = 0.5;
   double cfl = 0.5;
   double cg_tol = 1e-8;
   int cg_max_iter = 300;
   int max_tsteps = -1;
   bool visualization = false;
   int vis_steps = 5;
   bool visit = false;
   bool gfprint = false;
   const char *basename = "results/Laghos";
   int partition_type = 111;

   std::string device_info_str = "mode: 'Serial'";
   const char *device_info = "";
   const char *occa_config = "";
   bool occa_verbose = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption((int*) &problem, "-p", "--problem", "Problem setup to use.");
   args.AddOption(&order_v, "-ok", "--order-kinematic",
                  "Order (degree) of the kinematic finite element space.");
   args.AddOption(&order_e, "-ot", "--order-thermo",
                  "Order (degree) of the thermodynamic finite element space.");
   args.AddOption((int*) &odeSolverType, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&cfl, "-cfl", "--cfl", "CFL-condition number.");
   args.AddOption(&cg_tol, "-cgt", "--cg-tol",
                  "Relative CG tolerance (velocity linear solve).");
   args.AddOption(&cg_max_iter, "-cgm", "--cg-max-steps",
                  "Maximum number of CG iterations (velocity linear solve).");
   args.AddOption(&max_tsteps, "-ms", "--max-steps",
                  "Maximum number of steps (negative means no restriction).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit", "--no-visit",
                  "Enable or disable VisIt visualization.");
   args.AddOption(&gfprint, "-print", "--print", "-no-print", "--no-print",
                  "Enable or disable result output (files in mfem format).");
   args.AddOption(&basename, "-k", "--outputfilename",
                  "Name of the visit dump files");
   args.AddOption(&partition_type, "-pt", "--partition",
                  "Customized x/y/z Cartesian MPI partitioning of the serial mesh.\n\t"
                  "Here x,y,z are relative task ratios in each direction.\n\t"
                  "Example: with 48 mpi tasks and -pt 321, one would get a Cartesian\n\t"
                  "partition of the serial mesh by (6,4,2) MPI tasks in (x,y,z).\n\t"
                  "NOTE: the serially refined mesh must have the appropriate number\n\t"
                  "of zones in each direction, e.g., the number of zones in direction x\n\t"
                  "must be divisible by the number of MPI tasks in direction x.\n\t"
                  "Available options: 11, 21, 111, 211, 221, 311, 321, 322, 432.");
  args.AddOption(&device_info, "-d", "--device-info",
                 "Device information to run example on (default: \"mode: 'Serial'\").");
  args.AddOption(&occa_config,
                 "-oc", "--occa-config",
                 "Load OCCA information from the .json config file. --device-info overrides the config");
  args.AddOption(&occa_verbose,
                 "-ov", "--occa-verbose",
                 "--no-ov", "--no-occa-verbose",
                 "Print verbose information about OCCA kernel compilation.");
   args.Parse();
   if (!args.Good())
   {
      if (mpi.Root()) { args.PrintUsage(cout); }
      return 1;
   }
   if (mpi.Root()) { args.PrintOptions(cout); }

   if (strlen(occa_config)) {
     occa::json config = occa::json::parse(occa_config);
     if (!config.has("devices")) {
       std::cout << "Config file \"" << occa_config << "\" does not have 'devices'.\n";
       return 1;
     }

     occa::json devices = config["devices"];
     occa::json specificDevices = config["specificDevices"];

     const std::string mpiRankStr = occa::toString(myid);

     if (specificDevices.has(mpiRankStr)) {
       device_info_str = specificDevices[mpiRankStr].toString();
     } else {
       const int procsPerNode = devices.size();
       const int deviceID = (myid % procsPerNode);
       device_info_str = devices[deviceID].toString();
     }

     device_info = device_info_str.c_str();
   } else if (!strlen(device_info)) {
     device_info = device_info_str.c_str();
   }

   // Set the OCCA device to run example in
   occa::setDevice(device_info);

   // Load cached kernels
   occa::loadKernels();
   occa::loadKernels("mfem");
   occa::loadKernels("laghos");

   // Set as the background device
   occa::settings()["verboseCompilation"] = occa_verbose;

   // Set properties that determine the problem
   occa::properties props = GetProblemProperties();
   props["defines/PROBLEM"] = problem;

   // Read the serial mesh from the given mesh file on all processors.
   // Refine the mesh in serial to increase the resolution.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   const int dim = mesh->Dimension();
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }

   // Parallel partitioning of the mesh.
   ParMesh *pmesh = NULL;
   const int num_tasks = mpi.WorldSize(); int unit;
   int *nxyz = new int[dim];
   switch (partition_type)
   {
      case 11:
      case 111:
         unit = floor(pow(num_tasks, 1.0 / dim) + 1e-2);
         for (int d = 0; d < dim; d++) { nxyz[d] = unit; }
         if (dim == 2) { nxyz[2] = 0; }
         break;
      case 21: // 2D
         unit = floor(pow(num_tasks / 2, 1.0 / 2) + 1e-2);
         nxyz[0] = 2 * unit; nxyz[1] = unit; nxyz[2] = 0;
         break;
      case 211: // 3D.
         unit = floor(pow(num_tasks / 2, 1.0 / 3) + 1e-2);
         nxyz[0] = 2 * unit; nxyz[1] = unit; nxyz[2] = unit;
         break;
      case 221: // 3D.
         unit = floor(pow(num_tasks / 4, 1.0 / 3) + 1e-2);
         nxyz[0] = 2 * unit; nxyz[1] = 2 * unit; nxyz[2] = unit;
         break;
      case 311: // 3D.
         unit = floor(pow(num_tasks / 3, 1.0 / 3) + 1e-2);
         nxyz[0] = 3 * unit; nxyz[1] = unit; nxyz[2] = unit;
         break;
      case 321: // 3D.
         unit = floor(pow(num_tasks / 6, 1.0 / 3) + 1e-2);
         nxyz[0] = 3 * unit; nxyz[1] = 2 * unit; nxyz[2] = unit;
         break;
      case 322: // 3D.
         unit = floor(pow(2 * num_tasks / 3, 1.0 / 3) + 1e-2);
         nxyz[0] = 3 * unit / 2; nxyz[1] = unit; nxyz[2] = unit;
         break;
      case 432: // 3D.
         unit = floor(pow(num_tasks / 3, 1.0 / 3) + 1e-2);
         nxyz[0] = 2 * unit; nxyz[1] = 3 * unit / 2; nxyz[2] = unit;
         break;
      default:
         if (myid == 0)
         {
            cout << "Unknown partition type: " << partition_type << '\n';
         }
         delete mesh;
         MPI_Finalize();
         return 3;
   }
   int product = 1;
   for (int d = 0; d < dim; d++) { product *= nxyz[d]; }
   if (product == num_tasks)
   {
      int *partitioning = mesh->CartesianPartitioning(nxyz);
      pmesh = new ParMesh(MPI_COMM_WORLD, *mesh, partitioning);
      delete partitioning;
   }
   else
   {
      if (myid == 0)
      {
         cout << "Non-Cartesian partitioning through METIS will be used.\n";
#ifndef MFEM_USE_METIS
         cout << "MFEM was built without METIS. "
              << "Adjust the number of tasks to use a Cartesian split." << endl;
#endif
      }
#ifndef MFEM_USE_METIS
      return 1;
#endif
      pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   }
   delete [] nxyz;
   delete mesh;

   // Refine the mesh further in parallel to increase the resolution.
   for (int lev = 0; lev < rp_levels; lev++) { pmesh->UniformRefinement(); }

   int nzones = pmesh->GetNE(), nzones_min, nzones_max;
   MPI_Reduce(&nzones, &nzones_min, 1, MPI_INT, MPI_MIN, 0, pmesh->GetComm());
   MPI_Reduce(&nzones, &nzones_max, 1, MPI_INT, MPI_MAX, 0, pmesh->GetComm());
   if (myid == 0)
   { cout << "Zones min/max: " << nzones_min << " " << nzones_max << endl; }

   // Define the parallel finite element spaces. We use:
   // - H1 (Gauss-Lobatto, continuous) for position and velocity.
   // - L2 (Bernstein, discontinuous) for specific internal energy.
   L2_FECollection L2FEC(order_e, dim, BasisType::Positive);
   H1_FECollection H1FEC(order_v, dim);
   OccaFiniteElementSpace o_L2FESpace(pmesh, &L2FEC, Ordering::byNODES);
   OccaFiniteElementSpace o_H1FESpace(pmesh, &H1FEC, pmesh->Dimension(), Ordering::byNODES);

   // Boundary conditions: all tests use v.n = 0 on the boundary, and we assume
   // that the boundaries are straight.
   Array<int> ess_tdofs;
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max()), tdofs1d;
      for (int d = 0; d < pmesh->Dimension(); d++)
      {
         // Attributes 1/2/3 correspond to fixed-x/y/z boundaries, i.e., we must
         // enforce v_x/y/z = 0 for the velocity components.
         ess_bdr = 0; ess_bdr[d] = 1;
         o_H1FESpace.GetFESpace()->GetEssentialTrueDofs(ess_bdr, tdofs1d, d);
         ess_tdofs.Append(tdofs1d);
      }
   }

   // Define the explicit ODE solver used for time integration.
   ODESolver *ode_solver = NULL;
   switch (odeSolverType)
   {
      case ForwardEuler: ode_solver = new OccaForwardEulerSolver; break;
      case RK2: ode_solver = new OccaRK2Solver(0.5); break;
      case RK3: ode_solver = new OccaRK3SSPSolver; break;
      case RK4: ode_solver = new OccaRK4Solver; break;
      case RK6: ode_solver = new OccaRK6Solver; break;
      default:
         if (myid == 0)
         {
            cout << "Unknown ODE solver type: " << odeSolverType << '\n';
         }
         delete pmesh;
         MPI_Finalize();
         return 3;
   }

   HYPRE_Int glob_size_l2 = o_L2FESpace.GetGlobalTrueVSize();
   HYPRE_Int glob_size_h1 = o_H1FESpace.GetGlobalTrueVSize();

   if (mpi.Root())
   {
      cout << "Number of kinematic (position, velocity) dofs: "
           << glob_size_h1 << endl;
      cout << "Number of specific internal energy dofs: "
           << glob_size_l2 << endl;
   }

   const int Vsize_l2 = o_L2FESpace.GetVSize();
   const int Vsize_h1 = o_H1FESpace.GetVSize();

   // The monolithic BlockVector stores unknown fields as:
   // - (H1) position
   // - (H1) velocity
   // - (L2) specific internal energy
   OccaVector S(2*Vsize_h1 + Vsize_l2);

   // Define GridFunction objects for the position, velocity and specific
   // internal energy.  There is no function for the density, as we can always
   // compute the density values given the current mesh position, using the
   // property of pointwise mass conservation.
   ParGridFunction x_gf((ParFiniteElementSpace*) o_H1FESpace.GetFESpace());
   ParGridFunction v_gf((ParFiniteElementSpace*) o_H1FESpace.GetFESpace());
   ParGridFunction e_gf((ParFiniteElementSpace*) o_L2FESpace.GetFESpace());

   OccaGridFunction o_x_gf(&o_H1FESpace, S.GetRange(0         , Vsize_h1));
   OccaGridFunction o_v_gf(&o_H1FESpace, S.GetRange(Vsize_h1  , Vsize_h1));
   OccaGridFunction o_e_gf(&o_L2FESpace, S.GetRange(2*Vsize_h1, Vsize_l2));

   // Initialize x_gf using the starting mesh coordinates. This also links the
   // mesh positions to the values in x_gf.
   pmesh->SetNodalGridFunction(&x_gf);
   o_x_gf = x_gf;

   // Initialize the velocity.
   VectorFunctionCoefficient v_coeff(pmesh->Dimension(), v0);
   v_gf.ProjectCoefficient(v_coeff);
   o_v_gf = v_gf;

   // Initialize density and specific internal energy values. We interpolate in
   // a non-positive basis to get the correct values at the dofs.  Then we do an
   // L2 projection to the positive basis in which we actually compute. The goal
   // is to get a high-order representation of the initial condition. Note that
   // this density is a temporary function and it will not be updated during the
   // time evolution.
   ParGridFunction rho((ParFiniteElementSpace*) o_L2FESpace.GetFESpace());
   FunctionCoefficient rho_coeff(hydrodynamics::rho0);
   L2_FECollection l2_fec(order_e, pmesh->Dimension());
   OccaFiniteElementSpace o_l2_fes(pmesh, &l2_fec, Ordering::byNODES);

   ParFiniteElementSpace *l2_fes = (ParFiniteElementSpace*) o_l2_fes.GetFESpace();
   ParGridFunction l2_rho(l2_fes), l2_e(l2_fes);
   l2_rho.ProjectCoefficient(rho_coeff);
   rho.ProjectGridFunction(l2_rho);
   if (problem == sedov) {
     // For the Sedov test, we use a delta function at the origin.
     DeltaCoefficient e_coeff(0, 0, 0.25);
     l2_e.ProjectCoefficient(e_coeff);
   }
   else {
     FunctionCoefficient e_coeff(e0);
     l2_e.ProjectCoefficient(e_coeff);
   }
   e_gf.ProjectGridFunction(l2_e);

   OccaGridFunction o_rho(&o_L2FESpace);
   o_rho = rho;
   o_e_gf = e_gf;

   // Piecewise constant ideal gas coefficient over the Lagrangian mesh. The
   // gamma values are projected on a function that stays constant on the moving
   // mesh.
   L2_FECollection mat_fec(0, pmesh->Dimension());
   ParFiniteElementSpace mat_fes(pmesh, &mat_fec);
   ParGridFunction mat_gf(&mat_fes);
   FunctionCoefficient mat_coeff(hydrodynamics::gamma);
   mat_gf.ProjectCoefficient(mat_coeff);
   GridFunctionCoefficient *mat_gf_coeff = new GridFunctionCoefficient(&mat_gf);

   // Additional details, depending on the problem.
   bool use_viscosity; double gamma;
   switch (problem) {
   case vortex:
     use_viscosity = false; break;
   case sedov:
   case shockTube:
   case triplePoint:
     use_viscosity = true; break;
   default:
     MFEM_ABORT("Wrong problem specification!");
   }

   LagrangianHydroOperator oper(problem, o_H1FESpace, o_L2FESpace,
                                ess_tdofs, o_rho, cfl, mat_gf_coeff,
                                use_viscosity, cg_tol, cg_max_iter,
                                props);

   socketstream vis_rho, vis_v, vis_e;
   char vishost[] = "localhost";
   int  visport   = 19916;

   ParGridFunction rho_gf;
   if (visualization || visit) { oper.ComputeDensity(rho_gf); }

   if (visualization)
   {
      // Make sure all MPI ranks have sent their 'v' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh->GetComm());

      vis_rho.precision(8);
      vis_v.precision(8);
      vis_e.precision(8);

      int Wx = 0, Wy = 0; // window position
      const int Ww = 350, Wh = 350; // window size
      int offx = Ww+10; // window offsets

      VisualizeField(vis_rho, vishost, visport, rho_gf,
                     "Density", Wx, Wy, Ww, Wh);
      Wx += offx;
      VisualizeField(vis_v, vishost, visport, v_gf,
                     "Velocity", Wx, Wy, Ww, Wh);
      Wx += offx;
      VisualizeField(vis_e, vishost, visport, e_gf,
                     "Specific Internal Energy", Wx, Wy, Ww, Wh);
   }

   // Save data for VisIt visualization.
   VisItDataCollection visit_dc(basename, pmesh);
   if (visit)
   {
      visit_dc.RegisterField("Density",  &rho_gf);
      visit_dc.RegisterField("Velocity", &v_gf);
      visit_dc.RegisterField("Specific Internal Energy", &e_gf);
      visit_dc.SetCycle(0);
      visit_dc.SetTime(0.0);
      visit_dc.Save();
   }

   // Perform time-integration (looping over the time iterations, ti, with a
   // time-step dt). The object oper is of type LagrangianHydroOperator that
   // defines the Mult() method that used by the time integrators.
   ode_solver->Init(oper);
   oper.ResetTimeStepEstimate();
   double t = 0.0, dt = oper.GetTimeStepEstimate(S), t_old;
   bool last_step = false;
   int steps = 0;
   OccaVector S_old(S);
   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final)
      {
         dt = t_final - t;
         last_step = true;
      }

      S_old = S;
      t_old = t;
      oper.ResetTimeStepEstimate();

      // S is the vector of dofs, t is the current time, and dt is the time step
      // to advance.
      ode_solver->Step(S, t, dt);
      steps++;

      // Adaptive time step control.
      const double dt_est = oper.GetTimeStepEstimate(S);
      if (dt_est < dt)
      {
         // Repeat (solve again) with a decreased time step - decrease of the
         // time estimate suggests appearance of oscillations.
         dt *= 0.85;
         if (dt < numeric_limits<double>::epsilon())
         { MFEM_ABORT("The time step crashed!"); }
         t = t_old;
         S = S_old;
         oper.ResetQuadratureData();
         if (mpi.Root()) { cout << "Repeating step " << ti << endl; }
         ti--; continue;
      }
      else if (dt_est > 1.25 * dt) { dt *= 1.02; }

      // Make sure that the mesh corresponds to the new solution state.
      x_gf = o_x_gf;
      pmesh->NewNodes(x_gf, false);

      if (last_step || (ti % vis_steps) == 0)
      {
         double loc_norm = e_gf * e_gf, tot_norm;
         MPI_Allreduce(&loc_norm, &tot_norm, 1, MPI_DOUBLE, MPI_SUM,
                       pmesh->GetComm());
         if (mpi.Root())
         {
            cout << fixed;
            cout << "step " << setw(5) << ti
                 << ",\tt = " << setw(5) << setprecision(4) << t
                 << ",\tdt = " << setw(5) << setprecision(6) << dt
                 << ",\t|e| = " << setprecision(10)
                 << sqrt(tot_norm) << endl;
         }

         // Make sure all ranks have sent their 'v' solution before initiating
         // another set of GLVis connections (one from each rank):
         MPI_Barrier(pmesh->GetComm());

         if (visualization || visit || gfprint) { oper.ComputeDensity(rho_gf); }
         if (visualization)
         {
            int Wx = 0, Wy = 0; // window position
            int Ww = 350, Wh = 350; // window size
            int offx = Ww+10; // window offsets

            VisualizeField(vis_rho, vishost, visport, rho_gf,
                           "Density", Wx, Wy, Ww, Wh);
            Wx += offx;
            VisualizeField(vis_v, vishost, visport,
                           v_gf, "Velocity", Wx, Wy, Ww, Wh);
            Wx += offx;
            VisualizeField(vis_e, vishost, visport, e_gf,
                           "Specific Internal Energy", Wx, Wy, Ww,Wh);
            Wx += offx;
         }

         if (visit)
         {
            visit_dc.SetCycle(ti);
            visit_dc.SetTime(t);
            visit_dc.Save();
         }

         if (gfprint)
         {
            ostringstream mesh_name, rho_name, v_name, e_name;
            mesh_name << basename << "_" << ti
                      << "_mesh." << setfill('0') << setw(6) << myid;
            rho_name  << basename << "_" << ti
                      << "_rho." << setfill('0') << setw(6) << myid;
            v_name << basename << "_" << ti
                   << "_v." << setfill('0') << setw(6) << myid;
            e_name << basename << "_" << ti
                   << "_e." << setfill('0') << setw(6) << myid;

            ofstream mesh_ofs(mesh_name.str().c_str());
            mesh_ofs.precision(8);
            pmesh->Print(mesh_ofs);
            mesh_ofs.close();

            ofstream rho_ofs(rho_name.str().c_str());
            rho_ofs.precision(8);
            rho_gf.Save(rho_ofs);
            rho_ofs.close();

            ofstream v_ofs(v_name.str().c_str());
            v_ofs.precision(8);
            v_gf.Save(v_ofs);
            v_ofs.close();

            ofstream e_ofs(e_name.str().c_str());
            e_ofs.precision(8);
            e_gf.Save(e_ofs);
            e_ofs.close();
         }
      }
   }

   switch (odeSolverType) {
   case RK2: steps *= 2; break;
   case RK3: steps *= 3; break;
   case RK4: steps *= 4; break;
   case RK6: steps *= 6; break;
   default:;
   }
   oper.PrintTimingData(mpi.Root(), steps);

   if (visualization)
   {
      vis_v.close();
      vis_e.close();
   }

   // Free the used memory.
   delete ode_solver;
   delete pmesh;
   delete mat_gf_coeff;

   return 0;
}

namespace mfem
{

namespace hydrodynamics
{

double rho0(const Vector &x)
{
  switch (problem) {
  case vortex:
    return 1.0;
  case sedov:
    return 1.0;
  case shockTube:
    if (x(0) < 0.5) { return 1.0; }
    return 0.1;
  case triplePoint:
    if (x(0) > 1.0 && x(1) <= 1.5) { return 1.0; }
    return 0.125;
  default:
    MFEM_ABORT("Bad number given for problem id!");
    return 0.0;
  }
}

double gamma(const Vector &x)
{
  switch (problem) {
  case vortex:
    return 5./3.;
  case sedov:
    return 1.4;
  case shockTube:
    return 1.4;
  case triplePoint:
    if (x(0) > 1.0 && x(1) <= 1.5) { return 1.4; }
    return 1.5;
  default:
    MFEM_ABORT("Bad number given for problem id!");
    return 0.0;
  }
}

void v0(const Vector &x, Vector &v)
{
  switch (problem) {
  case vortex:
    v(0) =  sin(M_PI*x(0)) * cos(M_PI*x(1));
    v(1) = -cos(M_PI*x(0)) * sin(M_PI*x(1));
    if (x.Size() == 3) {
      v(0) *= cos(M_PI*x(2));
      v(1) *= cos(M_PI*x(2));
      v(2) = 0.0;
    }
    break;
  case sedov:
  case shockTube:
  case triplePoint:
    v = 0.0; break;
  default:
    MFEM_ABORT("Bad number given for problem id!");
  }
}

double e0(const Vector &x)
{
  switch (problem) {
  case vortex: {
    const double denom = 2.0 / 3.0;  // (5/3 - 1) * density.
    double val;
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
  case sedov:
    // This case in initialized in main().
    return 0.0;
  case shockTube:
    if (x(0) < 0.5) { return 1.0 / rho0(x) / (gamma(x) - 1.0); }
    return 0.1 / rho0(x) / (gamma(x) - 1.0);
  case triplePoint:
    if (x(0) > 1.0) { return 0.1 / rho0(x) / (gamma(x) - 1.0); }
    return 1.0 / rho0(x) / (gamma(x) - 1.0);
  default:
    MFEM_ABORT("Bad number given for problem id!");
    return 0.0;
  }
}

} // namespace hydrodynamics

} // namespace mfem

void display_banner(ostream & os)
{
   os << endl
      << "       __                __                 " << endl
      << "      / /   ____  ____  / /_  ____  _____   " << endl
      << "     / /   / __ `/ __ `/ __ \\/ __ \\/ ___/ " << endl
      << "    / /___/ /_/ / /_/ / / / / /_/ (__  )    " << endl
      << "   /_____/\\__,_/\\__, /_/ /_/\\____/____/  " << endl
      << "               /____/                       " << endl << endl;
}
