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
// Test problems:
//    p = 0  --> Taylor-Green vortex (smooth problem).
//    p = 1  --> Sedov blast.
//    p = 2  --> 1D Sod shock tube.
//    p = 3  --> Triple point.
//    p = 4  --> Gresho vortex (smooth problem).
//    p = 5  --> 2D Riemann problem, config. 12 of doi.org/10.1002/num.10025
//    p = 6  --> 2D Riemann problem, config.  6 of doi.org/10.1002/num.10025
//    p = 7  --> 2D Rayleigh-Taylor instability problem.
//
// Sample runs: see README.md, section 'Verification of Results'.
//
// Combinations resulting in 3D uniform Cartesian MPI partitionings of the mesh:
// -m data/cube01_hex.mesh   -pt 211 for  2 / 16 / 128 / 1024 ... tasks.
// -m data/cube_922_hex.mesh -pt 921 for    / 18 / 144 / 1152 ... tasks.
// -m data/cube_522_hex.mesh -pt 522 for    / 20 / 160 / 1280 ... tasks.
// -m data/cube_12_hex.mesh  -pt 311 for  3 / 24 / 192 / 1536 ... tasks.
// -m data/cube01_hex.mesh   -pt 221 for  4 / 32 / 256 / 2048 ... tasks.
// -m data/cube_922_hex.mesh -pt 922 for    / 36 / 288 / 2304 ... tasks.
// -m data/cube_522_hex.mesh -pt 511 for  5 / 40 / 320 / 2560 ... tasks.
// -m data/cube_12_hex.mesh  -pt 321 for  6 / 48 / 384 / 3072 ... tasks.
// -m data/cube01_hex.mesh   -pt 111 for  8 / 64 / 512 / 4096 ... tasks.
// -m data/cube_922_hex.mesh -pt 911 for  9 / 72 / 576 / 4608 ... tasks.
// -m data/cube_522_hex.mesh -pt 521 for 10 / 80 / 640 / 5120 ... tasks.
// -m data/cube_12_hex.mesh  -pt 322 for 12 / 96 / 768 / 6144 ... tasks.
//
// Wall BC tests:
// * penPar is the penalty Parameter
// * per is the perimeter of the bounding box of the domain
// TG 2D (rates):
// mpirun -np 4 laghos -m data/square01_quad.mesh -p 0 -rs 3 -tf 0.5 -s 4 -fa -vs 20 -vis
// 2D square:
// mpirun -np 4 laghos -m data/square01_quad.mesh -p 1 -rs 3 -tf 0.8 -s 7 -fa -vs 20 -vis
// 2D trapezoid:
// mpirun -np 4 laghos -m data/trapezoid_quad.mesh -p 1 -rs 2 -tf 1.5 -s 7 -fa -vs 20 -vis
// 2D circular hole:
// mpirun -np 4 laghos -m data/refined.mesh -p 1 -rs 2 -tf 0.8 -s 7 -fa -vs 20 -vis
// 2D disc (inside):
// mpirun -np 4 laghos -m data/disc-nurbs.mesh -p 1 -rs 3 -tf 10 -s 7 -fa -vs 20 -vis
// 2D disc (outside):
// mpirun -np 4 laghos -m data/convex.mesh -p 1 -rs 3 -tf 0.1 -s 7 -fa -vs 20 -vis
// 3D cube:
// mpirun -np 4 laghos -m data/cube01_hex.mesh -p 1 -rs 1 -tf 0.8 -s 7 -fa -vs 20 -vis
// 3D spherical hole:
// mpirun -np 4 laghos -m data/sphere_hole_V4.msh -p 1 -rs 0 -tf 0.8 -s 7 -fa -ok 3 -ot 2 -vs 1000
//
// ALE tests:
// mpirun -np 4 laghos -m data/wall_linear.mesh -p 1 -rs 0 -s 7 -fa -vs 20 -vis -tf 2.5 -ale 0.25
// remap gslib, dist = 0.05.
// mpirun -np 7 laghos -m data/wall_a02_b05_c15.mesh -p 1 -rs 0 -s 7 -fa -vs 20 -vis -tf 1.2 -ale 0.2
// remap gslib, dist = 0.02.
// mpirun -np 7 laghos -m data/circles3.mesh -p 1 -rs 0 -s 7 -fa -vs 20 -vis -tf 1.0 -ale 0.2
// mpirun -np 7 laghos -m data/circles4.mesh -p 1 -rs 0 -s 7 -fa -vs 20 -vis -tf 1.0 -ale 0.2

#include <fstream>
#include <sys/time.h>
#include <sys/resource.h>
#include "laghos_solver.hpp"
#include "laghos_remesh.hpp"
#include "laghos_remap.hpp"

using std::cout;
using std::endl;
using namespace mfem;
using namespace hydrodynamics;

// Choice for the problem setup.
static int problem, dim;

// Forward declarations.
double e0(const Vector &);
double rho0(const Vector &);
double gamma_func(const Vector &);
void v0(const Vector &, Vector &);

static long GetMaxRssMB();
static void display_banner(std::ostream&);
static void Checks(const int ti, const double norm, int &checks);

int main(int argc, char *argv[])
{
   // Initialize MPI.
   Mpi::Init();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // Print the banner.
   if (Mpi::Root()) { display_banner(cout); }

   // Parse command-line options.
   problem = 1;
   dim = 3;
   const char *mesh_file = "default";
   int rs_levels = 2;
   int rp_levels = 0;
   Array<int> cxyz;
   int order_v = 2;
   int order_e = 1;
   int order_q = -1;
   int ode_solver_type = 4;
   double t_final = 0.6;
   double penalty_param = 20.0;
   double ale_period = -1.0;
   double cfl = 0.5;
   double cg_tol = 1e-8;
   double ftz_tol = 0.0;
   int cg_max_iter = 300;
   int max_tsteps = -1;
   bool p_assembly = true;
   bool impose_visc = false;
   bool visualization = false;
   int vis_steps = 5;
   bool visit = false;
   bool gfprint = false;
   const char *basename = "results/Laghos";
   const char *device = "cpu";
   bool check = false;
   bool mem_usage = false;
   bool fom = false;
   int dev = 0;
   double blast_energy = 0.25;

   double blast_position[] = {1.0, 0.0, 0.0};
   double a, b, c;

   OptionsParser args(argc, argv);
   args.AddOption(&dim, "-dim", "--dimension", "Dimension of the problem.");
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&cxyz, "-c", "--cartesian-partitioning",
                  "Use Cartesian partitioning.");
   args.AddOption(&problem, "-p", "--problem", "Problem setup to use.");
   args.AddOption(&order_v, "-ok", "--order-kinematic",
                  "Order (degree) of the kinematic finite element space.");
   args.AddOption(&order_e, "-ot", "--order-thermo",
                  "Order (degree) of the thermodynamic finite element space.");
   args.AddOption(&order_q, "-oq", "--order-intrule",
                  "Order  of the integration rule.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6,\n\t"
                  "            7 - RK2Avg.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&penalty_param, "-pp", "--penalty-param",
                  "Penalty constant for the BC.");
   args.AddOption(&ale_period, "-ale", "--ale-period",
                  "ALE period interval in physical time.");
   args.AddOption(&cfl, "-cfl", "--cfl", "CFL-condition number.");
   args.AddOption(&cg_tol, "-cgt", "--cg-tol",
                  "Relative CG tolerance (velocity linear solve).");
   args.AddOption(&ftz_tol, "-ftz", "--ftz-tol",
                  "Absolute flush-to-zero tolerance.");
   args.AddOption(&cg_max_iter, "-cgm", "--cg-max-steps",
                  "Maximum number of CG iterations (velocity linear solve).");
   args.AddOption(&max_tsteps, "-ms", "--max-steps",
                  "Maximum number of steps (negative means no restriction).");
   args.AddOption(&p_assembly, "-pa", "--partial-assembly", "-fa",
                  "--full-assembly",
                  "Activate 1D tensor-based assembly (partial assembly).");
   args.AddOption(&impose_visc, "-iv", "--impose-viscosity", "-niv",
                  "--no-impose-viscosity",
                  "Use active viscosity terms even for smooth problems.");
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
   args.AddOption(&device, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&check, "-chk", "--checks", "-no-chk", "--no-checks",
                  "Enable 2D checks.");
   args.AddOption(&mem_usage, "-mb", "--mem", "-no-mem", "--no-mem",
                  "Enable memory usage.");
   args.AddOption(&fom, "-f", "--fom", "-no-fom", "--no-fom",
                  "Enable figure of merit output.");
   args.AddOption(&dev, "-dev", "--dev", "GPU device to use.");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(cout); }
      return 1;
   }
   if (Mpi::Root()) { args.PrintOptions(cout); }

   if (strcmp(mesh_file, "data/disc-nurbs.mesh") == 0)
   {
      blast_position[0] = - sqrt(8.0);
   }
   if (strcmp(mesh_file, "data/convex.mesh") == 0)
   {
      blast_position[0] = 0.7;
      blast_position[1] = 0.5;
   }
   if (strcmp(mesh_file, "data/sphere_hole_V4.msh") == 0)
   {
      blast_position[0] = -0.5;
      blast_position[1] = -0.5;
      blast_position[2] = -0.5;
   }
   if (strcmp(mesh_file, "data/wall_linear.mesh") == 0)
   {
      a = 0.0; b = 0.5; c = 0.0;
      blast_position[0] = 1.5;
      blast_position[1] = 1.5;
   }
   if (strcmp(mesh_file, "data/wall_a02_b05_c15.mesh") == 0)
   {
      a = 0.2; b = 0.5; c = 1.5;
      blast_position[0] = 1.0;
      blast_position[1] = 0.0;
   }
   if (strcmp(mesh_file, "data/circles3.mesh") == 0 ||
       strcmp(mesh_file, "data/circles4.mesh") == 0)
   {
      blast_position[0] =  -1.0;
      blast_position[1] =   0.0;
   }

   // Configure the device from the command line options
   Device backend;
   backend.Configure(device, dev);
   if (Mpi::Root()) { backend.Print(); }

   // On all processors, use the default builtin 1D/2D/3D mesh or read the
   // serial one given on the command line.
   Mesh *mesh;
   if (strncmp(mesh_file, "default", 7) != 0)
   {
      mesh = new Mesh(mesh_file, true, true);
   }
   else
   {
      if (dim == 1)
      {
         mesh = new Mesh(Mesh::MakeCartesian1D(2));
         mesh->GetBdrElement(0)->SetAttribute(1);
         mesh->GetBdrElement(1)->SetAttribute(1);
      }
      if (dim == 2)
      {
         mesh = new Mesh(Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL,
                                               true));
         const int NBE = mesh->GetNBE();
         for (int b = 0; b < NBE; b++)
         {
            Element *bel = mesh->GetBdrElement(b);
            const int attr = (b < NBE/2) ? 2 : 1;
            bel->SetAttribute(attr);
         }
      }
      if (dim == 3)
      {
         mesh = new Mesh(Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON,
                                               true));
         const int NBE = mesh->GetNBE();
         for (int b = 0; b < NBE; b++)
         {
            Element *bel = mesh->GetBdrElement(b);
            const int attr = (b < NBE/3) ? 3 : (b < 2*NBE/3) ? 1 : 2;
            bel->SetAttribute(attr);
         }
      }
   }
   dim = mesh->Dimension();

   // 1D vs partial assembly sanity check.
   if (p_assembly && dim == 1)
   {
      p_assembly = false;
      if (Mpi::Root())
      {
         cout << "Laghos does not support PA in 1D. Switching to FA." << endl;
      }
   }

   // Refine the mesh in serial to increase the resolution.
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   const int mesh_NE = mesh->GetNE();
   if (Mpi::Root())
   {
      cout << "Number of zones in the serial mesh: " << mesh_NE << endl;
   }

   // Parallel partitioning of the mesh.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // Refine the mesh further in parallel to increase the resolution.
   for (int lev = 0; lev < rp_levels; lev++) { pmesh->UniformRefinement(); }

   int NE = pmesh->GetNE(), ne_min, ne_max;
   MPI_Reduce(&NE, &ne_min, 1, MPI_INT, MPI_MIN, 0, pmesh->GetComm());
   MPI_Reduce(&NE, &ne_max, 1, MPI_INT, MPI_MAX, 0, pmesh->GetComm());
   if (myid == 0)
   { cout << "Zones min/max: " << ne_min << " " << ne_max << endl; }

   // Define the parallel finite element spaces. We use:
   // - H1 (Gauss-Lobatto, continuous) for position and velocity.
   // - L2 (Bernstein, discontinuous) for specific internal energy.
   L2_FECollection L2FEC(order_e, dim, BasisType::Positive);
   H1_FECollection H1FEC(order_v, dim);
   ParFiniteElementSpace L2FESpace(pmesh, &L2FEC);
   ParFiniteElementSpace H1FESpace(pmesh, &H1FEC, pmesh->Dimension());

   //
   // Construct the boundary curves.
   //
   // Mark which nodes to move tangentially.
   Array<bool> fit_marker_top(H1FESpace.GetNDofs());
   Array<bool> fit_marker_right(H1FESpace.GetNDofs());
   Array<bool> fit_marker_bottom(H1FESpace.GetNDofs());
   Array<bool> fit_marker_left(H1FESpace.GetNDofs());
   Array<bool> fit_marker_out(H1FESpace.GetNDofs());
   Array<bool> fit_marker_in(H1FESpace.GetNDofs());
   fit_marker_top    = false;
   fit_marker_right  = false;
   fit_marker_bottom = false;
   fit_marker_left   = false;
   fit_marker_out    = false;
   fit_marker_in     = false;
   Array<int> vdofs;
   Array<int> be_to_surface(pmesh->GetNBE());
   be_to_surface = -1;
   for (int e = 0; e < pmesh->GetNBE(); e++)
   {
      const int attr = pmesh->GetBdrElement(e)->GetAttribute();
      const int nd = H1FESpace.GetBE(e)->GetDof();
      H1FESpace.GetBdrElementVDofs(e, vdofs);

      // Top boundary.
      if (attr == 1)
      {
         be_to_surface[e] = 0;
         for (int j = 0; j < nd; j++) { fit_marker_top[vdofs[j]] = true; }
      }
      // Right boundary.
      else if (attr == 2)
      {
         be_to_surface[e] = 1;
         for (int j = 0; j < nd; j++) { fit_marker_right[vdofs[j]] = true; }
      }
      // Bottom boundary.
      else if (attr == 3)
      {
         be_to_surface[e] = 2;
         for (int j = 0; j < nd; j++) { fit_marker_bottom[vdofs[j]] = true; }
      }
      // Left boundary.
      else if (attr == 4)
      {
         be_to_surface[e] = 3;
         for (int j = 0; j < nd; j++) { fit_marker_left[vdofs[j]] = true; }
      }
      else if (attr == 5)
      {
         be_to_surface[e] = 0;
         for (int j = 0; j < nd; j++) { fit_marker_out[vdofs[j]] = true; }
      }
      else if (attr == 7)
      {
         be_to_surface[e] = 1;
         for (int j = 0; j < nd; j++) { fit_marker_in[vdofs[j]] = true; }
      }
   }
   Curve_Sine_Top curve_top(fit_marker_top, a, b, c);
   Curve_Sine_Right curve_right(fit_marker_right, a, b, c);
   Line_Bottom line_bottom(fit_marker_bottom);
   Line_Left line_left(fit_marker_left);
   Circle circle_out(fit_marker_out, 1.0);
   Circle circle_in(fit_marker_in, 0.4);
   Array<AnalyticSurface *> surf_array;
   if (strcmp(mesh_file, "data/wall_linear.mesh") == 0 ||
       strcmp(mesh_file, "data/wall_a02_b05_c15.mesh") == 0)
   {
      surf_array.Append(&curve_top);
      surf_array.Append(&curve_right);
      surf_array.Append(&line_bottom);
      surf_array.Append(&line_left);
   }
   else if (strcmp(mesh_file, "data/circles3.mesh") == 0 ||
            strcmp(mesh_file, "data/circles4.mesh") == 0)
   {
      surf_array.Append(&circle_out);
      surf_array.Append(&circle_in);
   }
   else { MFEM_ABORT("bad setup"); }

   AnalyticCompositeSurface surfaces(surf_array);

   // Boundary conditions: all tests use v.n = 0 on the boundary, and we assume
   // that the boundaries are straight.
   Array<int> ess_tdofs, ess_vdofs;
   const bool BC_strong = false;
   if (BC_strong)
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max()), dofs_marker, dofs_list;
      for (int d = 0; d < pmesh->Dimension(); d++)
      {
         // Attributes 1/2/3 correspond to fixed-x/y/z boundaries,
         // i.e., we must enforce v_x/y/z = 0 for the velocity components.
         ess_bdr = 0; ess_bdr[d] = 1;
         H1FESpace.GetEssentialTrueDofs(ess_bdr, dofs_list, d);
         ess_tdofs.Append(dofs_list);
         H1FESpace.GetEssentialVDofs(ess_bdr, dofs_marker, d);
         FiniteElementSpace::MarkerToList(dofs_marker, dofs_list);
         ess_vdofs.Append(dofs_list);
      }
   }

   // Define the explicit ODE solver used for time integration.
   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK2Solver(0.5); break;
      case 3: ode_solver = new RK3SSPSolver; break;
      case 4: ode_solver = new RK4Solver; break;
      case 6: ode_solver = new RK6Solver; break;
      case 7: ode_solver = new RK2AvgSolver; break;
      default:
         if (myid == 0)
         {
            cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         }
         delete pmesh;
         MPI_Finalize();
         return 3;
   }

   const HYPRE_Int glob_size_l2 = L2FESpace.GlobalTrueVSize();
   const HYPRE_Int glob_size_h1 = H1FESpace.GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of kinematic (position, velocity) dofs: "
           << glob_size_h1 << endl;
      cout << "Number of specific internal energy dofs: "
           << glob_size_l2 << endl;
   }

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
   BlockVector S(offset, Device::GetMemoryType());

   // Define GridFunction objects for the position, velocity and specific
   // internal energy. There is no function for the density, as we can always
   // compute the density values given the current mesh position, using the
   // property of pointwise mass conservation.
   ParGridFunction x_gf, v_gf, e_gf;
   x_gf.MakeRef(&H1FESpace, S, offset[0]);
   v_gf.MakeRef(&H1FESpace, S, offset[1]);
   e_gf.MakeRef(&L2FESpace, S, offset[2]);

   // Initialize x_gf using the starting mesh coordinates.
   pmesh->SetNodalGridFunction(&x_gf);
   // Sync the data location of x_gf with its base, S
   x_gf.SyncAliasMemory(S);

   // Initialize the velocity.
   VectorFunctionCoefficient v_coeff(pmesh->Dimension(), v0);
   v_gf.ProjectCoefficient(v_coeff);
   for (int i = 0; i < ess_vdofs.Size(); i++)
   {
      v_gf(ess_vdofs[i]) = 0.0;
   }
   // Sync the data location of v_gf with its base, S
   v_gf.SyncAliasMemory(S);

   // Initialize density and specific internal energy values. We interpolate in
   // a non-positive basis to get the correct values at the dofs. Then we do an
   // L2 projection to the positive basis in which we actually compute. The goal
   // is to get a high-order representation of the initial condition. Note that
   // this density is a temporary function and it will not be updated during the
   // time evolution.
   ParGridFunction rho0_gf(&L2FESpace);
   FunctionCoefficient rho0_coeff(rho0);
   L2_FECollection l2_fec(order_e, pmesh->Dimension());
   ParFiniteElementSpace l2_fes(pmesh, &l2_fec);
   ParGridFunction l2_rho0_gf(&l2_fes), l2_e(&l2_fes);
   l2_rho0_gf.ProjectCoefficient(rho0_coeff);
   rho0_gf.ProjectGridFunction(l2_rho0_gf);
   GridFunctionCoefficient rho0_gf_coeff(&rho0_gf);
   if (problem == 1)
   {
      // For the Sedov test, we use a delta function at the origin.
      DeltaCoefficient e_coeff(blast_position[0], blast_position[1],
                               blast_position[2], blast_energy);
      e_coeff.SetTol(1e-6);
      l2_e.ProjectCoefficient(e_coeff);

      // ParGridFunction sss(l2_e);
      // DeltaCoefficient e2_coeff(0.4, 0.0, 0.0, blast_energy);
      // e2_coeff.SetTol(1e-6);
      // sss.ProjectCoefficient(e2_coeff);

      // l2_e += sss;
   }
   else
   {
      FunctionCoefficient e_coeff(e0);
      l2_e.ProjectCoefficient(e_coeff);
   }
   e_gf.ProjectGridFunction(l2_e);
   // Sync the data location of e_gf with its base, S
   e_gf.SyncAliasMemory(S);

   // Piecewise constant ideal gas coefficient over the Lagrangian mesh. The
   // gamma values are projected on function that's constant on the moving mesh.
   L2_FECollection mat_fec(0, pmesh->Dimension());
   ParFiniteElementSpace mat_fes(pmesh, &mat_fec);
   ParGridFunction mat_gf(&mat_fes);
   FunctionCoefficient mat_coeff(gamma_func);
   mat_gf.ProjectCoefficient(mat_coeff);

   // Additional details, depending on the problem.
   int source = 0; bool visc = true, vorticity = false;
   switch (problem)
   {
      case 0: if (pmesh->Dimension() == 2) { source = 1; } visc = false; break;
      case 1: visc = true; break;
      case 2: visc = true; break;
      case 3: visc = true; S.HostRead(); break;
      case 4: visc = false; break;
      case 5: visc = true; break;
      case 6: visc = true; break;
      case 7: source = 2; visc = true; vorticity = true;  break;
      default: MFEM_ABORT("Wrong problem specification!");
   }
   if (impose_visc) { visc = true; }

   const double perimeter = 12.0; // TODO fix this.
   hydrodynamics::LagrangianHydroOperator hydro(S.Size(),
                                                H1FESpace, L2FESpace,
                                                ess_tdofs, BC_strong,
                                                rho0_gf_coeff, rho0_gf,
                                                mat_gf, source, cfl,
                                                visc, vorticity, p_assembly,
                                                cg_tol, cg_max_iter, ftz_tol,
                                                order_q, penalty_param, perimeter,
                                                surfaces, be_to_surface);

   socketstream vis_rho, vis_v, vis_e, vis_p, vis_q;
   char vishost[] = "localhost";
   int  visport   = 19916;

   QuadratureSpace qs(*pmesh, hydro.GetIntRule());
   QuadratureFunction rho_qf(qs), p_qf(qs);

   if (visualization || visit)
   {
      hydro.ComputeDensity(rho_qf);
      hydro.ComputePressure(e_gf, mat_gf(0), p_qf);
   }
   const double energy_init = hydro.InternalEnergy(e_gf) +
                              hydro.KineticEnergy(v_gf);
   const double momentum_in = hydro.Momentum(v_gf);

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
      if (problem != 0 && problem != 4)
      {
         hydrodynamics::VisualizeField(vis_rho, vishost, visport, rho_qf,
                                       "Density QF", Wx, Wy, Ww, Wh);
      }
      Wx += offx;
      hydrodynamics::VisualizeField(vis_v, vishost, visport, v_gf,
                                    "Velocity", Wx, Wy, Ww, Wh);
      Wx += offx;
      hydrodynamics::VisualizeField(vis_e, vishost, visport, e_gf,
                                    "Specific Internal Energy", Wx, Wy, Ww, Wh);
      Wx += offx;
      hydrodynamics::VisualizeField(vis_q, vishost, visport, p_qf,
                                    "Pressure QF", Wx, Wy, Ww, Wh);
   }

   // Save data for VisIt visualization.
   VisItDataCollection visit_dc(basename, pmesh);
   if (visit)
   {
      visit_dc.RegisterField("Velocity", &v_gf);
      visit_dc.RegisterField("Specific Internal Energy", &e_gf);
      visit_dc.SetCycle(0);
      visit_dc.SetTime(0.0);
      visit_dc.Save();
   }

   // Perform time-integration (looping over the time iterations, ti, with a
   // time-step dt). The object oper is of type LagrangianHydroOperator that
   // defines the Mult() method that used by the time integrators.
   ode_solver->Init(hydro);
   hydro.ResetTimeStepEstimate();
   double t = 0.0, dt = hydro.GetTimeStepEstimate(S), t_old;
   bool last_step = false;
   int steps = 0;
   BlockVector S_old(S);
   long mem=0, mmax=0, msum=0;
   int checks = 0;
   //   const double internal_energy = hydro.InternalEnergy(e_gf);
   //   const double kinetic_energy = hydro.KineticEnergy(v_gf);
   //   if (mpi.Root())
   //   {
   //      cout << std::fixed;
   //      cout << "step " << std::setw(5) << 0
   //            << ",\tt = " << std::setw(5) << std::setprecision(4) << t
   //            << ",\tdt = " << std::setw(5) << std::setprecision(6) << dt
   //            << ",\t|IE| = " << std::setprecision(10) << std::scientific
   //            << internal_energy
   //            << ",\t|KE| = " << std::setprecision(10) << std::scientific
   //            << kinetic_energy
   //            << ",\t|E| = " << std::setprecision(10) << std::scientific
   //            << kinetic_energy+internal_energy;
   //      cout << std::fixed;
   //      if (mem_usage)
   //      {
   //         cout << ", mem: " << mmax << "/" << msum << " MB";
   //      }
   //      cout << endl;
   //   }

   int ale_cnt = 0;
   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final)
      {
         dt = t_final - t;
         last_step = true;
      }
      if (steps == max_tsteps) { last_step = true; }
      S_old = S;
      t_old = t;
      hydro.ResetTimeStepEstimate();

      // S is the vector of dofs, t is the current time, and dt is the time step
      // to advance.
      ode_solver->Step(S, t, dt);
      steps++;

      // Adaptive time step control.
      const double dt_est = hydro.GetTimeStepEstimate(S);
      if (dt_est < dt)
      {
         // Repeat (solve again) with a decreased time step - decrease of the
         // time estimate suggests appearance of oscillations.
         dt *= 0.85;
         if (dt < std::numeric_limits<double>::epsilon())
         { MFEM_ABORT("The time step crashed!"); }
         t = t_old;
         S = S_old;
         hydro.ResetQuadratureData();
         if (Mpi::Root()) { cout << "Repeating step " << ti << endl; }
         if (steps < max_tsteps) { last_step = false; }
         ti--; continue;
      }
      else if (ale_period > 0.0 && t + 1e-12 > (ale_cnt + 1) * ale_period)
      {
         // ALE step - the next remap period has been reached, the dt was ok.
         if (myid == 0)
         {
            cout << std::fixed << std::setw(5)
                 << "ALE step [" << ale_cnt << "] at " << t << ": " << endl;
         }

         ParGridFunction x_gf_opt(&H1FESpace);
         OptimizeMesh(x_gf, surfaces,
                      hydro.GetIntRule(), hydro.GetIntRule_b(), x_gf_opt);

         const bool remap_v_gslib = true;
         const bool remap_v_adv   = !remap_v_gslib;

         // Setup and initialize the remap operator.
         const double cfl_remap = 0.1;
         RemapAdvector adv(*pmesh, order_v, order_e, cfl_remap, remap_v_adv);

         adv.InitFromLagr(x_gf, v_gf, hydro.GetIntRule(),
                          hydro.GetRhoDetJw(), e_gf);

         // Remap to x_gf_opt.
         adv.ComputeAtNewPosition(x_gf_opt, ess_tdofs);

         ParGridFunction v_new(&H1FESpace);
         if (remap_v_gslib)
         {
            InterpolationRemap interp;
            interp.Remap(v_gf, x_gf_opt, v_new);
         }

         // Move the mesh to x0 and transfer the result from the remap.
         x_gf = x_gf_opt;
         adv.TransferToLagr(rho0_gf, v_gf,
                            hydro.GetIntRule(), hydro.GetRhoDetJw(),
                            hydro.GetIntRule_b(), hydro.GetRhoDetJ_be(), e_gf);
         if (remap_v_gslib) { v_gf = v_new; }

         hydro.RemoveBdrNormalPart(v_gf, x_gf);

         // Update mass matrices.
         // Above we changed rho0_gf to reflect the mass matrices Coefficient.
         hydro.UpdateMassMatrices(rho0_gf_coeff);

         ale_cnt++;
      }
      else if (dt_est > 1.25 * dt) { dt *= 1.02; }

      // Make sure that the mesh corresponds to the new solution state. This is
      // needed, because some time integrators use different S-type vectors
      // and the oper object might have redirected the mesh positions to those.
      pmesh->NewNodes(x_gf, false);

      if (last_step || (ti % vis_steps) == 0)
      {
         double lnorm = e_gf * e_gf, norm;
         MPI_Allreduce(&lnorm, &norm, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
         if (mem_usage)
         {
            mem = GetMaxRssMB();
            MPI_Reduce(&mem, &mmax, 1, MPI_LONG, MPI_MAX, 0, pmesh->GetComm());
            MPI_Reduce(&mem, &msum, 1, MPI_LONG, MPI_SUM, 0, pmesh->GetComm());
         }
         // const double internal_energy = hydro.InternalEnergy(e_gf);
         // const double kinetic_energy = hydro.KineticEnergy(v_gf);
         if (Mpi::Root())
         {
            const double sqrt_norm = sqrt(norm);

            cout << std::fixed;
            cout << "step " << std::setw(5) << ti
                 << ",\tt = " << std::setw(5) << std::setprecision(4) << t
                 << ",\tdt = " << std::setw(5) << std::setprecision(6) << dt
                 << ",\t|e| = " << std::setprecision(10) << std::scientific
                 << sqrt_norm;
            //  << ",\t|IE| = " << std::setprecision(10) << std::scientific
            //  << internal_energy
            //   << ",\t|KE| = " << std::setprecision(10) << std::scientific
            //  << kinetic_energy
            //   << ",\t|E| = " << std::setprecision(10) << std::scientific
            //  << kinetic_energy+internal_energy;
            cout << std::fixed;
            if (mem_usage)
            {
               cout << ", mem: " << mmax << "/" << msum << " MB";
            }
            cout << endl;
         }

         // Make sure all ranks have sent their 'v' solution before initiating
         // another set of GLVis connections (one from each rank):
         MPI_Barrier(pmesh->GetComm());

         if (visualization || visit || gfprint)
         {
            hydro.ComputeDensity(rho_qf);
            hydro.ComputePressure(e_gf, mat_gf(0), p_qf);
         }
         if (visualization)
         {
            int Wx = 0, Wy = 0; // window position
            int Ww = 350, Wh = 350; // window size
            int offx = Ww+10; // window offsets
            if (problem != 0 && problem != 4)
            {
               hydrodynamics::VisualizeField(vis_rho, vishost, visport, rho_qf,
                                             "Density QF", Wx, Wy, Ww, Wh);
            }
            Wx += offx;
            hydrodynamics::VisualizeField(vis_v, vishost, visport,
                                          v_gf, "Velocity", Wx, Wy, Ww, Wh);
            Wx += offx;
            hydrodynamics::VisualizeField(vis_e, vishost, visport, e_gf,
                                          "Specific Internal Energy",
                                          Wx, Wy, Ww, Wh);
            Wx += offx;
            hydrodynamics::VisualizeField(vis_q, vishost, visport, p_qf,
                                          "Pressure QF", Wx, Wy, Ww, Wh);
         }

         if (visit)
         {
            visit_dc.SetCycle(ti);
            visit_dc.SetTime(t);
            visit_dc.Save();
         }

         if (gfprint)
         {
            std::ostringstream mesh_name, rho_name, v_name, e_name;
            mesh_name << basename << "_" << ti << "_mesh";
            rho_name  << basename << "_" << ti << "_rho";
            v_name << basename << "_" << ti << "_v";
            e_name << basename << "_" << ti << "_e";

            std::ofstream mesh_ofs(mesh_name.str().c_str());
            mesh_ofs.precision(8);
            pmesh->PrintAsOne(mesh_ofs);
            mesh_ofs.close();

            std::ofstream v_ofs(v_name.str().c_str());
            v_ofs.precision(8);
            v_gf.SaveAsOne(v_ofs);
            v_ofs.close();

            std::ofstream e_ofs(e_name.str().c_str());
            e_ofs.precision(8);
            e_gf.SaveAsOne(e_ofs);
            e_ofs.close();
         }
      }
   }
   MFEM_VERIFY(!check || checks == 2, "Check error!");

   switch (ode_solver_type)
   {
      case 2: steps *= 2; break;
      case 3: steps *= 3; break;
      case 4: steps *= 4; break;
      case 6: steps *= 6; break;
      case 7: steps *= 2;
   }

//   hydro.PrintTimingData(Mpi::Root(), steps, fom);

   if (mem_usage)
   {
      mem = GetMaxRssMB();
      MPI_Reduce(&mem, &mmax, 1, MPI_LONG, MPI_MAX, 0, pmesh->GetComm());
      MPI_Reduce(&mem, &msum, 1, MPI_LONG, MPI_SUM, 0, pmesh->GetComm());
   }

   const double energy_final = hydro.InternalEnergy(e_gf) +
                               hydro.KineticEnergy(v_gf);
   const double momentum_out = hydro.Momentum(v_gf);
   if (Mpi::Root())
   {
      cout << endl;
      cout << "Energy  diff: " << std::scientific << std::setprecision(2)
           << fabs(energy_init - energy_final) << endl;
      cout << "Moment  diff: " << std::scientific << std::setprecision(2)
           << fabs(momentum_in - momentum_out) << endl;
      if (mem_usage)
      {
         cout << "Maximum memory resident set size: "
              << mmax << "/" << msum << " MB" << endl;
      }
   }

   // Position errors w.r.t. a perfect circle.
   pmesh->ExchangeFaceNbrData();
   if (strcmp(mesh_file, "data/refined.mesh") == 0)
   {
      auto ir = IntRules.Get(Geometry::SEGMENT, 12);

      double perim = 0.0, pos_error = 0.0;
      for (int i = 0; i < pmesh->GetNBE(); i++)
      {
         auto Tr = pmesh->GetBdrFaceTransformations(i);
         if (Tr == NULL) { continue; }

         auto ip_c = Geometries.GetCenter(Geometry::SEGMENT);
         Vector pos(dim);
         Tr->Transform(ip_c, pos);
         if (!(pos(0) > 0.2 && pos(0) < 0.8 &&
               pos(1) > 0.2 && pos(1) < 0.8)) { continue; }

         for (int p = 0; p < ir.GetNPoints(); p++)
         {
            const IntegrationPoint &ip = ir.IntPoint(p);
            Tr->SetAllIntPoints(&ip);
            Vector vals;
            x_gf.GetVectorValue(*Tr, ip, vals);
            double err = sqrt((vals(0) - 0.5) * (vals(0) - 0.5) +
                              (vals(1) - 0.5) * (vals(1) - 0.5));
            err = fabs(err - 0.2);

            Vector nor(2);
            CalcOrtho(Tr->Jacobian(), nor);

            double w = ip.weight * nor.Norml2();
            perim     += w;
            pos_error += w * err;
         }
      }

      MPI_Allreduce(MPI_IN_PLACE, &perim, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
      MPI_Allreduce(MPI_IN_PLACE, &pos_error, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());

      if (Mpi::Root())
      {
         cout << "-- Points per face: " << ir.GetNPoints() << endl;
         cout << "-- Perimeter error: " << fabs(perim - 2 * M_PI * 0.2) << endl
              << "-- Position error:  " << pos_error << endl;
      }
   }

   // Position errors w.r.t. a perfect sphere.
   pmesh->ExchangeFaceNbrData();
   if (strcmp(mesh_file, "data/sphere_hole_V4.msh") == 0)
   {
      auto ir = IntRules.Get(Geometry::SQUARE, 12);

      double perim = 0.0, pos_error = 0.0;
      for (int i = 0; i < pmesh->GetNBE(); i++)
      {
         auto Tr = pmesh->GetBdrFaceTransformations(i);
         if (Tr == NULL) { continue; }

         auto ip_c = Geometries.GetCenter(Geometry::SQUARE);
         Vector pos(dim);
         Tr->Transform(ip_c, pos);
         if (!(pos(0) > -0.3 && pos(0) < 0.3 &&
               pos(1) > -0.3 && pos(1) < 0.3 &&
               pos(2) > -0.3 && pos(2) < 0.3)) { continue; }

         for (int p = 0; p < ir.GetNPoints(); p++)
         {
            const IntegrationPoint &ip = ir.IntPoint(p);
            Tr->SetAllIntPoints(&ip);
            Vector vals;
            x_gf.GetVectorValue(*Tr, ip, vals);
            double err = sqrt((vals(0) - 0.0) * (vals(0) - 0.0) +
                              (vals(1) - 0.0) * (vals(1) - 0.0) +
                              (vals(2) - 0.0) * (vals(2) - 0.0));
            err = fabs(err - 0.2);

            Vector nor(3);
            CalcOrtho(Tr->Jacobian(), nor);

            double w = ip.weight * nor.Norml2();
            perim     += w;
            pos_error += w * err;
         }
      }

      MPI_Allreduce(MPI_IN_PLACE, &perim, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
      MPI_Allreduce(MPI_IN_PLACE, &pos_error, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());

      if (Mpi::Root())
      {
         cout << "-- Points per face: " << ir.GetNPoints() << endl;
         cout << "-- Perimeter error: " << fabs(perim - 4 * M_PI * 0.04) << endl
              << "-- Position error:  " << pos_error << endl;
      }
   }

   // Print the error.
   // For problems 0 and 4 the exact velocity is constant in time.
   if (problem == 0 || problem == 4)
   {
      const double error_max = v_gf.ComputeMaxError(v_coeff),
                   error_l1  = v_gf.ComputeL1Error(v_coeff),
                   error_l2  = v_gf.ComputeL2Error(v_coeff);
      if (Mpi::Root())
      {
         cout << "L_inf  error: " << error_max << endl
              << "L_1    error: " << error_l1 << endl
              << "L_2    error: " << error_l2 << endl;
      }
   }

   if (visualization)
   {
      vis_v.close();
      vis_e.close();
   }

   // Free the used memory.
   delete ode_solver;
   delete pmesh;

   return 0;
}

double rho0(const Vector &x)
{
   switch (problem)
   {
      case 0: return 1.0;
      case 1: return 1.0;
      case 2: return (x(0) < 0.5) ? 1.0 : 0.1;
      case 3: return (dim == 2) ? (x(0) > 1.0 && x(1) > 1.5) ? 0.125 : 1.0
                        : x(0) > 1.0 && ((x(1) < 1.5 && x(2) < 1.5) ||
                                         (x(1) > 1.5 && x(2) > 1.5)) ? 0.125 : 1.0;
      case 4: return 1.0;
      case 5:
      {
         if (x(0) >= 0.5 && x(1) >= 0.5) { return 0.5313; }
         if (x(0) <  0.5 && x(1) <  0.5) { return 0.8; }
         return 1.0;
      }
      case 6:
      {
         if (x(0) <  0.5 && x(1) >= 0.5) { return 2.0; }
         if (x(0) >= 0.5 && x(1) <  0.5) { return 3.0; }
         return 1.0;
      }
      case 7: return x(1) >= 0.0 ? 2.0 : 1.0;
      default: MFEM_ABORT("Bad number given for problem id!"); return 0.0;
   }
}

double gamma_func(const Vector &x)
{
   switch (problem)
   {
      case 0: return 5.0 / 3.0;
      case 1: return 1.4;
      case 2: return 1.4;
      case 3:
         if (dim == 1) { return (x(0) > 0.5) ? 1.4 : 1.5; }
         else { return (x(0) > 1.0 && x(1) <= 1.5) ? 1.4 : 1.5; }
      case 4: return 5.0 / 3.0;
      case 5: return 1.4;
      case 6: return 1.4;
      case 7: return 5.0 / 3.0;
      default: MFEM_ABORT("Bad number given for problem id!"); return 0.0;
   }
}

static double rad(double x, double y) { return sqrt(x*x + y*y); }

void v0(const Vector &x, Vector &v)
{
   const double atn = dim!=1 ? pow((x(0)*(1.0-x(0))*4*x(1)*(1.0-x(1))*4.0),
                                   0.4) : 0.0;
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
      case 4:
      {
         v = 0.0;
         const double r = rad(x(0), x(1));
         if (r < 0.2)
         {
            v(0) =  5.0 * x(1);
            v(1) = -5.0 * x(0);
         }
         else if (r < 0.4)
         {
            v(0) =  2.0 * x(1) / r - 5.0 * x(1);
            v(1) = -2.0 * x(0) / r + 5.0 * x(0);
         }
         else { }
         break;
      }
      case 5:
      {
         v = 0.0;
         if (x(0) >= 0.5 && x(1) >= 0.5) { v(0)=0.0*atn, v(1)=0.0*atn; return;}
         if (x(0) <  0.5 && x(1) >= 0.5) { v(0)=0.7276*atn, v(1)=0.0*atn; return;}
         if (x(0) <  0.5 && x(1) <  0.5) { v(0)=0.0*atn, v(1)=0.0*atn; return;}
         if (x(0) >= 0.5 && x(1) <  0.5) { v(0)=0.0*atn, v(1)=0.7276*atn; return; }
         MFEM_ABORT("Error in problem 5!");
         return;
      }
      case 6:
      {
         v = 0.0;
         if (x(0) >= 0.5 && x(1) >= 0.5) { v(0)=+0.75*atn, v(1)=-0.5*atn; return;}
         if (x(0) <  0.5 && x(1) >= 0.5) { v(0)=+0.75*atn, v(1)=+0.5*atn; return;}
         if (x(0) <  0.5 && x(1) <  0.5) { v(0)=-0.75*atn, v(1)=+0.5*atn; return;}
         if (x(0) >= 0.5 && x(1) <  0.5) { v(0)=-0.75*atn, v(1)=-0.5*atn; return;}
         MFEM_ABORT("Error in problem 6!");
         return;
      }
      case 7:
      {
         v = 0.0;
         v(1) = 0.02 * exp(-2*M_PI*x(1)*x(1)) * cos(2*M_PI*x(0));
         break;
      }
      default: MFEM_ABORT("Bad number given for problem id!");
   }
}

double e0(const Vector &x)
{
   switch (problem)
   {
      case 0:
      {
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
      case 1: return 0.0; // This case in initialized in main().
      case 2: return (x(0) < 0.5) ? 1.0 / rho0(x) / (gamma_func(x) - 1.0)
                        : 0.1 / rho0(x) / (gamma_func(x) - 1.0);
      case 3: return (x(0) > 1.0) ? 0.1 / rho0(x) / (gamma_func(x) - 1.0)
                        : 1.0 / rho0(x) / (gamma_func(x) - 1.0);
      case 4:
      {
         const double r = rad(x(0), x(1)), rsq = x(0) * x(0) + x(1) * x(1);
         const double gamma = 5.0 / 3.0;
         if (r < 0.2)
         {
            return (5.0 + 25.0 / 2.0 * rsq) / (gamma - 1.0);
         }
         else if (r < 0.4)
         {
            const double t1 = 9.0 - 4.0 * log(0.2) + 25.0 / 2.0 * rsq;
            const double t2 = 20.0 * r - 4.0 * log(r);
            return (t1 - t2) / (gamma - 1.0);
         }
         else { return (3.0 + 4.0 * log(2.0)) / (gamma - 1.0); }
      }
      case 5:
      {
         const double irg = 1.0 / rho0(x) / (gamma_func(x) - 1.0);
         if (x(0) >= 0.5 && x(1) >= 0.5) { return 0.4 * irg; }
         if (x(0) <  0.5 && x(1) >= 0.5) { return 1.0 * irg; }
         if (x(0) <  0.5 && x(1) <  0.5) { return 1.0 * irg; }
         if (x(0) >= 0.5 && x(1) <  0.5) { return 1.0 * irg; }
         MFEM_ABORT("Error in problem 5!");
         return 0.0;
      }
      case 6:
      {
         const double irg = 1.0 / rho0(x) / (gamma_func(x) - 1.0);
         if (x(0) >= 0.5 && x(1) >= 0.5) { return 1.0 * irg; }
         if (x(0) <  0.5 && x(1) >= 0.5) { return 1.0 * irg; }
         if (x(0) <  0.5 && x(1) <  0.5) { return 1.0 * irg; }
         if (x(0) >= 0.5 && x(1) <  0.5) { return 1.0 * irg; }
         MFEM_ABORT("Error in problem 6!");
         return 0.0;
      }
      case 7:
      {
         const double rho = rho0(x), gamma = gamma_func(x);
         return (6.0 - rho * x(1)) / (gamma - 1.0) / rho;
      }
      default: MFEM_ABORT("Bad number given for problem id!"); return 0.0;
   }
}

static void display_banner(std::ostream &os)
{
   os << endl
      << "       __                __                 " << endl
      << "      / /   ____  ____  / /_  ____  _____   " << endl
      << "     / /   / __ `/ __ `/ __ \\/ __ \\/ ___/ " << endl
      << "    / /___/ /_/ / /_/ / / / / /_/ (__  )    " << endl
      << "   /_____/\\__,_/\\__, /_/ /_/\\____/____/  " << endl
      << "               /____/                       " << endl << endl;
}

static long GetMaxRssMB()
{
   struct rusage usage;
   if (getrusage(RUSAGE_SELF, &usage)) { return -1; }
#ifndef __APPLE__
   const long unit = 1024; // kilo
#else
   const long unit = 1024*1024; // mega
#endif
   return usage.ru_maxrss/unit; // mega bytes
}

static void Checks(const int ti, const double nrm, int &chk)
{
   const double eps = 1.e-13;
   //printf("\033[33m%.15e\033[m\n",nrm);

   auto check = [&](int p, int i, const double res)
   {
      auto rerr = [](const double a, const double v, const double eps)
      {
         MFEM_VERIFY(fabs(a) > eps && fabs(v) > eps, "One value is near zero!");
         const double err_a = fabs((a-v)/a);
         const double err_v = fabs((a-v)/v);
         return fmax(err_a, err_v) < eps;
      };
      if (problem == p && ti == i)
      { chk++; MFEM_VERIFY(rerr(nrm, res, eps), "P"<<problem<<", #"<<i); }
   };

   const double it_norms[2][8][2][2] = // dim, problem, {it,norm}
   {
      {
         {{5, 6.546538624534384e+00}, { 27, 7.588576357792927e+00}},
         {{5, 3.508254945225794e+00}, { 15, 2.756444596823211e+00}},
         {{5, 1.020745795651244e+01}, { 59, 1.721590205901898e+01}},
         {{5, 8.000000000000000e+00}, { 16, 8.000000000000000e+00}},
         {{5, 3.446324942352448e+01}, { 18, 3.446844033767240e+01}},
         {{5, 1.030899557252528e+01}, { 36, 1.057362418574309e+01}},
         {{5, 8.039707010835693e+00}, { 36, 8.316970976817373e+00}},
         {{5, 1.514929259650760e+01}, { 25, 1.514931278155159e+01}},
      },
      {
         {{5, 1.198510951452527e+03}, {188, 1.199384410059154e+03}},
         {{5, 1.339163718592566e+01}, { 28, 7.521073677397994e+00}},
         {{5, 2.041491591302486e+01}, { 59, 3.443180411803796e+01}},
         {{5, 1.600000000000000e+01}, { 16, 1.600000000000000e+01}},
         {{5, 6.892649884704898e+01}, { 18, 6.893688067534482e+01}},
         {{5, 2.061984481890964e+01}, { 36, 2.114519664792607e+01}},
         {{5, 1.607988713996459e+01}, { 36, 1.662736010353023e+01}},
         {{5, 3.029858112572883e+01}, { 24, 3.029858832743707e+01}}
      }
   };

   for (int p=0; p<8; p++)
   {
      for (int i=0; i<2; i++)
      {
         const int it = it_norms[dim-2][p][i][0];
         const double norm = it_norms[dim-2][p][i][1];
         check(p, it, norm);
      }
   }
}
