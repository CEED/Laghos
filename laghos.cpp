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
// Laghos (LAGrangian High-Order Solver) is a miniapp that solves the
// time-dependent Euler equation of compressible gas dynamics in a moving
// Lagrangian frame using unstructured high-order finite element spatial
// discretization and explicit high-order time-stepping. Laghos is based on the
// numerical algorithm described in the following article:
//
//    V. Dobrev, Tz. Kolev and R. Rieben, "High-order curvilinear finite element
//    methods for Lagrangian hydrodynamics", SIAM Journal on Scientific
//    Computing, (34) 2012, pp. B606–B641, https://doi.org/10.1137/120864672.
//
// Test problems: see README.

#include "laghos_solver.hpp"
#include "dist_solver.hpp"
#include "laghos_ale.hpp"
#include "riemann1D.hpp"

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

static void display_banner(std::ostream&);

int main(int argc, char *argv[])
{
   // Initialize MPI.
   MPI_Session mpi(argc, argv);
   const int myid = mpi.WorldRank();

   // Print the banner.
   if (mpi.Root()) { display_banner(cout); }

   // Parse command-line options.
   problem = 1;
   dim = 3;
   int zones = 50;
   const char *mesh_file = "default";
   int rs_levels = 2;
   int rp_levels = 0;
   int order_v = 2;
   int order_e = 1;
   int order_q = -1;
   int ode_solver_type = 4;
   double t_final = 0.6;
   double cfl = 0.5;
   double cg_tol = 1e-8;
   double ftz_tol = 0.0;
   int cg_max_iter = 300;
   int max_tsteps = -1;
   bool impose_visc = false;
   bool visualization = false;
   int vis_steps = 5;
   bool visit = false;
   bool gfprint = false;
   const char *basename = "results/Laghos";
   double blast_energy = 0.25;
   double blast_position[] = {0.0, 0.0, 0.0};

   OptionsParser args(argc, argv);
   args.AddOption(&dim, "-dim", "--dimension", "Dimension of the problem.");
   args.AddOption(&zones, "-z", "--zones_1d", "1D zones for problem 8.");
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
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
   args.AddOption(&cfl, "-cfl", "--cfl", "CFL-condition number.");
   args.AddOption(&cg_tol, "-cgt", "--cg-tol",
                  "Relative CG tolerance (velocity linear solve).");
   args.AddOption(&ftz_tol, "-ftz", "--ftz-tol",
                  "Absolute flush-to-zero tolerance.");
   args.AddOption(&cg_max_iter, "-cgm", "--cg-max-steps",
                  "Maximum number of CG iterations (velocity linear solve).");
   args.AddOption(&max_tsteps, "-ms", "--max-steps",
                  "Maximum number of steps (negative means no restriction).");
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
   args.Parse();
   if (!args.Good())
   {
      if (mpi.Root()) { args.PrintUsage(cout); }
      return 1;
   }
   if (mpi.Root()) { args.PrintOptions(cout); }

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
         int n = 2;
         if (problem == 8 || problem == 9) { n = zones; }
         mesh = new Mesh(n);
         mesh->GetBdrElement(0)->SetAttribute(1);
         mesh->GetBdrElement(1)->SetAttribute(1);
      }
      if (dim == 2)
      {
         if (problem == 10) {
             mesh = new Mesh(8, 4, Element::QUADRILATERAL, true, 7, 3);
             //mesh = new Mesh(2, 2, Element::QUADRILATERAL, true);
         }
         else
         { mesh = new Mesh(2, 2, Element::QUADRILATERAL, true); }

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
         mesh = new Mesh(2, 2, 2, Element::HEXAHEDRON, true);
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

   // Refine the mesh in serial to increase the resolution.
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   const int mesh_NE = mesh->GetNE();
   if (mpi.Root())
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

   // Boundary conditions: all tests use v.n = 0 on the boundary, and we assume
   // that the boundaries are straight.
   Array<int> ess_tdofs, ess_vdofs;
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
   if (mpi.Root())
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
   const int Vsize_h1 = H1FESpace.GetVSize();
   const int Vsize_l2 = L2FESpace.GetVSize();
   Array<int> offset(4);
   offset[0] = 0;
   offset[1] = offset[0] + Vsize_h1;
   offset[2] = offset[1] + Vsize_h1;
   offset[3] = offset[2] + Vsize_l2;
   BlockVector S(offset);

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

   // Initial mesh positions.
   ParGridFunction x0(x_gf);

   // Initialize the velocity.
   VectorFunctionCoefficient v_coeff(pmesh->Dimension(), v0);
   v_gf.ProjectCoefficient(v_coeff);
   for (int i = 0; i < ess_vdofs.Size(); i++) { v_gf(ess_vdofs[i]) = 0.0; }

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
   if (problem == 1)
   {
      // For the Sedov test, we use a delta function at the origin.
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

   // Piecewise constant ideal gas coefficient over the Lagrangian mesh. The
   // gamma values are projected on function that's constant on the moving mesh.
   L2_FECollection mat_fec(0, pmesh->Dimension());
   ParFiniteElementSpace mat_fes(pmesh, &mat_fec);
   ParGridFunction gamma_gf(&mat_fes);
   FunctionCoefficient mat_coeff(gamma_func);
   gamma_gf.ProjectCoefficient(mat_coeff);

   //
   // Shifted interface options.
   //
   // FE space for the pressure reconstruction.
   // L2 or H1.
   PressureFunction::PressureSpace p_space = PressureFunction::L2;
   // Integration of mass matrices.
   // true  -- the element mass matrices are integrated as mixed.
   // false -- the element mass matrices are integrated as pure.
   bool mix_mass = false;
   // 0 -- no shifting term.
   // 1 -- the momentum RHS gets this term:  - < [grad_p.d] psi >
   // 2 -- the momentum RHS gets this term:  - < [grad_p.d * grad_psi.d] n >
   // 3 -- the momentum RHS gets this term:  - < [(p + grad_p.d) * grad_psi.d] n >
   // 4 -- the momentum RHS gets this term:  - < [(p + grad_p.d)] [psi+grad_psi.d] n >
   // 5 -- the momentum RHS gets this term:  - < [grad_p.d] [psi+grad_psi.d] n >
   int v_shift_type = 0;
   bool shift_momentum = true;
   // 0 -- no shifting terms.
   // 1 -- the energy RHS gets the conservative momentum term:
   //      + < [grad_p.d] v phi >                         for v_shift_type = 1.
   //      + < [grad_p.d * sum_i grad_vi.d] n phi >       for v_shift_type = 2.
   //      + < [(p + grad_p.d) * sum_i grad_vi.d] n phi > for v_shift_type = 3.
   //      + < [(p + grad_p.d)] [sum_i grad_vi.d] n phi > for v_shift_type = 4.
   // 2 -- - <[[((nabla v d) . n)n]], {{p phi}} + <v, phi[[\grad p . d]]>
   // 3 -- - <[[((nabla v d) . n)n]], {{p}}{{phi}} - (1-gamma)(gamma)[[nabla p. d]].[[nabla phi]]>  + <v, phi[[\grad p . d]]>
   // 4 -- - <[[((nabla v d) . n)n]], {{p phi}}
   // 5 -- - <[[((nabla v d) . n)n]], {{p}}{{phi}} - (1-gamma)(gamma)[[nabla p. d]].[[nabla phi]]> - < {v},{phi}[[p + nabla p . d]]>
   // optionally, a stability term can be added:
   // + (dt / h) * [[ p + grad p . d ]], [[ phi + grad phi . d]]
   int e_shift_type = 0;
   // Scaling of both shifting terms.
   double shift_scale = 1.0;
   // Activate the diffusion.
   bool   v_shift_diffusion = false;
   double v_shift_diffusion_scale = 1.0;
   // Activate ALE. The ale_period is in physical time.
   const bool do_ale = false;
   const double ale_period = 0.1;

   const bool pure_test = (v_shift_type > 0 || e_shift_type > 0) ? false : true;
   bool calc_dist = (v_shift_type > 0 || e_shift_type > 0) ? true : false;
   calc_dist = true;

   if (e_shift_type > 1)
   {
      MFEM_VERIFY(mpi.WorldSize() == 1, "The e terms are not parallel yet.");
   }
   if (e_shift_type == 1)
   {
      MFEM_VERIFY(v_shift_type >= 1 || v_shift_type <= 5,
                 "doesn't match");
   }

//#define EXTRACT_1D

   // Interface function.
   ParFiniteElementSpace pfes_xi(pmesh, &H1FEC);
   ParGridFunction xi(&pfes_xi);
   hydrodynamics::InterfaceCoeff coeff_xi_0(problem, *pmesh, pure_test);
   xi.ProjectCoefficient(coeff_xi_0);
   GridFunctionCoefficient coeff_xi(&xi);

   // Material marking and visualization function.
   ParGridFunction materials(&mat_fes);
   int zone_id_L, zone_id_R;
   for (int i = 0; i < NE; i++)
   {
      int mat_id = hydrodynamics::material_id(i, xi);
      pmesh->SetAttribute(i, mat_id + 1);
      materials(i) = mat_id;
      if (i > 0 && materials(i-1) == 0 && materials(i) == 1)
      {
         // Relevant only for the 1D tests.
         zone_id_L = i-1;
         zone_id_R = i;
      }
   }
   hydrodynamics::MarkFaceAttributes(pfes_xi);

   // Set the initial condition based on the materials.
   Coefficient *rho_coeff = &rho0_coeff;
   GridFunctionCoefficient rho0_gf_coeff(&rho0_gf);
   rho_coeff = &rho0_gf_coeff;
   if (v_shift_type > 0 || e_shift_type > 0)
   {
      if (problem == 8)
      {
         hydrodynamics::InitSod2Mat(rho0_gf, v_gf, e_gf, gamma_gf);
         if (mix_mass == false) { rho_coeff = &rho0_gf_coeff; }
      }
      else if (problem == 9)
      {
         hydrodynamics::InitWaterAir(rho0_gf, v_gf, e_gf, gamma_gf);
         if (mix_mass == false) { rho_coeff = &rho0_gf_coeff; }
      }
      else if (problem == 10)
      {
         hydrodynamics::InitTriPoint2Mat(rho0_gf, v_gf, e_gf, gamma_gf);
         if (mix_mass == false) { rho_coeff = &rho0_gf_coeff; }
      }
   }

   v_gf.SyncAliasMemory(S);
   e_gf.SyncAliasMemory(S);

   // Distance vector.
   ParGridFunction dist(&H1FESpace);
   VectorGridFunctionCoefficient dist_coeff(&dist);
   PLapDistanceSolver dist_solver(7);
   //HeatDistanceSolver dist_solver(2.0);
   //dist_solver.diffuse_iter = 1;
   dist_solver.print_level = 0;
   if (calc_dist) { dist_solver.ComputeVectorDistance(coeff_xi, dist); }
   else           { dist = 0.0; }

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
      case 8: visc = true; break;
      case 9: visc = true; break;
      case 10: visc = true; S.HostRead(); break;
      default: MFEM_ABORT("Wrong problem specification!");
   }
   if (impose_visc) { visc = true; }

   double dt;
   PressureFunction p_gf(*pmesh, p_space, rho0_gf, order_e, gamma_gf);
   hydrodynamics::LagrangianHydroOperator hydro(S.Size(),
                                                H1FESpace, L2FESpace, ess_tdofs,
                                                *rho_coeff, rho0_gf, v_gf,
                                                gamma_gf, dist_coeff, p_gf,
                                                source, cfl,
                                                visc, vorticity,
                                                cg_tol, cg_max_iter, ftz_tol,
                                                order_q, &dt);
   hydro.SetShiftingOptions(problem, v_shift_type, e_shift_type,
                            shift_momentum, shift_scale,
                            v_shift_diffusion, v_shift_diffusion_scale);

   socketstream vis_rho, vis_v, vis_e, vis_p, vis_xi, vis_dist, vis_mat;
   char vishost[] = "localhost";
   int  visport   = 19916;

   ParGridFunction rho_gf(&L2FESpace);
   if (visualization || visit) { hydro.ComputeDensity(rho_gf); }
   const double energy_init = hydro.InternalEnergy(e_gf) +
                              hydro.KineticEnergy(v_gf);
   const double momentum_init = hydro.Momentum(v_gf);

   if (visualization)
   {
      // Make sure all MPI ranks have sent their 'v' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh->GetComm());
      vis_rho.precision(8);
      vis_v.precision(8);
      vis_e.precision(8);
      vis_p.precision(8);
      vis_xi.precision(8);
      vis_dist.precision(8);
      vis_mat.precision(8);
      int Wx = 0, Wy = 0; // window position
      const int Ww = 500, Wh = 500; // window size
      int offx = Ww + 10; // window offsets
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
      Wy += Wh + Wh/5;
      Wx = 0;
      hydrodynamics::VisualizeField(vis_p, vishost, visport,
                                    hydro.GetPressure(e_gf),
                                    "Pressure", Wx, Wy, Ww, Wh);
      Wx += offx;
      hydrodynamics::VisualizeField(vis_xi, vishost, visport, xi,
                                    "Interface", Wx, Wy, Ww, Wh);
      Wx += offx;
      hydrodynamics::VisualizeField(vis_dist, vishost, visport, dist,
                                    "Distances", Wx, Wy, Ww, Wh);
      hydrodynamics::VisualizeField(vis_mat, vishost, visport, materials,
                                    "Materials", 0, 0, Ww, Wh);
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
   ode_solver->Init(hydro);
   hydro.ResetTimeStepEstimate();
   double t = 0.0, t_old;
   dt = hydro.GetTimeStepEstimate(S);
   bool last_step = false;
   int steps = 0;
   BlockVector S_old(S);

   // Shifting - related extractors.
#ifdef EXTRACT_1D

   if (problem != 8 && problem != 9)
   {
      MFEM_ABORT("Please comment out extract1D\n.");
   }
   MFEM_VERIFY(H1FESpace.GetNRanks() == 1,
               "Point extraction works inly in serial.");
   const double dx = 1.0 / NE;
   ParGridFunction &pe_gf = hydro.GetPressure(e_gf);
   Vector point_interface(1), point_face(1);
   point_interface(0) = 0.5;
   if (problem == 9)
   {
      point_interface(0) = (pure_test) ? 0.7 : 0.7 + 0.5*dx;
   }
   point_face(0) = zone_id_R * dx;
   std::cout << zone_id_L << " " << zone_id_R << std::endl;
   std::cout << "True interface: " << point_interface(0) << std::endl
             << "Surrogate:      " << point_face(0) <<  std::endl;
   hydrodynamics::PrintCellNumbers(point_interface, H1FESpace);
   hydrodynamics::PrintCellNumbers(point_face, L2FESpace);
   // By construction, the interface is in the left zone.
   std::string vname, xname, pnameFL, pnameFR, pnameSL, pnameSR,
               enameFL, enameFR, enameSL, enameSR;

   std::string prefix = (problem == 8) ? "sod_" : "wa_";
   vname = prefix + "v.out";
   xname = prefix + "x.out";
   enameFL = prefix + "e_fit_L.out";
   enameFR = prefix + "e_fit_R.out";
   enameSL = prefix + "e_shift_L.out";
   enameSR = prefix + "e_shift_R.out";
   pnameFL = prefix + "p_fit_L.out";
   pnameFR = prefix + "p_fit_R.out";
   pnameSL = prefix + "p_shift_L.out";
   pnameSR = prefix + "p_shift_R.out";

   //hydrodynamics::PointExtractor v_extr(zone_id_L, point_interface, v_gf, vname);
   //hydrodynamics::PointExtractor x_extr(zone_id_L, point_interface, x_gf, xname);
   hydrodynamics::PointExtractor e_L_extr(zone_id_L, point_face, e_gf, enameFL);
   hydrodynamics::PointExtractor e_R_extr(zone_id_R, point_face, e_gf, enameFR);
   hydrodynamics::ShiftedPointExtractor e_LS_extr(zone_id_L, point_face, e_gf,
                                                  dist, enameSL);
   hydrodynamics::ShiftedPointExtractor e_RS_extr(zone_id_R, point_face, e_gf,
                                                  dist, enameSR);
   hydrodynamics::PointExtractor p_L_extr(zone_id_L, point_face, pe_gf, pnameFL);
   hydrodynamics::PointExtractor p_R_extr(zone_id_R, point_face, pe_gf, pnameFR);
   hydrodynamics::ShiftedPointExtractor p_LS_extr(zone_id_L, point_face, pe_gf,
                                                  dist, pnameSL);
   hydrodynamics::ShiftedPointExtractor p_RS_extr(zone_id_R, point_face, pe_gf,
                                                  dist, pnameSR);
   //v_extr.WriteValue(0.0);
   //x_extr.WriteValue(0.0);
   if (pure_test)
   {
      e_L_extr.WriteValue(0.0);
      e_R_extr.WriteValue(0.0);
      p_L_extr.WriteValue(0.0);
      p_R_extr.WriteValue(0.0);
   }
   else
   {
      e_LS_extr.WriteValue(0.0);
      e_RS_extr.WriteValue(0.0);
      p_LS_extr.WriteValue(0.0);
      p_RS_extr.WriteValue(0.0);
   }
#endif

   double energy_old = energy_init,
          energy_new = energy_init;

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
      // to advance. The function does t += dt.

      ode_solver->Step(S, t, dt);
      steps++;

      // Adaptive time step control.
      const double dt_est = hydro.GetTimeStepEstimate(S);
      if (dt_est < dt)
      {
         // Repeat (solve again) with a decreased time step - decrease of the
         // time estimate suggests appearance of oscillations.
         dt *= 0.85;
         if (dt < 1e-12) { MFEM_ABORT("The time step crashed!"); }
         t = t_old;
         S = S_old;
         hydro.ResetQuadratureData();
         if (mpi.Root()) { cout << "Repeating step " << ti << endl; }
         if (steps < max_tsteps) { last_step = false; }
         ti--; continue;
      }
      else if (do_ale && t + 1e-12 > (ale_cnt + 1) * ale_period)
      {
         // ALE step - the next remap period has been reached, the dt was ok.
         double mass_in     = hydro.Mass(),
                momentum_in = hydro.Momentum(v_gf),
                internal_in = hydro.InternalEnergy(e_gf),
                kinetic_in  = hydro.KineticEnergy(v_gf),
                total_in    = internal_in + kinetic_in;

         // Remap to x0 (the remesh always goes back to x0).
         RemapAdvector adv(*pmesh, order_v, order_e);
         adv.InitFromLagr(x_gf, xi, v_gf,
                          hydro.GetIntRule(), hydro.GetRhoDetJw(), e_gf);

         adv.ComputeAtNewPosition(x0);

         // Move the mesh back and transfer the result from the remap.
         x_gf = x0;
         adv.TransferToLagr(xi, v_gf,
                            hydro.GetIntRule(), hydro.GetRhoDetJw(),
                            rho0_gf, e_gf);

         // rho0_gf is changed to reflect the mass matrices Coefficient.
         rho_coeff = &rho0_gf_coeff;

         // Mass matrices.
         hydro.UpdateMassMatrices(*rho_coeff);

         // Material marking and visualization function.
         for (int k = 0; k < NE; k++)
         {
            int mat_id = hydrodynamics::material_id(k, xi);
            pmesh->SetAttribute(k, mat_id + 1);
            materials(k) = mat_id;
         }
         hydrodynamics::MarkFaceAttributes(pfes_xi);

         ale_cnt++;

         double mass_out     = hydro.Mass(),
                momentum_out = hydro.Momentum(v_gf),
                internal_out = hydro.InternalEnergy(e_gf),
                kinetic_out  = hydro.KineticEnergy(v_gf),
                total_out    = internal_out + kinetic_out;

         ConstantCoefficient zero(0.0);
         double err = xi.ComputeL1Error(zero);
         if (myid == 0)
         {
            cout << std::fixed << std::setw(5) << std::setprecision(4)
                 << "ALE step [" << ale_cnt << "] at " << t << ": "
                 << std::scientific << std::setprecision(4) << endl
                 << "mass err:       " << mass_out - mass_in << endl
                 << "momentum err:   " << momentum_out - momentum_in << endl
                 << "internal_e err: " << internal_out - internal_in << endl
                 << "kinetic_e err:  " << kinetic_out - kinetic_in << endl
                 << "total_e err:    " << total_out - total_in << endl;
            cout << "interface error:     " << err << endl;
         }
      }
      else if (dt_est > 1.25 * dt) { dt *= 1.02; }

      // Make sure that the mesh corresponds to the new solution state. This is
      // needed, because some time integrators use different S-type vectors
      // and the oper object might have redirected the mesh positions to those.
      pmesh->NewNodes(x_gf, false);

      // Shifting-related procedures.
      if (calc_dist) { dist_solver.ComputeVectorDistance(coeff_xi, dist); }
#ifdef EXTRACT_1D
      //v_extr.WriteValue(t);
      //x_extr.WriteValue(t);
      if (pure_test)
      {
         e_L_extr.WriteValue(t);
         e_R_extr.WriteValue(t);
         p_L_extr.WriteValue(t);
         p_R_extr.WriteValue(t);
      }
      else
      {
         e_LS_extr.WriteValue(t);
         e_RS_extr.WriteValue(t);
         p_LS_extr.WriteValue(t);
         p_RS_extr.WriteValue(t);
      }
#endif

      if (last_step || (ti % vis_steps) == 0)
      {
         energy_old = energy_new;
         energy_new = hydro.InternalEnergy(e_gf) + hydro.KineticEnergy(v_gf);


         double lnorm = e_gf * e_gf, norm;
         MPI_Allreduce(&lnorm, &norm, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
         // const double internal_energy = hydro.InternalEnergy(e_gf);
         // const double kinetic_energy = hydro.KineticEnergy(v_gf);
         if (mpi.Root())
         {
            const double sqrt_norm = sqrt(norm);

            cout << std::fixed;
            cout << "step " << std::setw(5) << ti
                 << ",\tt = " << std::setw(5) << std::setprecision(4) << t
                 << ",\tdt = " << std::setw(5) << std::setprecision(6) << dt
                 << ",\t|e| = " << std::setprecision(10) << std::scientific
                 << sqrt_norm
                 << ",\tde = " << std::setw(5) << std::setprecision(6) << energy_new-energy_old;

            //  << ",\t|IE| = " << std::setprecision(10) << std::scientific
            //  << internal_energy
            //   << ",\t|KE| = " << std::setprecision(10) << std::scientific
            //  << kinetic_energy
            //   << ",\t|E| = " << std::setprecision(10) << std::scientific
            //  << kinetic_energy+internal_energy;
            cout << std::fixed;
            cout << endl;
         }

         // Make sure all ranks have sent their 'v' solution before initiating
         // another set of GLVis connections (one from each rank):
         MPI_Barrier(pmesh->GetComm());

         if (visualization || visit || gfprint) { hydro.ComputeDensity(rho_gf); }
         if (visualization)
         {
            int Wx = 0, Wy = 0; // window position
            int Ww = 500, Wh = 500; // window size
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
                                          "Specific Internal Energy",
                                          Wx, Wy, Ww,Wh);
            Wy += Wh + Wh/5;
            Wx = 0;
            hydrodynamics::VisualizeField(vis_p, vishost, visport,
                                          hydro.GetPressure(e_gf),
                                          "Pressure", Wx, Wy, Ww, Wh);
            Wx += offx;
            hydrodynamics::VisualizeField(vis_xi, vishost, visport,
                                          xi, "Interface", Wx, Wy, Ww, Wh);
            Wx += offx;
            hydrodynamics::VisualizeField(vis_dist, vishost, visport, dist,
                                          "Distances", Wx, Wy, Ww, Wh);
            hydrodynamics::VisualizeField(vis_mat, vishost, visport, materials,
                                          "Materials", 0, 800, Ww, Wh);
         }

         if (visit)
         {
            visit_dc.SetCycle(ti);
            visit_dc.SetTime(t);
            visit_dc.Save();
         }

         if (last_step && gfprint)
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

            std::ofstream rho_ofs(rho_name.str().c_str());
            rho_ofs.precision(8);
            rho_gf.SaveAsOne(rho_ofs);
            rho_ofs.close();

            std::ofstream v_ofs(v_name.str().c_str());
            v_ofs.precision(8);
            v_gf.SaveAsOne(v_ofs);
            v_ofs.close();

            std::ofstream e_ofs(e_name.str().c_str());
            e_ofs.precision(8);
            e_gf.SaveAsOne(e_ofs);
            e_ofs.close();

            ParaViewDataCollection dacol("ParaViewLaghos", pmesh);
            dacol.SetLevelsOfDetail(10);
            dacol.SetHighOrderOutput(true);
            dacol.RegisterField("interface", &xi);
            dacol.RegisterField("density", &rho_gf);
            dacol.RegisterField("velocity", &v_gf);
            dacol.RegisterField("materials", &materials);
            dacol.SetTime(1.0);
            dacol.SetCycle(1);
            dacol.Save();
         }
      }
   }

   switch (ode_solver_type)
   {
      case 2: steps *= 2; break;
      case 3:
      case 10: steps *= 3; break;
      case 4: steps *= 4; break;
      case 6: steps *= 6; break;
      case 7: steps *= 2;
   }

   const double energy_final = hydro.InternalEnergy(e_gf) +
                               hydro.KineticEnergy(v_gf);
   const double momentum_final = hydro.Momentum(v_gf);
   if (mpi.Root())
   {
      cout << endl;
      cout << "Energy  diff: " << std::scientific << std::setprecision(2)
           << fabs(energy_init - energy_final) << endl;
      cout << "Momentum diff: " << std::scientific << std::setprecision(2)
           << fabs(momentum_init - momentum_final) << endl;
   }

   // Print the error.
   // For problems 0 and 4 the exact velocity is constant in time.
   if (problem == 0 || problem == 4)
   {
      const double error_max = v_gf.ComputeMaxError(v_coeff),
                   error_l1  = v_gf.ComputeL1Error(v_coeff),
                   error_l2  = v_gf.ComputeL2Error(v_coeff);
      if (mpi.Root())
      {
         cout << "L_inf  error: " << error_max << endl
              << "L_1    error: " << error_l1 << endl
              << "L_2    error: " << error_l2 << endl;
      }
   }

   if (problem == 2)
   {
      riemann1D::ExactEnergyCoefficient e_coeff;
      e_coeff.SetTime(t_final);

      const double error_max = e_gf.ComputeMaxError(e_coeff),
                   error_l1  = e_gf.ComputeL1Error(e_coeff),
                   error_l2  = e_gf.ComputeL2Error(e_coeff);
      if (mpi.Root())
      {
         cout << "Tot elements: " << mesh_NE << endl;
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
      case 8: return (x(0) < 0.5) ? 1.0 : 0.125;
      case 9: return (x(0) < 0.7) ? 1000.0 : 50.;
      case 10: return (x(0) > 1.1 && x(1) > 1.5) ? 0.125 : 1.0; // initialized by another function.
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
      case 3: return (x(0) > 1.0 && x(1) <= 1.5) ? 1.4 : 1.5;
      case 4: return 5.0 / 3.0;
      case 5: return 1.4;
      case 6: return 1.4;
      case 7: return 5.0 / 3.0;
      case 8: return (x(0) < 0.5) ? 2.0 : 1.4;
      case 9: return (x(0) < 0.7) ? 4.4 : 1.4;
      case 10: return 0.0; // initialized by another function.
      default: MFEM_ABORT("Bad number given for problem id!"); return 0.0;
   }
}

static double rad(double x, double y) { return sqrt(x*x + y*y); }

void v0(const Vector &x, Vector &v)
{
   const double atn = pow((x(0)*(1.0-x(0))*4*x(1)*(1.0-x(1))*4.0),0.4);
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
      case 8: v = 0.0; break;
      case 9: v = 0.0; break;
      case 10: v = 0.0; break;
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
         MFEM_ABORT("Error in problem 5!");
         return 0.0;
      }
      case 7:
      {
         const double rho = rho0(x), gamma = gamma_func(x);
         return (6.0 - rho * x(1)) / (gamma - 1.0) / rho;
      }
      case 8: return (x(0) < 0.5) ? 2.0 / rho0(x) / (gamma_func(x) - 1.0)
                                  : 0.1 / rho0(x) / (gamma_func(x) - 1.0);
      case 9: return (x(0) < 0.7) ? (1.0e9+gamma_func(x)*6.0e8) / rho0(x) / (gamma_func(x) - 1.0)
                                  : 1.0e5 / rho0(x) / (gamma_func(x) - 1.0);
      case 10: return 0.0; // initialized by another function.
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
