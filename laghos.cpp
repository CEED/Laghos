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
// Test problems: see README.
// mpirun -np 1 ./laghos -p 1 -dim 2 -rs 3 -tf 0.8 -s 7 -penPar 10.0 -vis -ok 1 -ot 0 -emb -tO 1 -nGT 0 -gPenCoef 1.0 -gS 1 -fP

#include "laghos_solver.hpp"

using std::cout;
using std::endl;
using namespace mfem;

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
  // penalty parameter is a user-defined non-dimensional constant for the penalty term: suggested value 10.0, but default set to 1.
  // Weak boundary imposition code is functional solely using solver option 7: i.e RK2AvgSolver
  problem = 1;
  dim = 3;
  const char *mesh_file = "default";
  int rs_levels = 2;
  int rp_levels = 0;
  int order_v = 2;
  int order_e = 1;
  int order_q = -1;
  int ode_solver_type = 4;
  double t_final = 0.6;
  double cfl = 0.5;
  double penaltyParameter = 1.0;
  double nitscheVersion = -1.0;
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
  bool useEmbedded = false;
  int geometricShape = 0;
  int nTerms = 1; 
  bool fullPenalty = false;
  int numberGhostTerms = 1;
  int numberEnergyGhostTerms = 1;
  double ghostPenaltyCoefficient = 1.0;
  double perimeter = 1.0;
  
  OptionsParser args(argc, argv);
  args.AddOption(&dim, "-dim", "--dimension", "Dimension of the problem.");
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
  args.AddOption(&penaltyParameter, "-penPar", "--penaltyParameter",
		 "Value of the penalty parameter");
  args.AddOption(&perimeter, "-per", "--perimeter",
		 "Perimeter of the bounding box of the domain");
  args.AddOption(&nitscheVersion, "-nitVer", "--nitscheVersion",
		 "-1 and 1 for skew-symmetric and symmetric versions of Nitsche");
  args.AddOption(&useEmbedded, "-emb", "--use-embedded", "-no-emb",
		 "--no-embedded",
		 "Use Embedded when there is surface that will be embedded in a pre-existing mesh");
  args.AddOption(&geometricShape, "-gS", "--geometricShape",
		 "Shape of the embedded geometry that will be embedded");
  args.AddOption(&nTerms, "-tO", "--taylorOrder",
		 "Number of terms in the Taylor expansion");
  args.AddOption(&fullPenalty, "-fP", "--full-Penalty", "-first-order-penalty",
		 "--first-order-penalty",
		 "Use full or first order for SBM penalty.");
  args.AddOption(&numberGhostTerms, "-nGT", "--numberGhostTerms",
                 "Number of terms in the  ghost penalty operator.");
  args.AddOption(&numberEnergyGhostTerms, "-nEGT", "--numberEnergyGhostTerms",
		  "Number of terms in the  energy equation ghost penalty operator.");
 
  args.AddOption(&ghostPenaltyCoefficient, "-gPenCoef", "--ghost-penalty-coefficient", "Ghost penalty scaling.");
  
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
  
  if (mesh->NURBSext){
    mesh->SetCurvature(order_v);
  }
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
  pmesh->ExchangeFaceNbrNodes();
    
  int NE = pmesh->GetNE(), ne_min, ne_max;
  MPI_Reduce(&NE, &ne_min, 1, MPI_INT, MPI_MIN, 0, pmesh->GetComm());
  MPI_Reduce(&NE, &ne_max, 1, MPI_INT, MPI_MAX, 0, pmesh->GetComm());
  if (myid == 0)
    { cout << "Zones min/max: " << ne_min << " " << ne_max << endl; }

  // Define the parallel finite element spaces. We use:
  // - H1 (Gauss-Lobatto, continuous) for position and velocity.
  // - L2 (Bernstein, discontinuous) for specific internal energy.
  // - L2 (Gauss-Legendre, discontinuous) for pressure.
  L2_FECollection L2FEC(order_e, dim, BasisType::GaussLobatto);
  // L2_FECollection L2FEC(order_e, dim, BasisType::Positive);
  H1_FECollection H1FEC(order_v, dim);
  ParFiniteElementSpace L2FESpace(pmesh, &L2FEC);
  ParFiniteElementSpace H1FESpace(pmesh, &H1FEC, pmesh->Dimension());
  // Quad rule for interior terms. Define the pressure ParGridFunction with the same rule. 
  int quadRule = (order_q > 0) ? order_q : 3 * H1FESpace.GetOrder(0) + L2FESpace.GetOrder(0) - 1;
  L2_FECollection P_L2FEC((int)(quadRule*0.5), dim, BasisType::GaussLegendre);
  //  L2_FECollection P_L2FEC(order_e, dim, BasisType::GaussLobatto);

  ParFiniteElementSpace P_L2FESpace(pmesh, &P_L2FEC);

  // Quad rule for interior terms. Define the pressure ParGridFunction with the same rule. 
  int quadRule_face =  H1FESpace.GetOrder(0) + L2FESpace.GetOrder(0) + (pmesh->GetBdrFaceTransformations(0))->OrderW();
  L2_FECollection PFace_L2FEC((int)(0.5*quadRule), dim, BasisType::GaussLobatto);
  // L2_FECollection PFace_L2FEC(order_e, dim, BasisType::GaussLobatto);

  ParFiniteElementSpace PFace_L2FESpace(pmesh, &PFace_L2FEC);
  ParFiniteElementSpace PFaceVector_L2FESpace(pmesh, &PFace_L2FEC, pmesh->Dimension()*pmesh->Dimension() );
  ParFiniteElementSpace PVector_L2FESpace(pmesh, &P_L2FEC, pmesh->Dimension()*pmesh->Dimension() );

  if (!(mesh->NURBSext))
   {
      pmesh->SetNodalFESpace(&H1FESpace);
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
  const int Vsize_l2 = L2FESpace.GetVSize();
  const int Vsize_h1 = H1FESpace.GetVSize();
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
  x_gf.ExchangeFaceNbrData();
  v_gf.ExchangeFaceNbrData();
  e_gf.ExchangeFaceNbrData();
  
  // Initialize the velocity.
  VectorFunctionCoefficient v_coeff(pmesh->Dimension(), v0);
  v_gf.ProjectCoefficient(v_coeff);

  v_gf.ExchangeFaceNbrData();
  // Initialize density and specific internal energy values. We interpolate in
  // a non-positive basis to get the correct values at the dofs. Then we do an
  // L2 projection to the positive basis in which we actually compute. The goal
  // is to get a high-order representation of the initial condition. Note that
  // this density is a temporary function and it will not be updated during the
  // time evolution.
  ParGridFunction rho0_gf(&L2FESpace);
  // Grid Functions for interior terms
  ParGridFunction p_gf(&P_L2FESpace);
  ParGridFunction cs_gf(&P_L2FESpace);
  ParGridFunction rho_gf(&P_L2FESpace);
  ParGridFunction rho0DetJ0_gf(&P_L2FESpace);
  ParGridFunction Jac0inv_gf(&PVector_L2FESpace);
  
  p_gf = 0.0;
  cs_gf = 0.0;
  rho_gf = 0.0;
  rho0DetJ0_gf = 0.0;
  Jac0inv_gf = 0.0;
  
  // Grid Functions for face terms
  ParGridFunction pface_gf(&PFace_L2FESpace);
  ParGridFunction csface_gf(&PFace_L2FESpace);
  ParGridFunction rhoface_gf(&PFace_L2FESpace);
  ParGridFunction viscousface_gf(&PFace_L2FESpace);
  ParGridFunction rho0DetJ0face_gf(&PFace_L2FESpace);
  ParGridFunction Jac0invface_gf(&PFaceVector_L2FESpace);
  
  pface_gf = 0.0;
  csface_gf = 0.0;
  rhoface_gf = 0.0;
  viscousface_gf = 0.0;
  rho0DetJ0face_gf = 0.0;
  Jac0invface_gf = 0.0;
  
  double globalmax_cs = 0.0;
  double globalmax_rho = 0.0;
  double globalmax_viscous_coef = 0.0;
 
  FunctionCoefficient rho0_coeff(rho0);
  hydrodynamics::Jac0InvVectorFunctionCoefficient Jac0inv_coeff(pmesh->Dimension(), pmesh->Dimension()*pmesh->Dimension());

  L2_FECollection l2_fec(order_e, pmesh->Dimension());
  ParFiniteElementSpace l2_fes(pmesh, &l2_fec);
  ParGridFunction l2_rho0_gf(&l2_fes), l2_e(&l2_fes);
  l2_rho0_gf.ProjectCoefficient(rho0_coeff);
  rho0_gf.ProjectGridFunction(l2_rho0_gf);
  rho0_gf.ExchangeFaceNbrData();
  
  rho_gf = rho0_gf;
  rho_gf.ExchangeFaceNbrData();
  
  Jac0invface_gf.ProjectCoefficient(Jac0inv_coeff);
  Jac0inv_gf.ProjectCoefficient(Jac0inv_coeff);
  
  Jac0inv_gf.ExchangeFaceNbrData();
  Jac0invface_gf.ExchangeFaceNbrData();
  
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

  hydrodynamics::LagrangianHydroOperator hydro(S.Size(),order_e, order_v, globalmax_rho, globalmax_cs, globalmax_viscous_coef,
					       H1FESpace, L2FESpace, P_L2FESpace, PFace_L2FESpace,
					       rho0_coeff, rho0_gf, rho_gf, rhoface_gf,
					       mat_gf, p_gf, pface_gf, v_gf, e_gf, cs_gf, csface_gf, viscousface_gf, rho0DetJ0_gf, rho0DetJ0face_gf, Jac0inv_gf, Jac0invface_gf, source, cfl, numberGhostTerms, numberEnergyGhostTerms, ghostPenaltyCoefficient,
					       visc, vorticity,
					       cg_tol, cg_max_iter, ftz_tol,
					       order_q, penaltyParameter, perimeter, nitscheVersion, useEmbedded, geometricShape, nTerms, fullPenalty);

  socketstream vis_rho, vis_v, vis_e;
  char vishost[] = "localhost";
  int  visport   = 19916;

  ParGridFunction rho_gf_vis;
  if (visualization || visit) { hydro.ComputeDensity(rho_gf_vis); }
  const double energy_init = hydro.InternalEnergy(e_gf) +
    hydro.KineticEnergy(v_gf);

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
	  hydrodynamics::VisualizeField(vis_rho, vishost, visport, rho_gf_vis,
					"Density", Wx, Wy, Ww, Wh);
	}
      Wx += offx;
      hydrodynamics::VisualizeField(vis_v, vishost, visport, v_gf,
                                    "Velocity", Wx, Wy, Ww, Wh);
      Wx += offx;
      hydrodynamics::VisualizeField(vis_e, vishost, visport, e_gf,
                                    "Specific Internal Energy", Wx, Wy, Ww, Wh);
    }

  // Save data for VisIt visualization.
  VisItDataCollection visit_dc(basename, pmesh);
  if (visit)
    {
      visit_dc.RegisterField("Density",  &rho_gf_vis);
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
	  if (mpi.Root()) { cout << "Repeating step " << ti << endl; }
	  if (steps < max_tsteps) { last_step = false; }
	  ti--; continue;
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
		   << sqrt_norm;
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

	  if (visualization || visit || gfprint) { hydro.ComputeDensity(rho_gf_vis); }
	  if (visualization)
	    {
	      int Wx = 0, Wy = 0; // window position
	      int Ww = 350, Wh = 350; // window size
	      int offx = Ww+10; // window offsets
	      if (problem != 0 && problem != 4)
		{
		  hydrodynamics::VisualizeField(vis_rho, vishost, visport, rho_gf_vis,
						"Density", Wx, Wy, Ww, Wh);
		}
	      Wx += offx;
	      hydrodynamics::VisualizeField(vis_v, vishost, visport,
					    v_gf, "Velocity", Wx, Wy, Ww, Wh);
	      Wx += offx;
	      hydrodynamics::VisualizeField(vis_e, vishost, visport, e_gf,
					    "Specific Internal Energy",
					    Wx, Wy, Ww,Wh);
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
	      rho_gf_vis.SaveAsOne(rho_ofs);
	      rho_ofs.close();

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

  switch (ode_solver_type)
    {
    case 2: steps *= 2; break;
    case 3: steps *= 3; break;
    case 4: steps *= 4; break;
    case 6: steps *= 6; break;
    case 7: steps *= 2;
    }

  const double energy_final = hydro.InternalEnergy(e_gf) +
    hydro.KineticEnergy(v_gf);
  if (mpi.Root())
    {
      cout << endl;
      cout << "Energy  diff: " << std::scientific << std::setprecision(2)
           << fabs(energy_init - energy_final) << endl;
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
    case 3: return (x(0) > 1.0 && x(1) <= 1.5) ? 1.4 : 1.5;
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
