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
//                            SERIAL version
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
//
// Sample runs: see README.md, section 'Verification of Results'
// All tests should be run in serial with the correct path to the mesh files.
//

#include "laghos_solver.hpp"
#include "laghos_timeinteg.hpp"
#include <fstream>

using namespace std;
using namespace mfem;
using namespace mfem::hydrodynamics;

// Choice for the problem setup.
int problem;

double rho0(const Vector &);
void v0(const Vector &, Vector &);
double e0(const Vector &);
double gamma(const Vector &);
void display_banner(ostream & os);

int main(int argc, char *argv[])
{
   // Print the banner.
   display_banner(cout);

   // Parse command-line options.
   problem = 1;
   const char *mesh_file = "../data/cube01_hex.mesh";
   int rs_levels = 2;
   int order_v = 2;
   int order_e = 1;
   int ode_solver_type = 4;
   double t_final = 0.6;
   double cfl = 0.5;
   double cg_tol = 1e-8;
   int cg_max_iter = 300;
   int max_tsteps = -1;
   bool p_assembly = true;
   bool impose_visc = false;
   bool visualization = false;
   int vis_steps = 5;
   bool visit = false;
   bool gfprint = false;
   const char *basename = "results/Laghos";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&problem, "-p", "--problem", "Problem setup to use.");
   args.AddOption(&order_v, "-ok", "--order-kinematic",
                  "Order (degree) of the kinematic finite element space.");
   args.AddOption(&order_e, "-ot", "--order-thermo",
                  "Order (degree) of the thermodynamic finite element space.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6,\n\t"
                  "            7 - RK2Avg.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&cfl, "-cfl", "--cfl", "CFL-condition number.");
   args.AddOption(&cg_tol, "-cgt", "--cg-tol",
                  "Relative CG tolerance (velocity linear solve).");
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
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // Read the serial mesh from the given mesh file on all processors.
   // Refine the mesh in serial to increase the resolution.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   const int dim = mesh->Dimension();
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }

   if (p_assembly && dim == 1)
   {
      p_assembly = false;
      cout << "Laghos does not support PA in 1D. Switching to FA." << endl;
   }

   // Define the parallel finite element spaces. We use:
   // - H1 (Gauss-Lobatto, continuous) for position and velocity.
   // - L2 (Bernstein, discontinuous) for specific internal energy.
   L2_FECollection L2FEC(order_e, dim, BasisType::Positive);
   H1_FECollection H1FEC(order_v, dim);
   FiniteElementSpace L2FESpace(mesh, &L2FEC);
   FiniteElementSpace H1FESpace(mesh, &H1FEC, mesh->Dimension());

   // Boundary conditions: all tests use v.n = 0 on the boundary, and we assume
   // that the boundaries are straight.
   Array<int> vdofs_marker, ess_vdofs;
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max()), vdofs1d;
      for (int d = 0; d < mesh->Dimension(); d++)
      {
         // Attributes 1/2/3 correspond to fixed-x/y/z boundaries, i.e., we must
         // enforce v_x/y/z = 0 for the velocity components.
         ess_bdr = 0; ess_bdr[d] = 1;
         H1FESpace.GetEssentialVDofs(ess_bdr, vdofs_marker, d);
         FiniteElementSpace::MarkerToList(vdofs_marker, vdofs1d);
         ess_vdofs.Append(vdofs1d);
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
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         delete mesh;
         return 3;
   }

   int Vsize_l2 = L2FESpace.GetVSize();
   int Vsize_h1 = H1FESpace.GetVSize();

   cout << "Number of kinematic (position, velocity) dofs: "
        << Vsize_h1 << endl;
   cout << "Number of specific internal energy dofs: "
        << Vsize_l2 << endl;

   // The monolithic BlockVector stores unknown fields as:
   // - 0 -> position
   // - 1 -> velocity
   // - 2 -> specific internal energy

   Array<int> true_offset(4);
   true_offset[0] = 0;
   true_offset[1] = true_offset[0] + Vsize_h1;
   true_offset[2] = true_offset[1] + Vsize_h1;
   true_offset[3] = true_offset[2] + Vsize_l2;
   BlockVector S(true_offset);

   // Define GridFunction objects for the position, velocity and specific
   // internal energy.  There is no function for the density, as we can always
   // compute the density values given the current mesh position, using the
   // property of pointwise mass conservation.
   GridFunction x_gf, v_gf, e_gf;
   x_gf.MakeRef(&H1FESpace, S, true_offset[0]);
   v_gf.MakeRef(&H1FESpace, S, true_offset[1]);
   e_gf.MakeRef(&L2FESpace, S, true_offset[2]);

   // Initialize x_gf using the starting mesh coordinates.
   mesh->SetNodalGridFunction(&x_gf);

   // Initialize the velocity.
   VectorFunctionCoefficient v_coeff(mesh->Dimension(), v0);
   v_gf.ProjectCoefficient(v_coeff);

   // Initialize density and specific internal energy values. We interpolate in
   // a non-positive basis to get the correct values at the dofs.  Then we do an
   // L2 projection to the positive basis in which we actually compute. The goal
   // is to get a high-order representation of the initial condition. Note that
   // this density is a temporary function and it will not be updated during the
   // time evolution.
   GridFunction rho(&L2FESpace);
   FunctionCoefficient rho_coeff(rho0);
   L2_FECollection l2_fec(order_e, mesh->Dimension());
   FiniteElementSpace l2_fes(mesh, &l2_fec);
   GridFunction l2_rho(&l2_fes), l2_e(&l2_fes);
   l2_rho.ProjectCoefficient(rho_coeff);
   rho.ProjectGridFunction(l2_rho);
   if (problem == 1)
   {
      // For the Sedov test, we use a delta function at the origin.
      DeltaCoefficient e_coeff(0, 0, 0.25);
      l2_e.ProjectCoefficient(e_coeff);
   }
   else
   {
      FunctionCoefficient e_coeff(e0);
      l2_e.ProjectCoefficient(e_coeff);
   }
   e_gf.ProjectGridFunction(l2_e);

   // Piecewise constant ideal gas coefficient over the Lagrangian mesh. The
   // gamma values are projected on a function that stays constant on the moving
   // mesh.
   L2_FECollection mat_fec(0, mesh->Dimension());
   FiniteElementSpace mat_fes(mesh, &mat_fec);
   GridFunction mat_gf(&mat_fes);
   FunctionCoefficient mat_coeff(gamma);
   mat_gf.ProjectCoefficient(mat_coeff);
   GridFunctionCoefficient *mat_gf_coeff = new GridFunctionCoefficient(&mat_gf);

   // Additional details, depending on the problem.
   int source = 0; bool visc = true;
   switch (problem)
   {
      case 0: if (mesh->Dimension() == 2) { source = 1; }
         visc = false; break;
      case 1: visc = true; break;
      case 2: visc = true; break;
      case 3: visc = true; break;
      case 4: visc = false; break;
      default: MFEM_ABORT("Wrong problem specification!");
   }
   if (impose_visc) { visc = true; }

   LagrangianHydroOperator oper(S.Size(), H1FESpace, L2FESpace,
                                ess_vdofs, rho, source, cfl, mat_gf_coeff,
                                visc, p_assembly, cg_tol, cg_max_iter,
                                H1FEC.GetBasisType());

   socketstream vis_rho, vis_v, vis_e;
   char vishost[] = "localhost";
   int  visport   = 19916;

   GridFunction rho_gf;
   if (visualization || visit) { oper.ComputeDensity(rho_gf); }

   const double energy_init = oper.InternalEnergy(e_gf) +
                              oper.KineticEnergy(v_gf);

   if (visualization)
   {
      vis_rho.precision(8);
      vis_v.precision(8);
      vis_e.precision(8);

      int Wx = 0, Wy = 0; // window position
      const int Ww = 350, Wh = 350; // window size
      int offx = Ww+10; // window offsets

      if (problem != 0 && problem != 4)
      {
         VisualizeField(vis_rho, vishost, visport, rho_gf,
                        "Density", Wx, Wy, Ww, Wh);
      }

      Wx += offx;
      VisualizeField(vis_v, vishost, visport, v_gf,
                     "Velocity", Wx, Wy, Ww, Wh);
      Wx += offx;
      VisualizeField(vis_e, vishost, visport, e_gf,
                     "Specific Internal Energy", Wx, Wy, Ww, Wh);
   }

   // Save data for VisIt visualization.
   VisItDataCollection visit_dc(basename, mesh);
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
         cout << "Repeating step " << ti << endl;
         if (steps < max_tsteps) { last_step = false; }
         ti--; continue;
      }
      else if (dt_est > 1.25 * dt) { dt *= 1.02; }

      // Make sure that the mesh corresponds to the new solution state. This is
      // needed, because some time integrators use different S-type vectors
      // and the oper object might have redirected the mesh positions to those.
      mesh->NewNodes(x_gf, false);

      if (last_step || (ti % vis_steps) == 0)
      {
         const double loc_norm = e_gf * e_gf;
         cout << fixed;
         cout << "step " << setw(5) << ti
              << ",\tt = " << setw(5) << setprecision(4) << t
              << ",\tdt = " << setw(5) << setprecision(6) << dt
              << ",\t|e| = " << setprecision(10)
              << sqrt(loc_norm) << endl;

         if (visualization || visit) { oper.ComputeDensity(rho_gf); }
         if (visualization)
         {
            int Wx = 0, Wy = 0; // window position
            int Ww = 350, Wh = 350; // window size
            int offx = Ww+10; // window offsets

            if (problem != 0 && problem != 4)
            {
               VisualizeField(vis_rho, vishost, visport, rho_gf,
                              "Density", Wx, Wy, Ww, Wh);
            }

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
            ostringstream v_name, rho_name, e_name, m_name;
            m_name << basename << "_" << ti << "_mesh";
            rho_name  << basename << "_" << ti << "_rho";
            v_name << basename << "_" << ti << "_v";
            e_name << basename << "_" << ti << "_e";

            ofstream mesh_ofs(m_name.str().c_str());
            mesh_ofs.precision(8);
            mesh->Print(mesh_ofs);
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

   switch (ode_solver_type)
   {
      case 2: steps *= 2; break;
      case 3: steps *= 3; break;
      case 4: steps *= 4; break;
      case 6: steps *= 6; break;
      case 7: steps *= 2;
   }
   oper.PrintTimingData(steps);

   const double energy_final = oper.InternalEnergy(e_gf) +
                               oper.KineticEnergy(v_gf);
   cout << endl;
   cout << "Energy  diff: " << scientific << setprecision(2)
        << fabs(energy_init - energy_final) << endl;

   // Print the error.
   // For problems 0 and 4 the exact velocity is constant in time.
   if (problem == 0 || problem == 4)
   {
      const double error_max = v_gf.ComputeMaxError(v_coeff),
                   error_l1  = v_gf.ComputeL1Error(v_coeff),
                   error_l2  = v_gf.ComputeL2Error(v_coeff);
      cout << "L_inf  error: " << error_max << endl
           << "L_1    error: " << error_l1 << endl
           << "L_2    error: " << error_l2 << endl;
   }

   if (visualization)
   {
      vis_v.close();
      vis_e.close();
   }

   // Free the used memory.
   delete ode_solver;
   delete mesh;
   delete mat_gf_coeff;

   return 0;
}

double rho0(const Vector &x)
{
   switch (problem)
   {
      case 0: return 1.0;
      case 1: return 1.0;
      case 2: return (x(0) < 0.5) ? 1.0 : 0.1;
      case 3: return (x(0) > 1.0 && x(1) <= 1.5) ? 1.0 : 0.125;
      case 4: return 1.0;
      default: MFEM_ABORT("Bad number given for problem id!"); return 0.0;
   }
}

double gamma(const Vector &x)
{
   switch (problem)
   {
      case 0: return 5./3.;
      case 1: return 1.4;
      case 2: return 1.4;
      case 3: return (x(0) > 1.0 && x(1) <= 1.5) ? 1.4 : 1.5;
      case 4: return 5.0 / 3.0;
      default: MFEM_ABORT("Bad number given for problem id!"); return 0.0;
   }
}

double rad(double x, double y)
{
   return sqrt(x*x + y*y);
}

void v0(const Vector &x, Vector &v)
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
      case 4:
      {
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
         else { v = 0.0; }
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
      case 2: return (x(0) < 0.5) ? 1.0 / rho0(x) / (gamma(x) - 1.0)
                        : 0.1 / rho0(x) / (gamma(x) - 1.0);
      case 3: return (x(0) > 1.0) ? 0.1 / rho0(x) / (gamma(x) - 1.0)
                        : 1.0 / rho0(x) / (gamma(x) - 1.0);
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
      default: MFEM_ABORT("Bad number given for problem id!"); return 0.0;
   }
}

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
