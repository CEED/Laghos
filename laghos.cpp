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
// ALE runs with latest remap:
// 3point 3 materials.
// mpirun -np 4 ./laghos -p 3 -m data/rectangle01_quad.mesh -rs 2 -tf 3.5 -pa
// Sedov 1 material.
// mpirun -np 4 ./laghos -p 1 -dim 2 -rs 3 -tf 0.8 -pa

#include <fstream>
#include <sys/time.h>
#include <sys/resource.h>
#include "laghos_solver.hpp"
#include "laghos_tmop.hpp"
#include "../remhos_remap/remhos_gslib.hpp"
#include "../remhos_remap/remhos_sync.hpp"

using std::cout;
using std::endl;
using namespace mfem;

// Choice for the problem setup.
static int problem;

// Forward declarations.
double e0(const Vector &);
double rho0(const Vector &);
double gamma_func(const Vector &);
void v0(const Vector &, Vector &);
int attr_id(const Vector &);

static long GetMaxRssMB();
static void display_banner(std::ostream&);
static void Checks(const int dim, const int ti, const double norm, int &checks);

int mat_id(const Vector &x)
{
   switch (problem)
   {
   case 0: return (x(0) < 0.5) ? 0 : 1;
   case 1: return 0;
   case 3:
   {
      if (x(0) < 1.0) { return 0; }
      return (x(1) > 1.5) ? 1 : 2;
   }
   case 4: return (x(0) < 0.25) ? 0 : 1;
   case 7: return (x(1) < 0.0) ? 0 : 1;
   default: MFEM_ABORT("Materials not setup for problem id!"); return -1;
   }
}

class IndicatorCoefficient : public Coefficient
{
protected:
   int ind_id;

public:
   IndicatorCoefficient(int id) : ind_id(id) { }

   virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      Vector coord(T.GetDimension());
      T.Transform(ip, coord);
      return (mat_id(coord) == ind_id) ? 1.0 : 0.0;
   }
};

double interface_rt(const Vector &x)
{
   // 0 - Taylor-Green.
   // 1 - Rayleigh-Taylor 2D and 3D.
   // 2 - 3point 2D.
   // 3 - 3point 2D, left part.
   // 4 - 3point 2D, right part.
   const int type = 1;

   // Taylor-Green.
   if (type == 0) { return tanh(2.0*(x(1) - 0.5)); }

   // Rayleigh-Taylor.
   if (type == 1) { return tanh(1.0*(x(1) - 0.0)); }

   // 3point.
   if (type == 2)
   {
      return tanh(0.1*(x(1) - 1.5));
   }

   // 3point left part
   if (type == 3)
   {
      if (x(0) <= 1.0)
      {
         return fabs(x(1) - 1.5);
      }
      else
      {
         double a = x(0) - 1.0;
         double b = x(1) - 1.5;
         return sqrt(a*a + b*b);
      }
   }

   // 3point right part
   if (type == 4)
   {
      if (x(0) >= 1.0)
      {
         return fabs(x(1) - 1.5);
      }
      else
      {
         double a = x(0) - 1.0;
         double b = x(1) - 1.5;
         return sqrt(a*a + b*b);
      }
   }
}

void SetInterfaces(const ParGridFunction &x, ParGridFunction &interfaces)
{
   // 0 - triple point; 1 - rayleigh-taylor.
   const int interface_type = 0;
   interfaces = 0.0;

   const double eps = 1.0e-8;
   ParFiniteElementSpace *x_pfes = x.ParFESpace(),
                         *i_pfes = interfaces.ParFESpace();
   ParMesh *pmesh = x_pfes->GetParMesh();
   Array<int> vdofs, vdofs_i;
   if (interface_type == 0)
   {
      for (int f = 0; f < pmesh->GetNumFaces(); f++)
      {
         x_pfes->GetFaceVDofs(f, vdofs);
         i_pfes->GetFaceVDofs(f, vdofs_i);

         const int size_i = vdofs_i.Size() / 2;
         const int size_x = vdofs.Size() / 2;

         // A little mismatch, but whatever.
         for (int i = 0; i < size_i; i++)
         {
            if (fabs(x(vdofs[i]) - 1.0) < eps)
            {
               interfaces(vdofs_i[i]) = 1.0;
            }
            if (fabs(x(vdofs[i+size_x]) - 1.5) < eps && x(vdofs[i]) >= 1.0)
            {
               interfaces(vdofs_i[i]) = 1.0;
            }
         }
      }
   }
   else
   {
      FunctionCoefficient coeff(interface_rt);
      interfaces.ProjectCoefficient(coeff);
   }
}

// Assumes size and spaces are already set.
void InitIndicators(std::vector<ParGridFunction> &ind)
{
   for (int k = 0; k < ind.size(); k++)
   {
      IndicatorCoefficient ic(k);
      ind[k].ProjectCoefficient(ic);
   }
}

void InitIndicators(std::vector<QuadratureFunction> &ind)
{
   auto qspace = dynamic_cast<QuadratureSpace *>(ind[0].GetSpace());
   const int NE  = qspace->GetMesh()->GetNE(), ind_cnt = ind.size();

   for (int e = 0; e < NE; e++)
   {
      const IntegrationRule &ir = qspace->GetElementIntRule(e);
      const int nip = ir.GetNPoints();

      // Transformation of the element with the pos_mesh coordinates.
      IsoparametricTransformation Tr;
      qspace->GetMesh()->GetElementTransformation(e, &Tr);

      Vector coord(Tr.GetSpaceDim());
      for (int q = 0; q < nip; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.Transform(ip, coord);
         for (int k = 0; k < ind_cnt; k++)
         {
            ind[k](e*nip + q) = (mat_id(coord) == k) ? 1.0 : 0.0;
         }
      }
   }
}


void PrintNewInterface(ParGridFunction &interfaces)
{
   ParMesh *pmesh = interfaces.ParFESpace()->GetParMesh();

   std::ostringstream mesh_name, interface_name;
   mesh_name << "interface.mesh";
   interface_name  << "interface.gf";

   std::ofstream mesh_ofs(mesh_name.str().c_str());
   mesh_ofs.precision(8);
   pmesh->PrintAsOne(mesh_ofs);
   mesh_ofs.close();

   std::ofstream interface_ofs(interface_name.str().c_str());
   interface_ofs.precision(8);
   interfaces.SaveAsOne(interface_ofs);
   interface_ofs.close();
}

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
   int dim = 3;
   const char *mesh_file = "default";
   int rs_levels = 2;
   int rp_levels = 0;
   Array<int> cxyz;
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
   bool p_assembly = true;
   bool impose_visc = false;
   bool visualization = false;
   int vis_steps = 5;
   bool visit = false;
   bool gfprint = false;
   const char *basename = "results/Laghos";
   int partition_type = 0;
   const char *device = "cpu";
   bool check = false;
   bool mem_usage = false;
   bool fom = false;
   bool gpu_aware_mpi = false;
   int dev = 0;
   double blast_energy = 0.25;
   double blast_position[] = {0.0, 0.0, 0.0};

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
   args.AddOption(&partition_type, "-pt", "--partition",
                  "Customized x/y/z Cartesian MPI partitioning of the serial mesh.\n\t"
                  "Here x,y,z are relative task ratios in each direction.\n\t"
                  "Example: with 48 mpi tasks and -pt 321, one would get a Cartesian\n\t"
                  "partition of the serial mesh by (6,4,2) MPI tasks in (x,y,z).\n\t"
                  "NOTE: the serially refined mesh must have the appropriate number\n\t"
                  "of zones in each direction, e.g., the number of zones in direction x\n\t"
                  "must be divisible by the number of MPI tasks in direction x.\n\t"
                  "Available options: 11, 21, 111, 211, 221, 311, 321, 322, 432.");
   args.AddOption(&device, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&check, "-chk", "--checks", "-no-chk", "--no-checks",
                  "Enable 2D checks.");
   args.AddOption(&mem_usage, "-mb", "--mem", "-no-mem", "--no-mem",
                  "Enable memory usage.");
   args.AddOption(&fom, "-f", "--fom", "-no-fom", "--no-fom",
                  "Enable figure of merit output.");
   args.AddOption(&gpu_aware_mpi, "-gam", "--gpu-aware-mpi", "-no-gam",
                  "--no-gpu-aware-mpi", "Enable GPU aware MPI communications.");
   args.AddOption(&dev, "-dev", "--dev", "GPU device to use.");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(cout); }
      return 1;
   }
   if (Mpi::Root()) { args.PrintOptions(cout); }

   // Configure the device from the command line options
   Device backend;
   backend.Configure(device, dev);
   if (Mpi::Root()) { backend.Print(); }
   backend.SetGPUAwareMPI(gpu_aware_mpi);

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
   ParMesh *pmesh = nullptr;
   const int num_tasks = Mpi::WorldSize(); int unit = 1;
   int *nxyz = new int[dim];
   switch (partition_type)
   {
      case 0:
         for (int d = 0; d < dim; d++) { nxyz[d] = unit; }
         break;
      case 11:
      case 111:
         unit = static_cast<int>(floor(pow(num_tasks, 1.0 / dim) + 1e-2));
         for (int d = 0; d < dim; d++) { nxyz[d] = unit; }
         break;
      case 21: // 2D
         unit = static_cast<int>(floor(pow(num_tasks / 2, 1.0 / 2) + 1e-2));
         nxyz[0] = 2 * unit; nxyz[1] = unit;
         break;
      case 31: // 2D
         unit = static_cast<int>(floor(pow(num_tasks / 3, 1.0 / 2) + 1e-2));
         nxyz[0] = 3 * unit; nxyz[1] = unit;
         break;
      case 32: // 2D
         unit = static_cast<int>(floor(pow(2 * num_tasks / 3, 1.0 / 2) + 1e-2));
         nxyz[0] = 3 * unit / 2; nxyz[1] = unit;
         break;
      case 49: // 2D
         unit = static_cast<int>(floor(pow(9 * num_tasks / 4, 1.0 / 2) + 1e-2));
         nxyz[0] = 4 * unit / 9; nxyz[1] = unit;
         break;
      case 51: // 2D
         unit = static_cast<int>(floor(pow(num_tasks / 5, 1.0 / 2) + 1e-2));
         nxyz[0] = 5 * unit; nxyz[1] = unit;
         break;
      case 211: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 2, 1.0 / 3) + 1e-2));
         nxyz[0] = 2 * unit; nxyz[1] = unit; nxyz[2] = unit;
         break;
      case 221: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 4, 1.0 / 3) + 1e-2));
         nxyz[0] = 2 * unit; nxyz[1] = 2 * unit; nxyz[2] = unit;
         break;
      case 311: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 3, 1.0 / 3) + 1e-2));
         nxyz[0] = 3 * unit; nxyz[1] = unit; nxyz[2] = unit;
         break;
      case 321: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 6, 1.0 / 3) + 1e-2));
         nxyz[0] = 3 * unit; nxyz[1] = 2 * unit; nxyz[2] = unit;
         break;
      case 322: // 3D.
         unit = static_cast<int>(floor(pow(2 * num_tasks / 3, 1.0 / 3) + 1e-2));
         nxyz[0] = 3 * unit / 2; nxyz[1] = unit; nxyz[2] = unit;
         break;
      case 432: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 3, 1.0 / 3) + 1e-2));
         nxyz[0] = 2 * unit; nxyz[1] = 3 * unit / 2; nxyz[2] = unit;
         break;
      case 511: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 5, 1.0 / 3) + 1e-2));
         nxyz[0] = 5 * unit; nxyz[1] = unit; nxyz[2] = unit;
         break;
      case 521: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 10, 1.0 / 3) + 1e-2));
         nxyz[0] = 5 * unit; nxyz[1] = 2 * unit; nxyz[2] = unit;
         break;
      case 522: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 20, 1.0 / 3) + 1e-2));
         nxyz[0] = 5 * unit; nxyz[1] = 2 * unit; nxyz[2] = 2 * unit;
         break;
      case 911: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 9, 1.0 / 3) + 1e-2));
         nxyz[0] = 9 * unit; nxyz[1] = unit; nxyz[2] = unit;
         break;
      case 921: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 18, 1.0 / 3) + 1e-2));
         nxyz[0] = 9 * unit; nxyz[1] = 2 * unit; nxyz[2] = unit;
         break;
      case 922: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 36, 1.0 / 3) + 1e-2));
         nxyz[0] = 9 * unit; nxyz[1] = 2 * unit; nxyz[2] = 2 * unit;
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
   const bool cartesian_partitioning = (cxyz.Size()>0)?true:false;
   if (product == num_tasks || cartesian_partitioning)
   {
      if (cartesian_partitioning)
      {
         int cproduct = 1;
         for (int d = 0; d < dim; d++) { cproduct *= cxyz[d]; }
         MFEM_VERIFY(!cartesian_partitioning || cxyz.Size() == dim,
                     "Expected " << mesh->SpaceDimension() << " integers with the "
                     "option --cartesian-partitioning.");
         MFEM_VERIFY(!cartesian_partitioning || num_tasks == cproduct,
                     "Expected cartesian partitioning product to match number of ranks.");
      }
      int *partitioning = cartesian_partitioning ?
                          mesh->CartesianPartitioning(cxyz):
                          mesh->CartesianPartitioning(nxyz);
      pmesh = new ParMesh(MPI_COMM_WORLD, *mesh, partitioning);
      delete [] partitioning;
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

   for (int i = 0; i < pmesh->GetNE(); i++)
   {
      Vector center;
      pmesh->GetElementCenter(i, center);
      int attrib = attr_id(center);
      pmesh->SetAttribute(i,attrib);
   }
   pmesh->SetAttributes();

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
   Array<int> true_offset(4);
   true_offset[0] = 0;
   true_offset[1] = true_offset[0] + Vsize_h1;
   true_offset[2] = true_offset[1] + Vsize_h1;
   true_offset[3] = true_offset[2] + Vsize_l2;
   BlockVector S(true_offset, Device::GetMemoryType());

   // Additional details, depending on the problem.
   int source = 0; bool visc = true, vorticity = false;
   int ind_cnt = 1;
   switch (problem)
   {
   case 0: if (pmesh->Dimension() == 2) { source = 1; }
           visc = false; ind_cnt = 2; break;
   case 1: visc = true; break;
   case 2: visc = true; break;
   case 3: visc = true; ind_cnt = 3; S.HostRead(); break;
   case 4: visc = false; ind_cnt = 2; break;
   case 5: visc = true; break;
   case 6: visc = true; break;
   case 7: source = 2; visc = true; vorticity = true; ind_cnt = 2; break;
   default: MFEM_ABORT("Wrong problem specification!");
   }
   if (impose_visc) { visc = true; }

   // Integration rule for all hydro integrals.
   auto ir = IntRules.Get(pmesh->GetElementBaseGeometry(0),
                          order_q > 0 ? order_q : 3 * order_v + order_e - 1);

   // Define GridFunction objects for the position, velocity and specific
   // internal energy. There is no function for the density, as we can always
   // compute the density values given the current mesh position, using the
   // property of pointwise mass conservation.
   ParGridFunction x_gf, v_gf, e_gf;
   x_gf.MakeRef(&H1FESpace, S, true_offset[0]);
   v_gf.MakeRef(&H1FESpace, S, true_offset[1]);
   e_gf.MakeRef(&L2FESpace, S, true_offset[2]);

   // Initialize x_gf using the starting mesh coordinates.
   pmesh->SetNodalGridFunction(&x_gf);
   // Sync the data location of x_gf with its base, S
   x_gf.SyncAliasMemory(S);

   // Quadrature-based data.
   QuadratureSpace qspace(*pmesh, ir);
   hydrodynamics::QuadratureData qdata(dim, NE, ir.GetNPoints(), ind_cnt);

   // Initialize material indicator functions.
   for (int k = 0; k < ind_cnt; k++) { qdata.ind[k].SetSpace(&qspace, 1); }
   L2_FECollection ind_fec(1, pmesh->Dimension(), BasisType::GaussLegendre);
   ParFiniteElementSpace ind_fes(pmesh, &ind_fec);
   std::vector<ParGridFunction> ind_L2(ind_cnt);
   for (int k = 0; k < ind_cnt; k++) { ind_L2[k].SetSpace(&ind_fes); }
   InitIndicators(ind_L2);
   InitIndicators(qdata.ind);

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
   // Sync the data location of e_gf with its base, S
   e_gf.SyncAliasMemory(S);

   // Piecewise constant ideal gas coefficient over the Lagrangian mesh. The
   // gamma values are projected on function that's constant on the moving mesh.
   L2_FECollection mat_fec(0, pmesh->Dimension());
   ParFiniteElementSpace mat_fes(pmesh, &mat_fec);
   ParGridFunction mat_gf(&mat_fes);
   FunctionCoefficient mat_coeff(gamma_func);
   mat_gf.ProjectCoefficient(mat_coeff);

   H1_FECollection interf_fec(order_v, pmesh->Dimension());
   ParFiniteElementSpace interf_fes(pmesh, &interf_fec);
   ParGridFunction interfaces(&interf_fes);
   SetInterfaces(x_gf, interfaces);

   hydrodynamics::LagrangianHydroOperator hydro(S.Size(),
                                                H1FESpace, L2FESpace, ess_tdofs,
                                                rho0_coeff, rho0_gf,
                                                mat_gf, qdata, source, cfl,
                                                visc, vorticity, p_assembly,
                                                cg_tol, cg_max_iter, ftz_tol,
                                                ir);

   socketstream vis_rho, vis_v, vis_e;
   char vishost[] = "localhost";
   int  visport   = 19916;

   ParGridFunction rho_gf;
   if (visualization || visit) { hydro.ComputeDensity(rho_gf); }
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
      else if (dt_est > 1.25 * dt) { dt *= 1.02; }

      // Ensure the sub-vectors x_gf, v_gf, and e_gf know the location of the
      // data in S. This operation simply updates the Memory validity flags of
      // the sub-vectors to match those of S.
      x_gf.SyncAliasMemory(S);
      v_gf.SyncAliasMemory(S);
      e_gf.SyncAliasMemory(S);

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

         if (visualization || visit || gfprint) { hydro.ComputeDensity(rho_gf); }
         if (visualization)
         {
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
         }
      }

      // Problems checks
      if (check)
      {
         double lnorm = e_gf * e_gf, norm;
         MPI_Allreduce(&lnorm, &norm, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
         const double e_norm = sqrt(norm);
         MFEM_VERIFY(rs_levels==0 && rp_levels==0, "check: rs, rp");
         MFEM_VERIFY(order_v==2, "check: order_v");
         MFEM_VERIFY(order_e==1, "check: order_e");
         MFEM_VERIFY(ode_solver_type==4, "check: ode_solver_type");
         MFEM_VERIFY(t_final == 0.6, "check: t_final");
         MFEM_VERIFY(cfl==0.5, "check: cfl");
         MFEM_VERIFY(strncmp(mesh_file, "default", 7) == 0, "check: mesh_file");
         MFEM_VERIFY(dim==2 || dim==3, "check: dimension");
         Checks(dim, ti, e_norm, checks);
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

   PrintNewInterface(interfaces);
   hydro.PrintTimingData(Mpi::Root(), steps, fom);

   if (mem_usage)
   {
      mem = GetMaxRssMB();
      MPI_Reduce(&mem, &mmax, 1, MPI_LONG, MPI_MAX, 0, pmesh->GetComm());
      MPI_Reduce(&mem, &msum, 1, MPI_LONG, MPI_SUM, 0, pmesh->GetComm());
   }

   const double energy_final = hydro.InternalEnergy(e_gf) +
                               hydro.KineticEnergy(v_gf);
   if (Mpi::Root())
   {
      cout << endl;
      cout << "Energy  diff: " << std::scientific << std::setprecision(2)
           << fabs(energy_init - energy_final) << endl;
      if (mem_usage)
      {
         cout << "Maximum memory resident set size: "
              << mmax << "/" << msum << " MB" << endl;
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

   // Project indicators to H1+P1.
   H1_FECollection ind_fec_H1(1, pmesh->Dimension());
   ParFiniteElementSpace ind_fes_H1(pmesh, &ind_fec_H1);
   std::vector<ParGridFunction> ind_H1(ind_cnt);
   for (int k = 0; k < ind_cnt; k++)
   {
      ind_H1[k].SetSpace(&ind_fes_H1);
      GridFunctionCoefficient ic(&ind_L2[k]);
      ind_H1[k].ProjectDiscCoefficient(ic, GridFunction::ARITHMETIC);
   }

   for (int k = 0; k < ind_cnt; k++)
   {
      VisQuadratureFunction(*pmesh, qdata.ind[k], "ind QF", 0 + k*400, 400);
   }

   //
   // Prepare for ALE.
   //
   // Setup the BlockVector (ordered ind-rho-e-v).
   const int size_qf   = qspace.GetSize(),
             size_gf_e = L2FESpace.GetVSize(),
             size_gf_v = H1FESpace.GetVSize();
   Array<int> offset(5);
   offset[0] = 0;
   offset[1] = offset[0] + size_qf;
   offset[2] = offset[1] + size_qf;
   offset[3] = offset[2] + size_gf_e;
   offset[4] = offset[3] + size_gf_v;
   BlockVector ind_rho_e_v_0(offset, Device::GetMemoryType());
   BlockVector ind_rho_e(offset);
   // Move indicators and densities to QuadratureFunctions.
   QuadratureFunction ind_0(&qspace, ind_rho_e_v_0.GetBlock(0).GetData()),
                      rho_0(&qspace, ind_rho_e_v_0.GetBlock(1).GetData());
   const int index = (problem == 3) ? 2 : 0;
   ParGridFunction ind_remap(ind_L2[index]);
   GridFunctionCoefficient ind_c(&ind_remap);
   InitializeQuadratureFunction(ind_c, x_gf, ind_0);
   hydro.ComputeDensity(rho_0);
   // Copy energy and velocity.
   ParGridFunction e_0(&L2FESpace, ind_rho_e_v_0.GetBlock(2).GetData());
   e_0 = e_gf;
   ParGridFunction v_0(&H1FESpace, ind_rho_e_v_0.GetBlock(3).GetData());
   v_0 = v_gf;
   Array<bool> ind_0_bool_el, ind_0_bool_dofs;
   ComputeBoolIndicators(NE, ind_0, ind_0_bool_el, ind_0_bool_dofs);
   const int nqp = ir.GetNPoints(), ndof = e_0.Size() / NE;
   for (int e = 0; e < NE; e++)
   {
      if (ind_0_bool_el[e] == false)
      {
         for (int q = 0; q < nqp; q++)
         {
            rho_0(e * nqp + q) = 0.0;
         }
         for (int i = 0; i < ndof; i++)
         {
            e_0(e*ndof + i) = 0.0;
         }
      }
   }

   InterpolationRemap interpolator(*pmesh);
   // Visualize starting state.
   VisQuadratureFunction(*pmesh, ind_0, "ind_0 QF", 0, 400);
   VisQuadratureFunction(*pmesh, rho_0, "rho_0 QF", 400, 400);
   {
      socketstream sock_e;
      hydrodynamics::VisualizeField(sock_e, "localhost", 19916, e_0, "e_0 GF",
                                    800, 400, 400, 400);

      socketstream sock_v;
      hydrodynamics::VisualizeField(sock_v, "localhost", 19916, v_0, "v_0 GF",
                                    1200, 400, 400, 400, true, "m");
   }
   QuadratureFunction p_0(&qspace);
   hydro.ComputePressure(rho_0, e_0, mat_gf(0), p_0);
   VisQuadratureFunction(*pmesh, p_0, "p_0 QF", 1600, 400);

   //
   // Remesh.
   //
   Vector x0(x_gf.Size());
   x0 = x_gf;
   ParGridFunction x_opt(x_gf);
   hydrodynamics::OptimizeMesh(x_opt, ess_vdofs, interfaces, ind_H1);

   //
   // Remap.
   //
   {
      interpolator.visualization = false;
      interpolator.h1_seminorm   = false;
      interpolator.max_iter      = 100;
      interpolator.subprob       = true;
      interpolator.weightedSpace = hiop::hiopInterfaceBase::WeightedSpaceType::Euclidean;
      interpolator.SetQuadratureSpace(qspace);
      interpolator.SetEnergyFESpace(L2FESpace);
      interpolator.SetVelocityFESpace(H1FESpace);

      BlockVector ind_rho_e(offset);
      const int optimization_type = 5;
      interpolator.RemapHydro(ind_rho_e_v_0, true, ind_0_bool_el, x_opt,
                              ind_rho_e, optimization_type);

      x_gf = x_opt;

      QuadratureFunction ind(&qspace, ind_rho_e.GetBlock(0).GetData()),
                         rho(&qspace, ind_rho_e.GetBlock(1).GetData());
      ParGridFunction e(&L2FESpace, ind_rho_e.GetBlock(2).GetData());
      ParGridFunction v(&H1FESpace, ind_rho_e.GetBlock(3).GetData());

      // Visualize starting state.
      VisQuadratureFunction(*pmesh, ind, "ind QF", 0, 800);
      VisQuadratureFunction(*pmesh, rho, "rho QF", 400, 800);
      {
         socketstream sock_e;
         hydrodynamics::VisualizeField(sock_e, "localhost", 19916, e, "e GF",
                                       800, 800, 400, 400);

         socketstream sock_v;
         hydrodynamics::VisualizeField(sock_v, "localhost", 19916, v, "v GF",
                                       1200, 800, 400, 400, true, "m");
      }
      QuadratureFunction p(&qspace);
      hydro.ComputePressure(rho, e, mat_gf(0), p);
      VisQuadratureFunction(*pmesh, p, "p QF", 1600, 800);
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

int attr_id(const Vector &x)
{
   if (problem == 1 || problem == 3)
   {
      return x(0) <= 1.0 ? 1 : (x(1) >= 1.5 ? 2 : 3);
   }
   else if (problem == 4)
   {
      return x(0) <= 0.25 ? 1 : 2;
   }
   else if (problem == 7)
   {
      return x(1) >= 0.0 ? 1 : 2;
   }
   else
   {
      MFEM_ABORT("attr_id: problem not supported!");
   }
}

double rho0(const Vector &x)
{
   switch (problem)
   {
      case 0: return 1.0;
      case 1: return 1.0;
      case 2: return (x(0) < 0.5) ? 1.0 : 0.1;
      case 3: return (x(0) > 1.0 && x(1) > 1.5) ? 0.125 : 1.0;
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
         if (x.Size() == 3) { v(1) *= cos(2*M_PI*x(2)); }
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

static bool rerr(const double a, const double v, const double eps)
{
   MFEM_VERIFY(fabs(a) > eps && fabs(v) > eps, "One value is near zero!");
   const double err_a = fabs((a-v)/a);
   const double err_v = fabs((a-v)/v);
   return fmax(err_a, err_v) < eps;
}

static void Checks(const int dim, const int ti, const double nrm, int &chk)
{
   const int pb = problem;
   const double eps = 1.e-13;
   printf("%.15e\n",nrm);
   if (dim==2)
   {
      const double p0_05 = 6.54653862453438e+00;
      const double p0_27 = 7.58857635779292e+00;
      if (pb==0 && ti==05) {chk++; MFEM_VERIFY(rerr(nrm,p0_05,eps),"P0, #05");}
      if (pb==0 && ti==27) {chk++; MFEM_VERIFY(rerr(nrm,p0_27,eps),"P0, #27");}
      const double p1_05 = 3.50825494522579e+00;
      const double p1_15 = 2.75644459682321e+00;
      if (pb==1 && ti==05) {chk++; MFEM_VERIFY(rerr(nrm,p1_05,eps),"P1, #05");}
      if (pb==1 && ti==15) {chk++; MFEM_VERIFY(rerr(nrm,p1_15,eps),"P1, #15");}
      const double p2_05 = 1.02074579565124e+01;
      const double p2_59 = 1.72159020590190e+01;
      if (pb==2 && ti==05) {chk++; MFEM_VERIFY(rerr(nrm,p2_05,eps),"P2, #05");}
      if (pb==2 && ti==59) {chk++; MFEM_VERIFY(rerr(nrm,p2_59,eps),"P2, #59");}
      const double p3_05 = 8.0;
      const double p3_16 = 8.0;
      if (pb==3 && ti==05) {chk++; MFEM_VERIFY(rerr(nrm,p3_05,eps),"P3, #05");}
      if (pb==3 && ti==16) {chk++; MFEM_VERIFY(rerr(nrm,p3_16,eps),"P3, #16");}
      const double p4_05 = 3.446324942352448e+01;
      const double p4_18 = 3.446844033767240e+01;
      if (pb==4 && ti==05) {chk++; MFEM_VERIFY(rerr(nrm,p4_05,eps),"P4, #05");}
      if (pb==4 && ti==18) {chk++; MFEM_VERIFY(rerr(nrm,p4_18,eps),"P4, #18");}
      const double p5_05 = 1.030899557252528e+01;
      const double p5_36 = 1.057362418574309e+01;
      if (pb==5 && ti==05) {chk++; MFEM_VERIFY(rerr(nrm,p5_05,eps),"P5, #05");}
      if (pb==5 && ti==36) {chk++; MFEM_VERIFY(rerr(nrm,p5_36,eps),"P5, #36");}
      const double p6_05 = 8.039707010835693e+00;
      const double p6_36 = 8.316970976817373e+00;
      if (pb==6 && ti==05) {chk++; MFEM_VERIFY(rerr(nrm,p6_05,eps),"P6, #05");}
      if (pb==6 && ti==36) {chk++; MFEM_VERIFY(rerr(nrm,p6_36,eps),"P6, #36");}
      const double p7_05 = 1.514929259650760e+01;
      const double p7_25 = 1.514931278155159e+01;
      if (pb==7 && ti==05) {chk++; MFEM_VERIFY(rerr(nrm,p7_05,eps),"P7, #05");}
      if (pb==7 && ti==25) {chk++; MFEM_VERIFY(rerr(nrm,p7_25,eps),"P7, #25");}
   }
   if (dim==3)
   {
      const double  p0_05 = 1.198510951452527e+03;
      const double p0_188 = 1.199384410059154e+03;
      if (pb==0 && ti==005) {chk++; MFEM_VERIFY(rerr(nrm,p0_05,eps),"P0, #05");}
      if (pb==0 && ti==188) {chk++; MFEM_VERIFY(rerr(nrm,p0_188,eps),"P0, #188");}
      const double p1_05 = 1.33916371859257e+01;
      const double p1_28 = 7.52107367739800e+00;
      if (pb==1 && ti==05) {chk++; MFEM_VERIFY(rerr(nrm,p1_05,eps),"P1, #05");}
      if (pb==1 && ti==28) {chk++; MFEM_VERIFY(rerr(nrm,p1_28,eps),"P1, #28");}
      const double p2_05 = 2.041491591302486e+01;
      const double p2_59 = 3.443180411803796e+01;
      if (pb==2 && ti==05) {chk++; MFEM_VERIFY(rerr(nrm,p2_05,eps),"P2, #05");}
      if (pb==2 && ti==59) {chk++; MFEM_VERIFY(rerr(nrm,p2_59,eps),"P2, #59");}
      const double p3_05 = 1.600000000000000e+01;
      const double p3_16 = 1.600000000000000e+01;
      if (pb==3 && ti==05) {chk++; MFEM_VERIFY(rerr(nrm,p3_05,eps),"P3, #05");}
      if (pb==3 && ti==16) {chk++; MFEM_VERIFY(rerr(nrm,p3_16,eps),"P3, #16");}
      const double p4_05 = 6.892649884704898e+01;
      const double p4_18 = 6.893688067534482e+01;
      if (pb==4 && ti==05) {chk++; MFEM_VERIFY(rerr(nrm,p4_05,eps),"P4, #05");}
      if (pb==4 && ti==18) {chk++; MFEM_VERIFY(rerr(nrm,p4_18,eps),"P4, #18");}
      const double p5_05 = 2.061984481890964e+01;
      const double p5_36 = 2.114519664792607e+01;
      if (pb==5 && ti==05) {chk++; MFEM_VERIFY(rerr(nrm,p5_05,eps),"P5, #05");}
      if (pb==5 && ti==36) {chk++; MFEM_VERIFY(rerr(nrm,p5_36,eps),"P5, #36");}
      const double p6_05 = 1.607988713996459e+01;
      const double p6_36 = 1.662736010353023e+01;
      if (pb==6 && ti==05) {chk++; MFEM_VERIFY(rerr(nrm,p6_05,eps),"P6, #05");}
      if (pb==6 && ti==36) {chk++; MFEM_VERIFY(rerr(nrm,p6_36,eps),"P6, #36");}
      const double p7_05 = 3.029858112572883e+01;
      const double p7_24 = 3.029858832743707e+01;
      if (pb==7 && ti==05) {chk++; MFEM_VERIFY(rerr(nrm,p7_05,eps),"P7, #05");}
      if (pb==7 && ti==24) {chk++; MFEM_VERIFY(rerr(nrm,p7_24,eps),"P7, #24");}
   }
}
