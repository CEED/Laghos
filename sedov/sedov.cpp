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

// Computes and writes out the Sedov shock solution
//
// See:
// James R. Kamm, Evaluation of the Sedov-von Neumann-Taylor Blast Wave Solution
// LA-UR-00-6055

#include "../sedov_sol.hpp"

#include <mfem.hpp>

#include <memory>
#include <sstream>

using namespace mfem;

// static void ProjectCoeff(ParGridFunction &u, VectorCoefficient &coeff,
//                          const mfem::IntegrationRule *ir);

int main(int argc, char *argv[]) {
  // Initialize MPI.
  Mpi::Init();
  int myid = Mpi::WorldRank();
  Hypre::Init();

  // Parse command-line options.
  int dim = 3;
  int rs_levels = 2;
  int rp_levels = 0;
  int nx = 2;
  int ny = 2;
  int nz = 2;
  int order_v = 2;
  int order_e = 1;
  int order_q = -1;
  double t_final = 0.6;
  const char *basename = "results/Sedov";
  real_t Sx = 1, Sy = 1, Sz = 1;

  OptionsParser args(argc, argv);
  args.AddOption(&dim, "-dim", "--dimension", "Dimension of the problem.");
  args.AddOption(&nx, "-nx", "--xelems", "Elements in x-dimension");
  args.AddOption(&ny, "-ny", "--yelems", "Elements in y-dimension");
  args.AddOption(&nz, "-nz", "--zelems", "Elements in z-dimension");
  args.AddOption(&Sx, "-Sx", "--xwidth", "Domain width in x-dimension");
  args.AddOption(&Sy, "-Sy", "--ywidth", "Domain width in y-dimension");
  args.AddOption(&Sz, "-Sz", "--zwidth", "Domain width in z-dimension");
  args.AddOption(&rs_levels, "-rs", "--refine-serial",
                 "Number of times to refine the mesh uniformly in serial.");
  args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                 "Number of times to refine the mesh uniformly in parallel.");
  args.AddOption(&order_v, "-ok", "--order-kinematic",
                 "Order (degree) of the kinematic finite element space.");
  args.AddOption(&order_e, "-ot", "--order-thermo",
                 "Order (degree) of the thermodynamic finite element space.");
  args.AddOption(&order_q, "-oq", "--order-intrule",
                 "Order  of the integration rule.");
  args.AddOption(&t_final, "-tf", "--t-final", "Final time; start time is 0.");
  args.AddOption(&basename, "-k", "--outputfilename",
                 "Name of the visit dump files");
  args.Parse();
  if (!args.Good()) {
    if (Mpi::Root()) {
      args.PrintUsage(std::cout);
    }
    return 1;
  }
  if (Mpi::Root()) {
    args.PrintOptions(std::cout);
  }

  Device::SetMemoryTypes(MemoryType::HOST, MemoryType::DEVICE);

  // On all processors, use the default builtin 1D/2D/3D mesh or read the
  // serial one given on the command line.
  std::unique_ptr<Mesh> mesh;
  switch (dim) {
  case 1:
    mesh.reset(new Mesh(Mesh::MakeCartesian1D(nx, Sx)));
    break;
  case 2:
    mesh.reset(new Mesh(
        Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL, true, Sx, Sy)));
    break;
  case 3:
    mesh.reset(new Mesh(Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON,
                                              Sx, Sy, Sz, true)));
    break;
  default:
    if (Mpi::Root()) {
      std::cout << "Invalid number of dims" << std::endl;
    }
    return -1;
  }

  // Refine the mesh in serial to increase the resolution.
  for (int lev = 0; lev < rs_levels; lev++) {
    mesh->UniformRefinement();
  }
  const int mesh_NE = mesh->GetNE();
  if (Mpi::Root()) {
    std::cout << "Number of zones in the serial mesh: " << mesh_NE << std::endl;
  }

  // Parallel partitioning of the mesh.
  std::unique_ptr<ParMesh> pmesh;
  const int num_tasks = Mpi::WorldSize();

  if (myid == 0) {
    std::cout << "Non-Cartesian partitioning through METIS will be used.\n";
#ifndef MFEM_USE_METIS
    std::cout << "MFEM was built without METIS. "
              << "Adjust the number of tasks to use a Cartesian split."
              << std::endl;
#endif
  }
#ifndef MFEM_USE_METIS
  return 1;
#endif
  pmesh.reset(new ParMesh(MPI_COMM_WORLD, *mesh));

  // Refine the mesh further in parallel to increase the resolution.
  for (int lev = 0; lev < rp_levels; lev++) {
    pmesh->UniformRefinement();
  }

  int NE = pmesh->GetNE(), ne_min, ne_max;
  MPI_Reduce(&NE, &ne_min, 1, MPI_INT, MPI_MIN, 0, pmesh->GetComm());
  MPI_Reduce(&NE, &ne_max, 1, MPI_INT, MPI_MAX, 0, pmesh->GetComm());
  if (myid == 0) {
    std::cout << "Zones min/max: " << ne_min << " " << ne_max << std::endl;
  }

  if (order_q <= 0) {
    order_q = (std::max(order_v, order_e) + 1) * 2;
  }

  const IntegrationRule &irule =
      IntRules.Get(pmesh->GetTypicalElementGeometry(), order_q);

  QuadratureSpace qspace(*pmesh, irule);
  QuadratureFunction qfunc(qspace, 2 + dim);

  double gamma = 1.4;
  double blast_energy = 1;
  double blast_position[] = {0.0, 0.0, 0.0};
  double rho0 = 1;
  double omega = 0;
  {
    SedovSol asol(dim, gamma, rho0, blast_energy, omega);
    asol.SetTime(t_final);
    if (myid == 0) {
        std::cout << "a = " << asol.a << std::endl;
        std::cout << "b = " << asol.b << std::endl;
        std::cout << "c = " << asol.c << std::endl;
        std::cout << "d = " << asol.d << std::endl;
        std::cout << "e = " << asol.e << std::endl;

        std::cout << "alpha0 = " << asol.alpha0 << std::endl;
        std::cout << "alpha1 = " << asol.alpha1 << std::endl;
        std::cout << "alpha2 = " << asol.alpha2 << std::endl;
        std::cout << "alpha3 = " << asol.alpha3 << std::endl;
        std::cout << "alpha4 = " << asol.alpha4 << std::endl;
        std::cout << "alpha5 = " << asol.alpha5 << std::endl;

        std::cout << "V0 = " << asol.V0 << std::endl;
        std::cout << "Vv = " << asol.Vv << std::endl;
        std::cout << "V2 = " << asol.V2 << std::endl;
        std::cout << "Vs = " << asol.Vs << std::endl;
        std::cout << "alpha = " << asol.alpha << std::endl;

        std::cout << "r2 (shock position) = " << asol.r2 << std::endl;
        std::cout << "U (shock speed) = " << asol.U << std::endl;
        std::cout << "rho1 (pre-shock density) = " << asol.rho1 << std::endl;
        std::cout << "rho2 (post-shock density) = " << asol.rho2 << std::endl;
        std::cout << "v2 (post-shock velocity) = " << asol.v2 << std::endl;
        std::cout << "p2 (post-shock pressure) = " << asol.p2 << std::endl;
    }
    auto slambda = [&](const Vector &x, Vector &res) {
      real_t tmp[3];
      Vector dr(tmp, dim);
      double r = 0;

      for (int i = 0; i < dim; ++i) {
        dr[i] = x[i] - blast_position[i];
        r += dr[i] * dr[i];
      }
      r = sqrt(r);
      if (r) {
        for (int i = 0; i < dim; ++i) {
          dr[i] /= r;
        }
      }
      else
      {
        dr = 0_r;
      }
      double rho, v, P;
      asol.EvalSol(r, rho, v, P);
      res[0] = rho;
      for (int i = 0; i < dim; ++i) {
        res[1 + i] = v * dr[i];
      }
      // internal energy
      res[1 + dim] = P / (gamma - 1);
    };
    VectorFunctionCoefficient asol_coeff(2 + dim, slambda);
    asol_coeff.Project(qfunc);
  }

  // Define the parallel finite element spaces. We use:
  // - H1 (Gauss-Lobatto, continuous) for position and velocity.
  // - L2 (Bernstein, discontinuous) for specific internal energy.
  L2_FECollection L2FEC(order_e, dim, BasisType::Positive);
  H1_FECollection H1FEC(order_v, dim);
  ParFiniteElementSpace L2FESpace(pmesh.get(), &L2FEC);
  ParFiniteElementSpace H1FESpace(pmesh.get(), &H1FEC, pmesh->Dimension());

  ParGridFunction rho_gf(&L2FESpace);
  ParGridFunction v_gf(&H1FESpace);
  ParGridFunction energy_gf(&L2FESpace);
#if 0
    // TODO: need to allow vector LF integrator to specify intrule
  VectorQuadratureFunctionCoefficient qcoeff(qfunc);
  {
    qcoeff.SetComponent(0, 1);
    ProjectCoeff(rho_gf, qcoeff, &irule);
  }
  {
    qcoeff.SetComponent(1, dim);
    ProjectCoeff(v_gf, qcoeff, &irule);
  }
  {
    qcoeff.SetComponent(1 + dim, 1);
    ProjectCoeff(energy_gf, qcoeff, &irule);
  }
#endif

  {
    std::stringstream fname;
    fname << basename << "_mesh";
    pmesh->Save(fname.str().c_str());
  }
  {
    std::stringstream fname;
    fname << basename << "_qfunc";
    std::ofstream out(fname.str());
    qfunc.Save(out);
  }
#if 0
  {
    std::stringstream fname;
    fname << basename << "_rho";
    rho_gf.Save(fname.str().c_str());
  }
  {
    std::stringstream fname;
    fname << basename << "_v";
    v_gf.Save(fname.str().c_str());
  }
  {
    std::stringstream fname;
    fname << basename << "_energy";
    energy_gf.Save(fname.str().c_str());
  }
#endif

  return 0;
}

// static void ProjectCoeff(ParGridFunction &u, VectorCoefficient &coeff,
//                          const IntegrationRule *ir) {
//   LinearForm b(u.FESpace());
//   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(coeff, ir));
//   b.UseFastAssembly(true);
//   b.Assemble();

//   BilinearForm a(u.FESpace());
//   a.SetAssemblyLevel(AssemblyLevel::FULL);
//   a.AddDomainIntegrator(new VectorFEMassIntegrator());
//   a.Assemble();
//   // Set solver and preconditioner
//   SparseMatrix A(a.SpMat());
//   GSSmoother prec(A);
//   CGSolver cg;
//   cg.SetPreconditioner(prec);
//   cg.SetOperator(A);
//   cg.SetRelTol(1e-12);
//   cg.SetMaxIter(1000);
//   cg.SetPrintLevel(0);

//   // Solve and get solution
//   u = 0.0;
//   cg.Mult(b, u);
// }
