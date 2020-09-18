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

#include "laghos_solver.hpp"
#include "laghos_timeinteg.hpp"
#include "laghos_rom.hpp"
#include "laghos_utils.hpp"
#include <fstream>

#ifndef _WIN32
#include <sys/stat.h>  // mkdir
#else
#include <direct.h>    // _mkdir
#define mkdir(dir, mode) _mkdir(dir)
#endif

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

void PrintParGridFunction(const int rank, const std::string& name, ParGridFunction *gf)
{
    Vector tv(gf->ParFESpace()->GetTrueVSize());
    gf->GetTrueDofs(tv);

    char tmp[100];
    sprintf(tmp, ".%06d", rank);

    std::string fullname = name + tmp;

    std::ofstream ofs(fullname.c_str(), std::ofstream::out);
    ofs.precision(16);

    for (int i=0; i<tv.Size(); ++i)
        ofs << tv[i] << std::endl;

    ofs.close();
}


void PrintDiffParGridFunction(NormType normtype, const int rank, const std::string& name, ParGridFunction *gf)
{
    Vector tv(gf->ParFESpace()->GetTrueVSize());

    char tmp[100];
    sprintf(tmp, ".%06d", rank);

    std::string fullname = name + tmp;

    std::ifstream ifs(fullname.c_str());

    for (int i=0; i<tv.Size(); ++i)
    {
        double d;
        ifs >> d;
        tv[i] = d;
    }

    ifs.close();

    ParGridFunction rgf(gf->ParFESpace());
    rgf.SetFromTrueDofs(tv);

    PrintNormsOfParGridFunctions(normtype, rank, name, &rgf, gf, true);
}

int main(int argc, char *argv[])
{
    // Initialize MPI.
    MPI_Session mpi(argc, argv);
    int myid = mpi.WorldRank();

    // Print the banner.
    if (mpi.Root()) {
        display_banner(cout);
    }

    // Parse command-line options.
    problem = 1;
    const char *mesh_file = "data/cube01_hex.mesh";
    int rs_levels = 2;
    int rp_levels = 0;
    int order_v = 2;
    int order_e = 1;
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
    const char *visit_basename = "results/Laghos";
    const char *basename = "";
    const char *twfile = "tw.csv";
    const char *twpfile = "twp.csv";
    int partition_type = 0;
    double blast_energy = 0.25;
    double blast_energyFactor = 1.0;
    double blast_position[] = {0.0, 0.0, 0.0};
    bool rom_offline = false;
    bool rom_online = false;
    bool rom_restore = false;
    double sFactorX = 2.0;
    double sFactorV = 20.0;
    double sFactorE = 2.0;
    int numWindows = 0;
    int windowNumSamples = 0;
    int windowOverlapSamples = 0;
    double dtc = 0.0;
    int visitDiffCycle = -1;
    bool writeSol = false;
    bool solDiff = false;
    bool match_end_time = false;
    double rhoFactor = 1.0;
    int rom_paramID = -1;
    const char *normtype_char = "l2";
    Array<double> twep;
    Array2D<int> twparam;
    ROM_Options romOptions;

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&rs_levels, "-rs", "--refine-serial",
                   "Number of times to refine the mesh uniformly in serial.");
    args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                   "Number of times to refine the mesh uniformly in parallel.");
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
    args.AddOption(&match_end_time, "-met", "--match-end-time", "-no-met", "--no-match-end-time",
                   "Match the end time of each window.");
    args.AddOption(&visit, "-visit", "--visit", "-no-visit", "--no-visit",
                   "Enable or disable VisIt visualization.");
    args.AddOption(&gfprint, "-print", "--print", "-no-print", "--no-print",
                   "Enable or disable result output (files in mfem format).");
    args.AddOption(&basename, "-o", "--outputfilename",
                   "Name of the sub-folder to dump files within the run directory");
    args.AddOption(&visit_basename, "-k", "--visitfilename",
                   "Name of the visit dump files");
    args.AddOption(&twfile, "-tw", "--timewindowfilename",
                   "Name of the CSV file defining offline time windows");
    args.AddOption(&twpfile, "-twp", "--timewindowparamfilename",
                   "Name of the CSV file defining online time window parameters");
    args.AddOption(&partition_type, "-pt", "--partition",
                   "Customized x/y/z Cartesian MPI partitioning of the serial mesh.\n\t"
                   "Here x,y,z are relative task ratios in each direction.\n\t"
                   "Example: with 48 mpi tasks and -pt 321, one would get a Cartesian\n\t"
                   "partition of the serial mesh by (6,4,2) MPI tasks in (x,y,z).\n\t"
                   "NOTE: the serially refined mesh must have the appropriate number\n\t"
                   "of zones in each direction, e.g., the number of zones in direction x\n\t"
                   "must be divisible by the number of MPI tasks in direction x.\n\t"
                   "Available options: 11, 21, 111, 211, 221, 311, 321, 322, 432.");
    args.AddOption(&rom_offline, "-offline", "--offline", "-no-offline", "--no-offline",
                   "Enable or disable ROM offline computations and output.");
    args.AddOption(&rom_online, "-online", "--online", "-no-online", "--no-online",
                   "Enable or disable ROM online computations and output.");
    args.AddOption(&rom_restore, "-restore", "--restore", "-no-restore", "--no-restore",
                   "Enable or disable ROM restoration phase where ROM solution is lifted to FOM size.");
    args.AddOption(&romOptions.dimX, "-rdimx", "--rom_dimx", "ROM dimension for X.");
    args.AddOption(&romOptions.dimV, "-rdimv", "--rom_dimv", "ROM dimension for V.");
    args.AddOption(&romOptions.dimE, "-rdime", "--rom_dime", "ROM dimension for E.");
    args.AddOption(&romOptions.dimFv, "-rdimfv", "--rom_dimfv", "ROM dimension for Fv.");
    args.AddOption(&romOptions.dimFe, "-rdimfe", "--rom_dimfe", "ROM dimension for Fe.");
    args.AddOption(&romOptions.sampX, "-nsamx", "--numsamplex", "number of samples for X.");
    args.AddOption(&romOptions.sampV, "-nsamv", "--numsamplev", "number of samples for V.");
    args.AddOption(&romOptions.sampE, "-nsame", "--numsamplee", "number of samples for E.");
    args.AddOption(&sFactorX, "-sfacx", "--sfactorx", "sample factor for X.");
    args.AddOption(&sFactorV, "-sfacv", "--sfactorv", "sample factor for V.");
    args.AddOption(&sFactorE, "-sface", "--sfactore", "sample factor for E.");
    args.AddOption(&romOptions.energyFraction, "-ef", "--rom-ef",
                   "Energy fraction for recommended ROM basis sizes.");
    args.AddOption(&numWindows, "-nwin", "--numwindows", "Number of ROM time windows.");
    args.AddOption(&windowNumSamples, "-nwinsamp", "--numwindowsamples", "Number of samples in ROM windows.");
    args.AddOption(&windowOverlapSamples, "-nwinover", "--numwindowoverlap", "Number of samples for ROM window overlap.");
    args.AddOption(&dtc, "-dtc", "--dtc", "Fixed (constant) dt.");
    args.AddOption(&visitDiffCycle, "-visdiff", "--visdiff", "VisIt DC cycle to diff.");
    args.AddOption(&writeSol, "-writesol", "--writesol", "-no-writesol", "--no-writesol",
                   "Enable or disable write solution.");
    args.AddOption(&solDiff, "-soldiff", "--soldiff", "-no-soldiff", "--no-soldiff",
                   "Enable or disable solution difference norm computation.");
    args.AddOption(&romOptions.hyperreduce, "-romhr", "--romhr", "-no-romhr", "--no-romhr",
                   "Enable or disable ROM hyperreduction.");
    args.AddOption(&romOptions.staticSVD, "-romsvds", "--romsvdstatic", "-no-romsvds", "--no-romsvds",
                   "Enable or disable ROM static SVD.");
    args.AddOption(&romOptions.useOffset, "-romos", "--romoffset", "-no-romoffset", "--no-romoffset",
                   "Enable or disable initial state offset for ROM.");
    args.AddOption(&normtype_char, "-normtype", "--norm_type", "Norm type for relative error computation.");
    args.AddOption(&romOptions.max_dim, "-sdim", "--sdim", "ROM max sample dimension");
    args.AddOption(&romOptions.RHSbasis, "-romsrhs", "--romsamplerhs", "-no-romsrhs", "--no-romsamplerhs",
                   "Sample RHS");
    args.AddOption(&romOptions.GramSchmidt, "-romgs", "--romgramschmidt", "-no-romgs", "--no-romgramschmidt",
                   "Enable or disable Gram-Schmidt orthonormalization on V and E induced by mass matrices.");
    args.AddOption(&rhoFactor, "-rhof", "--rhofactor", "Factor for scaling rho.");
    args.AddOption(&blast_energyFactor, "-bef", "--blastefactor", "Factor for scaling blast energy.");
    args.AddOption(&rom_paramID, "-rpar", "--romparam", "ROM offline parameter index.");
    args.AddOption(&romOptions.paramOffset, "-rparos", "--romparamoffset", "-no-rparos", "--no-romparamoffset",
                   "Enable or disable parametric offset.");
    args.Parse();
    if (!args.Good())
    {
        if (mpi.Root()) {
            args.PrintUsage(cout);
        }
        return 1;
    }
    std::string outputPath = "run";
    if (std::string(basename) != "") {
        outputPath += "/" + std::string(basename);
    }
    if (mpi.Root()) {
        const char path_delim = '/';
        std::string::size_type pos = 0;
        do {
          pos = outputPath.find(path_delim, pos+1);
          std::string subdir = outputPath.substr(0, pos);
          mkdir(subdir.c_str(), 0777);
        }
        while (pos != std::string::npos);
        mkdir((outputPath + "/ROMoffset").c_str(), 0777);
        mkdir((outputPath + "/ROMsol").c_str(), 0777);

        args.PrintOptions(cout);
    }

    romOptions.basename = &outputPath;

    MFEM_VERIFY(windowNumSamples == 0 || rom_offline, "-nwinsamp should be specified only in offline mode");
    MFEM_VERIFY(windowNumSamples == 0 || numWindows == 0, "-nwinsamp and -nwin cannot both be set");

    const bool usingWindows = (numWindows > 0 || windowNumSamples > 0);
    if (usingWindows)
    {
        if (rom_online || rom_restore)
        {
            double sFactor[]  = {sFactorX, sFactorV, sFactorE};
            const int err = ReadTimeWindowParameters(numWindows, outputPath + "/" + std::string(twpfile), twep, twparam, sFactor, myid == 0, romOptions.RHSbasis);
            MFEM_VERIFY(err == 0, "Error in ReadTimeWindowParameters");
        }
        else if (rom_offline && windowNumSamples == 0)
        {
            const int err = ReadTimeWindows(numWindows, twfile, twep, myid == 0);
            MFEM_VERIFY(err == 0, "Error in ReadTimeWindows");
        }
    }
    else  // not using windows
    {
        numWindows = 1;  // one window for the entire simulation
    }

    if (windowNumSamples > 0) romOptions.max_dim = windowNumSamples + windowOverlapSamples + 2;
    MFEM_VERIFY(windowOverlapSamples >= 0, "Negative window overlap");
    MFEM_VERIFY(windowOverlapSamples <= windowNumSamples, "Too many ROM window overlap samples.");

    StopWatch totalTimer;
    totalTimer.Start();

    static std::map<std::string, NormType> localmap;
    localmap["l2"] = l2norm;
    localmap["l1"] = l1norm;
    localmap["max"] = maxnorm;

    NormType normtype = localmap[normtype_char];

    // Read the serial mesh from the given mesh file on all processors.
    // Refine the mesh in serial to increase the resolution.
    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    const int dim = mesh->Dimension();
    for (int lev = 0; lev < rs_levels; lev++) {
        mesh->UniformRefinement();
    }

    if (p_assembly && dim == 1)
    {
        p_assembly = false;
        if (mpi.Root())
        {
            cout << "Laghos does not support PA in 1D. Switching to FA." << endl;
        }
    }

    // Parallel partitioning of the mesh.
    ParMesh *pmesh = NULL;
    const int num_tasks = mpi.WorldSize();
    int unit;
    int *nxyz = new int[dim];
    switch (partition_type)
    {
    case 0:
        for (int d = 0; d < dim; d++) {
            nxyz[d] = unit;
        }
        break;
    case 11:
    case 111:
        unit = floor(pow(num_tasks, 1.0 / dim) + 1e-2);
        for (int d = 0; d < dim; d++) {
            nxyz[d] = unit;
        }
        break;
    case 21: // 2D
        unit = floor(pow(num_tasks / 2, 1.0 / 2) + 1e-2);
        nxyz[0] = 2 * unit;
        nxyz[1] = unit;
        break;
    case 211: // 3D.
        unit = floor(pow(num_tasks / 2, 1.0 / 3) + 1e-2);
        nxyz[0] = 2 * unit;
        nxyz[1] = unit;
        nxyz[2] = unit;
        break;
    case 221: // 3D.
        unit = floor(pow(num_tasks / 4, 1.0 / 3) + 1e-2);
        nxyz[0] = 2 * unit;
        nxyz[1] = 2 * unit;
        nxyz[2] = unit;
        break;
    case 311: // 3D.
        unit = floor(pow(num_tasks / 3, 1.0 / 3) + 1e-2);
        nxyz[0] = 3 * unit;
        nxyz[1] = unit;
        nxyz[2] = unit;
        break;
    case 321: // 3D.
        unit = floor(pow(num_tasks / 6, 1.0 / 3) + 1e-2);
        nxyz[0] = 3 * unit;
        nxyz[1] = 2 * unit;
        nxyz[2] = unit;
        break;
    case 322: // 3D.
        unit = floor(pow(2 * num_tasks / 3, 1.0 / 3) + 1e-2);
        nxyz[0] = 3 * unit / 2;
        nxyz[1] = unit;
        nxyz[2] = unit;
        break;
    case 432: // 3D.
        unit = floor(pow(num_tasks / 3, 1.0 / 3) + 1e-2);
        nxyz[0] = 2 * unit;
        nxyz[1] = 3 * unit / 2;
        nxyz[2] = unit;
        break;
    case 511: // 3D.
        unit = floor(pow(num_tasks / 5, 1.0 / 3) + 1e-2);
        nxyz[0] = 5 * unit;
        nxyz[1] = unit;
        nxyz[2] = unit;
        break;
    case 521: // 3D.
        unit = floor(pow(num_tasks / 10, 1.0 / 3) + 1e-2);
        nxyz[0] = 5 * unit;
        nxyz[1] = 2 * unit;
        nxyz[2] = unit;
        break;
    case 522: // 3D.
        unit = floor(pow(num_tasks / 20, 1.0 / 3) + 1e-2);
        nxyz[0] = 5 * unit;
        nxyz[1] = 2 * unit;
        nxyz[2] = 2 * unit;
        break;
    case 911: // 3D.
        unit = floor(pow(num_tasks / 9, 1.0 / 3) + 1e-2);
        nxyz[0] = 9 * unit;
        nxyz[1] = unit;
        nxyz[2] = unit;
        break;
    case 921: // 3D.
        unit = floor(pow(num_tasks / 18, 1.0 / 3) + 1e-2);
        nxyz[0] = 9 * unit;
        nxyz[1] = 2 * unit;
        nxyz[2] = unit;
        break;
    case 922: // 3D.
        unit = floor(pow(num_tasks / 36, 1.0 / 3) + 1e-2);
        nxyz[0] = 9 * unit;
        nxyz[1] = 2 * unit;
        nxyz[2] = 2 * unit;
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
    for (int d = 0; d < dim; d++) {
        product *= nxyz[d];
    }
    if (product == num_tasks)
    {
        int *partitioning = mesh->CartesianPartitioning(nxyz);
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

    // Refine the mesh further in parallel to increase the resolution.
    for (int lev = 0; lev < rp_levels; lev++) {
        pmesh->UniformRefinement();
    }

    int nzones = pmesh->GetNE(), nzones_min, nzones_max;
    MPI_Reduce(&nzones, &nzones_min, 1, MPI_INT, MPI_MIN, 0, pmesh->GetComm());
    MPI_Reduce(&nzones, &nzones_max, 1, MPI_INT, MPI_MAX, 0, pmesh->GetComm());
    if (myid == 0)
    {
        cout << "Zones min/max: " << nzones_min << " " << nzones_max << endl;
    }

    // Define the parallel finite element spaces. We use:
    // - H1 (Gauss-Lobatto, continuous) for position and velocity.
    // - L2 (Bernstein, discontinuous) for specific internal energy.
    L2_FECollection L2FEC(order_e, dim, BasisType::Positive);
    H1_FECollection H1FEC(order_v, dim);
    ParFiniteElementSpace L2FESpace(pmesh, &L2FEC);
    ParFiniteElementSpace H1FESpace(pmesh, &H1FEC, pmesh->Dimension());

    cout << myid << ": pmesh->bdr_attributes.Max() " << pmesh->bdr_attributes.Max() << endl;
    // Boundary conditions: all tests use v.n = 0 on the boundary, and we assume
    // that the boundaries are straight.
    Array<int> ess_tdofs;
    {
        Array<int> ess_bdr(pmesh->bdr_attributes.Max()), tdofs1d;
        for (int d = 0; d < pmesh->Dimension(); d++)
        {
            // Attributes 1/2/3 correspond to fixed-x/y/z boundaries, i.e., we must
            // enforce v_x/y/z = 0 for the velocity components.
            ess_bdr = 0;
            ess_bdr[d] = 1;
            H1FESpace.GetEssentialTrueDofs(ess_bdr, tdofs1d, d);
            ess_tdofs.Append(tdofs1d);
        }
    }

    // Define the explicit ODE solver used for time integration.
    ODESolver *ode_solver = NULL;
    switch (ode_solver_type)
    {
    case 1:
        ode_solver = new ForwardEulerSolver;
        break;
    case 2:
        ode_solver = new RK2Solver(0.5);
        break;
    case 3:
        ode_solver = new RK3SSPSolver;
        break;
    case 4:
        ode_solver = new RK4Solver;
        break;
    case 6:
        ode_solver = new RK6Solver;
        break;
    case 7:
        ode_solver = new RK2AvgSolver(rom_online, &H1FESpace, &L2FESpace);
        break;
    default:
        if (myid == 0)
        {
            cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
        }
        delete pmesh;
        MPI_Finalize();
        return 3;
    }

    romOptions.RK2AvgSolver = (ode_solver_type == 7);

    HYPRE_Int glob_size_l2 = L2FESpace.GlobalTrueVSize();
    HYPRE_Int glob_size_h1 = H1FESpace.GlobalTrueVSize();

    if (mpi.Root())
    {
        cout << "Number of kinematic (position, velocity) dofs: "
             << glob_size_h1 << endl;
        cout << "Number of specific internal energy dofs: "
             << glob_size_l2 << endl;
    }

    int Vsize_l2 = L2FESpace.GetVSize();
    int Vsize_h1 = H1FESpace.GetVSize();

    int tVsize_l2 = L2FESpace.GetTrueVSize();
    int tVsize_h1 = H1FESpace.GetTrueVSize();

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
    ParGridFunction x_gf, v_gf, e_gf;
    x_gf.MakeRef(&H1FESpace, S, true_offset[0]);
    v_gf.MakeRef(&H1FESpace, S, true_offset[1]);
    e_gf.MakeRef(&L2FESpace, S, true_offset[2]);

    // Initialize x_gf using the starting mesh coordinates.
    pmesh->SetNodalGridFunction(&x_gf);

    // Initialize the velocity.
    VectorFunctionCoefficient v_coeff(pmesh->Dimension(), v0);
    v_gf.ProjectCoefficient(v_coeff);

    // Initialize density and specific internal energy values. We interpolate in
    // a non-positive basis to get the correct values at the dofs.  Then we do an
    // L2 projection to the positive basis in which we actually compute. The goal
    // is to get a high-order representation of the initial condition. Note that
    // this density is a temporary function and it will not be updated during the
    // time evolution.
    ParGridFunction rho(&L2FESpace);
    FunctionCoefficient rho_coeff0(rho0);
    ProductCoefficient rho_coeff(rhoFactor, rho_coeff0);
    L2_FECollection l2_fec(order_e, pmesh->Dimension());
    ParFiniteElementSpace l2_fes(pmesh, &l2_fec);
    ParGridFunction l2_rho(&l2_fes), l2_e(&l2_fes);
    l2_rho.ProjectCoefficient(rho_coeff);
    rho.ProjectGridFunction(l2_rho);
    if (problem == 1)
    {
        // For the Sedov test, we use a delta function at the origin.
        DeltaCoefficient e_coeff(blast_position[0], blast_position[1],
                                 blast_position[2], blast_energyFactor*blast_energy);
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
    L2_FECollection mat_fec(0, pmesh->Dimension());
    ParFiniteElementSpace mat_fes(pmesh, &mat_fec);
    ParGridFunction mat_gf(&mat_fes);
    FunctionCoefficient mat_coeff(gamma);
    mat_gf.ProjectCoefficient(mat_coeff);
    GridFunctionCoefficient *mat_gf_coeff = new GridFunctionCoefficient(&mat_gf);

    // Additional details, depending on the problem.
    int source = 0;
    bool visc = true;
    switch (problem)
    {
    case 0:
        if (pmesh->Dimension() == 2) {
            source = 1;
        }
        visc = false;
        break;
    case 1:
        visc = true;
        break;
    case 2:
        visc = true;
        break;
    case 3:
        visc = true;
        break;
    case 4:
        visc = false;
        break;
    default:
        MFEM_ABORT("Wrong problem specification!");
    }
    if (impose_visc) {
        visc = true;
    }

    LagrangianHydroOperator oper(S.Size(), H1FESpace, L2FESpace,
                                 ess_tdofs, rho, source, cfl, mat_gf_coeff,
                                 visc, p_assembly, cg_tol, cg_max_iter, ftz_tol,
                                 H1FEC.GetBasisType());

    socketstream vis_rho, vis_v, vis_e;
    char vishost[] = "localhost";
    int  visport   = 19916;

    ParGridFunction rho_gf;
    if (visualization || visit) {
        oper.ComputeDensity(rho_gf);
    }

    const double energy_init = oper.InternalEnergy(e_gf) +
                               oper.KineticEnergy(v_gf);

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
    const char *visit_outputPath = (outputPath + "/" + std::string(visit_basename)).c_str();
    VisItDataCollection visit_dc(visit_outputPath, pmesh);
    if (visit)
    {
        if (rom_offline || rom_restore)
            visit_dc.RegisterField("Position",  &x_gf);

        visit_dc.RegisterField("Density",  &rho_gf);
        visit_dc.RegisterField("Velocity", &v_gf);
        visit_dc.RegisterField("Specific Internal Energy", &e_gf);
        visit_dc.SetCycle(0);
        visit_dc.SetTime(0.0);
        visit_dc.Save();
    }

    cout << myid << ": pmesh number of elements " << pmesh->GetNE() << endl;

    romOptions.rank = myid;
    romOptions.H1FESpace = &H1FESpace;
    romOptions.L2FESpace = &L2FESpace;
    romOptions.window = 0;
    romOptions.FOMoper = &oper;
    romOptions.parameterID = rom_paramID;

    // Perform time-integration (looping over the time iterations, ti, with a
    // time-step dt). The object oper is of type LagrangianHydroOperator that
    // defines the Mult() method that is used by the time integrators.
    if (!rom_online) ode_solver->Init(oper);
    oper.ResetTimeStepEstimate();
    double t = 0.0, dt = oper.GetTimeStepEstimate(S), t_old, dt_old;
    bool use_dt_old = false;
    bool last_step = false;
    int steps = 0;
    BlockVector S_old(S);

    StopWatch samplerTimer;
    ROM_Sampler *sampler = NULL;
    ROM_Sampler *samplerLast = NULL;
    std::ofstream outfile_twp;
    Array<int> cutoff(5);
    if (rom_offline)
    {
        if (dtc > 0.0) dt = dtc;

        samplerTimer.Start();
        if (usingWindows && romOptions.parameterID == -1) {
            outfile_twp.open(outputPath + "/twpTemp.csv");
        }
        const double tf = (usingWindows && windowNumSamples == 0) ? twep[0] : t_final;
        romOptions.t_final = tf;
        romOptions.initial_dt = dt;
        sampler = new ROM_Sampler(romOptions, S);
        sampler->SampleSolution(0, 0, S);
        samplerTimer.Stop();
    }

    ROM_Basis *basis = NULL;
    Vector romS, romS_old;
    ROM_Operator *romOper = NULL;

    if (!usingWindows)
    {
        if (romOptions.sampX == 0) romOptions.sampX = sFactorX * romOptions.dimX;
        if (romOptions.sampV == 0) romOptions.sampV = sFactorV * romOptions.dimV;
        if (romOptions.sampE == 0) romOptions.sampE = sFactorE * romOptions.dimE;
    }

    StopWatch onlinePreprocessTimer;
    if (rom_online)
    {
        onlinePreprocessTimer.Start();
        if (dtc > 0.0) dt = dtc;
        if (usingWindows)
        {
            romOptions.dimX = twparam(0,0);
            romOptions.dimV = twparam(0,1);
            romOptions.dimE = twparam(0,2);
            if (romOptions.RHSbasis)
            {
                romOptions.dimFv = twparam(0,3);
                romOptions.dimFe = twparam(0,4);
            }
            const int oss = romOptions.RHSbasis ? 5 : 3;
            romOptions.sampX = twparam(0,oss);
            romOptions.sampV = twparam(0,oss+1);
            romOptions.sampE = twparam(0,oss+2);
        }
        basis = new ROM_Basis(romOptions, S, MPI_COMM_WORLD);
        romS.SetSize(romOptions.dimX + romOptions.dimV + romOptions.dimE);
        basis->ProjectFOMtoROM(S, romS);

        cout << myid << ": initial romS norm " << romS.Norml2() << endl;

        romOper = new ROM_Operator(romOptions, basis, rho_coeff, mat_coeff, order_e, source, visc, cfl, p_assembly,
                                   cg_tol, cg_max_iter, ftz_tol, &H1FEC, &L2FEC);

        ode_solver->Init(*romOper);
        onlinePreprocessTimer.Stop();
    }

    StopWatch restoreTimer, timeLoopTimer;
    if (rom_restore)
    {
        // -restore phase
        // No need to specify t_final because the loop in -restore phase is determined by the files in ROMsol folder.
        // When -romhr or --romhr are used in -online phase, then -restore phase needs to be called to project rom solution back to FOM size
        std::ifstream infile_tw_steps(outputPath + "/tw_steps");
        int nb_step(0);
        restoreTimer.Start();
        if (usingWindows) {
            romOptions.dimX = twparam(romOptions.window,0);
            romOptions.dimV = twparam(romOptions.window,1);
            romOptions.dimE = twparam(romOptions.window,2);
            if (romOptions.RHSbasis)
            {
                romOptions.dimFv = twparam(romOptions.window,3);
                romOptions.dimFe = twparam(romOptions.window,4);
            }
            basis = new ROM_Basis(romOptions, S, MPI_COMM_WORLD);
        } else {
            basis = new ROM_Basis(romOptions, S, MPI_COMM_WORLD);
        }
        int romSsize = romOptions.dimX + romOptions.dimV + romOptions.dimE;
        romS.SetSize(romSsize);
        if (infile_tw_steps.good())
        {
            infile_tw_steps >> nb_step;
        }
        int ti;
        for (ti = 1; !last_step; ti++)
        {
            // romS = readCurrentReduceSol(ti);
            // read ROM solution from a file.
            // TODO: it needs to be read from the format of HDF5 format
            // TODO: how about parallel version? introduce rank in filename
            std::string filename = outputPath + "/ROMsol/romS_" + std::to_string(ti);
            std::ifstream infile_romS(filename.c_str());
            if (infile_romS.good())
            {
                if ( (ti % vis_steps) == 0 )
                {
                    if (myid == 0)
                        cout << "Restoring " << ti << "-th solution" << endl;
                    for (int k=0; k<romSsize; ++k)
                    {
                        infile_romS >> romS(k);
                    }

                    infile_romS.close();
                    basis->LiftROMtoFOM(romS, S);

                    if (visit)
                    {
                        oper.ComputeDensity(rho_gf);
                        visit_dc.SetCycle(ti);
                        visit_dc.SetTime(t);
                        visit_dc.Save();
                    }
                }
            }
            else
            {
                // get out of the loop when no more file is found
                last_step = true;
                break;
            }
            if (ti == nb_step) {
                if (infile_tw_steps.good())
                {
                    infile_tw_steps >> nb_step;
                }
                romOptions.window++;
                romOptions.dimX = twparam(romOptions.window,0);
                romOptions.dimV = twparam(romOptions.window,1);
                romOptions.dimE = twparam(romOptions.window,2);
                if (romOptions.RHSbasis)
                {
                    romOptions.dimFv = twparam(romOptions.window,3);
                    romOptions.dimFe = twparam(romOptions.window,4);
                }
                basis->LiftROMtoFOM(romS, S);
                delete basis;
                basis = new ROM_Basis(romOptions, S, MPI_COMM_WORLD);
                romSsize = romOptions.dimX + romOptions.dimV + romOptions.dimE;
                romS.SetSize(romSsize);
            }
        } // time loop in "restore" phase
        ti--;
        std::string filename = outputPath + "/ROMsol/romS_" + std::to_string(ti);
        std::ifstream infile_romS(filename.c_str());
        if (myid == 0)
            cout << "Restoring " << ti << "-th solution" << endl;
        for (int k=0; k<romSsize; ++k)
        {
            infile_romS >> romS(k);
        }

        infile_romS.close();
        basis->LiftROMtoFOM(romS, S);

        if (visit)
        {
            oper.ComputeDensity(rho_gf);
            visit_dc.SetCycle(ti);
            visit_dc.SetTime(t);
            visit_dc.Save();
        }
        restoreTimer.Stop();
        infile_tw_steps.close();
    }
    else
    {
        // usual time loop when rom_restore phase is false.
        std::ofstream outfile_tw_steps;
        if (rom_online && usingWindows)
        {
            outfile_tw_steps.open(outputPath + "/tw_steps");
        }
        timeLoopTimer.Start();
        if (romOptions.hyperreduce && romOptions.GramSchmidt)
        {
            romOper->InducedGramSchmidtInitialize(romS);
        }
        double tOverlapMidpoint = 0.0;
        for (int ti = 1; !last_step; ti++)
        {
            if (t + dt >= t_final)
            {
                dt = t_final - t;
                last_step = true;
            }

            if ( use_dt_old )
            {
                dt = dt_old;
                use_dt_old = false;
            }

            if (rom_online && usingWindows && (t + dt >= twep[romOptions.window]) & match_end_time)
            {
                dt_old = dt;
                use_dt_old = true;
                dt = twep[romOptions.window] - t;
            }

            if (steps == max_tsteps) {
                last_step = true;
            }

            if (!rom_online || !romOptions.hyperreduce) S_old = S;
            t_old = t;
            oper.ResetTimeStepEstimate();

            // S is the vector of dofs, t is the current time, and dt is the time step
            // to advance.
            if (rom_online)
            {
                if (myid == 0)
                    cout << "ROM online at t " << t << ", dt " << dt << ", romS norm " << romS.Norml2() << endl;

                romS_old = romS;
                ode_solver->Step(romS, t, dt);

                // save ROM solution to a file.
                // TODO: it needs to be save in the format of HDF5 format
                // TODO: how about parallel version? introduce rank in filename
                // TODO: think about how to reuse "gfprint" option
                std::string filename = outputPath + "/ROMsol/romS_" + std::to_string(ti);
                std::ofstream outfile_romS(filename.c_str());
                outfile_romS.precision(16);
                romS.Print(outfile_romS, 1);
                outfile_romS.close();

                if (!romOptions.hyperreduce)
                    basis->LiftROMtoFOM(romS, S);

                romOper->UpdateSampleMeshNodes(romS);

                if (!romOptions.hyperreduce) oper.ResetQuadratureData();  // Necessary for oper.GetTimeStepEstimate(S);
            }
            else
            {
                if (myid == 0)
                    cout << "FOM simulation at t " << t << ", dt " << dt << endl;

                ode_solver->Step(S, t, dt);
            }

            steps++;

            const double last_dt = dt;

            // Adaptive time step control.
            const double dt_est = romOptions.hyperreduce ? romOper->GetTimeStepEstimateSP() : oper.GetTimeStepEstimate(S);

            //const double dt_est = oper.GetTimeStepEstimate(S);
            //cout << myid << ": dt_est " << dt_est << endl;
            if (dt_est < dt)
            {
                // Repeat (solve again) with a decreased time step - decrease of the
                // time estimate suggests appearance of oscillations.
                dt *= 0.85;
                if (dt < numeric_limits<double>::epsilon())
                {
                    MFEM_ABORT("The time step crashed!");
                }
                t = t_old;
                if (!rom_online || !romOptions.hyperreduce) S = S_old;
                if (rom_online) romS = romS_old;
                oper.ResetQuadratureData();
                if (mpi.Root()) {
                    cout << "Repeating step " << ti << endl;
                }
                if (steps < max_tsteps) {
                    last_step = false;
                }
                ti--;
                continue;
            }
            else if (dtc == 0.0 && dt_est > 1.25 * dt) {
                dt *= 1.02;
            }

            if (rom_offline)
            {
                timeLoopTimer.Stop();
                samplerTimer.Start();
                sampler->SampleSolution(t, last_dt, S);

                bool endWindow = false;
                if (usingWindows)
                {
                    if (numWindows > 0)
                    {
                        endWindow = (t >= twep[romOptions.window] && romOptions.window < numWindows-1);
                    }
                    else
                    {
                        endWindow = (sampler->MaxNumSamples() >= windowNumSamples);
                    }
                }

                if (samplerLast)
                {
                    samplerLast->SampleSolution(t, last_dt, S);
                    if (samplerLast->MaxNumSamples() == windowNumSamples + (windowOverlapSamples/2))
                        tOverlapMidpoint = t;

                    if (samplerLast->MaxNumSamples() >= windowNumSamples + windowOverlapSamples || last_step)
                    {
                        samplerLast->Finalize(t, last_dt, S, cutoff);
                        if (last_step)
                        {
                            // Let samplerLast define the final window, discarding the sampler window.
                            tOverlapMidpoint = t;
                            sampler = NULL;
                        }

                        MFEM_VERIFY(tOverlapMidpoint > 0.0, "Overlapping window endpoint undefined.");
                        if (myid == 0 && romOptions.parameterID == -1) {
                            outfile_twp << tOverlapMidpoint << ", ";
                            if (romOptions.RHSbasis)
                                outfile_twp << cutoff[0] << ", " << cutoff[1] << ", " << cutoff[2] << ", "
                                            << cutoff[3] << ", " << cutoff[4] << "\n";
                            else
                                outfile_twp << cutoff[0] << ", " << cutoff[1] << ", " << cutoff[2] << "\n";
                        }
                        delete samplerLast;
                        samplerLast = NULL;
                        tOverlapMidpoint = 0.0;
                    }
                }

                if (endWindow)
                {
                    if (numWindows == 0 && windowOverlapSamples > 0)
                    {
                        samplerLast = sampler;
                    }
                    else
                    {
                        sampler->Finalize(t, last_dt, S, cutoff);
                        if (myid == 0 && romOptions.parameterID == -1) {
                            outfile_twp << t << ", ";
                            if (romOptions.RHSbasis)
                                outfile_twp << cutoff[0] << ", " << cutoff[1] << ", " << cutoff[2] << ", "
                                            << cutoff[3] << ", " << cutoff[4] << "\n";
                            else
                                outfile_twp << cutoff[0] << ", " << cutoff[1] << ", " << cutoff[2] << "\n";
                        }
                        delete sampler;
                    }

                    romOptions.window++;
                    if (!last_step)
                    {
                        romOptions.t_final = (usingWindows && windowNumSamples == 0) ? twep[romOptions.window] : t_final;
                        romOptions.initial_dt = dt;
                        romOptions.window = romOptions.window;
                        sampler = new ROM_Sampler(romOptions, S);
                        sampler->SampleSolution(t, dt, S);
                    }
                }
                samplerTimer.Stop();
                timeLoopTimer.Start();
            }

            if (rom_online)
            {
                if (usingWindows && t >= twep[romOptions.window] && romOptions.window < numWindows-1)
                {
                    romOptions.window++;
                    outfile_tw_steps << ti << "\n";

                    if (myid == 0)
                        cout << "ROM online basis change for window " << romOptions.window << " at t " << t << ", dt " << dt << endl;

                    if (romOptions.hyperreduce && romOptions.GramSchmidt)
                    {
                        romOper->InducedGramSchmidtFinalize(romS);
                    }

                    romOptions.dimX = twparam(romOptions.window,0);
                    romOptions.dimV = twparam(romOptions.window,1);
                    romOptions.dimE = twparam(romOptions.window,2);
                    if (romOptions.RHSbasis)
                    {
                        romOptions.dimFv = twparam(romOptions.window,3);
                        romOptions.dimFe = twparam(romOptions.window,4);
                    }
                    const int oss = romOptions.RHSbasis ? 5 : 3;
                    romOptions.sampX = twparam(romOptions.window,oss);
                    romOptions.sampV = twparam(romOptions.window,oss+1);
                    romOptions.sampE = twparam(romOptions.window,oss+2);

                    if (romOptions.hyperreduce)
                    {
                        basis->LiftROMtoFOM(romS, S);
                    }
                    delete basis;
                    timeLoopTimer.Stop();
                    basis = new ROM_Basis(romOptions, S, MPI_COMM_WORLD);
                    romS.SetSize(romOptions.dimX + romOptions.dimV + romOptions.dimE);
                    timeLoopTimer.Start();

                    basis->ProjectFOMtoROM(S, romS);

                    delete romOper;
                    romOper = new ROM_Operator(romOptions, basis, rho_coeff, mat_coeff, order_e, source, visc, cfl, p_assembly,
                                               cg_tol, cg_max_iter, ftz_tol, &H1FEC, &L2FEC);

                    if (romOptions.hyperreduce && romOptions.GramSchmidt)
                    {
                        romOper->InducedGramSchmidtInitialize(romS);
                    }
                    ode_solver->Init(*romOper);
                }
            }

            // Make sure that the mesh corresponds to the new solution state. This is
            // needed, because some time integrators use different S-type vectors
            // and the oper object might have redirected the mesh positions to those.
            pmesh->NewNodes(x_gf, false);

            if (last_step || (ti % vis_steps) == 0)
            {
                double loc_norm = e_gf * e_gf, tot_norm;
                MPI_Allreduce(&loc_norm, &tot_norm, 1, MPI_DOUBLE, MPI_SUM,
                              pmesh->GetComm());

                if (romOptions.hyperreduce)
                    tot_norm = 0.0;  // e_gf is not updated in hyperreduction case

                if (mpi.Root())
                {
                    cout << fixed;
                    cout << "step " << setw(5) << ti
                         << ",\tt = " << setw(5) << setprecision(4) << t
                         << ",\tdt = " << setw(5) << setprecision(6) << dt
                         << ",\t|e| = " << setprecision(10)
                         << sqrt(tot_norm) << endl;
                    if (last_step) {
                        std::ofstream outfile(outputPath + "/num_steps");
                        outfile << ti;
                        outfile.close();
                    }
                }

                // Make sure all ranks have sent their 'v' solution before initiating
                // another set of GLVis connections (one from each rank):
                MPI_Barrier(pmesh->GetComm());

                if (visualization || visit || gfprint) {
                    oper.ComputeDensity(rho_gf);
                }
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
                    ostringstream mesh_name, rho_name, v_name, e_name;
                    mesh_name << visit_outputPath << "_" << ti
                              << "_mesh." << setfill('0') << setw(6) << myid;
                    rho_name  << visit_outputPath << "_" << ti
                              << "_rho." << setfill('0') << setw(6) << myid;
                    v_name << visit_outputPath << "_" << ti
                           << "_v." << setfill('0') << setw(6) << myid;
                    e_name << visit_outputPath << "_" << ti
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
        } // usual time loop
        timeLoopTimer.Stop();
        outfile_tw_steps.close();
    }

    if (romOptions.hyperreduce)
    {
        if (romOptions.GramSchmidt)
        {
            romOper->InducedGramSchmidtFinalize(romS);
        }
        basis->LiftROMtoFOM(romS, S);
    }

    if (rom_offline)
    {
        samplerTimer.Start();
        if (samplerLast)
            samplerLast->Finalize(t, dt, S, cutoff);
        else if (sampler)
            sampler->Finalize(t, dt, S, cutoff);

        if (myid == 0 && usingWindows && sampler != NULL && romOptions.parameterID == -1) {
            outfile_twp << t << ", ";

            if (romOptions.RHSbasis)
                outfile_twp << cutoff[0] << ", " << cutoff[1] << ", " << cutoff[2] << ", "
                            << cutoff[3] << ", " << cutoff[4] << "\n";
            else
                outfile_twp << cutoff[0] << ", " << cutoff[1] << ", " << cutoff[2] << "\n";
        }
        if (samplerLast == sampler)
            delete sampler;
        else
        {
            delete sampler;
            delete samplerLast;
        }

        samplerTimer.Stop();
        if(usingWindows && romOptions.parameterID == -1) outfile_twp.close();
    }

    if (writeSol)
    {
        PrintParGridFunction(myid, outputPath + "/Sol_Position", &x_gf);
        PrintParGridFunction(myid, outputPath + "/Sol_Velocity", &v_gf);
        PrintParGridFunction(myid, outputPath + "/Sol_Energy", &e_gf);
    }

    if (solDiff)
    {
        cout << "solDiff mode " << endl;
        PrintDiffParGridFunction(normtype, myid, outputPath + "/Sol_Position", &x_gf);
        PrintDiffParGridFunction(normtype, myid, outputPath + "/Sol_Velocity", &v_gf);
        PrintDiffParGridFunction(normtype, myid, outputPath + "/Sol_Energy", &e_gf);
    }

    if (visitDiffCycle >= 0)
    {
        VisItDataCollection dc(MPI_COMM_WORLD, outputPath + "/results/Laghos", pmesh);
        dc.Load(visitDiffCycle);
        cout << "Loaded VisIt DC cycle " << dc.GetCycle() << endl;

        ParGridFunction *dcfx = dc.GetParField("Position");
        ParGridFunction *dcfv = dc.GetParField("Velocity");
        ParGridFunction *dcfe = dc.GetParField("Specific Internal Energy");

        PrintNormsOfParGridFunctions(normtype, myid, "Position", dcfx, &x_gf, true);
        PrintNormsOfParGridFunctions(normtype, myid, "Velocity", dcfv, &v_gf, true);
        PrintNormsOfParGridFunctions(normtype, myid, "Energy", dcfe, &e_gf, true);
    }

    if (rom_online)
    {
        delete basis;
        delete romOper;
    }

    switch (ode_solver_type)
    {
    case 2:
        steps *= 2;
        break;
    case 3:
        steps *= 3;
        break;
    case 4:
        steps *= 4;
        break;
    case 6:
        steps *= 6;
        break;
    case 7:
        steps *= 2;
    }
    oper.PrintTimingData(mpi.Root(), steps);

    const double energy_final = oper.InternalEnergy(e_gf) +
                                oper.KineticEnergy(v_gf);
    if (mpi.Root())
    {
        cout << endl;
        cout << "Energy  diff: " << scientific << setprecision(2)
             << fabs(energy_init - energy_final) << endl;

    }

    PrintParGridFunction(myid, outputPath + "/x_gf", &x_gf);
    PrintParGridFunction(myid, outputPath + "/v_gf", &v_gf);
    PrintParGridFunction(myid, outputPath + "/e_gf", &e_gf);

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

    totalTimer.Stop();
    if (mpi.Root()) {
        if(rom_online) cout << "Elapsed time for online preprocess: " << onlinePreprocessTimer.RealTime() << " sec\n";
        if(rom_restore) cout << "Elapsed time for restore phase: " << restoreTimer.RealTime() << " sec\n";
        if(rom_offline) cout << "Elapsed time for sampling in the offline phase: " << samplerTimer.RealTime() << " sec\n";
        cout << "Elapsed time for time loop: " << timeLoopTimer.RealTime() << " sec\n";
        cout << "Total time: " << totalTimer.RealTime() << " sec\n";
    }

    // Free the used memory.
    delete ode_solver;
    delete pmesh;
    delete mat_gf_coeff;

    return 0;
}

double rho0(const Vector &x)
{
    switch (problem)
    {
    case 0:
        return 1.0;
    case 1:
        return 1.0;
    case 2:
        return (x(0) < 0.5) ? 1.0 : 0.1;
    case 3:
        return (x(0) > 1.0 && x(1) > 1.5) ? 0.125 : 1.0;
    case 4:
        return 1.0;
    default:
        MFEM_ABORT("Bad number given for problem id!");
        return 0.0;
    }
}

double gamma(const Vector &x)
{
    switch (problem)
    {
    case 0:
        return 5./3.;
    case 1:
        return 1.4;
    case 2:
        return 1.4;
    case 3:
        return (x(0) > 1.0 && x(1) <= 1.5) ? 1.4 : 1.5;
    case 4:
        return 5.0 / 3.0;
    default:
        MFEM_ABORT("Bad number given for problem id!");
        return 0.0;
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
    case 1:
        v = 0.0;
        break;
    case 2:
        v = 0.0;
        break;
    case 3:
        v = 0.0;
        break;
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
        else {
            v = 0.0;
        }
        break;
    }
    default:
        MFEM_ABORT("Bad number given for problem id!");
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
    case 1:
        return 0.0; // This case in initialized in main().
    case 2:
        return (x(0) < 0.5) ? 1.0 / rho0(x) / (gamma(x) - 1.0)
               : 0.1 / rho0(x) / (gamma(x) - 1.0);
    case 3:
        return (x(0) > 1.0) ? 0.1 / rho0(x) / (gamma(x) - 1.0)
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
        else {
            return (3.0 + 4.0 * log(2.0)) / (gamma - 1.0);
        }
    }
    default:
        MFEM_ABORT("Bad number given for problem id!");
        return 0.0;
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
