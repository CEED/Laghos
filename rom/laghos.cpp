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
//    p = 7  --> 2D Rayleigh-Taylor instability problem.//
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

#include "algo/greedy/GreedyRandomSampler.h"
#include "algo/AdaptiveDMD.h"
#include "algo/NonuniformDMD.h"

#include "laghos_solver.hpp"
#include "laghos_timeinteg.hpp"
#include "laghos_rom.hpp"
#include "laghos_utils.hpp"
#include <fstream>
#include <limits.h>

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
static int problem, dim;
static double rhoRatio; // For Rayleigh-Taylor instability problem

double rho0(const Vector &);
void v0(const Vector &, Vector &);
double e0(const Vector &);
double gamma_func(const Vector &);
void display_banner(ostream & os);

int main(int argc, char *argv[])
{
    // Initialize MPI.
    MPI_Session mpi(argc, argv);
    int myid = mpi.WorldRank();
    int nprocs = mpi.WorldSize();

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
    const char *visit_basename = "Laghos";
    const char *basename = "";
    const char *twfile = "tw.csv";
    const char *twpfile = "twp.csv";
    const char *initSamples_basename = "initSamples";
    const char *basisIdentifier = "";
    int partition_type = 0;
    double blast_energy = 0.25;
    double blast_position[] = {0.0, 0.0, 0.0};
    double dt_factor = 1.0;
    bool rom_build_database = false;
    bool rom_use_database = false;
    bool rom_sample_stages = false;
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
    const char *normtype_char = "l2";
    const char *testing_parameter_basename = "";
    const char *hyperreductionSamplingType = "gnat";
    const char *spaceTimeMethod = "spatial";
    const char *offsetType = "initial";
    const char *indicatorType = "time";
    const char *greedyParam = "bef";
    const char *greedySamplingType = "random";
    const char *greedyErrorIndicatorType = "useLastLifted";
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
    args.AddOption(&testing_parameter_basename, "-pardir", "--param_dir",
                   "Name of the subdirectory containing testing parameter files");
    args.AddOption(&twfile, "-tw", "--timewindowfilename",
                   "Name of the CSV file defining offline time windows");
    args.AddOption(&twpfile, "-twp", "--timewindowparamfilename",
                   "Name of the CSV file defining online time window parameters");
    args.AddOption(&initSamples_basename, "-is", "--initsamplesfilename",
                   "Prefix of the CSV file defining prescribed sample points");
    args.AddOption(&partition_type, "-pt", "--partition",
                   "Customized x/y/z Cartesian MPI partitioning of the serial mesh.\n\t"
                   "Here x,y,z are relative task ratios in each direction.\n\t"
                   "Example: with 48 mpi tasks and -pt 321, one would get a Cartesian\n\t"
                   "partition of the serial mesh by (6,4,2) MPI tasks in (x,y,z).\n\t"
                   "NOTE: the serially refined mesh must have the appropriate number\n\t"
                   "of zones in each direction, e.g., the number of zones in direction x\n\t"
                   "must be divisible by the number of MPI tasks in direction x.\n\t"
                   "Available options: 11, 21, 111, 211, 221, 311, 321, 322, 432.");
    args.AddOption(&rom_build_database, "-build-database", "--build-database", "-no-build-database", "--no-build-database",
                   "Enable or disable ROM database building.");
    args.AddOption(&rom_use_database, "-use-database", "--use-database", "-no-use-database", "--no-use-database",
                   "Enable or disable ROM database usage.");
    args.AddOption(&rom_sample_stages, "-sample-stages", "--sample-stages", "-no-sample-stages", "--no-sample-stages",
                   "Enable or disable sampling of intermediate Runge Kutta stages in ROM offline phase.");
    args.AddOption(&rom_offline, "-offline", "--offline", "-no-offline", "--no-offline",
                   "Enable or disable ROM offline computations and output.");
    args.AddOption(&rom_online, "-online", "--online", "-no-online", "--no-online",
                   "Enable or disable ROM online computations and output.");
    args.AddOption(&rom_restore, "-restore", "--restore", "-no-restore", "--no-restore",
                   "Enable or disable ROM restoration phase where ROM solution is lifted to FOM size.");
    args.AddOption(&romOptions.dimX, "-rdimx", "--rom_dimx", "ROM dimension for X.\n\t"
                   "Ceiling ROM dimension for X over all time windows.");
    args.AddOption(&romOptions.dimV, "-rdimv", "--rom_dimv", "ROM dimension for V.\n\t"
                   "Ceiling ROM dimension for V over all time windows.");
    args.AddOption(&romOptions.dimE, "-rdime", "--rom_dime", "ROM dimension for E.\n\t"
                   "Ceiling ROM dimension for E over all time windows.");
    args.AddOption(&romOptions.dimFv, "-rdimfv", "--rom_dimfv", "ROM dimension for Fv.\n\t"
                   "Ceiling ROM dimension for Fv over all time windows.");
    args.AddOption(&romOptions.dimFe, "-rdimfe", "--rom_dimfe", "ROM dimension for Fe.\n\t"
                   "Ceiling ROM dimension for Fe over all time windows.");
    args.AddOption(&romOptions.sampX, "-nsamx", "--numsamplex", "number of samples for X.");
    args.AddOption(&romOptions.sampV, "-nsamv", "--numsamplev", "number of samples for V.");
    args.AddOption(&romOptions.sampE, "-nsame", "--numsamplee", "number of samples for E.");
    args.AddOption(&romOptions.tsampV, "-ntsamv", "--numtsamplev", "number of time samples for V.");
    args.AddOption(&romOptions.tsampE, "-ntsame", "--numtsamplee", "number of time samples for E.");
    args.AddOption(&sFactorX, "-sfacx", "--sfactorx", "sample factor for X.");
    args.AddOption(&sFactorV, "-sfacv", "--sfactorv", "sample factor for V.");
    args.AddOption(&sFactorE, "-sface", "--sfactore", "sample factor for E.");
    args.AddOption(&romOptions.energyFraction, "-ef", "--rom-ef",
                   "Energy fraction for recommended ROM basis sizes.");
    args.AddOption(&romOptions.energyFraction_X, "-efx", "--rom-efx",
                   "Energy fraction for recommended X ROM basis size.");
    args.AddOption(&romOptions.sv_shift, "-sv-shift", "--sv-shift",
                   "Number of shifted singular values in energy fraction calculation when window-dependent offsets are not used.");
    args.AddOption(&basisIdentifier, "-bi", "--bi", "Basis identifier for parametric case.");
    args.AddOption(&numWindows, "-nwin", "--numwindows", "Number of ROM time windows.");
    args.AddOption(&windowNumSamples, "-nwinsamp", "--numwindowsamples", "Number of samples in ROM windows.");
    args.AddOption(&windowOverlapSamples, "-nwinover", "--numwindowoverlap", "Number of samples for ROM window overlap.");
    args.AddOption(&dt_factor, "-dtFactor", "--dtFactor", "Scaling factor for dt.");
    args.AddOption(&dtc, "-dtc", "--dtc", "Fixed (constant) dt.");
    args.AddOption(&romOptions.dmd, "-dmd", "--dmd", "-dmd", "--dmd",
                   "Do DMD calculations.");
    args.AddOption(&romOptions.dmd_tbegin, "-dmdtbegin", "--dmdtbegin",
                   "Time to begin DMD. If DMD starts from t = 0, it will not work due to an zero initial vectors.");
    args.AddOption(&romOptions.desired_dt, "-ddt", "--dtime-step",
                   "Desired Time step.");
    args.AddOption(&romOptions.dmd_closest_rbf, "-dmdcrbf", "--dmdcrbf",
                   "DMD RBF value between two closes training parameter points.");
    args.AddOption(&romOptions.dmd_nonuniform, "-dmdnuf", "--dmdnuf", "-no-dmdnuf", "--no-dmdnuf",
                   "Use NonuniformDMD rather than AdaptiveDMD.");
    args.AddOption(&visitDiffCycle, "-visdiff", "--visdiff", "VisIt DC cycle to diff.");
    args.AddOption(&writeSol, "-writesol", "--writesol", "-no-writesol", "--no-writesol",
                   "Enable or disable write solution.");
    args.AddOption(&solDiff, "-soldiff", "--soldiff", "-no-soldiff", "--no-soldiff",
                   "Enable or disable solution difference norm computation.");
    args.AddOption(&romOptions.hyperreduce, "-romhr", "--romhr", "-no-romhr", "--no-romhr",
                   "Enable or disable ROM hyperreduction.");
    args.AddOption(&romOptions.hyperreduce_prep, "-romhrprep", "--romhrprep", "-no-romhrprep", "--no-romhrprep",
                   "Enable or disable ROM hyperreduction preprocessing.");
    args.AddOption(&romOptions.staticSVD, "-romsvds", "--romsvdstatic", "-no-romsvds", "--no-romsvds",
                   "Enable or disable ROM static SVD.");
    args.AddOption(&romOptions.randomizedSVD, "-romsvdrm", "--romsvdrandom", "-no-romsvdrm", "--no-romsvdrm",
                   "Enable or disable ROM randomized SVD.");
    args.AddOption(&romOptions.randdimX, "-randdimx", "--rand_dimx", "Randomized SVD subspace dimension for X.");
    args.AddOption(&romOptions.randdimV, "-randdimv", "--rand_dimv", "Randomized SVD subspace dimension for V.");
    args.AddOption(&romOptions.randdimE, "-randdime", "--rand_dime", "Randomized SVD subspace dimension for E.");
    args.AddOption(&romOptions.randdimFv, "-randdimfv", "--rand_dimfv", "Randomized SVD subspace dimension for Fv.");
    args.AddOption(&romOptions.randdimFe, "-randdimfe", "--rand_dimfe", "Randomized SVD subspace dimension for Fe.");
    args.AddOption(&romOptions.useOffset, "-romos", "--romoffset", "-no-romoffset", "--no-romoffset",
                   "Enable or disable initial state offset for ROM.");
    args.AddOption(&normtype_char, "-normtype", "--norm_type", "Norm type for relative error computation.");
    args.AddOption(&romOptions.max_dim, "-sdim", "--sdim", "ROM max sample dimension");
    args.AddOption(&romOptions.incSVD_linearity_tol, "-lintol", "--linearitytol", "The incremental SVD model linearity tolerance.");
    args.AddOption(&romOptions.incSVD_singular_value_tol, "-svtol", "--singularvaluetol", "The incremental SVD model singular value tolerance.");
    args.AddOption(&romOptions.incSVD_sampling_tol, "-samptol", "--samplingtol", "The incremental SVD model sampling tolerance.");
    args.AddOption(&greedyParam, "-greedy-param", "--greedy-param", "The domain to parameterize.");
    args.AddOption(&romOptions.greedyParamSpaceMin, "-greedy-param-min", "--greedy-param-min", "The minimum value of the parameter point space.");
    args.AddOption(&romOptions.greedyParamSpaceMax, "-greedy-param-max", "--greedy-param-max", "The maximum value of the parameter point space.");
    args.AddOption(&romOptions.greedyParamSpaceSize, "-greedy-param-size", "--greedy-param-size", "The number of values to search in the parameter point space.");
    args.AddOption(&romOptions.greedytf, "-greedytf", "--greedytf", "The greedy algorithm error indicator final time.");
    args.AddOption(&romOptions.greedyTol, "-greedytol", "--greedytol", "The greedy algorithm tolerance.");
    args.AddOption(&romOptions.greedyAlpha, "-greedyalpha", "--greedyalpha", "The greedy algorithm alpha constant.");
    args.AddOption(&romOptions.greedyMaxClamp, "-greedymaxclamp", "--greedymaxclamp", "The greedy algorithm max clamp constant.");
    args.AddOption(&romOptions.greedySubsetSize, "-greedysubsize", "--greedysubsize", "The greedy algorithm subset size.");
    args.AddOption(&romOptions.greedyConvergenceSubsetSize, "-greedyconvsize", "--greedyconvsize", "The greedy algorithm convergence subset size.");
    args.AddOption(&greedySamplingType, "-greedysamptype", "--greedysamplingtype",
                   "Sampling type for the greedy algorithm.");
    args.AddOption(&greedyErrorIndicatorType, "-greedyerrindtype", "--greedyerrorindtype",
                   "Error indicator type for the greedy algorithm.");
    args.AddOption(&romOptions.SNS, "-romsns", "--romsns", "-no-romsns", "--no-romsns",
                   "Enable or disable SNS in hyperreduction on Fv and Fe");
    args.AddOption(&romOptions.GramSchmidt, "-romgs", "--romgramschmidt", "-no-romgs", "--no-romgramschmidt",
                   "Enable or disable Gram-Schmidt orthonormalization on V and E induced by mass matrices.");
    args.AddOption(&romOptions.rhoFactor, "-rhof", "--rhofactor", "Factor for scaling rho.");
    args.AddOption(&romOptions.atwoodFactor, "-af", "--atwoodfactor", "Factor for Atwood number in Rayleigh-Taylor instability problem.");
    args.AddOption(&romOptions.blast_energyFactor, "-bef", "--blastefactor", "Factor for scaling blast energy.");
    args.AddOption(&romOptions.parameterID, "-rpar", "--romparam", "ROM offline parameter index.");
    args.AddOption(&offsetType, "-rostype", "--romoffsettype",
                   "Offset type for initializing ROM windows.");
    args.AddOption(&indicatorType, "-loctype", "--romindicatortype",
                   "Indicator type for partitioning ROM windows.");
    args.AddOption(&spaceTimeMethod, "-romst", "--romspacetimetype",
                   "Space-time method.");
    args.AddOption(&romOptions.useXV, "-romxv", "--romusexv", "-no-romxv", "--no-romusexv",
                   "Enable or disable use of V basis for X-X0.");
    args.AddOption(&romOptions.useVX, "-romvx", "--romusevx", "-no-romvx", "--no-romusevx",
                   "Enable or disable use of X-X0 basis for V.");
    args.AddOption(&romOptions.mergeXV, "-romxandv", "--romusexandv", "-no-romxandv", "--no-romusexandv",
                   "Enable or disable merging of X-X0 and V bases.");
    args.AddOption(&hyperreductionSamplingType, "-hrsamptype", "--hrsamplingtype",
                   "Sampling type for the hyperreduction.");
    args.AddOption(&romOptions.EQP, "-eqp", "--eqp", "-no-eqp", "--no-eqp",
                   "Enable EQP.");
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

    std::string testing_parameter_outputPath = outputPath;
    if (std::string(testing_parameter_basename) != "") {
        testing_parameter_outputPath += "/" + std::string(testing_parameter_basename);
    }

    std::string hyperreduce_outputPath = (problem == 7) ? testing_parameter_outputPath : outputPath;

    romOptions.basename = &outputPath;
    romOptions.testing_parameter_basename = &testing_parameter_outputPath;
    romOptions.hyperreduce_basename = &hyperreduce_outputPath;
    romOptions.initSamples_basename = std::string(initSamples_basename);

    if (mpi.Root()) {
        const char path_delim = '/';
        std::string::size_type pos = 0;
        do {
            pos = outputPath.find(path_delim, pos+1);
            std::string subdir = outputPath.substr(0, pos);
            mkdir(subdir.c_str(), 0777);
        }
        while (pos != std::string::npos);
        if (std::string(testing_parameter_basename) != "")
            mkdir(testing_parameter_outputPath.c_str(), 0777);
        mkdir((testing_parameter_outputPath + "/ROMoffset").c_str(), 0777);
        mkdir((testing_parameter_outputPath + "/ROMsol").c_str(), 0777);
    }

    MFEM_VERIFY(!(romOptions.useXV && romOptions.useVX), "");
    MFEM_VERIFY(!(romOptions.useXV && romOptions.mergeXV) && !(romOptions.useVX && romOptions.mergeXV), "");
    MFEM_VERIFY(!(romOptions.hyperreduce && romOptions.hyperreduce_prep), "");

    if (romOptions.useXV) romOptions.dimX = romOptions.dimV;
    if (romOptions.useVX) romOptions.dimV = romOptions.dimX;

    romOptions.basisIdentifier = std::string(basisIdentifier);

    romOptions.hyperreductionSamplingType = getHyperreductionSamplingType(hyperreductionSamplingType);
    romOptions.spaceTimeMethod = getSpaceTimeMethod(spaceTimeMethod);
    const bool spaceTime = (romOptions.spaceTimeMethod != no_space_time);

    const bool fom_data = spaceTime || !(rom_online && romOptions.hyperreduce);  // Whether to construct FOM data structures

    static std::map<std::string, NormType> localmap;
    localmap["l2"] = l2norm;
    localmap["l1"] = l1norm;
    localmap["max"] = maxnorm;

    NormType normtype = localmap[normtype_char];

    bool usingWindows = (numWindows > 0 || windowNumSamples > 0);

    CAROM::GreedySampler* parameterPointGreedySampler = NULL;
    bool rom_calc_error_indicator = false;
    bool rom_calc_rel_error = false;
    bool rom_calc_rel_error_nonlocal = false;
    bool rom_calc_rel_error_local = false;
    bool rom_calc_rel_error_nonlocal_completed = false;
    bool rom_calc_rel_error_local_completed = false;
    bool greedy_write_solution = false;
    bool rom_read_greedy_twparam = false;

    // If using the greedy algorithm, initialize the parameter point greedy sampler.
    if (rom_build_database)
    {
        MFEM_VERIFY(!rom_offline && !rom_online && !rom_restore, "-offline, -online, -restore should be off when using -build-database");
        parameterPointGreedySampler = BuildROMDatabase(romOptions, t_final, myid, outputPath, rom_offline, rom_online, rom_restore, usingWindows, rom_calc_error_indicator, rom_calc_rel_error_nonlocal, rom_calc_rel_error_local, rom_read_greedy_twparam, greedyParam, greedyErrorIndicatorType, greedySamplingType);

        rom_calc_rel_error = rom_calc_rel_error_local || rom_calc_rel_error_nonlocal;
        rom_calc_rel_error_nonlocal_completed = rom_calc_rel_error_nonlocal && (rom_restore || (rom_online && !romOptions.hyperreduce));
        rom_calc_rel_error_local_completed = rom_calc_rel_error_local && (rom_restore || (rom_online && !romOptions.hyperreduce));
        greedy_write_solution = rom_offline || rom_calc_rel_error_nonlocal_completed;

        if (rom_offline)
        {
            if (romOptions.parameterID != -1)
            {
                if (windowNumSamples > 0)
                {
                    windowNumSamples = 0;
                    usingWindows = false;
                }
            }
        }

        if (rom_online || rom_restore)
        {
            romOptions.parameterID = -1;
            if (usingWindows)
            {
                windowNumSamples = 0;
                numWindows = countNumLines(outputPath + "/" + std::string(twpfile) + romOptions.basisIdentifier);
            }
        }
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
        if (std::string(testing_parameter_basename) != "")
            mkdir(testing_parameter_outputPath.c_str(), 0777);
        mkdir((testing_parameter_outputPath + "/ROMoffset" + romOptions.basisIdentifier).c_str(), 0777);
        mkdir((testing_parameter_outputPath + "/ROMsol").c_str(), 0777);
    }

    // Use the ROM database to run the parametric case on another parameter point.
    if (rom_use_database)
    {
        MFEM_VERIFY(!rom_offline, "-offline should be off when -use-database is turned on");
        MFEM_VERIFY(!rom_build_database, "-build-database should be off when -use-database is turned on");
        parameterPointGreedySampler = UseROMDatabase(romOptions, myid, outputPath, greedyParam);
    }

    if (mpi.Root())
    {
        args.PrintOptions(cout);
    }

    MFEM_VERIFY(windowNumSamples == 0 || rom_offline, "-nwinsamp should be specified only in offline mode");
    MFEM_VERIFY(windowNumSamples == 0 || numWindows == 0, "-nwinsamp and -nwin cannot both be set");

    if (usingWindows)
    {
        if (romOptions.dimX  > 0) romOptions.max_dimX  = romOptions.dimX;
        if (romOptions.dimV  > 0) romOptions.max_dimV  = romOptions.dimV;
        if (romOptions.dimE  > 0) romOptions.max_dimE  = romOptions.dimE;
        if (romOptions.dimFv > 0) romOptions.max_dimFv = romOptions.dimFv;
        if (romOptions.dimFe > 0) romOptions.max_dimFe = romOptions.dimFe;
        if (rom_online || rom_restore)
        {
            double sFactor[]  = {sFactorX, sFactorV, sFactorE};
            const int err = ReadTimeWindowParameters(numWindows, outputPath + "/" + std::string(twpfile) + romOptions.basisIdentifier, twep, twparam, sFactor, myid == 0, romOptions.SNS);
            MFEM_VERIFY(err == 0, "Error in ReadTimeWindowParameters");
            if (rom_build_database && rom_read_greedy_twparam)
            {
                ReadGreedyTimeWindowParameters(romOptions, numWindows, twparam, outputPath);
            }
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
        if (romOptions.SNS)
        {
            romOptions.dimFv = max(romOptions.dimFv, romOptions.dimV);
            romOptions.dimFe = max(romOptions.dimFe, romOptions.dimE);
        }
    }

    if (windowNumSamples > 0) romOptions.max_dim = windowNumSamples + windowOverlapSamples + 2;
    MFEM_VERIFY(windowOverlapSamples >= 0, "Negative window overlap");
    MFEM_VERIFY(windowOverlapSamples <= windowNumSamples, "Too many ROM window overlap samples.");

    StopWatch totalTimer;
    totalTimer.Start();

    // Read the serial mesh from the given mesh file on all processors.
    // Refine the mesh in serial to increase the resolution.
    Mesh* mesh = NULL;
    dim = 0;
    if (fom_data)
    {
        mesh = new Mesh(mesh_file, 1, 1);
        dim = mesh->Dimension();
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
    }

    // Parallel partitioning of the mesh.
    ParMesh* pmesh = NULL;
    if (fom_data)
    {
        const int num_tasks = mpi.WorldSize();
        int unit = 1;
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
    }

    int source = 0;
    double dt = 0.0;

    std::string offlineParam_outputPath = outputPath + "/offline_param" + romOptions.basisIdentifier + ".csv";
    romOptions.offsetType = getOffsetStyle(offsetType);
    romOptions.indicatorType = getlocalROMIndicator(indicatorType);
    if (rom_online)
    {
        std::string filename = testing_parameter_outputPath + "/ROMsol/romS_1";
        std::ifstream infile_romS(filename.c_str());
        MFEM_VERIFY(!infile_romS.good(), "ROMsol files already exist.")
        VerifyOfflineParam(dim, dt, romOptions, numWindows, twfile, offlineParam_outputPath, false);
    }

    // Define the parallel finite element spaces. We use:
    // - H1 (Gauss-Lobatto, continuous) for position and velocity.
    // - L2 (Bernstein, discontinuous) for specific internal energy.
    L2_FECollection L2FEC(order_e, dim, BasisType::Positive);
    H1_FECollection H1FEC(order_v, dim);
    ParFiniteElementSpace* L2FESpace = NULL;
    ParFiniteElementSpace* H1FESpace = NULL;
    if (fom_data)
    {
        L2FESpace = new ParFiniteElementSpace(pmesh, &L2FEC);
        H1FESpace = new ParFiniteElementSpace(pmesh, &H1FEC, pmesh->Dimension());
    }

    // Boundary conditions: all tests use v.n = 0 on the boundary, and we assume
    // that the boundaries are straight.
    Array<int> ess_tdofs, ess_vdofs;

    if (fom_data)
    {
        {
            Array<int> ess_bdr(pmesh->bdr_attributes.Max()), dofs_marker, dofs_list;
            for (int d = 0; d < pmesh->Dimension(); d++)
            {
                // Attributes 1/2/3 correspond to fixed-x/y/z boundaries, i.e., we must
                // enforce v_x/y/z = 0 for the velocity components.
                ess_bdr = 0;
                ess_bdr[d] = 1;
                H1FESpace->GetEssentialTrueDofs(ess_bdr, dofs_list, d);
                ess_tdofs.Append(dofs_list);
                H1FESpace->GetEssentialVDofs(ess_bdr, dofs_marker, d);
                FiniteElementSpace::MarkerToList(dofs_marker, dofs_list);
                ess_vdofs.Append(dofs_list);
            }
        }
    }

    // Define the explicit ODE solver used for time integration.
    ODESolver *ode_solver = NULL;
    int RKStepNumSamples;
    ODESolver *ode_solver_dat = NULL;
    HydroODESolver *ode_solver_samp = NULL;
    switch (ode_solver_type)
    {
    case 1:
        rom_sample_stages = false;
        ode_solver = new ForwardEulerSolver;
        if (rom_build_database) ode_solver_dat = new ForwardEulerSolver;
        break;
    case 2:
        rom_sample_stages = false;
        ode_solver = new RK2Solver(0.5);
        if (rom_build_database) ode_solver_dat = new RK2Solver(0.5);
        break;
    case 3:
        rom_sample_stages = false;
        ode_solver = new RK3SSPSolver;
        if (rom_build_database) ode_solver_dat = new RK3SSPSolver;
        break;
    case 4:
        if (rom_sample_stages) RKStepNumSamples = 3;
        ode_solver = new RK4ROMSolver(rom_online, RKStepNumSamples);
        if (rom_build_database) ode_solver_dat = new RK4ROMSolver();
        break;
    case 6:
        rom_sample_stages = false;
        ode_solver = new RK6Solver;
        if (rom_build_database) ode_solver_dat = new RK6Solver;
        break;
    case 7:
        if (rom_sample_stages) RKStepNumSamples = 1;
        ode_solver = new RK2AvgSolver(rom_online, H1FESpace, L2FESpace, RKStepNumSamples);
        if (rom_build_database) ode_solver_dat = new RK2AvgSolver(rom_online, H1FESpace, L2FESpace);
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

    if (rom_sample_stages) ode_solver_samp = dynamic_cast<HydroODESolver*> (ode_solver);
    romOptions.RK2AvgSolver = (ode_solver_type == 7);

    if (fom_data)
    {

        HYPRE_Int glob_size_l2 = L2FESpace->GlobalTrueVSize();
        HYPRE_Int glob_size_h1 = H1FESpace->GlobalTrueVSize();

        if (mpi.Root())
        {
            cout << "Number of kinematic (position, velocity) dofs: "
                 << glob_size_h1 << endl;
            cout << "Number of specific internal energy dofs: "
                 << glob_size_l2 << endl;
        }
    }

    int Vsize_l2 = 0;
    int Vsize_h1 = 0;

    int tVsize_l2 = 0;
    int tVsize_h1 = 0;
    if (fom_data)
    {
        Vsize_l2 = L2FESpace->GetVSize();
        Vsize_h1 = H1FESpace->GetVSize();

        tVsize_l2 = L2FESpace->GetTrueVSize();
        tVsize_h1 = H1FESpace->GetTrueVSize();
    }

    // The monolithic BlockVector stores unknown fields as:
    // - 0 -> position
    // - 1 -> velocity
    // - 2 -> specific internal energy

    Array<int> true_offset(4);
    if (fom_data)
    {
        true_offset[0] = 0;
        true_offset[1] = true_offset[0] + Vsize_h1;
        true_offset[2] = true_offset[1] + Vsize_h1;
        true_offset[3] = true_offset[2] + Vsize_l2;
    }

    BlockVector* S = fom_data ? new BlockVector(true_offset) : NULL;

    // Define GridFunction objects for the position, velocity and specific
    // internal energy.  There is no function for the density, as we can always
    // compute the density values given the current mesh position, using the
    // property of pointwise mass conservation.
    ParGridFunction* x_gf = NULL;
    ParGridFunction* v_gf = NULL;
    ParGridFunction* e_gf = NULL;
    if (fom_data)
    {
        x_gf = new ParGridFunction();
        v_gf = new ParGridFunction();
        e_gf = new ParGridFunction();
        x_gf->MakeRef(H1FESpace, *S, true_offset[0]);
        v_gf->MakeRef(H1FESpace, *S, true_offset[1]);
        e_gf->MakeRef(L2FESpace, *S, true_offset[2]);

        // Initialize x_gf using the starting mesh coordinates.
        pmesh->SetNodalGridFunction(x_gf);
    }

    // Initialize the velocity.
    VectorFunctionCoefficient* v_coeff = NULL;
    if (fom_data)
    {
        v_coeff = new VectorFunctionCoefficient(pmesh->Dimension(), v0);
        v_gf->ProjectCoefficient(*v_coeff);
        for (int i = 0; i < ess_vdofs.Size(); i++)
        {
            (*v_gf)(ess_vdofs[i]) = 0.0;
        }
    }

    if (rom_offline) // Set VTos
    {
        Vector Vtdof(tVsize_h1);
        v_gf->GetTrueDofs(Vtdof);
        CAROM::Vector VtdofDist(Vtdof.GetData(), tVsize_h1, true, false);
        const double vnorm = VtdofDist.norm();
        romOptions.VTos = (vnorm == 0.0);
    }

    // Initialize density and specific internal energy values. We interpolate in
    // a non-positive basis to get the correct values at the dofs.  Then we do an
    // L2 projection to the positive basis in which we actually compute. The goal
    // is to get a high-order representation of the initial condition. Note that
    // this density is a temporary function and it will not be updated during the
    // time evolution.
    ParGridFunction* rho = NULL;
    rhoRatio = (1.0 + romOptions.atwoodFactor) / (1.0 - romOptions.atwoodFactor); // Rayleigh-Taylor initial density
    FunctionCoefficient rho_coeff0(rho0);
    ProductCoefficient rho_coeff(romOptions.rhoFactor, rho_coeff0);
    if (fom_data)
    {
        rho = new ParGridFunction(L2FESpace);
        L2_FECollection l2_fec(order_e, pmesh->Dimension());
        ParFiniteElementSpace l2_fes(pmesh, &l2_fec);
        ParGridFunction l2_rho(&l2_fes), l2_e(&l2_fes);
        l2_rho.ProjectCoefficient(rho_coeff);
        rho->ProjectGridFunction(l2_rho);
        if (problem == 1)
        {
            // For the Sedov test, we use a delta function at the origin.
            DeltaCoefficient e_coeff(blast_position[0], blast_position[1],
                                     blast_position[2], romOptions.blast_energyFactor*blast_energy);
            l2_e.ProjectCoefficient(e_coeff);
        }
        else
        {
            FunctionCoefficient e_coeff0(e0);
            ProductCoefficient e_coeff(1.0 / romOptions.rhoFactor, e_coeff0);
            l2_e.ProjectCoefficient(e_coeff);
        }
        e_gf->ProjectGridFunction(l2_e);
    }

    // Piecewise constant ideal gas coefficient over the Lagrangian mesh. The
    // gamma values are projected on a function that stays constant on the moving
    // mesh.
    L2_FECollection* mat_fec = NULL;
    ParFiniteElementSpace* mat_fes = NULL;
    ParGridFunction* mat_gf = NULL;
    FunctionCoefficient mat_coeff(gamma_func);
    GridFunctionCoefficient *mat_gf_coeff = NULL;
    if (fom_data)
    {
        mat_fec = new L2_FECollection(0, pmesh->Dimension());
        mat_fes = new ParFiniteElementSpace(pmesh, mat_fec);
        mat_gf = new ParGridFunction(mat_fes);
        mat_gf->ProjectCoefficient(mat_coeff);
        mat_gf_coeff = new GridFunctionCoefficient(mat_gf);
    }

    // Additional details, depending on the problem.
    bool visc = true, vort = false;
    switch (problem)
    {
    case 0:
        if (pmesh && pmesh->Dimension() == 2) {
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
    case 5:
        visc = true;
        break;
    case 6:
        visc = true;
        break;
    case 7:
        visc = true;
        vort = true;
        source = 2;
        break;
    default:
        MFEM_ABORT("Wrong problem specification!");
    }
    if (impose_visc) {
        visc = true;
    }

    // Finding kinematic DOF on the end of the interface
    // which measure penetration distance in 2D Rayleigh-Taylor instability (problem 7)
    int pd1_vdof = -1, pd2_vdof = -1;
    if (problem == 7)
    {
        std::string pd_idx_outPath = outputPath + "/pd_idx";
        if (fom_data)
        {
            for (int i = 0; i < Vsize_h1/2; ++i)
            {
                if ((*S)(i) == 0.0 && (*S)(Vsize_h1/2+i) == 0.0)
                    pd1_vdof = Vsize_h1/2+i;
                if ((*S)(i) == 0.5 && (*S)(Vsize_h1/2+i) == 0.0)
                    pd2_vdof = Vsize_h1/2+i;
                if (pd1_vdof >= 0 && pd2_vdof >= 0)
                    break;
            }
        }
    }

    LagrangianHydroOperator* oper = NULL;
    if (fom_data)
    {
        const bool noMvSolve = rom_online && romOptions.EQP;
        const bool noMeSolve = false;  // TODO: implement EQP for E equation.
        oper = new LagrangianHydroOperator(S->Size(), *H1FESpace, *L2FESpace,
                                           ess_tdofs, *rho, source, cfl, mat_gf_coeff,
                                           visc, vort, p_assembly, cg_tol, cg_max_iter, ftz_tol,
                                           H1FEC.GetBasisType(), noMvSolve, noMeSolve,
                                           rom_online && romOptions.EQP);
    }

    socketstream* vis_rho = NULL;
    socketstream* vis_v = NULL;
    socketstream* vis_e = NULL;
    if (fom_data && (!rom_build_database || !rom_online))
    {
        vis_rho = new socketstream();
        vis_v = new socketstream();
        vis_e = new socketstream();
    }
    char vishost[] = "localhost";
    int  visport   = 19916;

    ParGridFunction* rho_gf = NULL;
    double energy_init;
    if (fom_data)
    {
        rho_gf = new ParGridFunction();
        if (visualization || visit) {
            oper->ComputeDensity(*rho_gf);
        }

        energy_init = oper->InternalEnergy(*e_gf) +
                      oper->KineticEnergy(*v_gf);

        if (visualization && (!rom_build_database || !rom_online))
        {
            // Make sure all MPI ranks have sent their 'v' solution before initiating
            // another set of GLVis connections (one from each rank):
            MPI_Barrier(pmesh->GetComm());

            vis_rho->precision(8);
            vis_v->precision(8);
            vis_e->precision(8);

            int Wx = 0, Wy = 0; // window position
            const int Ww = 350, Wh = 350; // window size
            int offx = Ww+10; // window offsets

            if (problem != 0 && problem != 4)
            {
                VisualizeField(*vis_rho, vishost, visport, *rho_gf,
                               "Density", Wx, Wy, Ww, Wh);
            }

            Wx += offx;
            VisualizeField(*vis_v, vishost, visport, *v_gf,
                           "Velocity", Wx, Wy, Ww, Wh);
            Wx += offx;
            VisualizeField(*vis_e, vishost, visport, *e_gf,
                           "Specific Internal Energy", Wx, Wy, Ww, Wh);
        }
    }

    // Save data for VisIt visualization.
    string visit_outputName = outputPath + "/" + std::string(visit_basename);
    const char *visit_outputPath = visit_outputName.c_str();
    VisItDataCollection* visit_dc = NULL;
    if (fom_data && (!rom_build_database || !rom_online))
    {
        visit_dc = new VisItDataCollection(visit_outputPath, pmesh);
        if (visit)
        {
            if (rom_offline || rom_restore)
                visit_dc->RegisterField("Position",  x_gf);

            visit_dc->RegisterField("Density",  rho_gf);
            visit_dc->RegisterField("Velocity", v_gf);
            visit_dc->RegisterField("Specific Internal Energy", e_gf);
            visit_dc->SetCycle(0);
            visit_dc->SetTime(0.0);
            visit_dc->Save();
        }

        cout << myid << ": pmesh number of elements " << pmesh->GetNE() << endl;
    }

    romOptions.rank = myid;
    romOptions.H1FESpace = H1FESpace;
    romOptions.L2FESpace = L2FESpace;
    romOptions.window = 0;
    romOptions.FOMoper = oper;
    romOptions.restore = rom_restore;

    // Perform time-integration (looping over the time iterations, ti, with a
    // time-step dt). The object oper is of type LagrangianHydroOperator that
    // defines the Mult() method that is used by the time integrators.
    if (!rom_online) ode_solver->Init(*oper);
    if (fom_data)
    {
        oper->ResetTimeStepEstimate();
    }
    double t = 0.0, t_old, dt_old;
    bool use_dt_old = false;
    bool last_step = false;
    int steps = 0;
    int unique_steps = 0;

    BlockVector* S_old = NULL;

    if (fom_data)
    {
        dt = oper->GetTimeStepEstimate(*S) * dt_factor;
        S_old = new BlockVector(*S);
    }

    if (rom_offline)
    {
        int err_rostype;
        err_rostype = (romOptions.parameterID == -1 && romOptions.offsetType == interpolateOffset);
        MFEM_VERIFY(err_rostype == 0, "-rostype interpolate is not compatible with non-parametric ROM.");
        err_rostype = (romOptions.parameterID != -1 && romOptions.offsetType == saveLoadOffset);
        MFEM_VERIFY(err_rostype == 0, "-rostype load is not compatible with parametric ROM.");
        if (romOptions.parameterID != -1 && romOptions.offsetType == interpolateOffset && myid == 0)
        {
            ofstream basisIdFile;
            basisIdFile.open(outputPath + "/basisIdentifier.txt");
            basisIdFile << romOptions.basisIdentifier;
            basisIdFile.close();
        }
        WriteOfflineParam(dim, dt, romOptions, numWindows, twfile, offlineParam_outputPath, myid == 0);
    }

    // Perform time-integration (looping over the time iterations, ti, with a
    // time-step dt). The object oper is of type LagrangianHydroOperator that
    // defines the Mult() method that is used by the time integrators.
    if (!rom_online) ode_solver->Init(*oper);
    if (fom_data) oper->ResetTimeStepEstimate();

    StopWatch samplerTimer, basisConstructionTimer;
    DMD_Sampler *dmd_sampler = NULL;
    DMD_Sampler *dmd_samplerLast = NULL;
    ROM_Sampler *sampler = NULL;
    ROM_Sampler *samplerLast = NULL;
    std::ofstream outfile_twp, outfile_time;
    const bool outputTimes = rom_offline && spaceTime;
    const bool outputSpaceTimeSolution = rom_offline && spaceTime;
    const bool inputTimes = rom_online && spaceTime;
    const bool readTimes = rom_online && spaceTime;
    Array<int> cutoff(5);
    if (rom_offline)
    {
        if (dtc > 0.0) dt = dtc;

        samplerTimer.Start();
        if (usingWindows && romOptions.parameterID == -1) {
            outfile_twp.open(outputPath + "/" + std::string(twpfile) + romOptions.basisIdentifier);
        }
        const double tf = (usingWindows && windowNumSamples == 0) ? twep[0] : t_final;
        romOptions.t_final = tf;
        romOptions.initial_dt = dt;
        if (romOptions.dmd)
        {
            dmd_sampler = new DMD_Sampler(romOptions, *S);
            dmd_sampler->SampleSolution(0, 0, *S);
        }
        else
        {
            sampler = new ROM_Sampler(romOptions, *S);
            sampler->SampleSolution(0, 0, (problem == 7) ? 0.0 : -1.0, *S);
        }
        samplerTimer.Stop();
    }

    if (outputTimes)
    {
        outfile_time.open(outputPath + "/timesteps.csv");
        outfile_time.precision(16);
    }

    std::ofstream ofs_STX, ofs_STV, ofs_STE;
    if (outputSpaceTimeSolution)
    {
        // TODO: output FOM solution at every timestep, including initial state at t=0.
        char fileExtension[100];
        sprintf(fileExtension, ".%06d", myid);

        std::string fullname = testing_parameter_outputPath + "/ST_Sol_Position" + fileExtension;
        ofs_STX.open(fullname.c_str(), std::ofstream::out);
        ofs_STX.precision(16);

        fullname = testing_parameter_outputPath + "/ST_Sol_Velocity" + fileExtension;
        ofs_STV.open(fullname.c_str(), std::ofstream::out);
        ofs_STV.precision(16);

        fullname = testing_parameter_outputPath + "/ST_Sol_Energy" + fileExtension;
        ofs_STE.open(fullname.c_str(), std::ofstream::out);
        ofs_STE.precision(16);

        AppendPrintParGridFunction(&ofs_STX, x_gf);
        AppendPrintParGridFunction(&ofs_STV, v_gf);
        AppendPrintParGridFunction(&ofs_STE, e_gf);
    }

    std::vector<double> timesteps;  // Used only for online space-time case.
    if (inputTimes)
    {
        const int err = ReadTimesteps(outputPath, timesteps);
        MFEM_VERIFY(err == 0, "Error in ReadTimesteps");
    }

    std::vector<ROM_Basis*> basis;
    basis.assign(std::max(numWindows, 1), nullptr);
    Vector romS, romS_old, lastLiftedSolution;
    std::vector<ROM_Operator*> romOper;
    romOper.assign(std::max(numWindows, 1), nullptr);
    std::vector<double> pd_weight;

    if (!usingWindows)
    {
        if (romOptions.sampX == 0 && !romOptions.mergeXV) romOptions.sampX = sFactorX * romOptions.dimX;
        if (romOptions.sampV == 0 && !romOptions.mergeXV) romOptions.sampV = sFactorV * romOptions.dimFv;
        if (romOptions.sampE == 0) romOptions.sampE = sFactorE * romOptions.dimFe;
    }

    StopWatch onlinePreprocessTimer;
    if (rom_online)
    {
        onlinePreprocessTimer.Start();
        if (dtc > 0.0) dt = dtc;
        if (usingWindows)
        {
            // Construct the ROM_Basis for each window.
            for (romOptions.window = numWindows-1; romOptions.window >= 0; --romOptions.window)
            {
                SetWindowParameters(twparam, romOptions);
                basis[romOptions.window] = new ROM_Basis(romOptions, MPI_COMM_WORLD, sFactorX, sFactorV);
                if (!romOptions.hyperreduce_prep)
                {
                    romOper[romOptions.window] = new ROM_Operator(romOptions, basis[romOptions.window], rho_coeff, mat_coeff, order_e, source,
                            visc, vort, cfl, p_assembly, cg_tol, cg_max_iter, ftz_tol, &H1FEC, &L2FEC);
                }
            }
            romOptions.window = 0;
        }
        else
        {
            basis[0] = new ROM_Basis(romOptions, MPI_COMM_WORLD, sFactorX, sFactorV, &timesteps);
            if (!romOptions.hyperreduce_prep)
            {
                romOper[0] = new ROM_Operator(romOptions, basis[0], rho_coeff, mat_coeff, order_e, source, visc, vort, cfl, p_assembly,
                                              cg_tol, cg_max_iter, ftz_tol, &H1FEC, &L2FEC, &timesteps);
            }
        }

        if (!romOptions.hyperreduce)
        {
            basis[0]->Init(romOptions, *S);
        }

        if (romOptions.hyperreduce_prep)
        {
            if (myid == 0)
            {
                cout << "Writing SP files for window: 0" << endl;
                basis[0]->writeSP(romOptions, 0);
            }
            for (int curr_window = 1; curr_window < numWindows; curr_window++)
            {
                basis[curr_window]->Init(romOptions, *S);
                basis[curr_window]->computeWindowProjection(*basis[curr_window - 1], romOptions, curr_window);
                if (myid == 0)
                {
                    cout << "Writing SP files for window: " << curr_window << endl;
                    basis[curr_window]->writeSP(romOptions, curr_window);
                }
            }
        }

        if (romOptions.mergeXV)
        {
            romOptions.dimX = basis[0]->GetDimX();
            romOptions.dimV = basis[0]->GetDimV();
        }

        romS.SetSize(romOptions.dimX + romOptions.dimV + romOptions.dimE);

        if (!romOptions.hyperreduce)
        {
            basis[0]->ProjectFOMtoROM(*S, romS);
            if (romOptions.hyperreduce_prep && myid == 0)
            {
                std::string romS_outPath = testing_parameter_outputPath + "/" + "romS" + "_0";
                std::ofstream outfile_romS(romS_outPath.c_str());
                outfile_romS.precision(16);
                romS.Print(outfile_romS, 1);
            }
        }
        else
        {
            std::string romS_outPath = testing_parameter_outputPath + "/" + "romS" + "_0";
            std::ifstream outfile_romS(romS_outPath.c_str());
            romS.Load(outfile_romS, romS.Size());
        }

        if (myid == 0)
        {
            cout << "Offset Style: " << offsetType << endl;
            cout << "Indicator Style: " << indicatorType << endl;
            cout << "Window " << romOptions.window << ": initial romS norm " << romS.Norml2() << endl;
        }

        if (rom_online && problem == 7 && romOptions.indicatorType == penetrationDistance)
        {
            if (!romOptions.hyperreduce)
            {
                int pd2_tdof = (pd2_vdof >= 0) ? H1FESpace->GetLocalTDofNumber(pd2_vdof) : -1;
                for (int curr_window = numWindows-1; curr_window >= 0; --curr_window)
                    basis[curr_window]->writePDweights(pd2_tdof, curr_window);
            }
            if (!romOptions.hyperreduce_prep)
            {
                std::string pd_weight_outputPath = testing_parameter_outputPath + "/pd_weight0";
                ReadPDweight(pd_weight, pd_weight_outputPath);
                if (myid == 0)
                {
                    MFEM_VERIFY(pd_weight.size() == basis[0]->GetDimX()+romOptions.useOffset, "Number of weights do not match.");
                }
            }
        }

        if (romOptions.hyperreduce_prep)
        {
            if (myid == 0)
            {
                cout << "Hyperreduction pre-processing completed. " << endl;
                if (rom_build_database)
                {
                    WriteGreedyPhase(rom_offline, rom_online, rom_restore, rom_calc_rel_error_nonlocal, rom_calc_rel_error_local, romOptions, outputPath + "/greedy_algorithm_stage.txt");
                }
            }
            return 0;
        }

        ode_solver->Init(*romOper[0]);
        onlinePreprocessTimer.Stop();
    }

    StopWatch restoreTimer, timeLoopTimer;
    bool greedy_converged = true;
    int ti;

    CAROM::DMD* dmd_X = NULL;
    CAROM::DMD* dmd_V = NULL;
    CAROM::DMD* dmd_E = NULL;

    CAROM::Vector* result_X = NULL;
    CAROM::Vector* result_V = NULL;
    CAROM::Vector* result_E = NULL;
    if (rom_restore)
    {
        if (romOptions.dmd)
        {
            restoreTimer.Start();
            std::string filename = outputPath + "/timeSamples.csv";
            std::ifstream infile_romS(filename.c_str());
            MFEM_VERIFY(infile_romS.good(), "timeSamples.csv can not be opened.")
            int prev_window = -1;
            int curr_window;
            double curr_time;
            double window_start_time;
            while (infile_romS >> curr_window >> curr_time)
            {
                if (prev_window != curr_window)
                {
                    if (curr_window != 0)
                    {
                        delete dmd_X;
                        delete dmd_V;
                        delete dmd_E;
                    }

                    if (romOptions.dmd_nonuniform)
                    {
                        dmd_X = new CAROM::NonuniformDMD(outputPath + "/" + "dmdX" + romOptions.basisIdentifier + "_" + to_string(curr_window));
                        dmd_V = new CAROM::NonuniformDMD(outputPath + "/" + "dmdV" + romOptions.basisIdentifier + "_" + to_string(curr_window));
                        dmd_E = new CAROM::NonuniformDMD(outputPath + "/" + "dmdE" + romOptions.basisIdentifier + "_" + to_string(curr_window));
                    }
                    else
                    {
                        dmd_X = new CAROM::DMD(outputPath + "/" + "dmdX" + romOptions.basisIdentifier + "_" + to_string(curr_window));
                        dmd_V = new CAROM::DMD(outputPath + "/" + "dmdV" + romOptions.basisIdentifier + "_" + to_string(curr_window));
                        dmd_E = new CAROM::DMD(outputPath + "/" + "dmdE" + romOptions.basisIdentifier + "_" + to_string(curr_window));
                    }

                    prev_window = curr_window;
                    window_start_time = curr_time;

                    // Skip first sample of each subsequent window except the first.
                    if (curr_window != 0)
                    {
                        continue;
                    }
                }

                if (result_X != NULL) delete result_X;
                if (result_V != NULL) delete result_V;
                if (result_E != NULL) delete result_E;

                if (myid == 0) cout << "Predicting time t " << curr_time << " using DMD window " << curr_window << " with initial start time " << window_start_time << std::endl;

                result_X = dmd_X->predict(curr_time);
                result_V = (romOptions.useVX && romOptions.dmd_nonuniform) ? dmd_X->predict(curr_time, 1) : dmd_V->predict(curr_time);
                result_E = dmd_E->predict(curr_time);
                Vector m_result_X(result_X->getData(), result_X->dim());
                Vector m_result_V(result_V->getData(), result_V->dim());
                Vector m_result_E(result_E->getData(), result_E->dim());
                x_gf->SetFromTrueDofs(m_result_X);
                v_gf->SetFromTrueDofs(m_result_V);
                e_gf->SetFromTrueDofs(m_result_E);

                if (visit)
                {
                    oper->ComputeDensity(*rho_gf);
                    visit_dc->SetCycle(ti);
                    visit_dc->SetTime(t);
                    visit_dc->Save();
                }
            }
            restoreTimer.Stop();
        }
        else
        {
            // -restore phase
            // No need to specify t_final because the loop in -restore phase is determined by the files in ROMsol folder.
            // When -romhr or --romhr are used in -online phase, then -restore phase needs to be called to project rom solution back to FOM size
            std::ifstream infile_tw_steps(testing_parameter_outputPath + "/tw_steps");
            int nb_step(0);
            restoreTimer.Start();
            if (usingWindows) {
                SetWindowParameters(twparam, romOptions);
            }

            basis[0] = new ROM_Basis(romOptions, MPI_COMM_WORLD, sFactorX, sFactorV);
            basis[0]->Init(romOptions, *S);

            if (romOptions.mergeXV)
            {
                romOptions.dimX = basis[0]->GetDimX();
                romOptions.dimV = basis[0]->GetDimV();
            }

            int romSsize = romOptions.dimX + romOptions.dimV + romOptions.dimE;
            romS.SetSize(romSsize);
            if (infile_tw_steps.good())
            {
                infile_tw_steps >> nb_step;
            }
            for (ti = 1; !last_step; ti++)
            {
                // romS = readCurrentReduceSol(ti);
                // read ROM solution from a file.
                // TODO: it needs to be read from the format of HDF5 format
                // TODO: how about parallel version? introduce rank in filename
                std::string filename = testing_parameter_outputPath + "/ROMsol/romS_" + std::to_string(ti);
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
                        basis[romOptions.window]->LiftROMtoFOM(romS, *S);

                        if (visit)
                        {
                            oper->ComputeDensity(*rho_gf);
                            visit_dc->SetCycle(ti);
                            visit_dc->SetTime(t);
                            visit_dc->Save();
                        }
                    }
                }
                else
                {
                    // exit the loop when no more files are found
                    last_step = true;
                    break;
                }
                if (ti == nb_step) {
                    if (infile_tw_steps.good())
                    {
                        infile_tw_steps >> nb_step;
                    }
                    romOptions.window++;
                    SetWindowParameters(twparam, romOptions);
                    basis[romOptions.window-1]->LiftROMtoFOM(romS, *S);
                    delete basis[romOptions.window-1];
                    basis[romOptions.window] = new ROM_Basis(romOptions, MPI_COMM_WORLD, sFactorX, sFactorV);
                    basis[romOptions.window]->Init(romOptions, *S);

                    if (romOptions.mergeXV)
                    {
                        romOptions.dimX = basis[romOptions.window]->GetDimX();
                        romOptions.dimV = basis[romOptions.window]->GetDimV();
                    }

                    romSsize = romOptions.dimX + romOptions.dimV + romOptions.dimE;
                    romS.SetSize(romSsize);
                }

                if (rom_build_database && !rom_calc_rel_error && romOptions.greedyErrorIndicatorType == useLastLiftedSolution)
                {
                    std::string nextfilename = testing_parameter_outputPath + "/ROMsol/romS_" + std::to_string(ti + 1);
                    std::string next2filename = testing_parameter_outputPath + "/ROMsol/romS_" + std::to_string(ti + 2);
                    std::ifstream nextfile_romS(nextfilename.c_str());
                    std::ifstream next2file_romS(next2filename.c_str());
                    if (nextfile_romS.good() && !next2file_romS.good())
                    {
                        lastLiftedSolution = *S;
                        ode_solver_dat->Init(*oper);
                        ode_solver_dat->Step(lastLiftedSolution, t, dt);
                    }
                    nextfile_romS.close();
                    next2file_romS.close();
                }
            } // time loop in "restore" phase
            ti--;
            std::string filename = testing_parameter_outputPath + "/ROMsol/romS_" + std::to_string(ti);
            std::ifstream infile_romS(filename.c_str());
            if (myid == 0)
                cout << "Restoring " << ti << "-th solution" << endl;
            for (int k=0; k<romSsize; ++k)
            {
                infile_romS >> romS(k);
            }
            infile_romS.close();
            basis[romOptions.window]->LiftROMtoFOM(romS, *S);

            if (visit)
            {
                oper->ComputeDensity(*rho_gf);
                visit_dc->SetCycle(ti);
                visit_dc->SetTime(t);
                visit_dc->Save();
            }
            restoreTimer.Stop();
            infile_tw_steps.close();
        }
    }
    else if (rom_online && spaceTime)
    {
        if (myid == 0)
            romOper[0]->SolveSpaceTimeGN(romS);

        MPI_Bcast(romS.GetData(), romS.Size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    else
    {
        // Usual time loop when not in restore or online space-time phase.
        std::ofstream outfile_tw_steps;
        if (rom_online && usingWindows)
        {
            outfile_tw_steps.open(testing_parameter_outputPath + "/tw_steps");
        }
        timeLoopTimer.Start();
        if (romOptions.hyperreduce)
        {
            romOper[0]->ApplyHyperreduction(romS);
        }
        double windowEndpoint = 0.0;
        double windowOverlapMidpoint = 0.0;
        for (ti = 1; !last_step; ti++)
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

            if (rom_online && usingWindows && ((t + dt) >= twep[romOptions.window]) && match_end_time)
            {
                dt_old = dt;
                use_dt_old = true;
                dt = twep[romOptions.window] - t;
            }

            if (steps == max_tsteps) {
                last_step = true;
            }

            if (!rom_online || !romOptions.hyperreduce) *S_old = *S;
            t_old = t;
            if (fom_data)
            {
                oper->ResetTimeStepEstimate();
            }

            // S is the vector of dofs, t is the current time, and dt is the time step
            // to advance.
            if (rom_online)
            {
                if (myid == 0)
                    cout << "ROM online at t " << t << ", dt " << dt << ", romS norm " << romS.Norml2() << endl;

                // MFEM may not converge if the ROM is too far away from the point
                // we are now trying to obtain an error indicator at.
                // This kills the simulation if it does not converge,
                // since romS will be full of NaN's.
                if (rom_build_database && !std::isfinite(romS.Norml2()))
                {
                    greedy_converged = false;
                    if (rom_calc_rel_error_local)
                    {
                        MFEM_ABORT("The greedy algorithm has failed. The local ROM did not converge. Decrease your parameter space range.")
                    }
                    rom_calc_rel_error_nonlocal_completed = rom_calc_rel_error_nonlocal;
                    break;
                }
                romS_old = romS;
                ode_solver->Step(romS, t, dt);

                // save ROM solution to a file.
                // TODO: it needs to be save in the format of HDF5 format
                // TODO: how about parallel version? introduce rank in filename
                // TODO: think about how to reuse "gfprint" option
                std::string filename = testing_parameter_outputPath + "/ROMsol/romS_" + std::to_string(ti);
                std::ofstream outfile_romS(filename.c_str());
                outfile_romS.precision(16);
                if (romOptions.hyperreduce && romOptions.GramSchmidt)
                {
                    Vector romCoord(romS);
                    romOper[romOptions.window]->PostprocessHyperreduction(romCoord, true);
                    romCoord.Print(outfile_romS, 1);
                }
                else
                {
                    romS.Print(outfile_romS, 1);
                }
                outfile_romS.close();

                if (!romOptions.hyperreduce)
                {
                    basis[romOptions.window]->LiftROMtoFOM(romS, *S);

                    // If using the greedy algorithm, take only the last step in the FOM space
                    // when using the useLastLiftedSolution error indicator type
                    if (rom_build_database && !rom_calc_rel_error && last_step && romOptions.greedyErrorIndicatorType == useLastLiftedSolution)
                    {
                        lastLiftedSolution = *S;
                        ode_solver_dat->Init(*oper);
                        ode_solver_dat->Step(lastLiftedSolution, t, dt);
                    }
                }

                romOper[romOptions.window]->UpdateSampleMeshNodes(romS);

                if (fom_data)
                {
                    oper->ResetQuadratureData();  // Necessary for oper->GetTimeStepEstimate(*S);
                }
            }
            else
            {
                if (myid == 0)
                    cout << "FOM simulation at t " << t << ", dt " << dt << endl;

                ode_solver->Step(*S, t, dt);
            }

            steps++;

            const double last_dt = dt;

            // Adaptive time step control.
            const double dt_est = romOptions.hyperreduce ? romOper[romOptions.window]->GetTimeStepEstimateSP() : oper->GetTimeStepEstimate(*S);

            if (dt_est < dt)
            {
                // Repeat (solve again) with a decreased time step - decrease of the
                // time estimate suggests appearance of oscillations.
                dt *= 0.85;
                if (dt < 1e-7)
                {
                    if (rom_build_database)
                    {
                        greedy_converged = false;
                        if (rom_calc_rel_error_local)
                        {
                            MFEM_ABORT("The greedy algorithm has failed. The local ROM did not converge. Decrease your parameter space range.")
                        }
                        rom_calc_rel_error_nonlocal_completed = rom_calc_rel_error_nonlocal;
                        break;
                    }
                }
                if (dt < numeric_limits<double>::epsilon())
                {
                    MFEM_ABORT("The time step crashed!");
                }
                t = t_old;
                if (!rom_online || !romOptions.hyperreduce) *S = *S_old;
                if (rom_online) romS = romS_old;
                if (fom_data)
                {
                    oper->ResetQuadratureData();
                }
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

            unique_steps++;

            if (outputTimes) outfile_time << t << "\n";

            if (outputSpaceTimeSolution)
            {
                // TODO: time this?
                AppendPrintParGridFunction(&ofs_STX, x_gf);
                AppendPrintParGridFunction(&ofs_STV, v_gf);
                AppendPrintParGridFunction(&ofs_STE, e_gf);
            }

            if (rom_offline)
            {
                timeLoopTimer.Stop();
                samplerTimer.Start();

                double real_pd;
                if (rom_sample_stages)
                {
                    std::vector<Vector>& RKStages = ode_solver_samp->GetRKStages();
                    std::vector<double>& RKTime = ode_solver_samp->GetRKTime();
                    MFEM_VERIFY(RKStages.size() == RKStepNumSamples, "Inconsistent number of Runge Kutta stages.");
                    for (int RKidx = 0; RKidx < RKStepNumSamples; ++RKidx)
                    {
                        real_pd = -1.0;
                        if (problem == 7)
                        {
                            // 2D Rayleigh-Taylor penetration distance
                            if (romOptions.indicatorType == penetrationDistance)
                            {
                                double proc_pd = (pd2_vdof >= 0) ? -RKStages[RKidx](pd2_vdof) : 0.0;
                                MPI_Reduce(&proc_pd, &real_pd, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                                MFEM_VERIFY(myid > 0 || real_pd >= 0.0, "Incorrect computation of penetration distance");
                            }
                            else if (romOptions.indicatorType == parameterTime)
                            {
                                real_pd = romOptions.atwoodFactor * RKTime[RKidx] * RKTime[RKidx];
                            }
                        }
                        if (romOptions.dmd)
                        {
                            dmd_sampler->SampleSolution(RKTime[RKidx], last_dt, RKStages[RKidx]);
                            if (dmd_samplerLast) dmd_samplerLast->SampleSolution(RKTime[RKidx], last_dt, RKStages[RKidx]);
                        }
                        else
                        {
                            sampler->SampleSolution(RKTime[RKidx], last_dt, real_pd, RKStages[RKidx]);
                            if (samplerLast) samplerLast->SampleSolution(RKTime[RKidx], last_dt, real_pd, RKStages[RKidx]);
                        }
                        if (mpi.Root()) cout << "Runge-Kutta stage " << RKidx+1 << " sampled" << endl;
                    }
                }

                real_pd = -1.0;
                if (problem == 7)
                {
                    // 2D Rayleigh-Taylor penetration distance
                    if (romOptions.indicatorType == penetrationDistance)
                    {
                        double proc_pd = (pd2_vdof >= 0) ? -(*S)(pd2_vdof) : 0.0;
                        MPI_Reduce(&proc_pd, &real_pd, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                        MFEM_VERIFY(myid > 0 || real_pd >= 0.0, "Incorrect computation of penetration distance");
                    }
                    else if (romOptions.indicatorType == parameterTime)
                    {
                        real_pd = romOptions.atwoodFactor * t * t;
                    }
                }
                if (romOptions.dmd)
                {
                    dmd_sampler->SampleSolution(t, last_dt, *S);
                }
                else
                {
                    sampler->SampleSolution(t, last_dt, real_pd, *S);
                }

                bool endWindow = false;
                if (usingWindows)
                {
                    if (numWindows > 0)
                    {
                        endWindow = (t >= twep[romOptions.window] && romOptions.window < numWindows-1);
                    }
                    else
                    {
                        if (romOptions.dmd)
                        {
                            endWindow = (dmd_sampler->MaxNumSamples() >= windowNumSamples);
                        }
                        else
                        {
                            endWindow = (sampler->MaxNumSamples() >= windowNumSamples);
                        }
                    }
                }

                if (romOptions.dmd)
                {
                    if (dmd_samplerLast)
                    {
                        dmd_samplerLast->SampleSolution(t, last_dt, *S);
                        if (dmd_samplerLast->MaxNumSamples() == windowNumSamples + (windowOverlapSamples/2))
                            windowOverlapMidpoint = (romOptions.indicatorType == physicalTime) ? t : real_pd;

                        if (dmd_samplerLast->MaxNumSamples() >= windowNumSamples + windowOverlapSamples || last_step)
                        {
                            dmd_samplerLast->Finalize(romOptions);
                            if (last_step)
                            {
                                // Let samplerLast define the final window, discarding the sampler window.
                                windowOverlapMidpoint = (romOptions.indicatorType == physicalTime) ? t : real_pd;
                                dmd_sampler = NULL;
                            }

                            MFEM_VERIFY(windowOverlapMidpoint > 0.0, "Overlapping window endpoint undefined.");
                            if (myid == 0 && romOptions.parameterID == -1) {
                                outfile_twp << windowOverlapMidpoint << ", " << cutoff[0] << ", " << cutoff[1] << ", " << cutoff[2];
                                if (romOptions.SNS)
                                    outfile_twp << "\n";
                                else
                                    outfile_twp << ", " << cutoff[3] << ", " << cutoff[4] << "\n";
                            }
                            delete dmd_samplerLast;
                            dmd_samplerLast = NULL;
                            windowOverlapMidpoint = 0.0;
                        }
                    }
                }
                else
                {
                    if (samplerLast)
                    {
                        samplerLast->SampleSolution(t, last_dt, real_pd, *S);
                        if (samplerLast->MaxNumSamples() == windowNumSamples + (windowOverlapSamples/2))
                            windowOverlapMidpoint = (romOptions.indicatorType == physicalTime) ? t : real_pd;

                        if (samplerLast->MaxNumSamples() >= windowNumSamples + windowOverlapSamples || last_step)
                        {
                            samplerLast->Finalize(cutoff, romOptions);
                            if (last_step)
                            {
                                // Let samplerLast define the final window, discarding the sampler window.
                                windowOverlapMidpoint = (romOptions.indicatorType == physicalTime) ? t : real_pd;
                                sampler = NULL;
                            }

                            MFEM_VERIFY(windowOverlapMidpoint > 0.0, "Overlapping window endpoint undefined.");
                            if (myid == 0 && romOptions.parameterID == -1) {
                                outfile_twp << windowOverlapMidpoint << ", " << cutoff[0] << ", " << cutoff[1] << ", " << cutoff[2];
                                if (romOptions.SNS)
                                    outfile_twp << "\n";
                                else
                                    outfile_twp << ", " << cutoff[3] << ", " << cutoff[4] << "\n";
                            }
                            delete samplerLast;
                            samplerLast = NULL;
                            windowOverlapMidpoint = 0.0;
                        }
                    }
                }

                if (endWindow)
                {
                    windowEndpoint = (romOptions.indicatorType == physicalTime) ? t : real_pd;
                    if (numWindows == 0 && windowOverlapSamples > 0)
                    {
                        if (romOptions.dmd)
                        {
                            dmd_samplerLast = dmd_sampler;
                        }
                        else
                        {
                            samplerLast = sampler;
                        }
                    }
                    else
                    {
                        if (romOptions.dmd)
                        {
                            dmd_sampler->Finalize(romOptions);
                        }
                        else
                        {
                            sampler->Finalize(cutoff, romOptions);
                        }
                        if (myid == 0 && romOptions.parameterID == -1) {
                            outfile_twp << windowEndpoint << ", " << cutoff[0] << ", " << cutoff[1] << ", " << cutoff[2];
                            if (romOptions.SNS)
                                outfile_twp << "\n";
                            else
                                outfile_twp << ", " << cutoff[3] << ", " << cutoff[4] << "\n";
                        }
                        if (romOptions.dmd)
                        {
                            delete dmd_sampler;
                        }
                        else
                        {
                            delete sampler;
                        }
                    }

                    romOptions.window++;
                    if (!last_step)
                    {
                        romOptions.t_final = (usingWindows && windowNumSamples == 0) ? twep[romOptions.window] : t_final;
                        romOptions.initial_dt = dt;
                        romOptions.window = romOptions.window;

                        if (romOptions.dmd)
                        {
                            dmd_sampler = new DMD_Sampler(romOptions, *S);
                            dmd_sampler->SampleSolution(t, dt, *S);
                        }
                        else
                        {
                            sampler = new ROM_Sampler(romOptions, *S);
                            sampler->SampleSolution(t, dt, real_pd, *S);
                        }
                    }
                    else
                    {
                        if (romOptions.dmd)
                        {
                            dmd_sampler = NULL;
                        }
                        else
                        {
                            sampler = NULL;
                        }
                    }
                }
                samplerTimer.Stop();
                timeLoopTimer.Start();
            }

            if (rom_online)
            {
                double window_par = t;
                if (problem == 7)
                {
                    if (romOptions.indicatorType == penetrationDistance)
                    {
                        // 2D Rayleigh-Taylor penetration distance
                        window_par = (romOptions.useOffset) ? -pd_weight.back() : 0.0;
                        for (int i=0; i<basis[romOptions.window]->GetDimX(); ++i)
                            window_par -= pd_weight[i]*romS[i];
                    }
                    else if (romOptions.indicatorType == parameterTime)
                    {
                        window_par = romOptions.atwoodFactor * t * t;
                    }
                }

                if (usingWindows && window_par >= twep[romOptions.window] && romOptions.window < numWindows-1)
                {
                    romOptions.window++;
                    outfile_tw_steps << ti << "\n";

                    if (myid == 0)
                        cout << "ROM online basis change for window " << romOptions.window << " at t " << t << ", dt " << dt << endl;

                    if (romOptions.hyperreduce)
                    {
                        romOper[romOptions.window-1]->PostprocessHyperreduction(romS);
                    }

                    int rdimxprev = romOptions.dimX;
                    int rdimvprev = romOptions.dimV;
                    int rdimeprev = romOptions.dimE;

                    SetWindowParameters(twparam, romOptions);
                    if (romOptions.hyperreduce)
                    {
                        basis[romOptions.window]->ProjectFromPreviousWindow(romOptions, romS, romOptions.window, rdimxprev, rdimvprev, rdimeprev);
                    }

                    delete basis[romOptions.window-1];
                    timeLoopTimer.Stop();

                    if (!romOptions.hyperreduce)
                    {
                        basis[romOptions.window]->Init(romOptions, *S);
                    }

                    if (romOptions.mergeXV)
                    {
                        romOptions.dimX = basis[romOptions.window]->GetDimX();
                        romOptions.dimV = basis[romOptions.window]->GetDimV();
                    }

                    if (!romOptions.hyperreduce)
                    {
                        romS.SetSize(romOptions.dimX + romOptions.dimV + romOptions.dimE);
                    }
                    timeLoopTimer.Start();

                    if (!romOptions.hyperreduce)
                    {
                        basis[romOptions.window]->ProjectFOMtoROM(*S, romS);
                    }
                    if (myid == 0)
                    {
                        cout << "Window " << romOptions.window << ": initial romS norm " << romS.Norml2() << endl;
                    }

                    delete romOper[romOptions.window-1];

                    if (romOptions.hyperreduce)
                    {
                        romOper[romOptions.window]->ApplyHyperreduction(romS);
                    }

                    if (problem == 7 && romOptions.indicatorType == penetrationDistance)
                    {
                        std::string pd_weight_outputPath = testing_parameter_outputPath + "/pd_weight" + to_string(romOptions.window);
                        ReadPDweight(pd_weight, pd_weight_outputPath);
                        if (myid == 0)
                        {
                            MFEM_VERIFY(pd_weight.size() == basis[romOptions.window]->GetDimX()+romOptions.useOffset, "Number of weights do not match.")
                        }
                    }

                    ode_solver->Init(*romOper[romOptions.window]);
                }
            }

            if (mpi.Root())
            {
                if (last_step) {
                    std::ofstream outfile(testing_parameter_outputPath + "/num_steps");
                    outfile << ti;
                    outfile.close();
                }
            }

            // Make sure that the mesh corresponds to the new solution state. This is
            // needed, because some time integrators use different S-type vectors
            // and the oper object might have redirected the mesh positions to those.
            if (fom_data && (!rom_build_database || !rom_online))
            {
                pmesh->NewNodes(*x_gf, false);

                if (last_step || (ti % vis_steps) == 0)
                {
                    double loc_norm = (*e_gf) * (*e_gf), tot_norm;
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

                    if (visualization || visit || gfprint) {
                        oper->ComputeDensity(*rho_gf);
                    }
                    if (visualization)
                    {
                        int Wx = 0, Wy = 0; // window position
                        int Ww = 350, Wh = 350; // window size
                        int offx = Ww+10; // window offsets

                        if (problem != 0 && problem != 4)
                        {
                            VisualizeField(*vis_rho, vishost, visport, *rho_gf,
                                           "Density", Wx, Wy, Ww, Wh);
                        }

                        Wx += offx;
                        VisualizeField(*vis_v, vishost, visport,
                                       *v_gf, "Velocity", Wx, Wy, Ww, Wh);
                        Wx += offx;
                        VisualizeField(*vis_e, vishost, visport, *e_gf,
                                       "Specific Internal Energy", Wx, Wy, Ww,Wh);
                        Wx += offx;
                    }

                    if (visit)
                    {
                        visit_dc->SetCycle(ti);
                        visit_dc->SetTime(t);
                        visit_dc->Save();
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
                        rho_gf->Save(rho_ofs);
                        rho_ofs.close();

                        ofstream v_ofs(v_name.str().c_str());
                        v_ofs.precision(8);
                        v_gf->Save(v_ofs);
                        v_ofs.close();

                        ofstream e_ofs(e_name.str().c_str());
                        e_ofs.precision(8);
                        e_gf->Save(e_ofs);
                        e_ofs.close();
                    }
                }
            }
        } // usual time loop
        timeLoopTimer.Stop();
        outfile_tw_steps.close();
    }

    if (romOptions.hyperreduce)
    {
        if (romOptions.GramSchmidt && !spaceTime)
        {
            romOper[romOptions.window]->PostprocessHyperreduction(romS);
        }
        if (!rom_online || spaceTime)
        {
            basis[romOptions.window]->LiftROMtoFOM(romS, *S);
        }
    }

    if (rom_offline)
    {
        samplerTimer.Start();
        basisConstructionTimer.Start();
        if (romOptions.dmd)
        {
            if (dmd_samplerLast)
                dmd_samplerLast->Finalize(romOptions);
            else if (dmd_sampler)
                dmd_sampler->Finalize(romOptions);
        }
        else
        {
            if (samplerLast)
                samplerLast->Finalize(cutoff, romOptions);
            else if (sampler)
                sampler->Finalize(cutoff, romOptions);
        }
        basisConstructionTimer.Stop();

        if (outputTimes)
        {
            outfile_time.close();
            MFEM_VERIFY(romOptions.window == 0, "Time windows not implemented in this case");
            MFEM_VERIFY(unique_steps + 1 == sampler->FinalNumberOfSamples(), "");
            // TODO: for now, we just write out the simulation timestep times, not the ROM basis generator
            // snapshot times. So far, in our tests snapshots are taken on every timestep, so the timesteps
            // and snapshots coincide. In general, this needs to be extended to allow for snapshots on a
            // subset of timesteps. Both the timesteps and snapshot times will be needed for space-time ROM.
            // The timesteps are needed for time integration (which defines the space-time system), and the
            // snapshot times are needed because the temporal bases and temporal samples are based on
            // snapshots.
        }

        if (myid == 0 && usingWindows && (sampler != NULL || dmd_sampler != NULL) && romOptions.parameterID == -1) {
            double real_pd = -1.0;
            if (problem == 7)
            {
                // 2D Rayleigh-Taylor penetration distance
                if (romOptions.indicatorType == penetrationDistance)
                {
                    double proc_pd = (pd2_vdof >= 0) ? -(*S)(pd2_vdof) : 0.0;
                    MPI_Reduce(&proc_pd, &real_pd, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                    MFEM_VERIFY(myid || real_pd >= 0.0, "Incorrect computation of penetration distance");
                }
                else if (romOptions.indicatorType == parameterTime)
                {
                    real_pd = romOptions.atwoodFactor * t * t;
                }
            }
            double windowEndpoint = (romOptions.indicatorType == physicalTime) ? t : real_pd;
            outfile_twp << windowEndpoint << ", " << cutoff[0] << ", " << cutoff[1] << ", " << cutoff[2];

            if (romOptions.SNS)
                outfile_twp << "\n";
            else
                outfile_twp << ", " << cutoff[3] << ", " << cutoff[4] << "\n";
        }
        if (romOptions.dmd)
        {
            if (dmd_samplerLast == dmd_sampler)
                delete dmd_sampler;
            else
            {
                delete dmd_sampler;
                delete dmd_samplerLast;
            }
        }
        else
        {
            if (samplerLast == sampler)
                delete sampler;
            else
            {
                delete sampler;
                delete samplerLast;
            }
        }

        samplerTimer.Stop();
        if (usingWindows && romOptions.parameterID == -1) outfile_twp.close();
    }

    double relative_error = 10.0 * romOptions.greedyTol;
    if (rom_build_database && (rom_calc_rel_error_nonlocal_completed || rom_calc_rel_error_local_completed) && greedy_converged)
    {
        cout << "Comparing to: " << testing_parameter_outputPath + "/Sol" + "_" + to_string(romOptions.blast_energyFactor) << endl;
        relative_error = PrintDiffParGridFunction(normtype, myid, testing_parameter_outputPath + "/Sol_Position" + "_" + to_string(romOptions.blast_energyFactor), x_gf);
        relative_error = std::max(relative_error, PrintDiffParGridFunction(normtype, myid, testing_parameter_outputPath + "/Sol_Velocity" + "_" + to_string(romOptions.blast_energyFactor), v_gf));
        relative_error = std::max(relative_error, PrintDiffParGridFunction(normtype, myid, testing_parameter_outputPath + "/Sol_Energy" + "_" + to_string(romOptions.blast_energyFactor), e_gf));
    }
    if (rom_calc_rel_error_local_completed && relative_error > romOptions.greedyTol)
    {
        MFEM_ABORT("The greedy algorithm has failed. The local ROM did not meet the relative error tolerance. Increase your relative error tolerance.");
    }

    if (outputSpaceTimeSolution)
    {
        ofs_STX.close();
        ofs_STV.close();
        ofs_STE.close();
    }

    if (fom_data && (!rom_build_database || greedy_write_solution))
    {
        if (writeSol)
        {
            PrintParGridFunction(myid, testing_parameter_outputPath + "/Sol_Position" + romOptions.basisIdentifier, x_gf);
            PrintParGridFunction(myid, testing_parameter_outputPath + "/Sol_Velocity" + romOptions.basisIdentifier, v_gf);
            PrintParGridFunction(myid, testing_parameter_outputPath + "/Sol_Energy" + romOptions.basisIdentifier, e_gf);
        }

        if (solDiff)
        {
            if (myid == 0) cout << "solDiff mode " << endl;
            PrintDiffParGridFunction(normtype, myid, testing_parameter_outputPath + "/Sol_Position" + romOptions.basisIdentifier, x_gf);
            PrintDiffParGridFunction(normtype, myid, testing_parameter_outputPath + "/Sol_Velocity" + romOptions.basisIdentifier, v_gf);
            PrintDiffParGridFunction(normtype, myid, testing_parameter_outputPath + "/Sol_Energy" + romOptions.basisIdentifier, e_gf);
        }

        if (visitDiffCycle >= 0)
        {
            VisItDataCollection dc(MPI_COMM_WORLD, visit_outputPath, pmesh);
            dc.Load(visitDiffCycle);
            if (myid == 0) cout << "Loaded VisIt DC cycle " << dc.GetCycle() << endl;

            ParGridFunction *dcfx = dc.GetParField("Position");
            ParGridFunction *dcfv = dc.GetParField("Velocity");
            ParGridFunction *dcfe = dc.GetParField("Specific Internal Energy");

            PrintNormsOfParGridFunctions(normtype, myid, "Position", dcfx, x_gf, true);
            PrintNormsOfParGridFunctions(normtype, myid, "Velocity", dcfv, v_gf, true);
            PrintNormsOfParGridFunctions(normtype, myid, "Energy", dcfe, e_gf, true);
        }
    }

    double errorIndicator = INT_MAX;
    bool errorIndicatorComputed = false;
    int errorIndicatorVecSize = 0;

    // If using the greedy algorithm, calculate the error indicator
    if (rom_build_database && !rom_calc_rel_error)
    {
        if (rom_online && !greedy_converged)
        {
            if (romOptions.greedyErrorIndicatorType == varyBasisSize)
            {
                char tmp[100];
                sprintf(tmp, ".%06d", myid);

                std::string fullname = outputPath + "/" + std::string("errorIndicatorVec") + tmp;

                std::ifstream checkfile(fullname);
                if (checkfile.good())
                {
                    checkfile.close();
                    remove(fullname.c_str());
                }
            }
            errorIndicatorComputed = true;
        }
        else if ((rom_online && !romOptions.hyperreduce) || (rom_restore) ||
                 (rom_offline && rom_calc_error_indicator && romOptions.greedyErrorIndicatorType == fom))
        {
            if (rom_online)
            {
                basis[romOptions.window]->LiftROMtoFOM(romS, *S);
            }

            // calculate the error indicator using the FOM lifted during
            // the second to last step compared against the FOM lifted at the last step.
            if (romOptions.greedyErrorIndicatorType == useLastLiftedSolution)
            {
                Vector errorIndicatorVec = Vector(lastLiftedSolution.Size());
                subtract(lastLiftedSolution, *S, errorIndicatorVec);

                errorIndicator = errorIndicatorVec.Norml2();
                errorIndicatorVecSize = errorIndicatorVec.Size();

                errorIndicatorComputed = true;
            }
            // calculate the error indicator using the last step of the two FOM
            // solutions with varied basis size
            else if (romOptions.greedyErrorIndicatorType == varyBasisSize)
            {
                char tmp[100];
                sprintf(tmp, ".%06d", myid);

                std::string fullname = outputPath + "/" + std::string("errorIndicatorVec") + tmp;

                std::ifstream checkfile(fullname);
                if (checkfile.good())
                {
                    Vector finalSolution = *S;
                    Vector previousFinalSolution;
                    previousFinalSolution.Load(checkfile, finalSolution.Size());

                    Vector errorIndicatorVec = Vector(finalSolution.Size());
                    subtract(finalSolution, previousFinalSolution, errorIndicatorVec);

                    errorIndicator = errorIndicatorVec.Norml2();
                    errorIndicatorVecSize = errorIndicatorVec.Size();

                    checkfile.close();
                    remove(fullname.c_str());

                    errorIndicatorComputed = true;
                }
                else
                {
                    Vector finalSolution = *S;

                    std::ofstream ofs(fullname.c_str(), std::ofstream::out);
                    ofs.precision(16);

                    for (int i=0; i<finalSolution.Size(); ++i)
                        ofs << finalSolution[i] << std::endl;

                    ofs.close();

                    if (myid == 0)
                    {
                        DeleteROMSolution(outputPath);
                    }
                }
            }
            else if (romOptions.greedyErrorIndicatorType == fom)
            {
                char tmp[100];
                sprintf(tmp, ".%06d", myid);

                std::string fullname = outputPath + "/" + std::string("errorIndicatorVec") + tmp;

                std::ifstream checkfile(fullname);
                if (checkfile.good())
                {
                    errorIndicator = PrintDiffParGridFunction(normtype, myid, outputPath + "/Sol_Position_error_indicator", x_gf);
                    errorIndicator = std::max(errorIndicator, PrintDiffParGridFunction(normtype, myid, outputPath + "/Sol_Velocity_error_indicator", v_gf));
                    errorIndicator = std::max(errorIndicator, PrintDiffParGridFunction(normtype, myid, outputPath + "/Sol_Energy_error_indicator", e_gf));
                    errorIndicatorVecSize = 1;

                    checkfile.close();
                    remove(fullname.c_str());

                    errorIndicatorComputed = true;
                }
                else
                {
                    std::ofstream ofs(fullname.c_str(), std::ofstream::out);
                    ofs.close();

                    if (myid == 0)
                    {
                        DeleteROMSolution(outputPath);
                    }
                }
            }
        }
    }
    if (rom_online)
    {
        delete basis[romOptions.window];
        delete romOper[romOptions.window];
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
    if (fom_data && (!rom_build_database || !rom_online))
    {
        oper->PrintTimingData(mpi.Root(), steps);

        const double energy_final = oper->InternalEnergy(*e_gf) +
                                    oper->KineticEnergy(*v_gf);
        if (mpi.Root())
        {
            cout << endl;
            cout << "Energy diff: " << scientific << setprecision(2)
                 << fabs(energy_init - energy_final) << endl;
        }

        PrintParGridFunction(myid, testing_parameter_outputPath + "/x_gf" + romOptions.basisIdentifier, x_gf);
        PrintParGridFunction(myid, testing_parameter_outputPath + "/v_gf" + romOptions.basisIdentifier, v_gf);
        PrintParGridFunction(myid, testing_parameter_outputPath + "/e_gf" + romOptions.basisIdentifier, e_gf);

        // Print the error.
        // For problems 0 and 4 the exact velocity is constant in time.
        if (problem == 0 || problem == 4)
        {
            const double error_max = v_gf->ComputeMaxError(*v_coeff),
                         error_l1  = v_gf->ComputeL1Error(*v_coeff),
                         error_l2  = v_gf->ComputeL2Error(*v_coeff);
            if (mpi.Root())
            {
                cout << "L_inf  error: " << error_max << endl
                     << "L_1    error: " << error_l1 << endl
                     << "L_2    error: " << error_l2 << endl;
            }
        }

        // 2D Rayleigh-Taylor penetration distance
        if (problem == 7 && fom_data)
        {
            double proc_pd[2], real_pd[2];
            proc_pd[0] = (pd1_vdof >= 0) ?  (*S)(pd1_vdof) : 0.0;
            proc_pd[1] = (pd2_vdof >= 0) ? -(*S)(pd2_vdof) : 0.0;
            MPI_Reduce(proc_pd, real_pd, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

            if (mpi.Root())
                cout << "Penetration distance (upward, downward): " << real_pd[0] << ", " << real_pd[1] << endl;
        }

        if (visualization)
        {
            vis_v->close();
            vis_e->close();
        }
    }

    totalTimer.Stop();
    if (mpi.Root()) {
        if(rom_online) cout << "Elapsed time for online preprocess: " << onlinePreprocessTimer.RealTime() << " sec\n";
        if(rom_restore) cout << "Elapsed time for restore phase: " << restoreTimer.RealTime() << " sec\n";
        if(rom_offline) cout << "Elapsed time for sampling in the offline phase: " << samplerTimer.RealTime() << " sec\n";
        if(rom_offline) cout << "Elapsed time for basis construction in the offline phase: " << basisConstructionTimer.RealTime() << " sec\n";
        cout << "Elapsed time for time loop: " << timeLoopTimer.RealTime() << " sec\n";
        cout << "Total time: " << totalTimer.RealTime() << " sec\n";
    }

    // If using the greedy algorithm, save the error indicator and any information
    // for use during the next iteration.
    if (rom_build_database && myid == 0)
    {
        if (!(rom_offline && rom_calc_error_indicator && romOptions.greedyErrorIndicatorType == fom))
        {
            WriteGreedyPhase(rom_offline, rom_online, rom_restore, rom_calc_rel_error_nonlocal, rom_calc_rel_error_local, romOptions, outputPath + "/greedy_algorithm_stage.txt");
        }
        if (rom_calc_rel_error_local_completed)
        {
            DeleteROMSolution(outputPath);
        }
    }
    if(rom_build_database && (rom_offline || rom_calc_rel_error_nonlocal_completed || errorIndicatorComputed))
    {
        if (rom_calc_rel_error_nonlocal_completed)
        {
            parameterPointGreedySampler->setPointRelativeError(relative_error);
        }
        else if (errorIndicatorComputed)
        {
            parameterPointGreedySampler->setPointErrorIndicator(errorIndicator, errorIndicatorVecSize);
        }

        if (myid == 0)
        {
            std::string outputFile = outputPath + "/greedy_algorithm_stage.txt";
            remove(outputFile.c_str());

            parameterPointGreedySampler->save(outputPath + "/greedy_algorithm_data");
            DeleteROMSolution(outputPath);
        }

        if (parameterPointGreedySampler->isComplete())
        {
            // The greedy algorithm procedure has ended
            if (myid == 0)
            {
                cout << "The greedy algorithm procedure has completed!" << endl;
            }
            return 1;
        }
    }

    // Free the used memory.
    if (ode_solver != nullptr) delete ode_solver;
    if (pmesh != nullptr) delete pmesh;
    if (oper != nullptr) delete oper;
    if (rho != nullptr) delete rho;
    if (mat_fec != nullptr) delete mat_fec;
    if (mat_fes != nullptr) delete mat_fes;
    if (mat_gf != nullptr) delete mat_gf;
    if (mat_gf_coeff != nullptr) delete mat_gf_coeff;
    if (L2FESpace != nullptr) delete L2FESpace;
    if (H1FESpace != nullptr) delete H1FESpace;
    if (S != nullptr) delete S;
    if (S_old != nullptr) delete S_old;
    if (x_gf != nullptr) delete x_gf;
    if (v_gf != nullptr) delete v_gf;
    if (e_gf != nullptr) delete e_gf;
    if (rho_gf != nullptr) delete rho_gf;
    if (vis_rho != nullptr) delete vis_rho;
    if (vis_v != nullptr) delete vis_v;
    if (vis_e != nullptr) delete vis_e;
    if (visit_dc != nullptr) delete visit_dc;
    if (v_coeff != nullptr) delete v_coeff;

    if (romOptions.dmd && rom_restore)
    {
        delete dmd_X;
        delete dmd_V;
        delete dmd_E;
        delete result_X;
        delete result_V;
        delete result_E;
    }

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
    //return (dim == 2) ? (x(0) > 1.0 && x(1) > 1.5) ? 0.125 : 1.0
    //       : x(0) > 1.0 && ((x(1) < 1.5 && x(2) < 1.5) ||
    //                        (x(1) > 1.5 && x(2) > 1.5)) ? 0.125 : 1.0;
    case 4:
        return 1.0;
    case 5:
    {
        if (x(0) >= 0.5 && x(1) >= 0.5) {
            return 0.5313;
        }
        if (x(0) <  0.5 && x(1) <  0.5) {
            return 0.8;
        }
        return 1.0;
    }
    case 6:
    {
        if (x(0) <  0.5 && x(1) >= 0.5) {
            return 2.0;
        }
        if (x(0) >= 0.5 && x(1) <  0.5) {
            return 3.0;
        }
        return 1.0;
    }
    case 7:
        return x(1) >= 0.0 ? rhoRatio : 1.0;
    default:
        MFEM_ABORT("Bad number given for problem id!");
        return 0.0;
    }
}

double gamma_func(const Vector &x)
{
    switch (problem)
    {
    case 0:
        return 5.0 / 3.0;
    case 1:
        return 1.4;
    case 2:
        return 1.4;
    case 3:
        return (x(0) > 1.0 && x(1) <= 1.5) ? 1.4 : 1.5;
    case 4:
        return 5.0 / 3.0;
    case 5:
        return 1.4;
    case 6:
        return 1.4;
    case 7:
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
    case 5:
    {
        const double atn = pow((x(0)*(1.0-x(0))*4*x(1)*(1.0-x(1))*4.0),0.4);
        v = 0.0;
        if (x(0) >= 0.5 && x(1) >= 0.5) {
            v(0)=0.0*atn, v(1)=0.0*atn;
            return;
        }
        if (x(0) <  0.5 && x(1) >= 0.5) {
            v(0)=0.7276*atn, v(1)=0.0*atn;
            return;
        }
        if (x(0) <  0.5 && x(1) <  0.5) {
            v(0)=0.0*atn, v(1)=0.0*atn;
            return;
        }
        if (x(0) >= 0.5 && x(1) <  0.5) {
            v(0)=0.0*atn, v(1)=0.7276*atn;
            return;
        }
        MFEM_ABORT("Error in problem 5!");
        return;
    }
    case 6:
    {
        const double atn = pow((x(0)*(1.0-x(0))*4*x(1)*(1.0-x(1))*4.0),0.4);
        v = 0.0;
        if (x(0) >= 0.5 && x(1) >= 0.5) {
            v(0)=+0.75*atn, v(1)=-0.5*atn;
            return;
        }
        if (x(0) <  0.5 && x(1) >= 0.5) {
            v(0)=+0.75*atn, v(1)=+0.5*atn;
            return;
        }
        if (x(0) <  0.5 && x(1) <  0.5) {
            v(0)=-0.75*atn, v(1)=+0.5*atn;
            return;
        }
        if (x(0) >= 0.5 && x(1) <  0.5) {
            v(0)=-0.75*atn, v(1)=-0.5*atn;
            return;
        }
        MFEM_ABORT("Error in problem 6!");
        return;
    }
    case 7:
    {
        v = 0.0;
        v(1) = 0.02 * exp(-2*M_PI*x(1)*x(1)) * cos(2*M_PI*x(0));
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
        return (x(0) < 0.5) ? 1.0 / rho0(x) / (gamma_func(x) - 1.0)
               : 0.1 / rho0(x) / (gamma_func(x) - 1.0);
    case 3:
        return (x(0) > 1.0) ? 0.1 / rho0(x) / (gamma_func(x) - 1.0)
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
        else {
            return (3.0 + 4.0 * log(2.0)) / (gamma - 1.0);
        }
    }
    case 5:
    {
        const double irg = 1.0 / rho0(x) / (gamma_func(x) - 1.0);
        if (x(0) >= 0.5 && x(1) >= 0.5) {
            return 0.4 * irg;
        }
        if (x(0) <  0.5 && x(1) >= 0.5) {
            return 1.0 * irg;
        }
        if (x(0) <  0.5 && x(1) <  0.5) {
            return 1.0 * irg;
        }
        if (x(0) >= 0.5 && x(1) <  0.5) {
            return 1.0 * irg;
        }
        MFEM_ABORT("Error in problem 5!");
        return 0.0;
    }
    case 6:
    {
        const double irg = 1.0 / rho0(x) / (gamma_func(x) - 1.0);
        if (x(0) >= 0.5 && x(1) >= 0.5) {
            return 1.0 * irg;
        }
        if (x(0) <  0.5 && x(1) >= 0.5) {
            return 1.0 * irg;
        }
        if (x(0) <  0.5 && x(1) <  0.5) {
            return 1.0 * irg;
        }
        if (x(0) >= 0.5 && x(1) <  0.5) {
            return 1.0 * irg;
        }
        MFEM_ABORT("Error in problem 5!");
        return 0.0;
    }
    case 7:
    {
        const double rho = rho0(x), gamma = gamma_func(x);
        return (4.0 + rhoRatio - rho * x(1)) / (gamma - 1.0) / rho;
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
