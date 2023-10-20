#ifndef MFEM_LAGHOS_ROM
#define MFEM_LAGHOS_ROM

#include "mfem.hpp"

#include "linalg/BasisGenerator.h"
#include "linalg/BasisReader.h"
#include "algo/AdaptiveDMD.h"
#include "algo/NonuniformDMD.h"
#include "algo/greedy/GreedyRandomSampler.h"

#include "laghos_solver.hpp"

#include "mfem/SampleMesh.hpp"

using namespace mfem;


//#define STXV  // TODO: remove this?

enum NormType { l1norm=1, l2norm=2, maxnorm=0 };

double PrintNormsOfParGridFunctions(NormType normtype, const int rank, const std::string& name, ParGridFunction *f1, ParGridFunction *f2,
                                    const bool scalar);
void PrintL2NormsOfParGridFunctions(const int rank, const std::string& name, ParGridFunction *f1, ParGridFunction *f2,
                                    const bool scalar);

namespace ROMBasisName {
const char* const X = "basisX";
const char* const V = "basisV";
const char* const E = "basisE";
const char* const Fv = "basisFv";
const char* const Fe = "basisFe";
};

enum VariableName { X, V, E, Fv, Fe };

enum offsetStyle
{
    useInitialState,
    saveLoadOffset,
    interpolateOffset
};

enum samplingType
{
    randomSampling,
    latinHypercubeSampling
};

enum errorIndicatorType
{
    useLastLiftedSolution,
    varyBasisSize,
    fom
};

static offsetStyle getOffsetStyle(const char* offsetType)
{
    static std::unordered_map<std::string, offsetStyle> offsetMap =
    {
        {"initial", useInitialState},
        {"load", saveLoadOffset},
        {"interpolate", interpolateOffset}
    };
    auto iter = offsetMap.find(offsetType);
    MFEM_VERIFY(iter != std::end(offsetMap), "Invalid input for offset type");
    return iter->second;
}

enum HyperreductionSamplingType
{
    gnat,       // Default, GNAT
    qdeim,      // QDEIM
    sopt,       // S-OPT
    eqp,        // EQP
	eqp_energy  // Energy-conserving EQP
};

static HyperreductionSamplingType getHyperreductionSamplingType(const char* sampling_type)
{
    static std::unordered_map<std::string, HyperreductionSamplingType> SamplingTypeMap =
    {
        {"gnat", gnat},
        {"qdeim", qdeim},
        {"sopt", sopt},
        {"eqp", eqp},
		{"eqp_energy", eqp_energy}
    };
    auto iter = SamplingTypeMap.find(sampling_type);
    MFEM_VERIFY(iter != std::end(SamplingTypeMap), "Invalid input for hyperreduction sampling type");
    return iter->second;
}

enum SpaceTimeMethod
{
    no_space_time, // Default, spatial ROM
    gnat_lspg,     // LSPG (least-squares Petrov-Galerkin) using GNAT for hyperreduction
    coll_lspg,     // LSPG (least-squares Petrov-Galerkin) using collocation for hyperreduction
    galerkin       // Galerkin space-time system (not recommended)  TODO: remove this option?
};

static SpaceTimeMethod getSpaceTimeMethod(const char* spaceTime)
{
    static std::unordered_map<std::string, SpaceTimeMethod> spaceTimeMap =
    {
        {"spatial", no_space_time},
        {"gnat_lspg", gnat_lspg},
        {"coll_lspg", coll_lspg},
        {"galerkin", galerkin}
    };
    auto iter = spaceTimeMap.find(spaceTime);
    MFEM_VERIFY(iter != std::end(spaceTimeMap), "Invalid input for space time method");
    return iter->second;
}

static samplingType getSamplingType(const char* samp)
{
    static std::unordered_map<std::string, samplingType> sampMap =
    {
        {"random", randomSampling},
        {"latin-hypercube", latinHypercubeSampling}
    };
    auto iter = sampMap.find(samp);
    MFEM_VERIFY(iter != std::end(sampMap), "Invalid input of sampling type");
    return iter->second;
}

static errorIndicatorType getErrorIndicatorType(const char* errorIndicator)
{
    static std::unordered_map<std::string, errorIndicatorType> errorIndicatorMap =
    {
        {"useLastLifted", useLastLiftedSolution},
        {"varyBasisSize", varyBasisSize},
        {"fom", fom}
    };
    auto iter = errorIndicatorMap.find(errorIndicator);
    MFEM_VERIFY(iter != std::end(errorIndicatorMap), "Invalid input of error indicator type");
    return iter->second;
}

enum localROMIndicator
{
    physicalTime,
    penetrationDistance,
    parameterTime
};

static localROMIndicator getlocalROMIndicator(const char* indicatorType)
{
    static std::unordered_map<std::string, localROMIndicator> indicatorMap =
    {
        {"time", physicalTime},
        {"distance", penetrationDistance},
        {"parameter-time", parameterTime}
    };
    auto iter = indicatorMap.find(indicatorType);
    MFEM_VERIFY(iter != std::end(indicatorMap), "Invalid input of local ROM indicator type");
    return iter->second;
}

struct ROM_Options
{
    int rank = 0;  // MPI rank
    ParFiniteElementSpace *H1FESpace = NULL; // FOM H1 FEM space
    ParFiniteElementSpace *L2FESpace = NULL; // FOM L2 FEM space

    std::string *basename = NULL;
    std::string *testing_parameter_basename = NULL;
    std::string *hyperreduce_basename = NULL;
    std::string initSamples_basename = "";

    std::string basisIdentifier = "";
    std::string greedyParam = "bef";
    double greedytf = -1.0; // error indicator final time for the greedy algorithm (only used in varyBasisSize and fom error indicator types)
    double greedyTol = 0.1; // relative error tolerance for the greedy algorithm
    double greedyAlpha = 1.05; // alpha constant for the greedy algorithm
    double greedyMaxClamp = 2.0; // max clamp constant for the greedy algorithm
    double greedyParamSpaceMin = 0; // min value of the greedy algorithm 1D parameter domain
    double greedyParamSpaceMax = 0; // max value of the greedy algorithm 1D parameter domain
    int greedyParamSpaceSize = 0; // the maximum number of local ROMS to create in the greedy algorithm 1D parameter domain
    int greedySubsetSize = 0; // subset size of parameter points whose error indicators are checked during the greedy algorithm
    int greedyConvergenceSubsetSize = 0; // convergence subset size for terminating the greedy algorithm
    samplingType greedySamplingType = randomSampling; // sampling type for the greedy algorithm
    errorIndicatorType greedyErrorIndicatorType = useLastLiftedSolution; // error indicator type for the greedy algorithm

    double t_final = 0.0; // simulation final time
    double initial_dt = 0.0; // initial timestep size
    bool   dmd = false;
    double dmd_tbegin = 0.0;
    double desired_dt = -1.0;
    double dmd_closest_rbf = 0.9;
    bool   dmd_nonuniform = false;
    double rhoFactor = 1.0; // factor for scaling rho
    double atwoodFactor = 1.0 / 3.0; // factor for Atwood number in Rayleigh-Taylor instability problem
    double blast_energyFactor = 1.0; // factor for scaling blast energy

    bool restore = false; // if true, restore phase
    bool staticSVD = true; // true: use StaticSVD
    bool useOffset = true; // if true, sample variables minus initial state as an offset
    bool SNS = false; // if true, use SNS relation to obtain nonlinear RHS bases by multiplying mass matrix to a solution matrix. See arXiv 1809.04064.
    double energyFraction = 0.9999; // used for recommending basis sizes, depending on singular values
    double energyFraction_X = 0.9999; // used for recommending basis sizes, depending on singular values
    int sv_shift = 1; // Number of shifted singular values in energy fraction calculation (to avoid one singular occupies almost all energy when window-dependent offsets are not used)
    int window = 0; // Laghos-ROM time window index
    int max_dim = 0; // maximum dimension for libROM basis generator time interval
    int parameterID = -1; // index of parameters chosen for this Laghos simulation
    hydrodynamics::LagrangianHydroOperator *FOMoper = NULL; // FOM operator

    // Variable basis dimensions
    int dimX = -1;
    int dimV = -1;
    int dimE = -1;
    int dimFv = -1;
    int dimFe = -1;
    int max_dimX = std::numeric_limits<int>::max();
    int max_dimV = std::numeric_limits<int>::max();
    int max_dimE = std::numeric_limits<int>::max();
    int max_dimFv = std::numeric_limits<int>::max();
    int max_dimFe = std::numeric_limits<int>::max();

    // Randomized SVD options
    bool randomizedSVD = false; // true: use RandomizedSVD
    int randdimX = -1;
    int randdimV = -1;
    int randdimE = -1;
    int randdimFv = -1;
    int randdimFe = -1;

    // Incremental SVD options
    double incSVD_linearity_tol = 1.e-7;
    double incSVD_singular_value_tol = 1.e-14;
    double incSVD_sampling_tol = 1.e-7;

    // Number of spatial samples for each variable
    int sampX = 0;
    int sampV = 0;
    int sampE = 0;

    // Number of temporal samples for each variable
    int tsampV = 1;
    int tsampE = 1;

    bool hyperreduce = false; // whether to use hyperreduction on ROM online phase
    bool hyperreduce_prep = false; // whether to do hyperreduction pre-processing on ROM online phase
    bool use_sample_mesh = false; // whether to use sample mesh; True only when hyperreduce mode with GNAT, QDEIM, S-OPT
    bool GramSchmidt = true; // whether to use Gram-Schmidt with respect to mass matrices
    bool RK2AvgSolver = false; // true if RK2Avg solver is used for time integration
    offsetStyle offsetType = useInitialState; // type of offset in time windows
    localROMIndicator indicatorType = physicalTime; // type of local ROM indicator in time windows

    bool mergeXV = false; // If true, merge bases for V and X-X0 by using BasisGenerator on normalized basis vectors for V and X-X0.

    bool useXV = false; // If true, use V basis for X-X0.
    bool useVX = false; // If true, use X-X0 basis for V.

    HyperreductionSamplingType hyperreductionSamplingType = gnat;
    SpaceTimeMethod spaceTimeMethod = no_space_time;

    bool VTos = false;

	int maxNNLSnnz = 0; // max number of NNLS solution nonzeros
	double tolNNLS = 1.0e-14; // NNLS solver error tolerance

	// snapshot sampling frequency (sample every sampfreq timestep)
	int sampfreq = 1;
};

static double* getGreedyParam(ROM_Options& romOptions, const char* greedyParam)
{
    static std::unordered_map<std::string, double*> paramMap =
    {
        {"bef", &romOptions.blast_energyFactor}
    };
    auto iter = paramMap.find(greedyParam);
    MFEM_VERIFY(iter != std::end(paramMap), "Invalid input for greedy parameter.");
    return iter->second;
}

class DMD_Sampler
{
public:
    DMD_Sampler(ROM_Options const& input, Vector const& S_init)
        : rank(input.rank), window(input.window), tbegin(input.dmd_tbegin), tH1size(input.H1FESpace->GetTrueVSize()), tL2size(input.L2FESpace->GetTrueVSize()),
          H1size(input.H1FESpace->GetVSize()), L2size(input.L2FESpace->GetVSize()),
          X(tH1size), dXdt(tH1size), V(tH1size), dVdt(tH1size), E(tL2size), dEdt(tL2size),
          gfH1(input.H1FESpace), gfL2(input.L2FESpace), offsetInit(input.useOffset), energyFraction(input.energyFraction),
          energyFraction_X(input.energyFraction_X), sns(input.SNS), lhoper(input.FOMoper),
          parameterID(input.parameterID), basename(*input.basename),
          useXV(input.useXV), useVX(input.useVX), VTos(input.VTos)
    {
        SetStateVariables(S_init);

        dXdt = 0.0;
        dVdt = 0.0;
        dEdt = 0.0;

        X0 = 0.0;
        V0 = 0.0;
        E0 = 0.0;

        if (offsetInit)
        {
            std::string path_init = (input.offsetType == interpolateOffset) ? basename + "/ROMoffset" + input.basisIdentifier + "/param" + std::to_string(parameterID) + "_init" : basename + "/ROMoffset" + input.basisIdentifier + "/init";
            initX = new CAROM::Vector(tH1size, true);
            initV = new CAROM::Vector(tH1size, true);
            initE = new CAROM::Vector(tL2size, true);
            Xdiff.SetSize(tH1size);
            Ediff.SetSize(tL2size);

            if (input.offsetType == useInitialState && input.window > 0)
            {
                // Read the initial state in the offline phase
                initX->read(path_init + "X0");
                initV->read(path_init + "V0");
                initE->read(path_init + "E0");
                first_sv = input.sv_shift;
            }
            else
            {
                // Compute and save offsets for the current window in the offline phase
                for (int i=0; i<tH1size; ++i)
                {
                    (*initX)(i) = X[i];
                }

                for (int i=0; i<tH1size; ++i)
                {
                    (*initV)(i) = V[i];
                }

                for (int i=0; i<tL2size; ++i)
                {
                    (*initE)(i) = E[i];
                }

                initX->write(path_init + "X" + std::to_string(window));
                initV->write(path_init + "V" + std::to_string(window));
                initE->write(path_init + "E" + std::to_string(window));

                if (window == 0)
                {
                    const double Vnorm = initV->norm();
                    int osVT = (Vnorm == 0.0) ? 1 : 0;

                    MFEM_VERIFY(VTos == osVT, "");
                }
            }
        }
        else
        {
            first_sv = input.sv_shift;
            initX = NULL;
            initV = NULL;
            initE = NULL;
        }

        if (input.dmd_nonuniform)
        {
            dmd_X = new CAROM::NonuniformDMD(tH1size, initX, initV);
            dmd_V = new CAROM::NonuniformDMD(tH1size, NULL, NULL);
            dmd_E = new CAROM::NonuniformDMD(tL2size, NULL, NULL);
        }
        else
        {
            dmd_X = new CAROM::AdaptiveDMD(tH1size, input.desired_dt, "G", "LS", input.dmd_closest_rbf, initX);
            dmd_V = new CAROM::AdaptiveDMD(tH1size, input.desired_dt, "G", "LS", input.dmd_closest_rbf, initV);
            dmd_E = new CAROM::AdaptiveDMD(tL2size, input.desired_dt, "G", "LS", input.dmd_closest_rbf, initE);
        }
    }

    void SampleSolution(const double t, const double dt, Vector const& S);

    void Finalize(ROM_Options& input);

    int MaxNumSamples()
    {
        return std::max(std::max(dmd_X->getNumSamples(), dmd_V->getNumSamples()), dmd_E->getNumSamples());
    }

    int FinalNumberOfSamples()
    {
        MFEM_VERIFY(finalized, "DMD_Sampler not finalized");
        return finalNumSamples;
    }

    int GetRank()
    {
        return rank;
    }

    CAROM::DMD *dmd_X, *dmd_V, *dmd_E;

private:
    const int H1size;
    const int L2size;
    const int tH1size;
    const int tL2size;

    const int window;

    const double tbegin;

    const int rank;
    int first_sv = 0;
    double energyFraction;
    double energyFraction_X;

    const int parameterID;

    std::string basename = "run";

    Vector X, X0, Xdiff, Ediff, dXdt, V, V0, dVdt, E, E0, dEdt;

    const bool offsetInit;
    CAROM::Vector *initX = 0;
    CAROM::Vector *initV = 0;
    CAROM::Vector *initE = 0;

    ParGridFunction gfH1, gfL2;

    const bool sns;

    const bool useXV;
    const bool useVX;

    bool finalized = false;
    int finalNumSamples = 0;

    hydrodynamics::LagrangianHydroOperator *lhoper;

    int VTos = 0; // Velocity temporal index offset, used for V and Fe. This fixes the issue that V and Fe are not sampled at t=0, since they are initially zero. This is valid for the Sedov test but not in general when the initial velocity is nonzero.

    void SetStateVariables(Vector const& S)
    {
        X0 = X;
        V0 = V;
        E0 = E;

        for (int i=0; i<H1size; ++i)
        {
            gfH1[i] = S[i];
        }

        gfH1.GetTrueDofs(X);

        for (int i=0; i<H1size; ++i)
        {
            gfH1[i] = S[H1size + i];
        }

        gfH1.GetTrueDofs(V);

        for (int i=0; i<L2size; ++i)
        {
            gfL2[i] = S[(2*H1size) + i];
        }

        gfL2.GetTrueDofs(E);
    }

    void SetStateVariableRates(const double dt)
    {
        for (int i=0; i<tH1size; ++i)
        {
            dXdt[i] = (X[i] - X0[i]) / dt;
            dVdt[i] = (V[i] - V0[i]) / dt;
        }

        for (int i=0; i<tL2size; ++i)
        {
            dEdt[i] = (E[i] - E0[i]) / dt;
        }
    }
};

class ROM_Sampler
{
public:
    ROM_Sampler(ROM_Options const& input, Vector const& S_init)
        : rank(input.rank), tH1size(input.H1FESpace->GetTrueVSize()), tL2size(input.L2FESpace->GetTrueVSize()),
          H1size(input.H1FESpace->GetVSize()), L2size(input.L2FESpace->GetVSize()),
          X(tH1size), dXdt(tH1size), V(tH1size), dVdt(tH1size), E(tL2size), dEdt(tL2size),
          gfH1(input.H1FESpace), gfL2(input.L2FESpace), offsetInit(input.useOffset), energyFraction(input.energyFraction),
          energyFraction_X(input.energyFraction_X), sns(input.SNS), lhoper(input.FOMoper), writeSnapshots(input.parameterID >= 0),
          parameterID(input.parameterID), basename(*input.basename), Voffset(!input.useXV && !input.useVX && !input.mergeXV),
          useXV(input.useXV), useVX(input.useVX), VTos(input.VTos), spaceTime(input.spaceTimeMethod != no_space_time),
          rhsBasis(input.hyperreductionSamplingType != eqp && input.hyperreductionSamplingType != eqp_energy),
		  hyperreductionSamplingType(input.hyperreductionSamplingType)
    {
        const int window = input.window;

        // TODO: update the following comment, since there should now be a maximum of 1 time interval now.
        const int max_model_dim_est = int(input.t_final/input.initial_dt + 0.5) + 100;  // Note that this is a rough estimate which may be exceeded, resulting in multiple libROM basis time intervals.
        const int max_model_dim = (input.max_dim > 0) ? input.max_dim + 1 : max_model_dim_est;

        std::cout << rank << ": max_model_dim " << max_model_dim << std::endl;

        const bool output_rightSV = spaceTime;

        CAROM::Options x_options = CAROM::Options(tH1size, max_model_dim, 1, output_rightSV);
        CAROM::Options e_options = CAROM::Options(tL2size, max_model_dim, 1, output_rightSV);
        bool staticSVD = (input.staticSVD || input.randomizedSVD);
        if (!staticSVD)
        {
            x_options.setIncrementalSVD(input.incSVD_linearity_tol,
                                        input.initial_dt,
                                        input.incSVD_sampling_tol,
                                        input.t_final,
                                        true);
            x_options.setMaxBasisDimension(max_model_dim);
            x_options.setSingularValueTol(input.incSVD_singular_value_tol);

            e_options.setIncrementalSVD(input.incSVD_linearity_tol,
                                        input.initial_dt,
                                        input.incSVD_sampling_tol,
                                        input.t_final,
                                        true);
            e_options.setMaxBasisDimension(max_model_dim);
            e_options.setSingularValueTol(input.incSVD_singular_value_tol);
        }

        if (input.randomizedSVD)
        {
            x_options.setRandomizedSVD(true, input.randdimX);
        }

        generator_X = new CAROM::BasisGenerator(
            x_options,
            !staticSVD,
            staticSVD ? BasisFileName(basename, VariableName::X, window, parameterID, input.basisIdentifier) : basename + "/" + ROMBasisName::X + std::to_string(window) + input.basisIdentifier);

        if (input.randomizedSVD)
        {
            x_options.setRandomizedSVD(true, input.randdimV);
        }
        generator_V = new CAROM::BasisGenerator(
            x_options,
            !staticSVD,
            staticSVD ? BasisFileName(basename, VariableName::V, window, parameterID, input.basisIdentifier) : basename + "/" + ROMBasisName::V + std::to_string(window) + input.basisIdentifier);

        if (input.randomizedSVD)
        {
            e_options.setRandomizedSVD(true, input.randdimE);
        }
        generator_E = new CAROM::BasisGenerator(
            e_options,
            !staticSVD,
            staticSVD ? BasisFileName(basename, VariableName::E, window, parameterID, input.basisIdentifier) : basename + "/" + ROMBasisName::E + std::to_string(window) + input.basisIdentifier);

        if (!sns && rhsBasis)
        {
            if (input.randomizedSVD)
            {
                x_options.setRandomizedSVD(true, input.randdimFv);
            }
            generator_Fv = new CAROM::BasisGenerator(
                x_options,
                !staticSVD,
                staticSVD ? BasisFileName(basename, VariableName::Fv, window, parameterID, input.basisIdentifier) : basename + "/" + ROMBasisName::Fv + std::to_string(window) + input.basisIdentifier);

            if (input.randomizedSVD)
            {
                e_options.setRandomizedSVD(true, input.randdimFe);
            }
            generator_Fe = new CAROM::BasisGenerator(
                e_options,
                !staticSVD,
                staticSVD ? BasisFileName(basename, VariableName::Fe, window, parameterID, input.basisIdentifier) : basename + "/" + ROMBasisName::Fe + std::to_string(window) + input.basisIdentifier);
        }

        SetStateVariables(S_init);

        dXdt = 0.0;
        dVdt = 0.0;
        dEdt = 0.0;

        X0 = 0.0;
        V0 = 0.0;
        E0 = 0.0;

        if (offsetInit)
        {
            std::string path_init = (input.offsetType == interpolateOffset) ? basename + "/ROMoffset" + input.basisIdentifier + "/param" + std::to_string(parameterID) + "_init" : basename + "/ROMoffset" + input.basisIdentifier + "/init";
            initX = new CAROM::Vector(tH1size, true);
            initV = new CAROM::Vector(tH1size, true);
            initE = new CAROM::Vector(tL2size, true);
            Xdiff.SetSize(tH1size);
            Ediff.SetSize(tL2size);

            if (input.offsetType == useInitialState && input.window > 0)
            {
                // Read the initial state in the offline phase
                initX->read(path_init + "X0");
                initV->read(path_init + "V0");
                initE->read(path_init + "E0");
                first_sv = input.sv_shift;
            }
            else
            {
                // Compute and save offsets for the current window in the offline phase
                for (int i=0; i<tH1size; ++i)
                {
                    (*initX)(i) = X[i];
                }

                for (int i=0; i<tH1size; ++i)
                {
                    (*initV)(i) = V[i];
                }

                for (int i=0; i<tL2size; ++i)
                {
                    (*initE)(i) = E[i];
                }

                initX->write(path_init + "X" + std::to_string(window));
                initV->write(path_init + "V" + std::to_string(window));
                initE->write(path_init + "E" + std::to_string(window));

                if (window == 0)
                {
                    const double Vnorm = initV->norm();
                    int osVT = (Vnorm == 0.0) ? 1 : 0;

                    MFEM_VERIFY(VTos == osVT, "");
                }
            }
        }
        else first_sv = input.sv_shift;
    }

    void SampleSolution(const double t, const double dt, const double pd, Vector const& S);

    void Finalize(Array<int> &cutoff, ROM_Options& input, Vector const& sol);

    int MaxNumSamples()
    {
        return std::max(std::max(generator_X->getNumSamples(), generator_V->getNumSamples()), generator_E->getNumSamples());
    }

    int FinalNumberOfSamples()
    {
        MFEM_VERIFY(finalized, "ROM_Sampler not finalized");
        return finalNumSamples;
    }

    int GetRank()
    {
        return rank;
    }

private:
    const int H1size;
    const int L2size;
    const int tH1size;
    const int tL2size;

    const int rank;
    int first_sv = 0;
    double energyFraction;
    double energyFraction_X;

    const int parameterID;
    const bool writeSnapshots;
    std::vector<double> tSnapX, tSnapV, tSnapE, tSnapFv, tSnapFe, pdSnap;

    std::string basename = "run";

    CAROM::BasisGenerator *generator_X, *generator_V, *generator_E, *generator_Fv, *generator_Fe;

    Vector X, X0, Xdiff, Ediff, dXdt, V, V0, dVdt, E, E0, dEdt;

    const bool offsetInit;
    CAROM::Vector *initX = 0;
    CAROM::Vector *initV = 0;
    CAROM::Vector *initE = 0;

    ParGridFunction gfH1, gfL2;

    const bool sns;
    const bool rhsBasis;

    const bool Voffset;
    const bool useXV;
    const bool useVX;

    bool finalized = false;
    int finalNumSamples = 0;

    const bool spaceTime;

	const bool hyperreductionSamplingType; 

    hydrodynamics::LagrangianHydroOperator *lhoper;

    int VTos = 0; // Velocity temporal index offset, used for V and Fe. This fixes the issue that V and Fe are not sampled at t=0, since they are initially zero. This is valid for the Sedov test but not in general when the initial velocity is nonzero.

    void SetStateVariables(Vector const& S)
    {
        X0 = X;
        V0 = V;
        E0 = E;

        for (int i=0; i<H1size; ++i)
        {
            gfH1[i] = S[i];
        }

        gfH1.GetTrueDofs(X);

        for (int i=0; i<H1size; ++i)
        {
            gfH1[i] = S[H1size + i];
        }

        gfH1.GetTrueDofs(V);

        for (int i=0; i<L2size; ++i)
        {
            gfL2[i] = S[(2*H1size) + i];
        }

        gfL2.GetTrueDofs(E);
    }

    void SetStateVariableRates(const double dt)
    {
        for (int i=0; i<tH1size; ++i)
        {
            dXdt[i] = (X[i] - X0[i]) / dt;
            dVdt[i] = (V[i] - V0[i]) / dt;
        }

        for (int i=0; i<tL2size; ++i)
        {
            dEdt[i] = (E[i] - E0[i]) / dt;
        }
    }

    void SetStateFromTrueDOFs(Vector const& x, Vector const& v, Vector const& e, Vector & S)
    {
        MFEM_VERIFY(S.Size() == 2*H1size + L2size, "");
        MFEM_VERIFY(x.Size() == tH1size, "");
        MFEM_VERIFY(v.Size() == tH1size, "");
        MFEM_VERIFY(e.Size() == tL2size, "");

        // Set X component of S
        Xdiff = x;
        if (offsetInit)
        {
            for (int i=0; i<tH1size; ++i)
            {
                Xdiff[i] += (*initX)(i);
            }
        }

        gfH1.SetFromTrueDofs(Xdiff);

        for (int i=0; i<H1size; ++i)
        {
            S[i] = gfH1[i];
        }

        // Set V component of S
        Xdiff = v;
        if (offsetInit)
        {
            for (int i=0; i<tH1size; ++i)
            {
                Xdiff[i] += (*initV)(i);
            }
        }

        gfH1.SetFromTrueDofs(Xdiff);

        for (int i=0; i<H1size; ++i)
        {
            S[H1size + i] = gfH1[i];
        }

        // Set E component of S
        Ediff = e;
        if (offsetInit)
        {
            for (int i=0; i<tL2size; ++i)
            {
                Ediff[i] += (*initE)(i);
            }
        }

        gfL2.SetFromTrueDofs(Ediff);

        for (int i=0; i<L2size; ++i)
        {
            S[(2 * H1size) + i] = gfL2[i];
        }
    }

    std::string BasisFileName(const std::string basename, VariableName v, const int window, const int parameter, const std::string basisIdentifier)
    {
        std::string fileName, path;

        const std::string prefix = (parameter >= 0) ? "var" : "basis";

        switch (v)
        {
        case VariableName::V:
            fileName = "V" + std::to_string(window) + basisIdentifier;
            break;
        case VariableName::E:
            fileName = "E" + std::to_string(window) + basisIdentifier;
            break;
        case VariableName::Fv:
            fileName = "Fv" + std::to_string(window) + basisIdentifier;
            break;
        case VariableName::Fe:
            fileName = "Fe" + std::to_string(window) + basisIdentifier;
            break;
        default:
            fileName = "X" + std::to_string(window) + basisIdentifier;
        }

        path = (parameter >= 0) ? basename + "/param" + std::to_string(parameter) + "_" : basename + "/";
        return path + prefix + fileName;
    }

    void SetupEQP_Force(const CAROM::Matrix* snapX, const CAROM::Matrix* snapV, const CAROM::Matrix* snapE,
                        const CAROM::Matrix* basisV, const CAROM::Matrix* basisE, ROM_Options const& input,
                        Vector const& sol);

    void SetupEQP_Force_Eq(const CAROM::Matrix* snapX, const CAROM::Matrix* snapV, const CAROM::Matrix* snapE,
                           const CAROM::Matrix* basisV, const CAROM::Matrix* basisE, ROM_Options const& input,
                           bool equationE);

	void SetupEQP_En_Force_Eq(const CAROM::Matrix* snapX, const CAROM::Matrix* snapV, const CAROM::Matrix* snapE,
							  const CAROM::Matrix* basisV, const CAROM::Matrix* basisE, ROM_Options const& input);
};

class ROM_Basis
{
    friend class STROM_Basis;

public:
    ROM_Basis(ROM_Options const& input, MPI_Comm comm_,
              const double sFactorX=1.0, const double sFactorV=1.0,
              const std::vector<double> *timesteps=NULL);

    ~ROM_Basis();

    void Init(ROM_Options const& input, Vector const& S);

    void ReadSolutionBases(const int window);
    void ReadTemporalBases(const int window);

    void ProjectFOMtoROM(Vector const& f, Vector & r,
                         const bool timeDerivative=false);

    void ProjectFOMtoROM_V(Vector const& f, Vector & r,
                           const bool timeDerivative=false);

	void AddLastCol_V(Vector const& f);
	void AddLastCol_E(Vector const& f);

    void LiftROMtoFOM(Vector const& r, Vector & f);
    void LiftROMtoFOM_dVdt(Vector const& r, Vector & f);
    void LiftROMtoFOM_dEdt(Vector const& r, Vector & f);

    ParMesh *GetSampleMesh() {
        return sample_pmesh;
    }

    int SolutionSize() const;
    int SolutionSizeSP() const;
    int SolutionSizeFOM() const;

    int SolutionSizeH1SP() const {
        return size_H1_sp;
    }

    int SolutionSizeL2SP() const {
        return size_L2_sp;
    }

    int SolutionSizeH1FOM() const {
        return H1size;
    }

    int SolutionSizeL2FOM() const {
        return L2size;
    }

    void LiftToSampleMesh(const Vector &x, Vector &xsp) const;
    void RestrictFromSampleMesh(const Vector &xsp, Vector &x,
                                const bool timeDerivative=false,
                                const bool rhs_without_mass_matrix=false,
                                const DenseMatrix *invMvROM=NULL,
                                const DenseMatrix *invMeROM=NULL) const;

    void RestrictFromSampleMesh_V(const Vector &xsp, Vector &x) const;
    void RestrictFromSampleMesh_E(const Vector &xsp, Vector &x) const;

    void ProjectFromSampleMesh(const Vector &usp, Vector &u,
                               const bool timeDerivative) const;

    void HyperreduceRHS_V(Vector &v) const;
    void HyperreduceRHS_E(Vector &e) const;

    void ProjectFromPreviousWindow(ROM_Options const& input, Vector& romS, int window, int rdimxPrev, int rdimvPrev, int rdimePrev);
    void computeWindowProjection(const ROM_Basis& basisPrev, ROM_Options const& input, const int window);

    void writeSP(ROM_Options const& input, const int window = 0) const;
    void readSP(ROM_Options const& input, const int window = 0);

    void writePDweights(const int id, const int window = 0) const;

    double GetOffsetX(const int idx) const {
        return (*initX)(idx);
    }

    void Set_dxdt_Reduced(const Vector &x, Vector &y) const;

    int GetRank() const {
        return rank;
    }

    int GetDimX() const {
        return rdimx;
    }

    int GetDimV() const {
        return rdimv;
    }

    int GetDimE() const {
        return rdime;
    }

    int GetDimFv() const {
        return rdimfv;
    }

    int GetDimFe() const {
        return rdimfe;
    }

    void ApplyEssentialBCtoInitXsp(Array<int> const& ess_tdofs);

    void GetBasisVectorV(const bool sp, const int id, Vector &v) const;
    void GetBasisVectorE(const bool sp, const int id, Vector &v) const;

    CAROM::Matrix *GetBVsp() {
        return BVsp;
    }

    CAROM::Matrix *GetBEsp() {
        return BEsp;
    }

    void ComputeReducedMatrices(bool sns1);

    void SetSpaceTimeInitialGuess(ROM_Options const& input);  // TODO: private function?
    void GetSpaceTimeInitialGuess(Vector& st) const;

    // TODO: should these be public?
    int GetTemporalSize() const {
        return temporalSize;
    }
    void ScaleByTemporalBasis(const int t, Vector const& u, Vector &ut);

    MPI_Comm comm;

    CAROM::SampleMeshManager *smm = NULL;
    CAROM::SampleDOFSelector *sampleSelector = NULL;

    CAROM::Matrix* PiXtransPiV = 0;  // TODO: make this private and use a function to access its mult
    CAROM::Matrix* PiXtransPiX = 0;  // TODO: make this private and use a function to access its mult
    CAROM::Matrix* PiXtransPiXlag = 0;  // TODO: make this private and use a function to access its mult

private:
    const bool hyperreduce;
    const bool hyperreduce_prep;
    const bool use_sample_mesh; // whether to use sample mesh; True only when hyperreduce mode with GNAT, QDEIM, S-OPT
    const bool offsetInit;
    const bool use_sns;
    hydrodynamics::LagrangianHydroOperator *lhoper; // for SNS
    const bool useGramSchmidt;
    int rdimx, rdimv, rdime, rdimfv, rdimfe;
    int nprocs, rank, rowOffsetH1, rowOffsetL2;

    const bool useXV;  // If true, use V basis for X-X0.
    const bool useVX;  // If true, use X-X0 for V.
    const bool mergeXV;  // If true, merge bases for X-X0 and V.

    int H1size;
    int L2size;
    int tH1size;
    int tL2size;

    std::string basisIdentifier;

    CAROM::Matrix* basisX = 0;
    CAROM::Matrix* basisV = 0;
    CAROM::Matrix* basisE = 0;
    CAROM::Matrix* basisFv = 0;
    CAROM::Matrix* basisFe = 0;

    std::string basename = "run";
    std::string testing_parameter_basename = "run";
    std::string hyperreduce_basename = "run";
    std::string initSamples_basename = "";

    CAROM::Vector *fH1, *fL2;

    mutable Vector mfH1, mfL2;

    ParGridFunction* gfH1;
    ParGridFunction* gfL2;

    CAROM::Vector *rX = 0;
    CAROM::Vector *rV = 0;
    CAROM::Vector *rE = 0;

    CAROM::Vector *rX2 = 0;
    CAROM::Vector *rV2 = 0;
    CAROM::Vector *rE2 = 0;

    // For hyperreduction
    ParMesh* sample_pmesh = 0;

    CAROM::Matrix *BXsp = NULL;
    CAROM::Matrix *BVsp = NULL;
    CAROM::Matrix *BEsp = NULL;
    CAROM::Matrix *BFvsp = NULL;
    CAROM::Matrix *BFesp = NULL;

    int size_H1_sp = 0;
    int size_L2_sp = 0;

protected:
    CAROM::Vector *spX = NULL;
    CAROM::Vector *spV = NULL;
    CAROM::Vector *spE = NULL;

    CAROM::Vector *sX = NULL;
    CAROM::Vector *sV = NULL;
    CAROM::Vector *sE = NULL;

    CAROM::Matrix *BsinvX = NULL;
    CAROM::Matrix *BsinvV = NULL;
    CAROM::Matrix *BsinvE = NULL;

    CAROM::Matrix *BwinX = NULL;
    CAROM::Matrix *BwinV = NULL;
    CAROM::Matrix *BwinE = NULL;

    CAROM::Vector *initX = 0;
    CAROM::Vector *initV = 0;
    CAROM::Vector *initE = 0;
    CAROM::Vector *initXsp = 0;
    CAROM::Vector *initVsp = 0;
    CAROM::Vector *initEsp = 0;
    CAROM::Vector *BX0 = NULL;

    CAROM::Vector *BtInitDiffX = 0;  // TODO: destructor
    CAROM::Vector *BtInitDiffV = 0;
    CAROM::Vector *BtInitDiffE = 0;

    int numSamplesX = 0;
    int numSamplesV = 0;
    int numSamplesE = 0;

    std::vector<int> initSamplesV;
    std::vector<int> initSamplesE;

    int numTimeSamplesV = 0;
    int numTimeSamplesE = 0;

    const bool Voffset;

    const bool RK2AvgFormulation;
    CAROM::Matrix *BXXinv = NULL;
    CAROM::Matrix *BVVinv = NULL;
    CAROM::Matrix *BEEinv = NULL;

    double energyFraction_X;

    void SetupHyperreduction(ParFiniteElementSpace *H1FESpace, ParFiniteElementSpace *L2FESpace, Array<int>& nH1, const int window,
                             const std::vector<double> *timesteps);

    std::vector<int> paramID_list;
    std::vector<double> coeff_list;

private:
    void SetSpaceTimeInitialGuessComponent(Vector& st, std::string const& name,
                                           ParFiniteElementSpace *fespace,
                                           const CAROM::Matrix* basis,
                                           const CAROM::Matrix* tbasis,
                                           const int nt,
                                           const int rdim) const;

    void SampleMeshAddInitialState(Vector &usp) const;

    // Space-time data
    const double t_initial = 0.0;  // Note that the initial time is hard-coded as 0.0
    const HyperreductionSamplingType hyperreductionSamplingType;
    const SpaceTimeMethod spaceTimeMethod;
    const bool spaceTime;  // whether space-time is used
    int temporalSize = 0;
    int VTos = 0;  // Velocity temporal index offset, used for V and Fe. This fixes the issue that V and Fe are not sampled at t=0, since they are initially zero. This is valid for the Sedov test but not in general when the initial velocity is nonzero.
    // TODO: generalize for nonzero initial velocity.

    CAROM::Matrix* tbasisX = 0;
    CAROM::Matrix* tbasisV = 0;
    CAROM::Matrix* tbasisE = 0;
    CAROM::Matrix* tbasisFv = 0;
    CAROM::Matrix* tbasisFe = 0;

    CAROM::Matrix* PiVtransPiFv = 0;
    CAROM::Matrix* PiEtransPiFe = 0;

    // TODO: delete the new pointers added for space-time

    std::vector<int> timeSamples;  // merged V and E time samples

    Vector st0;
};

class STROM_Basis
{
public:
    STROM_Basis(ROM_Options const& input, ROM_Basis *b_, std::vector<double> *timesteps_)
        : b(b_), u_ti(b_->SolutionSize()), timesteps(timesteps_), spaceTimeMethod(input.spaceTimeMethod)
    {
    }

    int SolutionSizeST() const {
        // TODO: this assumes 1 temporal basis vector for each spatial vector. Generalize to allow for multiple temporal basis vectors per spatial vector.
        return b->SolutionSize();
    }

    int GetNumSpatialSamples() const {
        if (spaceTimeMethod == gnat_lspg || spaceTimeMethod == coll_lspg)
            return (2*b->numSamplesV) + b->numSamplesE;  // use V samples for X
        else  // Galerkin case
            return b->numSamplesX + b->numSamplesV + b->numSamplesE;
    }

    int GetNumSampledTimes() const {
        return b->timeSamples.size();
    }

    int GetNumSamplesV() const {
        // NOTE: this assumes the space-time samples are the Cartesian product of spatial and temporal samples.
        // This may need to be generalized in the future, depending on the space-time sampling algorithm.
        return b->numSamplesV * GetNumSampledTimes();
    }

    int GetNumSamplesE() const {
        // NOTE: this assumes the space-time samples are the Cartesian product of spatial and temporal samples.
        // This may need to be generalized in the future, depending on the space-time sampling algorithm.
        return b->numSamplesE * GetNumSampledTimes();
    }

    int GetTotalNumSamples() const {
        // NOTE: this assumes the space-time samples are the Cartesian product of spatial and temporal samples.
        // This may need to be generalized in the future, depending on the space-time sampling algorithm.
        return GetNumSpatialSamples() * GetNumSampledTimes();
    }

    int GetTimeSampleIndex(const int i) const {
        return b->timeSamples[i];
    }

    double GetTimeSample(const int i) const {
        MFEM_VERIFY(i <= timesteps->size(), "");
        return (i == 0) ? t_initial : (*timesteps)[i-1];
    }

    double GetTimestep(const int i) const {
        MFEM_VERIFY(i < timesteps->size(), "");
        return GetTimeSample(i+1) - GetTimeSample(i);
    }

    void LiftToSampleMesh(const int ti, Vector const& x, Vector &xsp) const;

    void RestrictFromSampleMesh(const int ti, Vector const& usp, Vector &u) const;

    void ApplySpaceTimeHyperreductionInverses(Vector const& u, Vector &w) const;

private:
    ROM_Basis *b;  // stores spatial and temporal bases
    mutable Vector u_ti;
    const double t_initial = 0.0;  // Note that the initial time is hard-coded as 0.0
    std::vector<double> *timesteps; // Positive timestep times (excluding initial time which is assumed to be zero).

    const SpaceTimeMethod spaceTimeMethod;

    const bool GaussNewton = true; // TODO: eliminate this
};

class ROM_Operator : public TimeDependentOperator
{
public:
    ROM_Operator(ROM_Options const& input, ROM_Basis *b, Coefficient& rho_coeff,
                 FunctionCoefficient& mat_coeff, const int order_e, const int source,
                 const bool visc, const bool vort, const double cfl, const bool p_assembly, const double cg_tol,
                 const int cg_max_iter, const double ftz_tol,
                 H1_FECollection *H1fec = NULL, FiniteElementCollection *L2fec = NULL,
                 std::vector<double> *timesteps = NULL);

    virtual void Mult(const Vector &x, Vector &y) const;

    void UpdateSampleMeshNodes(Vector const& romSol);
    double GetTimeStepEstimateSP() const {
        if (!use_sample_mesh) return 0.0;

        if (rank == 0)
        {
            operSP->ResetTimeStepEstimate();
            dt_est_SP = operSP->GetTimeStepEstimate(fx);
        }

        MPI_Bcast(&dt_est_SP, 1, MPI_DOUBLE, 0, basis->comm);

        return dt_est_SP;
    }

    void StepRK2Avg(Vector &S, double &t, double &dt) const;

    void ApplyHyperreduction(Vector &S);
    void PostprocessHyperreduction(Vector &S, bool keep_data=false);

    // TODO: should the following space time functions be refactored into a new space time ROM operator class?
    void EvalSpaceTimeResidual_RK4(Vector const& S, Vector &f) const;  // TODO: private function?
    void EvalSpaceTimeJacobian_RK4(Vector const& S, DenseMatrix &J) const;  // TODO: private function?

    void SolveSpaceTime(Vector &S);
    void SolveSpaceTimeGN(Vector &S);

	void ForceIntegratorEQP_FOM(Vector & rhs, bool energy_conserve = false) const;
	void ForceIntegratorEQP(Vector & res, bool energy_conserve = false) const;

	void ForceIntegratorEQP_E_FOM(Vector const& v, Vector & rhs,
			bool energy_conserve = false) const;
	void ForceIntegratorEQP_E(Vector const& v, Vector & res,
			bool energy_conserve = false) const;

	HyperreductionSamplingType getSamplingType() const;

    void InitEQP() const;

    ~ROM_Operator()
    {
        operFOM->ResetEQP();

        delete mat_gf_coeff;
        delete mat_gf;
        delete L2FESpaceSP;
        delete H1FESpaceSP;
        delete mat_fes;
        delete mat_fec;
        delete spmesh;
        delete xsp_gf;
    }

private:
    hydrodynamics::LagrangianHydroOperator *operFOM = NULL;
    hydrodynamics::LagrangianHydroOperator *operSP = NULL;

    Array<int> ess_tdofs;

    ROM_Basis *basis;

    // Space-time data
    STROM_Basis *STbasis = 0;
    ODESolver *ST_ode_solver = 0;
    mutable Vector Sr;

    mutable Vector fx, fy;

    const bool hyperreduce;
    HyperreductionSamplingType hyperreductionSamplingType = gnat;
    bool use_sample_mesh = false; // whether to use sample mesh; True only when hyperreduce mode with GNAT, QDEIM, S-OPT

    int Vsize_l2sp, Vsize_h1sp;
    ParFiniteElementSpace *L2FESpaceSP = 0;
    ParFiniteElementSpace *H1FESpaceSP = 0;
    ParMesh *spmesh = 0;

    ParFiniteElementSpace *mat_fes = 0;
    ParGridFunction *mat_gf = 0;
    GridFunctionCoefficient *mat_gf_coeff = 0;
    L2_FECollection *mat_fec = 0;

    ParGridFunction *xsp_gf = 0;

    const int rank;

    mutable double dt_est_SP = 0.0;

    bool sns1 = false; // Simplify calculation by Eq. (4.4) in arXiv 1809.04064 when using 1st choice of SNS.
    bool noMsolve = false;
    bool useReducedM = false;  // TODO: remove this?

    DenseMatrix invMvROM, invMeROM;

    void ComputeReducedMv();
    void ComputeReducedMe();

    const bool useGramSchmidt;
    DenseMatrix CoordinateBVsp, CoordinateBEsp;  // TODO: use DenseSymmetricMatrix in mfem/linalg/symmat.hpp
    void InducedInnerProduct(const int id1, const int id2, const int var, const int dim, double& ip);
    void InducedGramSchmidt(const int var, Vector &S);

    const SpaceTimeMethod spaceTimeMethod;

    const bool GaussNewton = true; // TODO: eliminate this

    void UndoInducedGramSchmidt(const int var, Vector &S, bool keep_data);

    void ReadSolutionNNLS(ROM_Options const& input, string basename,
                          std::vector<int> & indices,
                          std::vector<double> & weights);

    // Data for EQP
    std::vector<int> eqpI, eqpI_E;
    std::vector<double> eqpW, eqpW_E;

    mutable bool eqp_init = false;
    mutable bool eqp_init_E = false;
    mutable int nvdof = 0;
    mutable int nedof = 0;

    mutable DenseMatrix W_elems, W_E_elems;

    CAROM::Matrix* Wmat = 0;
    CAROM::Matrix* Wmat_E = 0;

    ParFiniteElementSpace *H1spaceFOM = nullptr; // FOM H1 FEM space
    ParFiniteElementSpace *L2spaceFOM = nullptr; // FOM L2 FEM space
};

CAROM::GreedySampler* BuildROMDatabase(ROM_Options& romOptions, double& t_final, const int myid, const std::string outputPath,
                                       bool& rom_offline, bool& rom_online, bool& rom_restore, const bool usingWindows, bool& rom_calc_error_indicator, bool& rom_calc_rel_error_nonlocal, bool& rom_calc_rel_error_local, bool& rom_read_greedy_twparam, const char* greedyParamString, const char* greedyErrorIndicatorType, const char* greedySamplingType);

CAROM::GreedySampler* UseROMDatabase(ROM_Options& romOptions, const int myid, const std::string outputPath, const char* greedyParamString);

#endif // MFEM_LAGHOS_ROM
