#ifndef MFEM_LAGHOS_ROM
#define MFEM_LAGHOS_ROM

#include "mfem.hpp"

#include "StaticSVDBasisGenerator.h"
#include "IncrementalSVDBasisGenerator.h"
#include "BasisReader.h"

#include "laghos_solver.hpp"

//using namespace CAROM;
using namespace mfem;

enum NormType { l1norm=1, l2norm=2, maxnorm=0 };

void PrintNormsOfParGridFunctions(NormType normtype, const int rank, const std::string& name, ParGridFunction *f1, ParGridFunction *f2,
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

struct ROM_Options
{
    int rank = 0;  // MPI rank
    ParFiniteElementSpace *H1FESpace = NULL; // FOM H1 FEM space
    ParFiniteElementSpace *L2FESpace = NULL; // FOM L2 FEM space

    std::string *basename = NULL;

    double t_final = 0.0; // simulation final time
    double initial_dt = 0.0; // initial timestep size

    bool staticSVD = false; // true: use StaticSVDBasisGenerator; false: use IncrementalSVDBasisGenerator
    bool useOffset = false; // if true, sample variables minus initial state as an offset
    bool RHSbasis = false; // if true, use bases for nonlinear RHS terms without mass matrix inverses applied
    double energyFraction = 0.9999; // used for recommending basis sizes, depending on singular values
    int window = 0; // Laghos-ROM time window index
    int max_dim = 0; // maximimum dimension for libROM basis generator time interval
    int parameterID = 0; // index of parameters chosen for this Laghos simulation
    hydrodynamics::LagrangianHydroOperator *FOMoper = NULL; // FOM operator

    // Variable basis dimensions
    int dimX = -1;
    int dimV = -1;
    int dimE = -1;
    int dimFv = -1;
    int dimFe = -1;

    // Number of samples for each variable
    int sampX = 0;
    int sampV = 0;
    int sampE = 0;

    bool hyperreduce = false; // whether to use hyperreduction on ROM online phase
    bool GramSchmidt = false; // whether to use Gram-Schmidt with respect to mass matrices
    bool RK2AvgSolver = false; // true if RK2Avg solver is used for time integration
    bool paramOffset = false; // used for determining offset options in the online stage, depending on parametric ROM or non-parametric
};

class ROM_Sampler
{
public:
    ROM_Sampler(ROM_Options const& input, Vector const& S_init)
        : rank(input.rank), tH1size(input.H1FESpace->GetTrueVSize()), tL2size(input.L2FESpace->GetTrueVSize()),
          H1size(input.H1FESpace->GetVSize()), L2size(input.L2FESpace->GetVSize()),
          X(tH1size), dXdt(tH1size), V(tH1size), dVdt(tH1size), E(tL2size), dEdt(tL2size),
          gfH1(input.H1FESpace), gfL2(input.L2FESpace), offsetInit(input.useOffset), energyFraction(input.energyFraction),
          sampleF(input.RHSbasis), lhoper(input.FOMoper), writeSnapshots(input.parameterID >= 0), parameterID(input.parameterID), basename(*input.basename)
    {
        const int window = input.window;

        if (sampleF)
        {
            MFEM_VERIFY(offsetInit, "");
        }

        // TODO: read the following parameters from input?
        double model_linearity_tol = 1.e-7;
        double model_sampling_tol = 1.e-7;

        const int max_model_dim_est = int(input.t_final/input.initial_dt + 0.5) + 100;  // Note that this is a rough estimate which may be exceeded, resulting in multiple libROM basis time intervals.
        const int max_model_dim = (input.max_dim > 0) ? input.max_dim : max_model_dim_est;

        std::cout << rank << ": max_model_dim " << max_model_dim << std::endl;

        if (input.staticSVD)
        {
            CAROM::StaticSVDOptions static_x_options(
                tH1size,
                max_model_dim
            );
            static_x_options.max_time_intervals = 1;
            CAROM::StaticSVDOptions static_e_options(
                tL2size,
                max_model_dim
            );
            static_e_options.max_time_intervals = 1;
            generator_X = new CAROM::StaticSVDBasisGenerator(
                static_x_options,
                BasisFileName(basename, VariableName::X, window, parameterID));
            generator_V = new CAROM::StaticSVDBasisGenerator(
                static_x_options,
                BasisFileName(basename, VariableName::V, window, parameterID));
            generator_E = new CAROM::StaticSVDBasisGenerator(
                static_e_options,
                BasisFileName(basename, VariableName::E, window, parameterID));

            if (sampleF)
            {
                generator_Fv = new CAROM::StaticSVDBasisGenerator(
                    static_x_options,
                    BasisFileName(basename, VariableName::Fv, window, parameterID));
                generator_Fe = new CAROM::StaticSVDBasisGenerator(
                    static_e_options,
                    BasisFileName(basename, VariableName::Fe, window, parameterID));
            }
        }
        else
        {
            CAROM::IncrementalSVDOptions inc_x_options(
                tH1size,
                max_model_dim,
                model_linearity_tol,
                max_model_dim,
                input.initial_dt,
                model_sampling_tol,
                input.t_final,
                false,
                true
            );
            inc_x_options.max_time_intervals = 1;
            CAROM::IncrementalSVDOptions inc_e_options(
                tL2size,
                max_model_dim,
                model_linearity_tol,
                max_model_dim,
                input.initial_dt,
                model_sampling_tol,
                input.t_final,
                false,
                true
            );
            inc_e_options.max_time_intervals = 1;
            generator_X = new CAROM::IncrementalSVDBasisGenerator(
                inc_x_options,
                basename + "/" + ROMBasisName::X + std::to_string(window));

            generator_V = new CAROM::IncrementalSVDBasisGenerator(
                inc_x_options,
                basename + "/" + ROMBasisName::V + std::to_string(window));

            generator_E = new CAROM::IncrementalSVDBasisGenerator(
                inc_e_options,
                basename + "/" + ROMBasisName::E + std::to_string(window));

            if (sampleF)
            {
                generator_Fv = new CAROM::IncrementalSVDBasisGenerator(
                    inc_x_options,
                    basename + "/" + ROMBasisName::Fv + std::to_string(window));

                generator_Fe = new CAROM::IncrementalSVDBasisGenerator(
                    inc_e_options,
                    basename + "/" + ROMBasisName::Fe + std::to_string(window));
            }
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
            //std::string path_init = (parameterID >= 0) ? "run/ROMoffset/param" + std::to_string(parameterID) + "_init" : "run/ROMoffset/init"; // TODO: Tony PR77
            std::string path_init = basename + "/ROMoffset/init";
            initX = new CAROM::Vector(tH1size, true);
            initV = new CAROM::Vector(tH1size, true);
            initE = new CAROM::Vector(tL2size, true);
            Xdiff.SetSize(tH1size);
            Ediff.SetSize(tL2size);

            for (int i=0; i<tH1size; ++i)
            {
                (*initX)(i) = X[i];
            }
            initX->write(path_init + "X" + std::to_string(window));

            for (int i=0; i<tH1size; ++i)
            {
                (*initV)(i) = V[i];
            }
            initV->write(path_init + "V" + std::to_string(window));
            for (int i=0; i<tL2size; ++i)
            {
                (*initE)(i) = E[i];
            }
            initE->write(path_init + "E" + std::to_string(window));
        }
    }

    void SampleSolution(const double t, const double dt, Vector const& S);

    void Finalize(const double t, const double dt, Vector const& S, Array<int> &cutoff);

    int MaxNumSamples()
    {
        return std::max(std::max(generator_X->getNumSamples(), generator_V->getNumSamples()), generator_E->getNumSamples());
    }

private:
    const int H1size;
    const int L2size;
    const int tH1size;
    const int tL2size;

    const int rank;
    double energyFraction;

    const int parameterID;
    const bool writeSnapshots;
    std::vector<double> tSnapX, tSnapV, tSnapE, tSnapFv, tSnapFe;

    std::string basename = "run";

    CAROM::SVDBasisGenerator *generator_X, *generator_V, *generator_E, *generator_Fv, *generator_Fe;

    Vector X, X0, Xdiff, Ediff, dXdt, V, V0, dVdt, E, E0, dEdt;

    const bool offsetInit;
    CAROM::Vector *initX = 0;
    CAROM::Vector *initV = 0;
    CAROM::Vector *initE = 0;

    ParGridFunction gfH1, gfL2;

    const bool sampleF;

    hydrodynamics::LagrangianHydroOperator *lhoper;

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

    std::string BasisFileName(const std::string basename, VariableName v, const int window, const int parameter)
    {
        std::string fileName, path;

        const std::string prefix = (parameter >= 0) ? "var" : "basis";

        switch (v)
        {
        case VariableName::V:
            fileName = "V" + std::to_string(window);
            break;
        case VariableName::E:
            fileName = "E" + std::to_string(window);
            break;
        case VariableName::Fv:
            fileName = "Fv" + std::to_string(window);
            break;
        case VariableName::Fe:
            fileName = "Fe" + std::to_string(window);
            break;
        default:
            fileName = "X" + std::to_string(window);
        }

        path = (parameter >= 0) ? basename + "/param" + std::to_string(parameter) + "_" : basename + "/";
        return path + prefix + fileName;
    }
};

class ROM_Basis
{
public:
    ROM_Basis(ROM_Options const& input, Vector const& S, MPI_Comm comm_);

    ~ROM_Basis()
    {
        delete rX;
        delete rV;
        delete rE;
        delete basisX;
        delete basisV;
        delete basisE;
        delete basisFv;
        delete basisFe;
        delete fH1;
        delete fL2;
        delete spX;
        delete spV;
        delete spE;
        delete sX;
        delete sV;
        delete sE;
        delete BXsp;
        delete BVsp;
        delete BEsp;
        delete BsinvX;
        delete BsinvV;
        delete BsinvE;
        delete BX0;
    }

    void ReadSolutionBases(const int window);

    void ProjectFOMtoROM(Vector const& f, Vector & r,
                         const bool timeDerivative=false);
    void LiftROMtoFOM(Vector const& r, Vector & f);
    int TotalSize() const {
        return rdimx + rdimv + rdime;
    }

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

    void ApplyEssentialBCtoInitXsp(Array<int> const& ess_tdofs);

    void GetBasisVectorV(const bool sp, const int id, Vector &v) const;
    void GetBasisVectorE(const bool sp, const int id, Vector &v) const;

    CAROM::Matrix *GetBVsp() {
        return BVsp;
    }

    CAROM::Matrix *GetBEsp() {
        return BEsp;
    }

    void ComputeReducedRHS();

    MPI_Comm comm;

private:
    const bool hyperreduce;
    const bool offsetInit;
    const bool RHSbasis;
    const bool useGramSchmidt;
    int rdimx, rdimv, rdime, rdimfv, rdimfe;
    int nprocs, rank, rowOffsetH1, rowOffsetL2;

    const int H1size;
    const int L2size;
    const int tH1size;
    const int tL2size;

    CAROM::Matrix* basisX = 0;
    CAROM::Matrix* basisV = 0;
    CAROM::Matrix* basisE = 0;
    CAROM::Matrix* basisFv = 0;
    CAROM::Matrix* basisFe = 0;

    std::string basename = "run";

    CAROM::Vector *fH1, *fL2;

    Vector mfH1, mfL2;

    ParGridFunction gfH1, gfL2;

    CAROM::Vector *rX = 0;
    CAROM::Vector *rV = 0;
    CAROM::Vector *rE = 0;

    CAROM::Vector *rX2 = 0;
    CAROM::Vector *rV2 = 0;
    CAROM::Vector *rE2 = 0;

    // For hyperreduction
    std::vector<int> s2sp_X, s2sp_V, s2sp_E;
    ParMesh* sample_pmesh = 0;
    std::vector<int> st2sp;  // mapping from stencil dofs in original mesh (st) to stencil dofs in sample mesh (s+)
    std::vector<int> s2sp_H1;  // mapping from sample dofs in original mesh (s) to stencil dofs in sample mesh (s+)
    std::vector<int> s2sp_L2;  // mapping from sample dofs in original mesh (s) to stencil dofs in sample mesh (s+)

    CAROM::Matrix *BXsp = NULL;
    CAROM::Matrix *BVsp = NULL;
    CAROM::Matrix *BEsp = NULL;
    CAROM::Matrix *BFvsp = NULL;
    CAROM::Matrix *BFesp = NULL;

    int size_H1_sp = 0;
    int size_L2_sp = 0;

    CAROM::Vector *spX = NULL;
    CAROM::Vector *spV = NULL;
    CAROM::Vector *spE = NULL;

    CAROM::Vector *sX = NULL;
    CAROM::Vector *sV = NULL;
    CAROM::Vector *sE = NULL;

    CAROM::Matrix *BsinvX = NULL;
    CAROM::Matrix *BsinvV = NULL;
    CAROM::Matrix *BsinvE = NULL;

    CAROM::Vector *initX = 0;
    CAROM::Vector *initV = 0;
    CAROM::Vector *initE = 0;
    CAROM::Vector *initXsp = 0;
    CAROM::Vector *initVsp = 0;
    CAROM::Vector *initEsp = 0;
    CAROM::Vector *BX0 = NULL;

    int numSamplesX = 0;
    int numSamplesV = 0;
    int numSamplesE = 0;

    const bool RK2AvgFormulation;
    CAROM::Matrix *BXXinv = NULL;
    CAROM::Matrix *BVVinv = NULL;
    CAROM::Matrix *BEEinv = NULL;

    void SetupHyperreduction(ParFiniteElementSpace *H1FESpace, ParFiniteElementSpace *L2FESpace, Array<int>& nH1, const int window);
};

class ROM_Operator : public TimeDependentOperator
{
public:
    ROM_Operator(ROM_Options const& input, ROM_Basis *b, Coefficient& rho_coeff,
                 FunctionCoefficient& mat_coeff, const int order_e, const int source,
                 const bool visc, const double cfl, const bool p_assembly, const double cg_tol,
                 const int cg_max_iter, const double ftz_tol,
                 H1_FECollection *H1fec = NULL, FiniteElementCollection *L2fec = NULL);

    virtual void Mult(const Vector &x, Vector &y) const;

    void UpdateSampleMeshNodes(Vector const& romSol);
    double GetTimeStepEstimateSP() const {
        if (!hyperreduce) return 0.0;

        if (rank == 0)
        {
            operSP->ResetTimeStepEstimate();
            dt_est_SP = operSP->GetTimeStepEstimate(fx);
        }

        MPI_Bcast(&dt_est_SP, 1, MPI_DOUBLE, 0, basis->comm);

        return dt_est_SP;
    }

    void StepRK2Avg(Vector &S, double &t, double &dt) const;

    void InducedGramSchmidtInitialize(Vector &S);
    void InducedGramSchmidtFinalize(Vector &S);

    ~ROM_Operator()
    {
        delete mat_gf_coeff;
        delete mat_gf;
        delete L2FESpaceSP;
        delete H1FESpaceSP;
        delete mat_fes;
        delete mat_fec;
        delete spmesh;
    }

private:
    hydrodynamics::LagrangianHydroOperator *operFOM = NULL;
    hydrodynamics::LagrangianHydroOperator *operSP = NULL;

    Array<int> ess_tdofs;

    ROM_Basis *basis;

    mutable Vector fx, fy;

    const bool hyperreduce;

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

    bool useReducedMv = false;  // TODO: remove this?
    bool useReducedMe = false;  // TODO: remove this?

    DenseMatrix invMvROM, invMeROM;

    void ComputeReducedMv();
    void ComputeReducedMe();

    const bool useGramSchmidt;
    DenseMatrix CoordinateBVsp, CoordinateBEsp;
    void InducedInnerProduct(const int id1, const int id2, const int var, const int dim, double& ip);
    void InducedGramSchmidt(const int var, Vector &S);
    void UndoInducedGramSchmidt(const int var, Vector &S);
};

#endif // MFEM_LAGHOS_ROM
