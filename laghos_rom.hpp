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
};

class ROM_Sampler
{
public:
    ROM_Sampler(const int rank_, ParFiniteElementSpace *H1FESpace, ParFiniteElementSpace *L2FESpace,
                const double t_final, const double initial_dt, Vector const& S_init,
                const bool staticSVD = false, const bool useXoffset = false, double energyFraction_=0.9999,
                const int window=0)
        : rank(rank_), tH1size(H1FESpace->GetTrueVSize()), tL2size(L2FESpace->GetTrueVSize()),
          H1size(H1FESpace->GetVSize()), L2size(L2FESpace->GetVSize()),
          X(tH1size), dXdt(tH1size), V(tH1size), dVdt(tH1size), E(tL2size), dEdt(tL2size),
          gfH1(H1FESpace), gfL2(L2FESpace), offsetXinit(useXoffset), energyFraction(energyFraction_)
    {
        // TODO: read the following parameters from input?
        double model_linearity_tol = 1.e-7;
        double model_sampling_tol = 1.e-7;

        int max_model_dim = int(t_final/initial_dt + 0.5) + 100;

        std::cout << rank << ": max_model_dim " << max_model_dim << std::endl;

        if (staticSVD)
        {
            generator_X = new CAROM::StaticSVDBasisGenerator(tH1size, max_model_dim,
                    ROMBasisName::X + std::to_string(window));
            generator_V = new CAROM::StaticSVDBasisGenerator(tH1size, max_model_dim,
                    ROMBasisName::V + std::to_string(window));
            generator_E = new CAROM::StaticSVDBasisGenerator(tL2size, max_model_dim,
                    ROMBasisName::E + std::to_string(window));
        }
        else
        {
            generator_X = new CAROM::IncrementalSVDBasisGenerator(tH1size,
                    model_linearity_tol,
                    false,
                    true,
                    tH1size,
                    initial_dt,
                    max_model_dim,
                    model_sampling_tol,
                    t_final,
                    ROMBasisName::X + std::to_string(window));
            generator_V = new CAROM::IncrementalSVDBasisGenerator(tH1size,
                    model_linearity_tol,
                    false,
                    true,
                    tH1size,
                    initial_dt,
                    max_model_dim,
                    model_sampling_tol,
                    t_final,
                    ROMBasisName::V + std::to_string(window));
            generator_E = new CAROM::IncrementalSVDBasisGenerator(tL2size,
                    model_linearity_tol,
                    false,
                    true,
                    tL2size,
                    initial_dt,
                    max_model_dim,
                    model_sampling_tol,
                    t_final,
                    ROMBasisName::E + std::to_string(window));
        }

        SetStateVariables(S_init);

        dXdt = 0.0;
        dVdt = 0.0;
        dEdt = 0.0;

        X0 = 0.0;
        V0 = 0.0;
        E0 = 0.0;

        if (offsetXinit)
        {
            initX = new CAROM::Vector(tH1size, true);
            Xdiff.SetSize(tH1size);
            for (int i=0; i<tH1size; ++i)
            {
                (*initX)(i) = X[i];
            }

            initX->write("initX");
        }
    }

    void SampleSolution(const double t, const double dt, Vector const& S);

    void Finalize(const double t, const double dt, Vector const& S);

private:
    const int H1size;
    const int L2size;
    const int tH1size;
    const int tL2size;

    const int rank;
    double energyFraction;

    CAROM::SVDBasisGenerator *generator_X, *generator_V, *generator_E;

    Vector X, X0, Xdiff, dXdt, V, V0, dVdt, E, E0, dEdt;

    const bool offsetXinit;
    CAROM::Vector *initX = 0;

    ParGridFunction gfH1, gfL2;

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

class ROM_Basis
{
public:
    ROM_Basis(MPI_Comm comm_, ParFiniteElementSpace *H1FESpace, ParFiniteElementSpace *L2FESpace,
              int & dimX, int & dimV, int & dimE, int nsamx, int nsamv, int nsame,
              const bool staticSVD_ = false, const bool hyperreduce_ = false, const bool useXoffset = false,
              const int window=0);

    ~ROM_Basis()
    {
        delete rX;
        delete rV;
        delete rE;
        delete basisX;
        delete basisV;
        delete basisE;
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
    }

    void ReadSolutionBases(const int window);

    void ProjectFOMtoROM(Vector const& f, Vector & r);
    void LiftROMtoFOM(Vector const& r, Vector & f);
    int TotalSize() {
        return rdimx + rdimv + rdime;
    }

    ParMesh *GetSampleMesh() {
        return sample_pmesh;
    }

    int SolutionSize() const;
    int SolutionSizeSP() const;
    int SolutionSizeFOM() const;

    void LiftToSampleMesh(const Vector &x, Vector &xsp) const;
    void RestrictFromSampleMesh(const Vector &xsp, Vector &x) const;

    int GetRank() const {
        return rank;
    }

    void ApplyEssentialBCtoInitXsp(Array<int> const& ess_tdofs);

    MPI_Comm comm;

private:
    const bool staticSVD;
    const bool hyperreduce;
    const bool offsetXinit;
    int rdimx, rdimv, rdime;

    int nprocs, rank, rowOffsetH1, rowOffsetL2;

    const int H1size;
    const int L2size;
    const int tH1size;
    const int tL2size;

    CAROM::Matrix* basisX = 0;
    CAROM::Matrix* basisV = 0;
    CAROM::Matrix* basisE = 0;

    CAROM::Vector *fH1, *fL2;

    Vector mfH1, mfL2;

    ParGridFunction gfH1, gfL2;

    CAROM::Vector *rX = 0;
    CAROM::Vector *rV = 0;
    CAROM::Vector *rE = 0;

    // For hyperreduction
    std::vector<int> s2sp_X, s2sp_V, s2sp_E;
    ParMesh* sample_pmesh = 0;
    std::vector<int> st2sp;  // mapping from stencil dofs in original mesh (st) to stencil dofs in sample mesh (s+)
    std::vector<int> s2sp_H1;  // mapping from sample dofs in original mesh (s) to stencil dofs in sample mesh (s+)
    std::vector<int> s2sp_L2;  // mapping from sample dofs in original mesh (s) to stencil dofs in sample mesh (s+)

    CAROM::Matrix *BXsp = NULL;
    CAROM::Matrix *BVsp = NULL;
    CAROM::Matrix *BEsp = NULL;

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
    CAROM::Vector *initXsp = 0;

    int numSamplesX = 0;
    int numSamplesV = 0;
    int numSamplesE = 0;

    void SetupHyperreduction(ParFiniteElementSpace *H1FESpace, ParFiniteElementSpace *L2FESpace, Array<int>& nH1);
};

class ROM_Operator : public TimeDependentOperator
{
public:
    ROM_Operator(hydrodynamics::LagrangianHydroOperator *lhoper, ROM_Basis *b, FunctionCoefficient& rho_coeff,
                 FunctionCoefficient& mat_coeff, const int order_e, const int source,
                 const bool visc, const double cfl, const bool p_assembly, const double cg_tol,
                 const int cg_max_iter, const double ftz_tol, const bool hyperreduce_ = false,
                 H1_FECollection *H1fec = NULL, FiniteElementCollection *L2fec = NULL);

    virtual void Mult(const Vector &x, Vector &y) const;

    void UpdateSampleMeshNodes(Vector const& romSol);
    double GetTimeStepEstimateSP() const {
        return dt_est_SP;
    }

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
};

#endif // MFEM_LAGHOS_ROM
