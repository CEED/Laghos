#ifndef MFEM_LAGHOS_ROM
#define MFEM_LAGHOS_ROM

#include "mfem.hpp"

#include "StaticSVDBasisGenerator.h"
#include "IncrementalSVDBasisGenerator.h"
#include "BasisReader.h"

#include "laghos_solver.hpp"

//using namespace CAROM;
using namespace mfem;

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
	      const bool staticSVD = false)
    : rank(rank_), tH1size(H1FESpace->GetTrueVSize()), tL2size(L2FESpace->GetTrueVSize()),
      H1size(H1FESpace->GetVSize()), L2size(L2FESpace->GetVSize()),
      X(tH1size), dXdt(tH1size), V(tH1size), dVdt(tH1size), E(tL2size), dEdt(tL2size),
      gfH1(H1FESpace), gfL2(L2FESpace)
  {
    // TODO: read the following parameters from input?
    double model_linearity_tol = 1.e-7;
    double model_sampling_tol = 1.e-7;

    int max_model_dim = int(t_final/initial_dt + 0.5);

    if (staticSVD)
      {
	generator_X = new CAROM::StaticSVDBasisGenerator(tH1size, max_model_dim,
							 ROMBasisName::X);
	generator_V = new CAROM::StaticSVDBasisGenerator(tH1size, max_model_dim,
							 ROMBasisName::V);
	generator_E = new CAROM::StaticSVDBasisGenerator(tL2size, max_model_dim,
							 ROMBasisName::E);
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
							      ROMBasisName::X);
	generator_V = new CAROM::IncrementalSVDBasisGenerator(tH1size,
							      model_linearity_tol,
							      false,
							      true,
							      tH1size,
							      initial_dt,
							      max_model_dim,
							      model_sampling_tol,
							      t_final,
							      ROMBasisName::V);
	generator_E = new CAROM::IncrementalSVDBasisGenerator(tL2size,
							      model_linearity_tol,
							      false,
							      true,
							      tL2size,
							      initial_dt,
							      max_model_dim,
							      model_sampling_tol,
							      t_final,
							      ROMBasisName::E);
      }

    SetStateVariables(S_init);

    dXdt = 0.0;
    dVdt = 0.0;
    dEdt = 0.0;

    X0 = 0.0;
    V0 = 0.0;
    E0 = 0.0;
  }
  
  void SampleSolution(const double t, const double dt, Vector const& S);

  void Finalize(const double t, const double dt, Vector const& S);
  
private:
  const int H1size;
  const int L2size;
  const int tH1size;
  const int tL2size;

  const int rank;
  
  CAROM::SVDBasisGenerator *generator_X, *generator_V, *generator_E;

  Vector X, X0, dXdt, V, V0, dVdt, E, E0, dEdt;

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
	    int & dimX, int & dimV, int & dimE,
	    const bool staticSVD_ = false);

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
  }
  
  void ReadSolutionBases();

  void ProjectFOMtoROM(Vector const& f, Vector & r);
  void LiftROMtoFOM(Vector const& r, Vector & f);
  int TotalSize() { return rdimx + rdimv + rdime; }
  
private:
  const bool staticSVD;
  int rdimx, rdimv, rdime;
  MPI_Comm comm;

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
};

class ROM_Operator : public TimeDependentOperator
{
public:
  ROM_Operator(hydrodynamics::LagrangianHydroOperator *lhoper, ROM_Basis *b)
    : TimeDependentOperator(b->TotalSize()), operFOM(lhoper), basis(b),
      fx(lhoper->Height()), fy(lhoper->Height())
  {
    MFEM_VERIFY(lhoper->Height() == lhoper->Width(), "");
  }

  virtual void Mult(const Vector &x, Vector &y) const;

private:
  hydrodynamics::LagrangianHydroOperator *operFOM;
  ROM_Basis *basis;

  mutable Vector fx, fy;
};

#endif // MFEM_LAGHOS_ROM
