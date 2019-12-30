#ifndef MFEM_LAGHOS_ROM
#define MFEM_LAGHOS_ROM

#include "mfem.hpp"

#include "StaticSVDBasisGenerator.h"
#include "IncrementalSVDBasisGenerator.h"
#include "BasisReader.h"

//using namespace CAROM;
using namespace mfem;

namespace ROMBasisName {
  const char* const X = "basisX";
  const char* const V = "basisV";
  const char* const E = "basisE";
};

class ROM_Sampler
{  
public:
  ROM_Sampler(const int rank_, const int H1size_, const int L2size_, const double t_final,
	      const double initial_dt, Vector const& S_init, const bool staticSVD = false)
    : rank(rank_), H1size(H1size_), L2size(L2size_), X(H1size_), dXdt(H1size_),
      V(H1size_), dVdt(H1size_), E(L2size_), dEdt(L2size_)
  {
    // TODO: read the following parameters from input?
    double model_linearity_tol = 1.e-7;
    double model_sampling_tol = 1.e-7;

    int max_model_dim = int(t_final/initial_dt + 0.5);

    if (staticSVD)
      {
	generator_X = new CAROM::StaticSVDBasisGenerator(H1size, max_model_dim,
							 ROMBasisName::X);
	generator_V = new CAROM::StaticSVDBasisGenerator(H1size, max_model_dim,
							 ROMBasisName::V);
	generator_E = new CAROM::StaticSVDBasisGenerator(L2size, max_model_dim,
							 ROMBasisName::E);
      }
    else
      {
	generator_X = new CAROM::IncrementalSVDBasisGenerator(H1size,
							      model_linearity_tol,
							      false,
							      true,
							      H1size,
							      initial_dt,
							      max_model_dim,
							      model_sampling_tol,
							      t_final,
							      ROMBasisName::X);
	generator_V = new CAROM::IncrementalSVDBasisGenerator(H1size,
							      model_linearity_tol,
							      false,
							      true,
							      H1size,
							      initial_dt,
							      max_model_dim,
							      model_sampling_tol,
							      t_final,
							      ROMBasisName::V);
	generator_E = new CAROM::IncrementalSVDBasisGenerator(L2size,
							      model_linearity_tol,
							      false,
							      true,
							      L2size,
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
  }

  void SampleSolution(const double t, const double dt, Vector const& S);

  void Finalize(const double t, const double dt, Vector const& S);
  
private:
  const int H1size;
  const int L2size;
  const int rank;
  
  CAROM::SVDBasisGenerator *generator_X, *generator_V, *generator_E;

  Vector X, dXdt, V, dVdt, E, dEdt;

  void SetStateVariables(Vector const& S)
  {
    for (int i=0; i<H1size; ++i)
      {
	X[i] = S[i];
	V[i] = S[H1size + i];
      }
    
    for (int i=0; i<L2size; ++i)
      {
	E[i] = S[(2*H1size) + i];
      }
  }
  
  void SetStateVariableRates(const double dt, Vector const& S)
  {
    for (int i=0; i<H1size; ++i)
      {
	dXdt[i] = (S[i] - X[i]) / dt;
	dVdt[i] = (S[H1size + i] - V[i]) / dt;
      }
    
    for (int i=0; i<L2size; ++i)
      {
	dEdt[i] = (S[(2*H1size) + i] - E[i]) / dt;
      }
  }
};

class ROM_Basis
{
public:
  ROM_Basis(MPI_Comm comm_, const int H1size_, const int L2size_,
	    const int dimX, const int dimV, const int dimE,
	    const bool staticSVD_ = false);

  void ReadSolutionBases();

private:
  const bool staticSVD;
  int rdimx, rdimv, rdime;
  MPI_Comm comm;

  int nprocs, rank, rowOffsetH1, rowOffsetL2;

  const int H1size;
  const int L2size;

  CAROM::Matrix* basisX = 0;
  CAROM::Matrix* basisV = 0;
  CAROM::Matrix* basisE = 0;
};

#endif // MFEM_LAGHOS_ROM
