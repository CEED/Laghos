#ifndef MFEM_LAGHOS_ROM
#define MFEM_LAGHOS_ROM

#include "mfem.hpp"

#include "StaticSVDBasisGenerator.h"
#include "IncrementalSVDBasisGenerator.h"

//using namespace CAROM;
using namespace mfem;

class ROM_Sampler
{  
public:
  ROM_Sampler(const int rank_, const int H1size_, const int L2size_, const double t_final,
	      const double initial_dt, Vector const& S_init, const bool staticSVD = false)
    : rank(rank_), H1size(H1size_), L2size(L2size_), X(H1size_), dXdt(H1size_)
  {
    // TODO: read the following parameters from input?
    double model_linearity_tol = 1.e-7;
    double model_sampling_tol = 1.e-7;

    int max_model_dim = int(t_final/initial_dt + 0.5);

    if (staticSVD)
      {
	generator_X = new CAROM::StaticSVDBasisGenerator(H1size, max_model_dim,
							 "basisX");
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
							      "basisX");
      }

    for (int i=0; i<H1size; ++i)
      {
	X[i] = S_init[i];
      }
  }

  void SampleSolution(const double t, const double dt, Vector const& S);

  void Finalize(const double t, const double dt, Vector const& S);
  
private:
  const int H1size;
  const int L2size;
  const int rank;
  
  CAROM::SVDBasisGenerator *generator_X, *generator_V, *generator_E;

  Vector X, dXdt;
};

#endif // MFEM_LAGHOS_ROM
