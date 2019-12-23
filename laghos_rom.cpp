#include "laghos_rom.hpp"

using namespace std;

void ROM_Sampler::SampleSolution(const double t, const double dt, Vector const& S)
{
  SetStateVariableRates(dt, S);
  SetStateVariables(S);

  const bool sampleX = generator_X->isNextSample(t);

  if (sampleX)
    {
      if (rank == 0)
	{
	  cout << "X taking sample at t " << t << endl;
	}
      
      generator_X->takeSample(X.GetData(), t, dt);
      generator_X->computeNextSampleTime(X.GetData(), dXdt.GetData(), t);
    }

  const bool sampleV = generator_V->isNextSample(t);

  if (sampleV)
    {
      if (rank == 0)
	{
	  cout << "V taking sample at t " << t << endl;
	}
      
      generator_V->takeSample(V.GetData(), t, dt);
      generator_V->computeNextSampleTime(V.GetData(), dVdt.GetData(), t);
    }

  const bool sampleE = generator_E->isNextSample(t);

  if (sampleE)
    {
      if (rank == 0)
	{
	  cout << "E taking sample at t " << t << endl;
	}
      
      generator_E->takeSample(E.GetData(), t, dt);
      generator_E->computeNextSampleTime(E.GetData(), dEdt.GetData(), t);
    }
}

void BasisGeneratorFinalSummary(CAROM::SVDBasisGenerator* bg)
{
  const int rom_dim = bg->getSpatialBasis()->numColumns();
  cout << "ROM dimension = " << rom_dim << endl;

  const CAROM::Matrix* sing_vals = bg->getSingularValues();
            
  cout << "Singular Values:" << endl;
  for (int sv = 0; sv < sing_vals->numColumns(); ++sv) {
    cout << (*sing_vals)(sv, sv) << endl;
  }
}

void ROM_Sampler::Finalize(const double t, const double dt, Vector const& S)
{
  SetStateVariables(S);
  
  generator_X->takeSample(X.GetData(), t, dt);
  generator_X->endSamples();

  generator_V->takeSample(V.GetData(), t, dt);
  generator_V->endSamples();

  generator_E->takeSample(E.GetData(), t, dt);
  generator_E->endSamples();

  if (rank == 0)
    {
      cout << "X basis summary output" << endl;
      BasisGeneratorFinalSummary(generator_X);

      cout << "V basis summary output" << endl;
      BasisGeneratorFinalSummary(generator_V);

      cout << "E basis summary output" << endl;
      BasisGeneratorFinalSummary(generator_E);
    }
}
