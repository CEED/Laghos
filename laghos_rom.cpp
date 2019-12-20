#include "laghos_rom.hpp"

using namespace std;

void ROM_Sampler::SampleSolution(const double t, const double dt, Vector const& S)
{
  for (int i=0; i<H1size; ++i)
    {
      dXdt[i] = (S[i] - X[i]) / dt;
      X[i] = S[i];
    }

  const bool sampleX = generator_X->isNextSample(t);

  if (sampleX)
    {
      cout << "X taking sample at t " << t << endl;
      generator_X->takeSample(X.GetData(), t, dt);
      generator_X->computeNextSampleTime(X.GetData(), dXdt.GetData(), t);
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
  for (int i=0; i<H1size; ++i)
    {
      X[i] = S[i];
    }
  
  generator_X->takeSample(X.GetData(), t, dt);
  generator_X->endSamples();

  if (rank == 0)
    {
      cout << "X basis summary output" << endl;
      BasisGeneratorFinalSummary(generator_X);
    }
}
