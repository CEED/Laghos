#include "laghos_rom.hpp"

using namespace std;

void ROM_Sampler::SampleSolution(const double t, const double dt, Vector const& S)
{
  for (int i=0; i<H1size; ++i)
    {
      dXdt[i] = (S[i] - X[i]) / dt;
    }

  const bool sampleX = generator_X->isNextSample(t);

  if (sampleX)
    {
      cout << "X taking sample at t " << t << endl;
      generator_X->takeSample(X.GetData(), t, dt);
      generator_X->computeNextSampleTime(X.GetData(), dXdt.GetData(), t);
    }
}
