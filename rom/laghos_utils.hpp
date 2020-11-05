#ifndef MFEM_LAGHOS_UTILS
#define MFEM_LAGHOS_UTILS

#include "StaticSVDBasisGenerator.h"
#include "IncrementalSVDBasisGenerator.h"

#include "mfem.hpp"

using namespace std;
using namespace mfem;

void BasisGeneratorFinalSummary(CAROM::SVDBasisGenerator* bg, const double energyFraction, int & cutoff, const bool printout=true);

void PrintSingularValues(const int rank, const std::string& basename, const std::string& name, CAROM::SVDBasisGenerator* bg, const bool usingWindows = false, const int window = -1);

int ReadTimeWindows(const int nw, std::string twfile, Array<double>& twep, const bool printStatus);

int ReadTimeWindowParameters(const int nw, std::string twfile, Array<double>& twep, Array2D<int>& twparam, double sFactor[], const bool printStatus, const bool rhs);

void split_line(const string &line, vector<string> &words);

void SetWindowParameters(Array2D<int> const& twparam, ROM_Options & romOptions);

#endif // MFEM_LAGHOS_UTILS
