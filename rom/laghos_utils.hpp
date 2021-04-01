#ifndef MFEM_LAGHOS_UTILS
#define MFEM_LAGHOS_UTILS

#include "BasisGenerator.h"

#include "mfem.hpp"

#include "laghos_rom.hpp"

using namespace std;
using namespace mfem;

void BasisGeneratorFinalSummary(CAROM::BasisGenerator* bg, const double energyFraction, int & cutoff, const std::string cutoffOutputPath = "", const bool printout=true);

void PrintSingularValues(const int rank, const std::string& basename, const std::string& name, CAROM::BasisGenerator* bg, const bool usingWindows = false, const int window = -1);

int ReadTimeWindows(const int nw, std::string twfile, Array<double>& twep, const bool printStatus);

int ReadTimeWindowParameters(const int nw, std::string twfile, Array<double>& twep, Array2D<int>& twparam, double sFactor[], const bool printStatus, const bool rhs);

void split_line(const string &line, vector<string> &words);

void SetWindowParameters(Array2D<int> const& twparam, ROM_Options & romOptions);

void writeNum(int num, std::string file_name);

// read data from from text.txt and store it in vector v
void readNum(int& num, std::string file_name);

void writeDouble(double num, std::string file_name);

// read data from from text.txt and store it in vector v
void readDouble(double& num, std::string file_name);

void writeVec(vector<int> v, std::string file_name);

// read data from from text.txt and store it in vector v
void readVec(vector<int> &v, std::string file_name);

#endif // MFEM_LAGHOS_UTILS
