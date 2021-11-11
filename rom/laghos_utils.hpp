#ifndef MFEM_LAGHOS_UTILS
#define MFEM_LAGHOS_UTILS

#include "linalg/BasisGenerator.h"

#include "mfem.hpp"

#include "laghos_rom.hpp"

using namespace std;
using namespace mfem;

void split_line(const string &line, vector<string> &words);

void DeleteROMSolution(std::string outputPath);

void ReadGreedyPhase(bool& rom_offline, bool& rom_online, bool& rom_restore, bool& rom_calc_rel_error_nonlocal, bool& rom_calc_rel_error_local, ROM_Options& romOptions, std::string greedyfile);

void WriteGreedyPhase(bool& rom_offline, bool& rom_online, bool& rom_restore, bool& rom_calc_rel_error_nonlocal, bool& rom_calc_rel_error_local, ROM_Options& romOptions, std::string greedyfile);

void WriteOfflineParam(int dim, double dt, ROM_Options& romOptions, const int numWindows, const char* twfile, std::string paramfile, const bool printStatus);

void VerifyOfflineParam(int& dim, double &dt, ROM_Options& romOptions, const int numWindows, const char* twfile, std::string paramfile, const bool rom_offline);

void BasisGeneratorFinalSummary(CAROM::BasisGenerator* bg, const double energyFraction, int & cutoff, const std::string cutoffOutputPath = "", const bool printout=true);

void PrintSingularValues(const int rank, const std::string& basename, const std::string& name, CAROM::BasisGenerator* bg, const bool usingWindows = false, const int window = -1);

int ReadTimesteps(std::string const& path, std::vector<double>& time);

int ReadTimeWindows(const int nw, std::string twfile, Array<double>& twep, const bool printStatus);

int ReadTimeWindowParameters(const int nw, std::string twfile, Array<double>& twep, Array2D<int>& twparam, double sFactor[], const bool printStatus, const bool rhs);

void ReadGreedyTimeWindowParameters(ROM_Options& romOptions, const int nw, Array2D<int>& twparam, std::string outputPath);

void ReadPDweight(std::vector<double>& pd_weight, std::string outputPath);

void SetWindowParameters(Array2D<int> const& twparam, ROM_Options& romOptions);

void AppendPrintParGridFunction(std::ofstream *ofs, ParGridFunction *gf);
void PrintParGridFunction(const int rank, const std::string& name, ParGridFunction *gf);
double PrintDiffParGridFunction(NormType normtype, const int rank, const std::string& name, ParGridFunction *gf);

void writeNum(int num, std::string file_name, bool append = false);

// read data from from text.txt and store it in vector v
void readNum(int& num, std::string file_name);

void writeDouble(double num, std::string file_name, bool append = false);

// read data from from text.txt and store it in vector v
void readDouble(double& num, std::string file_name);

void writeVec(vector<int> v, std::string file_name, bool append = false);

// read data from from text.txt and store it in vector v
void readVec(vector<int> &v, std::string file_name);

// count the number of lines in a file
int countNumLines(std::string file_name);

#endif // MFEM_LAGHOS_UTILS
