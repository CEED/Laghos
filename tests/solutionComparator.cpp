#include <fstream>
#include <string>
#include <cmath>
#include <iostream>
#include "mfem.hpp"

using namespace std;
using namespace mfem;

int getDimensions(string &filePath) {
  string line;
  int count = 0;

  ifstream file(filePath);
  while (getline(file, line)) {
    count++;
  }
  return count;
}

void compareSolutions(string &baselineFile, string &targetFile, double errorBound, int numProcessors) {
    int* baselineDim = new int[numProcessors];
    istream** baselineFiles = new istream*[numProcessors];
    int* targetDim = new int[numProcessors];
    istream** targetFiles = new istream*[numProcessors];
    std::filebuf* baselinefb = new filebuf[numProcessors];
    std::filebuf* targetfb = new filebuf[numProcessors];
    
    for (int i = 0; i < numProcessors; i++) {
      if (i > 0) {
        baselineFile.back() = '0' + i;
        targetFile.back() = '0' + i;
      }
      cout << "Opening file: " << baselineFile << endl;
      if (baselinefb[i].open(baselineFile, ios::in)) {
        baselineFiles[i] = new istream(&baselinefb[i]);
        baselineDim[i] = getDimensions(baselineFile);
      }
      else {
        cerr << "Something went wrong with opening the following file. Most likely it doesn't exist: " << baselineFile << endl;
        abort();
      }
      cout << "Opening file: " << targetFile << endl;
      if (targetfb[i].open(targetFile, ios::in)) {
        targetFiles[i] = new istream(&targetfb[i]);
        targetDim[i] = getDimensions(targetFile);
      }
      else {
        cerr << "Something went wrong with opening the following file. Most likely it doesn't exist: " << targetFile << endl;
        abort();
      }
    }
    Vector baseline = Vector();
    Vector target = Vector();
    baseline.Load(baselineFiles, numProcessors, baselineDim);
    target.Load(targetFiles, numProcessors, targetDim);

    double baselineNormL2 = baseline.Norml2();
    double targetNormL2 = target.Norml2();

    // Test whether l2 norm is smaller than error bound
    if (baselineNormL2 == 0.0) {
      if (abs(baselineNormL2 - targetNormL2) > errorBound) {
        cerr << "TargetNormL2 = " << targetNormL2 << ", baselineNormL2 = " << baselineNormL2 << endl;
        cerr << "abs(baselineNormL2 - targetNormL2) = " << abs(baselineNormL2 - targetNormL2) / baselineNormL2 << endl;
        cerr << "Error bound was surpassed for the l2 norm of the difference of the solutions." << endl;
        abort();
      }
    }
    else {
      if (abs(baselineNormL2 - targetNormL2) / baselineNormL2 > errorBound) {
        cerr << "TargetNormL2 = " << targetNormL2 << ", baselineNormL2 = " << baselineNormL2 << endl;
        cerr << "abs(baselineNormL2 - targetNormL2) / baselineNormL2 = " << abs(baselineNormL2 - targetNormL2) / baselineNormL2 << endl;
        cerr << "Error bound was surpassed for the l2 norm of the difference of the solutions." << endl;
        abort();
      }
    }
}

int main(int argc, char *argv[]) {
    string baselinePath((string) argv[1]);
    string targetPath((string) argv[2]);
    double errorBound = stod(argv[3]);
    int numProcessors = stoi(argv[4]);

    compareSolutions(baselinePath, targetPath, errorBound, numProcessors);
    return 0;
}
