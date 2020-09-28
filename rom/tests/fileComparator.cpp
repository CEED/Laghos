#include <fstream>
#include <string>
#include <cmath>
#include <cassert>
#include <iostream>
#include <sstream>

using namespace std;

void compareFiles(ifstream &baselineFile, ifstream &targetFile, double errorBound) {
    string baselineLine, targetLine;
    double baselineNum, targetNum;
    int fileLine = 1;
    bool outOfRange;

    while(!baselineFile.eof()) {
        outOfRange = false;
        getline(baselineFile, baselineLine);
        getline(targetFile, targetLine);
        if (baselineLine == "" || targetLine == "") {
            assert(baselineLine == targetLine || !(cerr << "The files are not the same length." << endl));
            break;
        }

        auto posOfData = baselineLine.find_last_of(' ');
        auto stripped = baselineLine.substr(posOfData != string::npos ? posOfData : 0);
        baselineNum = stod(stripped);

        posOfData = targetLine.find_last_of(' ');
        stripped =targetLine.substr(posOfData != string::npos ? posOfData : 0);
        targetNum = stod(stripped);

        if (abs(baselineNum - targetNum) > errorBound) {
            cerr << "errorBound = " << errorBound << endl;
            cerr << "abs(baselineNum - targetNum) = " << abs(baselineNum - targetNum) << endl;
            cerr << "TargetNum = " << targetNum << ", BaselineNum = " << baselineNum << endl;
            cerr << "Error bound was surpassed on line: " << fileLine << endl;
            abort();
        }
        fileLine++;
    }
    assert(targetFile.eof() || !(cerr << "The files are not the same length." << endl));
}

int main(int argc, char *argv[]) {
    ifstream baselineFile, targetFile;
    baselineFile.open((string) argv[1]);
    targetFile.open((string) argv[2]);
    double errorBound = stod(argv[3]);
    compareFiles(baselineFile, targetFile, errorBound);

    return 0;
}
