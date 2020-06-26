#include <fstream>
#include <string>
#include <cmath>
#include <cassert>
#include <iostream>
#include <sstream>

using namespace std;

void compareFiles(ifstream &origFile, ifstream &testFile, double errorBound) {
    string origLine, testLine;
    double origNum, testNum;
    int fileLine = 1;
    bool outOfRange;

    while(!origFile.eof()) {
        outOfRange = false;
        getline(origFile, origLine);
        getline(testFile, testLine);
        if (origLine == "" || testLine == "") {
            assert(origLine == testLine || !(cerr << "The files are not the same length. "));
            break;
        }

        auto posOfData = origLine.find_last_of(' ');
        auto stripped = origLine.substr(posOfData != string::npos ? posOfData : 0);

        // If one number underflows/overflows, the number from the other file should as well
        try {
            origNum = stod(stripped);
        }
        catch (exception& e) {
            outOfRange = !outOfRange;
        }

        posOfData = testLine.find_last_of(' ');
        stripped =testLine.substr(posOfData != string::npos ? posOfData : 0);

        try {
            testNum = stod(stripped);
        }
        catch (exception& e) {
            outOfRange = !outOfRange;
        }

        // If only one number out of two was out of range, abort
        if (outOfRange) {
            cerr << "Error bound was surpassed on line: " << fileLine << ". ";
            abort();
        }
        assert(abs(origNum - testNum) <= errorBound || !(cerr << "Error bound was surpassed \
on line: " << fileLine << ". "));
        fileLine++;
    }
    assert(testFile.eof() || !(cerr << "The files are not the same length. "));
}

int main(int argc, char *argv[]) {
    ifstream origFile, testFile;
    origFile.open((string) argv[1]);
    testFile.open((string) argv[2]);
    double errorBound = stod(argv[3]);
    compareFiles(origFile, testFile, errorBound);

    return 0;
}
