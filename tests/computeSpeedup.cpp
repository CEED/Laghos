#include <string>
#include <cmath>
#include <cassert>
#include <iostream>

using namespace std;

int main(int argc, char *argv[]) {
    double offlineSpeed = stod(argv[1]);
    double onlineSpeed = stod(argv[2]);
    double errorBound = stod(argv[3]);
    if (offlineSpeed / onlineSpeed * 100 < errorBound) {
        cerr << "speedupTol = " << errorBound << endl;
        cerr << "onlineSpeed / offlineSpeed = " << offlineSpeed / onlineSpeed * 100 << endl;
        cerr << "speedupTol: " << errorBound << " was not surpassed." << endl;
        abort();
    }

    return 0;
}
