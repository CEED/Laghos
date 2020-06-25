#include "BasisReader.h"
#include "Matrix.h"
#include <string>
#include <iostream>
#include <cassert>
#include <cmath>

using namespace std;

void compareBasis(CAROM::Matrix *origBasis, CAROM::Matrix *testBasis, double errorBound) {

    // Get basis dimensions
    int orignumRows = origBasis->numRows();
    int orignumColumns = origBasis->numColumns();
    int testnumRows = testBasis->numRows();
    int testnumColumns = testBasis->numColumns();

    // Test basis matrices have the same dimensions
    assert(orignumRows == testnumRows || !(cerr << "The number of rows of the two basis matrices \
are not equal. "));
    assert(orignumColumns == testnumColumns || !(cerr << "The number of columns of the two basis matrices \
are not equal. "));

    try {
        *origBasis -=(*testBasis);
    }
    catch (const exception& e) {
        cerr << "Something went wrong when calculating the difference \
between the basis matrices. ";
        abort();
    }
    double matrixl2Norm = 0;

    // Compute l2-norm
    for (unsigned int j = 0; j < orignumRows; j++) {
        double vecl2Norm = 0;
        for (unsigned int i = 0; i < orignumColumns; i++) {
            vecl2Norm += pow(origBasis->operator()(i,j), 2);
        }
        matrixl2Norm += vecl2Norm;
    }
    matrixl2Norm = sqrt(matrixl2Norm);

    // Test whether l2 norm is smaller than error bound
    assert(abs(matrixl2Norm) <= errorBound || !(cerr << "Error bound was surpassed \
for the l2 norm of the difference of the basis matrices. "));

}

int main (int argc, char *argv[]) {
    CAROM::BasisReader origReader((string) argv[1]);
    CAROM::Matrix *origBasis = (CAROM::Matrix*) origReader.getSpatialBasis(0.0);;
    CAROM::BasisReader testReader((string) argv[2]);
    CAROM::Matrix *testBasis = (CAROM::Matrix*) testReader.getSpatialBasis(0.0);

    double errorBound = stod(argv[3]);
    compareBasis(origBasis, testBasis, errorBound);
    return 0;
}
