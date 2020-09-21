#include "BasisReader.h"
#include "Matrix.h"
#include <string>
#include <iostream>
#include <cmath>
#include <vector>
#include <mpi.h>

using namespace std;

void compareBasis(string &baselineFile, string &targetFile, double errorBound, int numProcessors) {

    MPI_Init(NULL, NULL);

    // Get the number of processes
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    vector<double> vecNormL2;
    vector<double> reducedVecNormL2;

    CAROM::BasisReader baselineReader(baselineFile);
    CAROM::Matrix *baselineBasis = (CAROM::Matrix*) baselineReader.getSpatialBasis(0.0);
    CAROM::BasisReader targetReader(targetFile);
    CAROM::Matrix *targetBasis = (CAROM::Matrix*) targetReader.getSpatialBasis(0.0);

    // Get basis dimensions
    int baselineNumRows = baselineBasis->numRows();
    int baselineNumColumns = baselineBasis->numColumns();
    int targetNumRows = targetBasis->numRows();
    int targetNumColumns = targetBasis->numColumns();

    // Test basis matrices have the same dimensions
    if (baselineNumRows != targetNumRows) {
        cerr << "The number of rows of the two basis matrices \
are not equal in the following files: " << baselineFile << " and " << targetFile << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (baselineNumColumns != targetNumColumns) {
        cerr << "The number of columns of the two basis matrices \
are not equal in the following file: " << baselineFile << " and " << targetFile << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    vecNormL2.resize(baselineNumColumns, 0.0);
    reducedVecNormL2.resize(baselineNumColumns, 0.0);

    try {
        *baselineBasis -=(*targetBasis);
    }
    catch (const exception& e) {
        cerr << "Something went wrong when calculating the difference \
between the basis matrices in the following files: " << baselineFile << " and " << targetFile << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Compute l2-norm
    for (unsigned int i = 0; i < baselineNumColumns; i++) {
        for (unsigned int j = 0; j < baselineNumRows; j++) {
            vecNormL2[i] += pow(baselineBasis->operator()(j,i), 2);
        }
    }

    for (int i = 0; i < vecNormL2.size(); i++) {
        MPI_Reduce(&vecNormL2[i], &reducedVecNormL2[i], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        double matrixNormL2 = 0;
        for (int i = 0; i < vecNormL2.size(); i++) {
            matrixNormL2 += sqrt(vecNormL2[i]);
        }
        // Test whether l2 norm is smaller than error bound
        if (matrixNormL2 > errorBound) {
            cerr << "The matrixNormL2 of the difference of the basis matrices is "
                 << matrixNormL2 << " and it surpassed the error bound." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Finalize the MPI environment.
    MPI_Finalize();
}

int main (int argc, char *argv[]) {
    string baselinePath((string) argv[1]);
    string targetPath((string) argv[2]);
    double errorBound = stod(argv[3]);
    int numProcessors = stoi(argv[4]);

    compareBasis(baselinePath, targetPath, errorBound, numProcessors);
    return 0;
}
