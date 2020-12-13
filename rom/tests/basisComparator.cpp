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
    vector<double> diffVecNormL2;
    vector<double> reducedDiffVecNormL2;

    CAROM::BasisReader baselineReader(baselineFile);
    CAROM::Matrix *baselineBasis = (CAROM::Matrix*) baselineReader.getSpatialBasis(0.0);
    CAROM::BasisReader targetReader(targetFile);
    CAROM::Matrix *targetBasis = (CAROM::Matrix*) targetReader.getSpatialBasis(0.0);
    CAROM::Matrix *diffBasis = (CAROM::Matrix*) baselineReader.getSpatialBasis(0.0);

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
    diffVecNormL2.resize(baselineNumColumns, 0.0);
    reducedDiffVecNormL2.resize(baselineNumColumns, 0.0);

    try {
        *diffBasis -=(*targetBasis);
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
            diffVecNormL2[i] += pow(diffBasis->operator()(j,i), 2);
        }
    }

    for (int i = 0; i < diffVecNormL2.size(); i++) {
        MPI_Reduce(&vecNormL2[i], &reducedVecNormL2[i], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&diffVecNormL2[i], &reducedDiffVecNormL2[i], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        double baselineNormL2 = 0;
        double diffNormL2 = 0;
        for (int i = 0; i < reducedDiffVecNormL2.size(); i++) {
            baselineNormL2 += sqrt(reducedVecNormL2[i]);
            diffNormL2 += sqrt(reducedDiffVecNormL2[i]);
        }
        double error = diffNormL2 / baselineNormL2;

        // Test whether l2 norm is smaller than error bound
        if (error > errorBound) {
            cerr << "baselineNormL2 = " << baselineNormL2 << ", diffNormL2 = " << diffNormL2 << endl;
            cerr << "error = " << error << endl;
            cerr << "Error bound: " << errorBound << " was surpassed for the l2 norm of the difference of the basis matrices." << endl;
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
