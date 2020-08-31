#include "mfem.hpp"

#include "StaticSVDBasisGenerator.h"
#include "BasisReader.h"

using namespace std;
using namespace mfem;


void BasisGeneratorFinalSummary(CAROM::SVDBasisGenerator* bg, const double energyFraction)
{
    int cutoff = 0;
    const int rom_dim = bg->getSpatialBasis()->numColumns();
    const CAROM::Matrix* sing_vals = bg->getSingularValues();

    MFEM_VERIFY(rom_dim == sing_vals->numColumns(), "");

    double sum = 0.0;
    for (int sv = 0; sv < sing_vals->numColumns(); ++sv) {
        sum += (*sing_vals)(sv, sv);
    }

    double partialSum = 0.0;
    for (int sv = 0; sv < sing_vals->numColumns(); ++sv) {
        partialSum += (*sing_vals)(sv, sv);
        if (partialSum / sum > energyFraction)
        {
            cutoff = sv+1;
            break;
        }
    }

    cout << "Take first " << cutoff << " of " << sing_vals->numColumns() << " basis vectors" << endl;
}

void PrintSingularValues(const int rank, const std::string& basename, const std::string& name, CAROM::SVDBasisGenerator* bg)
{
    const CAROM::Matrix* sing_vals = bg->getSingularValues();

    char tmp[100];
    sprintf(tmp, ".%06d", rank);

    std::string fullname = basename + "/sVal" + name + tmp;

    std::ofstream ofs(fullname.c_str(), std::ofstream::out);
    ofs.precision(16);

    for (int sv = 0; sv < sing_vals->numColumns(); ++sv) {
        ofs << (*sing_vals)(sv, sv) << endl;
    }

    ofs.close();
}

void LoadSampleSets(const int rank, const double energyFraction, const int nsets, const std::string& basename, const std::string& varName,
                    const int dim, const int totalSamples)
{
    std::unique_ptr<CAROM::SVDBasisGenerator> basis_generator;

    std::string basis_filename = basename + "/basis" + varName + "0";  // 0 is the time window index
    basis_generator.reset(new CAROM::StaticSVDBasisGenerator(dim, totalSamples, basis_filename));

    cout << "Loading snapshots for " << varName << endl;

    for (int i=0; i<nsets; ++i)
    {
        std::string filename =  basename + "/param" + std::to_string(i) + "_var" + varName + "0_snapshot";  // 0 is time window index
        basis_generator->loadSamples(filename,"snapshot");
    }

    //cout << "Saving data uploaded as a snapshot" << endl;
    //basis_generator->writeSnapshot();

    cout << "Computing SVD for " << varName << endl;
    basis_generator->endSamples();  // save the basis file

    if (rank == 0)
    {
        cout << varName << " basis summary output: ";
        BasisGeneratorFinalSummary(basis_generator.get(), energyFraction);
        PrintSingularValues(rank, basename, varName, basis_generator.get());
    }
}

// id is snapshot index, 0-based
void GetSnapshotDim(const int id, const std::string& basename, const std::string& varName, int& varDim, int& numSnapshots)
{
    std::string filename = basename + "/param" + std::to_string(id) + "_var" + varName + "0_snapshot";  // 0 is time window index

    CAROM::BasisReader reader(filename);
    const CAROM::Matrix *S = reader.getSnapshotMatrix(0.0);
    varDim = S->numRows();
    numSnapshots = S->numColumns();
}

int main(int argc, char *argv[])
{
    // Initialize MPI.
    MPI_Session mpi(argc, argv);
    int myid = mpi.WorldRank();

    // Parse command-line options.
    int nset = 0;
    double energyFraction = 0.9999;
    bool rhsBasis = false;
    const char *basename = "";

    OptionsParser args(argc, argv);
    args.AddOption(&nset, "-nset", "--numsets", "Number of sample sets to merge");
    args.AddOption(&energyFraction, "-ef", "--rom-ef", "Energy fraction for recommended ROM basis sizes.");
    args.AddOption(&rhsBasis, "-rhs", "--rhsbasis", "-no-rhs", "--no-rhsbasis",
                   "Enable or disable merging of RHS bases for Fv and Fe.");
    args.AddOption(&basename, "-k", "--outputfilename",
                   "Name of the sub-folder to dump files within the run directory");

    args.Parse();
    if (!args.Good())
    {
        if (mpi.Root()) {
            args.PrintUsage(cout);
        }
        return 1;
    }
    if (mpi.Root()) {
        args.PrintOptions(cout);
    }
    std::string outputPath = "run";
    if (basename != "") {
        outputPath += "/" + std::string(basename);
    }

    if (nset < 2)
        cout << "More than one set must be specified. No merging is being done." << endl;

    Array<int> snapshotSize(nset);
    Array<int> snapshotSizeFv(nset);
    int dimX, dimV, dimE, dimFv, dimFe;

    GetSnapshotDim(0, outputPath, "X", dimX, snapshotSize[0]);
    {
        int dummy = 0;
        GetSnapshotDim(0, outputPath, "V", dimV, dummy);
        MFEM_VERIFY(dummy == snapshotSize[0], "Inconsistent snapshot sizes");
        GetSnapshotDim(0, outputPath, "E", dimE, dummy);
        MFEM_VERIFY(dummy == snapshotSize[0], "Inconsistent snapshot sizes");

        if (rhsBasis)
        {
            GetSnapshotDim(0, outputPath, "Fv", dimFv, snapshotSizeFv[0]);
            MFEM_VERIFY(snapshotSizeFv[0] >= snapshotSize[0], "Inconsistent snapshot sizes");
            GetSnapshotDim(0, outputPath, "Fe", dimFe, dummy);
            MFEM_VERIFY(dummy == snapshotSize[0], "Inconsistent snapshot sizes");
        }
    }

    MFEM_VERIFY(dimX == dimV, "Different sizes for X and V");
    MFEM_VERIFY(dimFv == dimV, "Different sizes for V and Fv");
    MFEM_VERIFY(dimFe == dimE, "Different sizes for E and Fe");

    int totalSnapshotSize = snapshotSize[0];
    int totalSnapshotSizeFv = snapshotSizeFv[0];
    for (int i=1; i<nset; ++i)
    {
        int dummy = 0;
        int dim = 0;
        GetSnapshotDim(i, outputPath, "X", dim, snapshotSize[i]);
        MFEM_VERIFY(dim == dimX, "Inconsistent snapshot sizes");
        GetSnapshotDim(i, outputPath, "V", dim, dummy);
        MFEM_VERIFY(dim == dimV && dummy == snapshotSize[i], "Inconsistent snapshot sizes");
        GetSnapshotDim(i, outputPath, "E", dim, dummy);
        MFEM_VERIFY(dim == dimE && dummy == snapshotSize[i], "Inconsistent snapshot sizes");

        if (rhsBasis)
        {
            GetSnapshotDim(i, outputPath, "Fv", dim, snapshotSizeFv[i]);
            MFEM_VERIFY(dim == dimV && snapshotSizeFv[i] >= snapshotSize[i], "Inconsistent snapshot sizes");
            GetSnapshotDim(i, outputPath, "Fe", dim, dummy);
            MFEM_VERIFY(dim == dimE && dummy == snapshotSize[i], "Inconsistent snapshot sizes");
        }

        totalSnapshotSize += snapshotSize[i];
        totalSnapshotSizeFv += snapshotSizeFv[i];
    }

    LoadSampleSets(myid, energyFraction, nset, outputPath, "X", dimX, totalSnapshotSize);
    LoadSampleSets(myid, energyFraction, nset, outputPath, "V", dimV, totalSnapshotSize);
    LoadSampleSets(myid, energyFraction, nset, outputPath, "E", dimE, totalSnapshotSize);

    if (rhsBasis)
    {
        LoadSampleSets(myid, energyFraction, nset, outputPath, "Fv", dimV, totalSnapshotSizeFv);
        LoadSampleSets(myid, energyFraction, nset, outputPath, "Fe", dimE, totalSnapshotSize);
    }

    return 0;
}
