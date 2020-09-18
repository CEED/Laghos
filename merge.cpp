#include "mfem.hpp"
#include "laghos_utils.hpp"

#include "StaticSVDBasisGenerator.h"
#include "BasisReader.h"

using namespace std;
using namespace mfem;

void LoadSampleSets(const int rank, const double energyFraction, const int nsets, const std::string& basename, const std::string& varName,
                    const bool usingWindows, const int window, const int dim, const int totalSamples, int& cutoff)
{
    std::unique_ptr<CAROM::SVDBasisGenerator> basis_generator;

    std::string basis_filename = basename + "/basis" + varName + std::to_string(window);
    CAROM::StaticSVDOptions static_svd_options(dim, totalSamples);
    static_svd_options.max_time_intervals = 1;
    basis_generator.reset(new CAROM::StaticSVDBasisGenerator(static_svd_options, basis_filename));

    cout << "Loading snapshots for " << varName << " in time window " << window << endl;

    for (int i=0; i<nsets; ++i)
    {
        std::string filename = basename + "/param" + std::to_string(i) + "_var" + varName + std::to_string(window) + "_snapshot";
        basis_generator->loadSamples(filename,"snapshot");
    }

    //cout << "Saving data uploaded as a snapshot" << endl;
    //basis_generator->writeSnapshot();

    cout << "Computing SVD for " << varName << " in time window " << window << endl;
    basis_generator->endSamples();  // save the basis file

    if (rank == 0)
    {
        cout << varName << " basis summary output: ";
        BasisGeneratorFinalSummary(basis_generator.get(), energyFraction, cutoff);
        PrintSingularValues(rank, basename, varName, basis_generator.get(), usingWindows, window);
    }
}

// id is snapshot index, 0-based
void GetSnapshotDim(const int id, const std::string& basename, const std::string& varName, const int window, int& varDim, int& numSnapshots)
{
    std::string filename = basename + "/param" + std::to_string(id) + "_var" + varName + std::to_string(window) + "_snapshot";

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
    int numWindows = 0;
    double energyFraction = 0.9999;
    bool rhsBasis = false;
    const char *basename = "";
    const char *twfile = "tw.csv";

    OptionsParser args(argc, argv);
    args.AddOption(&nset, "-nset", "--numsets", "Number of sample sets to merge.");
    args.AddOption(&numWindows, "-nwin", "--numwindows", "Number of ROM time windows.");
    args.AddOption(&energyFraction, "-ef", "--rom-ef", "Energy fraction for recommended ROM basis sizes.");
    args.AddOption(&rhsBasis, "-rhs", "--rhsbasis", "-no-rhs", "--no-rhsbasis",
                   "Enable or disable merging of RHS bases for Fv and Fe.");
    args.AddOption(&basename, "-o", "--outputfilename",
                   "Name of the sub-folder to dump files within the run directory");
    args.AddOption(&twfile, "-tw", "--timewindowfilename",
                   "Name of the CSV file defining offline time windows");

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

    const bool usingWindows = (numWindows > 0); // TODO: Tony PR77 || windowNumSamples > 0);
    Array<double> twep;
    std::ofstream outfile_twp;
    Array<int> cutoff(5);

    if (usingWindows) {
        const int err = ReadTimeWindows(numWindows, twfile, twep, myid == 0);
        MFEM_VERIFY(err == 0, "Error in ReadTimeWindows");
        outfile_twp.open(outputPath + "/twpTemp.csv");
    }
    else {
        numWindows = 1;
    }

    Array<int> snapshotSize(nset);
    Array<int> snapshotSizeFv(nset);
    Array<int> snapshotSizeFe(nset);
    int dimX, dimV, dimE, dimFv, dimFe;

    for (int t=0; t<numWindows; ++t)
    {
        GetSnapshotDim(0, outputPath, "X", t, dimX, snapshotSize[0]);
        {
            int dummy = 0;
            GetSnapshotDim(0, outputPath, "V", t, dimV, dummy);
            MFEM_VERIFY(dummy == snapshotSize[0], "Inconsistent snapshot sizes");
            GetSnapshotDim(0, outputPath, "E", t, dimE, dummy);
            MFEM_VERIFY(dummy == snapshotSize[0], "Inconsistent snapshot sizes");

            if (rhsBasis)
            {
                GetSnapshotDim(0, outputPath, "Fv", t, dimFv, snapshotSizeFv[0]);
                MFEM_VERIFY(snapshotSizeFv[0] >= snapshotSize[0], "Inconsistent snapshot sizes");
                GetSnapshotDim(0, outputPath, "Fe", t, dimFe, snapshotSizeFe[0]);
                MFEM_VERIFY(dummy >= snapshotSize[0], "Inconsistent snapshot sizes");
            }
        }

        MFEM_VERIFY(dimX == dimV, "Different sizes for X and V");
        MFEM_VERIFY(dimFv == dimV, "Different sizes for V and Fv");
        MFEM_VERIFY(dimFe == dimE, "Different sizes for E and Fe");

        int totalSnapshotSize = snapshotSize[0];
        int totalSnapshotSizeFv = snapshotSizeFv[0];
        int totalSnapshotSizeFe = snapshotSizeFe[0];
        for (int i=1; i<nset; ++i)
        {
            int dummy = 0;
            int dim = 0;
            GetSnapshotDim(i, outputPath, "X", t, dim, snapshotSize[i]);
            MFEM_VERIFY(dim == dimX, "Inconsistent snapshot sizes");
            GetSnapshotDim(i, outputPath, "V", t, dim, dummy);
            MFEM_VERIFY(dim == dimV && dummy == snapshotSize[i], "Inconsistent snapshot sizes");
            GetSnapshotDim(i, outputPath, "E", t, dim, dummy);
            MFEM_VERIFY(dim == dimE && dummy == snapshotSize[i], "Inconsistent snapshot sizes");

            if (rhsBasis)
            {
                GetSnapshotDim(i, outputPath, "Fv", t, dim, snapshotSizeFv[i]);
                MFEM_VERIFY(dim == dimV && snapshotSizeFv[i] >= snapshotSize[i], "Inconsistent snapshot sizes");
                GetSnapshotDim(i, outputPath, "Fe", t, dim, snapshotSizeFe[i]);
                MFEM_VERIFY(dim == dimE && snapshotSizeFe[i] >= snapshotSize[i], "Inconsistent snapshot sizes");
            }

            totalSnapshotSize += snapshotSize[i];
            totalSnapshotSizeFv += snapshotSizeFv[i];
            totalSnapshotSizeFe += snapshotSizeFe[i];
        }

        LoadSampleSets(myid, energyFraction, nset, outputPath, "X", usingWindows, t, dimX, totalSnapshotSize, cutoff[0]);
        LoadSampleSets(myid, energyFraction, nset, outputPath, "V", usingWindows, t, dimV, totalSnapshotSize, cutoff[1]);
        LoadSampleSets(myid, energyFraction, nset, outputPath, "E", usingWindows, t, dimE, totalSnapshotSize, cutoff[2]);

        if (rhsBasis)
        {
            LoadSampleSets(myid, energyFraction, nset, outputPath, "Fv", usingWindows, t, dimV, totalSnapshotSizeFv, cutoff[3]);
            LoadSampleSets(myid, energyFraction, nset, outputPath, "Fe", usingWindows, t, dimE, totalSnapshotSizeFe, cutoff[4]);
        }

        if (myid == 0 && usingWindows)
        {
            outfile_twp << twep[t] << ", ";
            if (rhsBasis)
                outfile_twp << cutoff[0] << ", " << cutoff[1] << ", " << cutoff[2] << ", "
                            << cutoff[3] << ", " << cutoff[4] << "\n";
            else
                outfile_twp << cutoff[0] << ", " << cutoff[1] << ", " << cutoff[2] << "\n";
        }
    }

    return 0;
}
