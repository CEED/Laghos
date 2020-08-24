#include "mfem.hpp"

#include "StaticSVDBasisGenerator.h"
#include "BasisReader.h"

using namespace std;
using namespace mfem;


void BasisGeneratorFinalSummary(CAROM::SVDBasisGenerator* bg, const double energyFraction, int& cutoff)
{
    //int cutoff = 0;
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

void PrintSingularValues(const int rank, const std::string& name, const int window, CAROM::SVDBasisGenerator* bg)
{
    const CAROM::Matrix* sing_vals = bg->getSingularValues();

    char tmp[100];
    sprintf(tmp, ".%06d", rank);

    std::string fullname = "run/sVal" + name + std::to_string(window) + tmp;

    std::ofstream ofs(fullname.c_str(), std::ofstream::out);
    ofs.precision(16);

    for (int sv = 0; sv < sing_vals->numColumns(); ++sv) {
        ofs << (*sing_vals)(sv, sv) << endl;
    }

    ofs.close();
}

void LoadSampleSets(const int rank, const double energyFraction, const int nsets, const std::string& varName,
                    const int window, const int dim, const int totalSamples, int& cutoff)
{
    std::unique_ptr<CAROM::SVDBasisGenerator> basis_generator;

    std::string basis_filename = "run/basis" + varName + std::to_string(window);
    basis_generator.reset(new CAROM::StaticSVDBasisGenerator(dim, totalSamples, 1, basis_filename));

    cout << "Loading snapshots for " << varName << " in time window " << window << endl;

    for (int i=0; i<nsets; ++i)
    {
        std::string filename = "run/param" + std::to_string(i) + "_var" + varName + std::to_string(window) + "_snapshot";
        basis_generator->loadSamples(filename,"snapshot");
    }

    //cout << "Saving data uploaded as a snapshot" << endl;
    //basis_generator->writeSnapshot();

    cout << "Computing SVD for " << varName << " in time window " << window << endl;
    basis_generator->endSamples();  // save the basis file

    cout << varName << " basis in time window " << window << " summary output: ";
    BasisGeneratorFinalSummary(basis_generator.get(), energyFraction, cutoff);
    PrintSingularValues(rank, varName, window, basis_generator.get());
}

// id is snapshot index, 0-based
void GetSnapshotDim(const int id, const std::string& varName, const int window, int& varDim, int& numSnapshots)
{
    std::string filename = "run/param" + std::to_string(id) + "_var" + varName + std::to_string(window) + "_snapshot";

    CAROM::BasisReader reader(filename);
    const CAROM::Matrix *S = reader.getSnapshotMatrix(0.0);
    varDim = S->numRows();
    numSnapshots = S->numColumns();
}

int ReadTimeWindows(const int nw, std::string twfile, Array<double>& twep, const bool printStatus)
{
    if (printStatus) cout << "Reading time windows from file " << twfile << endl;

    std::ifstream ifs(twfile.c_str());

    if (!ifs.is_open())
    {
        cout << "Error: invalid file" << endl;
        return 1;  // invalid file
    }

    twep.SetSize(nw);

    string line, word;
    int count = 0;
    while (getline(ifs, line))
    {
        if (count >= nw)
        {
            cout << "Error reading CSV file. Read more than " << nw << " lines" << endl;
            ifs.close();
            return 3;
        }

        stringstream s(line);
        vector<string> row;

        while (getline(s, word, ','))
            row.push_back(word);

        if (row.size() != 1)
        {
            cout << "Error: CSV file does not specify exactly 1 parameter" << endl;
            ifs.close();
            return 2;  // incorrect number of parameters
        }

        twep[count] = stod(row[0]);

        if (printStatus) cout << "Using time window " << count << " with end time " << twep[count] << endl;
        count++;
    }

    ifs.close();

    if (count != nw)
    {
        cout << "Error reading CSV file. Read " << count << " lines but expected " << nw << endl;
        return 3;
    }

    return 0;
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
    const char *twfile = "tw.csv";

    OptionsParser args(argc, argv);
    args.AddOption(&nset, "-nset", "--numsets", "Number of sample sets to merge.");
    args.AddOption(&numWindows, "-nwin", "--numwindows", "Number of ROM time windows.");
    args.AddOption(&energyFraction, "-ef", "--rom-ef", "Energy fraction for recommended ROM basis sizes.");
    args.AddOption(&rhsBasis, "-rhs", "--rhsbasis", "-no-rhs", "--no-rhsbasis",
                   "Enable or disable merging of RHS bases for Fv and Fe.");
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

    if (nset < 2)
        cout << "More than one set must be specified. No merging is being done." << endl;

    const bool usingWindows = (numWindows > 0); // TODO: || windowNumSamples > 0);
    Array<double> twep;
    std::ofstream outfile_twp;
    Array<int> cutoff(5);

    if (usingWindows) {
        const int err = ReadTimeWindows(numWindows, twfile, twep, myid == 0);
        MFEM_VERIFY(err == 0, "Error in ReadTimeWindows");
        outfile_twp.open("twpTemp.csv");
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
        GetSnapshotDim(0, "X", t, dimX, snapshotSize[0]);
        {
            int dummy = 0;
            GetSnapshotDim(0, "V", t, dimV, dummy);
            MFEM_VERIFY(dummy == snapshotSize[0], "Inconsistent snapshot sizes");
            GetSnapshotDim(0, "E", t, dimE, dummy);
            MFEM_VERIFY(dummy == snapshotSize[0], "Inconsistent snapshot sizes");

            if (rhsBasis)
            {
                GetSnapshotDim(0, "Fv", t, dimFv, snapshotSizeFv[0]);
                MFEM_VERIFY(snapshotSizeFv[0] >= snapshotSize[0], "Inconsistent snapshot sizes");
                GetSnapshotDim(0, "Fe", t, dimFe, snapshotSizeFe[0]);
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
            GetSnapshotDim(i, "X", t, dim, snapshotSize[i]);
            MFEM_VERIFY(dim == dimX, "Inconsistent snapshot sizes");
            GetSnapshotDim(i, "V", t, dim, dummy);
            MFEM_VERIFY(dim == dimV && dummy == snapshotSize[i], "Inconsistent snapshot sizes");
            GetSnapshotDim(i, "E", t, dim, dummy);
            MFEM_VERIFY(dim == dimE && dummy == snapshotSize[i], "Inconsistent snapshot sizes");

            if (rhsBasis)
            {
                GetSnapshotDim(i, "Fv", t, dim, snapshotSizeFv[i]);
                MFEM_VERIFY(dim == dimV && snapshotSizeFv[i] >= snapshotSize[i], "Inconsistent snapshot sizes");
                GetSnapshotDim(i, "Fe", t, dim, snapshotSizeFe[i]);
                MFEM_VERIFY(dim == dimE && snapshotSizeFe[i] >= snapshotSize[i], "Inconsistent snapshot sizes");
            }

            totalSnapshotSize += snapshotSize[i];
            totalSnapshotSizeFv += snapshotSizeFv[i];
            totalSnapshotSizeFe += snapshotSizeFe[i];
        }

        outfile_twp << twep[t] << ", ";

        LoadSampleSets(myid, energyFraction, nset, "X", t, dimX, totalSnapshotSize, cutoff[0]);
        LoadSampleSets(myid, energyFraction, nset, "V", t, dimV, totalSnapshotSize, cutoff[1]);
        LoadSampleSets(myid, energyFraction, nset, "E", t, dimE, totalSnapshotSize, cutoff[2]);

        if (rhsBasis)
        {
            LoadSampleSets(myid, energyFraction, nset, "Fv", t, dimV, totalSnapshotSizeFv, cutoff[3]);
            LoadSampleSets(myid, energyFraction, nset, "Fe", t, dimE, totalSnapshotSizeFe, cutoff[4]);
            outfile_twp << cutoff[0] << ", " << cutoff[1] << ", " << cutoff[2] << ", "
                        << cutoff[3] << ", " << cutoff[4] << "\n";
        }
        else
            outfile_twp << cutoff[0] << ", " << cutoff[1] << ", " << cutoff[2] << "\n";
    }

    return 0;
}
