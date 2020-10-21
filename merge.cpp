#include "mfem.hpp"
#include "laghos_utils.hpp"

#include "StaticSVDBasisGenerator.h"
#include "BasisReader.h"

using namespace std;
using namespace mfem;

enum VariableName { X, V, E, Fv, Fe };

enum offsetStyle
{
    usePreviousSolution,
    useInitialState,
    saveLoadOffset,
    interpolateOffset
};

static offsetStyle getOffsetStyle(const char* offsetType)
{
    static std::unordered_map<std::string, offsetStyle> offsetMap =
    {
        {"previous", usePreviousSolution},
        {"initial", useInitialState},
        {"load", saveLoadOffset},
        {"interpolate", interpolateOffset}
    };
    auto iter = offsetMap.find(offsetType);
    MFEM_VERIFY(iter != std::end(offsetMap), "Invalid input of offset type");
    return iter->second;
}

void MergePhysicalTimeWindow(const int rank, const double energyFraction, const int nsets, const std::string& basename, const std::string& varName, const std::string& basis_filename,
                             const bool usingWindows, const int basisWindow, const int dim, const int totalSamples,
                             const std::vector<std::vector<int>> &offsetAllWindows, int& cutoff)
{
    std::unique_ptr<CAROM::SVDBasisGenerator> basis_generator;
    CAROM::StaticSVDOptions static_svd_options(dim, totalSamples);
    static_svd_options.max_time_intervals = 1;
    basis_generator.reset(new CAROM::StaticSVDBasisGenerator(static_svd_options, basis_filename));

    if (usingWindows)
    {
        cout << "Loading snapshots for " << varName << " in sample time window " << basisWindow << endl; // sampleWindow = basisWindow
    }
    else
    {
        cout << "Loading snapshots for " << varName << endl;
    }

    for (int i=0; i<nsets; ++i)
    {
        std::string filename = basename + "/param" + std::to_string(i) + "_var" + varName + std::to_string(basisWindow) + "_snapshot";
        basis_generator->loadSamples(filename,"snapshot");
    }

    if (usingWindows)
    {
        cout << "Computing SVD for " << varName << " in basis time window " << basisWindow << endl;
    }
    else
    {
        cout << "Computing SVD for " << varName << endl;
    }
    basis_generator->endSamples();  // save the basis file

    if (rank == 0)
    {
        cout << varName << " basis summary output: ";
        BasisGeneratorFinalSummary(basis_generator.get(), energyFraction, cutoff);
        PrintSingularValues(rank, basename, varName, basis_generator.get(), usingWindows, basisWindow);
    }
}

void MergeSamplingTimeWindow(const int rank, const double energyFraction, const int nsets, const std::string& basename, VariableName v, const std::string& varName, const std::string& basis_filename,
                             const int windowOverlapSamples, const int basisWindow, const bool useOffset, const offsetStyle offsetType, const int dim, const int totalSamples,
                             const std::vector<std::vector<int>> &offsetAllWindows, int& cutoff)
{
    bool offsetInit = (useOffset && offsetType != useInitialState && basisWindow > 0) && (v == X || v == V || v == E);
    std::unique_ptr<CAROM::SVDBasisGenerator> basis_generator, window_basis_generator;
    CAROM::StaticSVDOptions static_svd_options(dim, totalSamples);
    static_svd_options.max_time_intervals = 1;
    basis_generator.reset(new CAROM::StaticSVDBasisGenerator(static_svd_options, basis_filename));

    int windowSamples = 0;
    for (int paramID=0; paramID<nsets; ++paramID)
    {
        int num_snap = offsetAllWindows[offsetAllWindows.size()-1][paramID+nsets*v]+1;
        int col_lb = offsetAllWindows[basisWindow][paramID+nsets*v];
        int col_ub = std::min(offsetAllWindows[basisWindow+1][paramID+nsets*v]+windowOverlapSamples+1, num_snap);
        windowSamples += col_ub - col_lb - offsetInit;
    }

    CAROM::StaticSVDOptions window_static_svd_options(dim, windowSamples);
    window_basis_generator.reset(new CAROM::StaticSVDBasisGenerator(window_static_svd_options, basis_filename));

    cout << "Loading snapshots for " << varName << " in basis time window " << basisWindow << endl;

    for (int paramID=0; paramID<nsets; ++paramID)
    {
        std::string snapshot_filename = basename + "/param" + std::to_string(paramID) + "_var" + varName + "0_snapshot";
        basis_generator->loadSamples(snapshot_filename,"snapshot");

        int num_snap = offsetAllWindows[offsetAllWindows.size()-1][paramID+nsets*v]+1;
        const CAROM::Matrix* mat = basis_generator->getSnapshotMatrix();
        MFEM_VERIFY(dim == mat->numRows(), "Inconsistent snapshot size");
        MFEM_VERIFY(num_snap == mat->numColumns(), "Inconsistent number of snapshots");
        int col_lb = offsetAllWindows[basisWindow][paramID+nsets*v];
        int col_ub = std::min(offsetAllWindows[basisWindow+1][paramID+nsets*v]+windowOverlapSamples+1, num_snap);

        if (offsetInit && offsetType == interpolateOffset)
        {
            std::string path_init = basename + "/ROMoffset/param" + std::to_string(paramID) + "_init";
            CAROM::Vector *init = new CAROM::Vector(dim, true);
            init->read(path_init + varName + "0");

            for (int i = 0; i<dim; ++i)
            {
                (*init)(i) += mat->item(i,col_lb);
            }
            init->write(path_init + varName + std::to_string(basisWindow));
        }

        Vector tmp;
        tmp.SetSize(dim);
        for (int j = col_lb; j < col_ub; ++j)
        {
            for (int i = 0; i < dim; ++i)
            {
                tmp[i] = (offsetInit) ? mat->item(i,j) - mat->item(i,col_lb) : mat->item(i,j);
            }
            window_basis_generator->takeSample(tmp.GetData(), 0.0, 1.0);
        }
    }

    cout << "Computing SVD for " << varName << " in basis time window " << basisWindow << endl;
    window_basis_generator->endSamples();  // save the basis file

    if (rank == 0)
    {
        cout << varName << " basis summary output: ";
        BasisGeneratorFinalSummary(window_basis_generator.get(), energyFraction, cutoff);
        PrintSingularValues(rank, basename, varName, window_basis_generator.get(), true, basisWindow);
    }
}

void LoadSampleSets(const int rank, const double energyFraction, const int nsets, const std::string& basename, VariableName v, const bool usingWindows, const int windowNumSamples,
                    const int windowOverlapSamples, const int basisWindow, const bool useOffset, const offsetStyle offsetType, const int dim, const int totalSamples,
                    const std::vector<std::vector<int>> &offsetAllWindows, int& cutoff)
{
    std::string varName;
    switch (v)
    {
    case VariableName::V:
        varName = "V";
        break;
    case VariableName::E:
        varName = "E";
        break;
    case VariableName::Fv:
        varName = "Fv";
        break;
    case VariableName::Fe:
        varName = "Fe";
        break;
    default:
        varName = "X";
    }
    std::string basis_filename = basename + "/basis" + varName + std::to_string(basisWindow);

    if (windowNumSamples > 0)
    {
        MergeSamplingTimeWindow(rank, energyFraction, nsets, basename, v, varName, basis_filename, windowOverlapSamples, basisWindow, useOffset, offsetType, dim, totalSamples, offsetAllWindows, cutoff);
    }
    else
    {
        MergePhysicalTimeWindow(rank, energyFraction, nsets, basename, varName, basis_filename, usingWindows, basisWindow, dim, totalSamples, offsetAllWindows, cutoff);
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

void GetSnapshotTime(const int id, const std::string& basename, const std::string& varName, std::vector<double> &tSnap)
{
    std::string filename = basename + "/param" + std::to_string(id) + "_tSnap" + varName;
    std::ifstream infile_tSnap(filename);
    MFEM_VERIFY(infile_tSnap.is_open(), "Snapshot time input file does not exists.");

    tSnap.clear();
    double t = 0.0;
    while (infile_tSnap >> t)
    {
        tSnap.push_back(t);
    }
}

void GetParametricTimeWindows(const int nset, const bool rhsBasis, const std::string& basename, const int windowNumSamples, int &numBasisWindows, Array<double> &twep, std::vector<std::vector<int>> &offsetAllWindows)
{
    std::vector<double> tVec;
    std::vector<std::vector<double>> tSnapX, tSnapV, tSnapE, tSnapFv, tSnapFe;
    const int numVar = (rhsBasis) ? 5 : 3;
    std::vector<int> numSnap(nset*numVar);
    for (int paramID = 0; paramID < nset; ++paramID)
    {
        GetSnapshotTime(paramID, basename, "X", tVec);
        reverse(tVec.begin(), tVec.end());
        tSnapX.push_back(tVec);
        numSnap[paramID+nset*VariableName::X] = tVec.size();
        GetSnapshotTime(paramID, basename, "V", tVec);
        reverse(tVec.begin(), tVec.end());
        tSnapV.push_back(tVec);
        numSnap[paramID+nset*VariableName::V] = tVec.size();
        GetSnapshotTime(paramID, basename, "E", tVec);
        reverse(tVec.begin(), tVec.end());
        tSnapE.push_back(tVec);
        numSnap[paramID+nset*VariableName::E] = tVec.size();

        if (rhsBasis)
        {
            GetSnapshotTime(paramID, basename, "Fv", tVec);
            reverse(tVec.begin(), tVec.end());
            tSnapFv.push_back(tVec);
            numSnap[paramID+nset*VariableName::Fv] = tVec.size();
            GetSnapshotTime(paramID, basename, "Fe", tVec);
            reverse(tVec.begin(), tVec.end());
            tSnapFe.push_back(tVec);
            numSnap[paramID+nset*VariableName::Fe] = tVec.size();
        }
    }

    bool lastBasisWindow = false;
    std::vector<double> tTemp(nset*numVar, 0.0);
    std::vector<int> offsetCurrentWindow(nset*numVar, 0);
    std::vector<double> twepTemp;

    offsetAllWindows.push_back(offsetCurrentWindow);
    numBasisWindows = 0;

    while (!lastBasisWindow)
    {
        // The snapshot time vectors are placed in descending order, with the last element is when the last snapshot in previous time window is taken
        // Find the smallest time, windowRight, such that at most windowNumSamples+1 new snapshots are counted for every variable and parameter
        for (int paramID = 0; paramID < nset; ++paramID)
        {
            tTemp[paramID+nset*VariableName::X] = *(tSnapX[paramID].rbegin() + std::min(windowNumSamples + 1, static_cast<int>(tSnapX[paramID].size()) - 1));
            tTemp[paramID+nset*VariableName::V] = *(tSnapV[paramID].rbegin() + std::min(windowNumSamples + 1, static_cast<int>(tSnapV[paramID].size()) - 1));
            tTemp[paramID+nset*VariableName::E] = *(tSnapE[paramID].rbegin() + std::min(windowNumSamples + 1, static_cast<int>(tSnapE[paramID].size()) - 1));

            if (rhsBasis)
            {
                tTemp[paramID+nset*VariableName::Fv] = *(tSnapFv[paramID].rbegin() + std::min(windowNumSamples + 1, static_cast<int>(tSnapFv[paramID].size()) - 1));
                tTemp[paramID+nset*VariableName::Fe] = *(tSnapFe[paramID].rbegin() + std::min(windowNumSamples + 1, static_cast<int>(tSnapFe[paramID].size()) - 1));
            }
        }

        double windowRight = *min_element(tTemp.begin(), tTemp.end());

        // Record a vector, offsetCurrentWindow, of the largest snapshot index whose time taken is smaller than windowRight for every variable and parameter
        // A matrix offsetAllWindows is assembled by appending offsetCurrentWindow for each basis window
        // A basis window then takes the snapshots with indices between two consecutive vectors in offsetAllWindows inclusively,
        // which include the last overlapping snapshot in previous time window, all the snapshots taken strictly before windowRight, 
        // and the overlapping snapshot just taken at or after windowRight, making sure no data is missed by closing the basis window at or before windowRight 
        for (int paramID = 0; paramID < nset; ++paramID)
        {
            for (int t = 0; t < windowNumSamples + 2; ++t)
            {
                if (tSnapX[paramID].back() < windowRight)
                {
                    tSnapX[paramID].pop_back();
                    offsetCurrentWindow[paramID+nset*VariableName::X] += 1;
                }
                if (tSnapV[paramID].back() < windowRight)
                {
                    tSnapV[paramID].pop_back();
                    offsetCurrentWindow[paramID+nset*VariableName::V] += 1;
                }
                if (tSnapE[paramID].back() < windowRight)
                {
                    tSnapE[paramID].pop_back();
                    offsetCurrentWindow[paramID+nset*VariableName::E] += 1;
                }

                if (rhsBasis)
                {
                    if (tSnapFv[paramID].back() < windowRight)
                    {
                        tSnapFv[paramID].pop_back();
                        offsetCurrentWindow[paramID+nset*VariableName::Fv] += 1;
                    }
                    if (tSnapFe[paramID].back() < windowRight)
                    {
                        tSnapFe[paramID].pop_back();
                        offsetCurrentWindow[paramID+nset*VariableName::Fe] += 1;
                    }
                }
            }
        }
        offsetAllWindows.push_back(offsetCurrentWindow);
        numBasisWindows += 1;

        // Find the largest time, windowLeft, such that the last snapshot is counted for every variable and parameter
        // The next basis window takes this snapshot, making sure no data is missed by opening the next basis window at or after windowLeft 
        // By opening new basis window at the midpoint of windowLeft and windowRight, 
        // we make sure no data is missed in both basis windows 
        for (int paramID = 0; paramID < nset; ++paramID)
        {
            tTemp[paramID+nset*VariableName::X] = tSnapX[paramID].back();
            tTemp[paramID+nset*VariableName::V] = tSnapV[paramID].back();
            tTemp[paramID+nset*VariableName::E] = tSnapE[paramID].back();

            if (rhsBasis)
            {
                tTemp[paramID+nset*VariableName::Fv] = tSnapFv[paramID].back();
                tTemp[paramID+nset*VariableName::Fe] = tSnapFe[paramID].back();
            }
        }

        double windowLeft = *max_element(tTemp.begin(), tTemp.end());
        double overlapMidpoint = (windowLeft + windowRight) / 2;
        twepTemp.push_back(overlapMidpoint);

        if (numSnap[0] == offsetCurrentWindow[0]+1)
        {
            MFEM_VERIFY(windowLeft == windowRight, "");
            lastBasisWindow = true;
        }
    }

    twep.SetSize(numBasisWindows);
    for (int basisWindow = 0; basisWindow < numBasisWindows; ++basisWindow)
    {
        twep[basisWindow] = twepTemp[basisWindow];
    }
}

int main(int argc, char *argv[])
{
    // Initialize MPI.
    MPI_Session mpi(argc, argv);
    int myid = mpi.WorldRank();

    // Parse command-line options.
    int nset = 0;
    int numWindows = 0;
    int windowNumSamples = 0;
    int windowOverlapSamples = 0;
    double energyFraction = 0.9999;
    bool useOffset = false;
    const char *offsetType = "previous";
    bool rhsBasis = false;
    const char *basename = "";
    const char *twfile = "tw.csv";

    OptionsParser args(argc, argv);
    args.AddOption(&nset, "-nset", "--numsets", "Number of sample sets to merge.");
    args.AddOption(&numWindows, "-nwin", "--numwindows", "Number of ROM time windows.");
    args.AddOption(&windowNumSamples, "-nwinsamp", "--numwindowsamples", "Number of samples in ROM windows.");
    args.AddOption(&windowOverlapSamples, "-nwinover", "--numwindowoverlap", "Number of samples for ROM window overlap.");
    args.AddOption(&energyFraction, "-ef", "--rom-ef", "Energy fraction for recommended ROM basis sizes.");
    args.AddOption(&useOffset, "-romos", "--romoffset", "-no-romoffset", "--no-romoffset",
                   "Enable or disable initial state offset for ROM.");
    args.AddOption(&offsetType, "-rostype", "--romoffsettype",
                   "Offset type for initializing ROM windows.");
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

    MFEM_VERIFY(windowNumSamples == 0 || numWindows == 0, "-nwinsamp and -nwin cannot both be set");
    MFEM_VERIFY(windowNumSamples >= 0, "Negative window");
    MFEM_VERIFY(windowOverlapSamples >= 0, "Negative window overlap");
    MFEM_VERIFY(windowOverlapSamples <= windowNumSamples, "Too many ROM window overlap samples.");

    const bool usingWindows = (numWindows > 0 || windowNumSamples > 0);
    Array<double> twep;
    std::ofstream outfile_twp;
    Array<int> cutoff(5);
    int numBasisWindows = 0;
    std::vector<std::vector<int>> offsetAllWindows;

    if (numWindows > 0) {
        numBasisWindows = numWindows;
        const int err = ReadTimeWindows(numWindows, twfile, twep, myid == 0);
        MFEM_VERIFY(err == 0, "Error in ReadTimeWindows");
        outfile_twp.open(outputPath + "/twpTemp.csv");
    }
    else if (windowNumSamples > 0) {
        numWindows = 1;
        GetParametricTimeWindows(nset, rhsBasis, outputPath, windowNumSamples, numBasisWindows, twep, offsetAllWindows);
        outfile_twp.open(outputPath + "/twpTemp.csv");
    }
    else {
        numWindows = 1;
        numBasisWindows = 1;
    }

    offsetStyle trueOffsetType = getOffsetStyle(offsetType);
    MFEM_VERIFY(trueOffsetType != saveLoadOffset, "-rostype load is not compatible with parametric ROM")

    std::ifstream infile_offlineParam(outputPath + "/offline_param.csv");
    MFEM_VERIFY(infile_offlineParam.is_open(), "Parameter record file does not exist.");
    std::string line;
    std::vector<std::string> words;
    std::getline(infile_offlineParam, line);
    split_line(line, words);
    MFEM_VERIFY(std::stoi(words[0]) == useOffset, "-romos option does not match record.");
    MFEM_VERIFY(std::stoi(words[1]) == trueOffsetType, "-romostype option does not match record.");
    MFEM_VERIFY(std::stoi(words[2]) == rhsBasis, "-romsrhs option does not match record.");
    MFEM_VERIFY(std::stoi(words[3]) == numWindows, "-nwin option does not match record.");
    MFEM_VERIFY(std::strcmp(words[4].c_str(), twfile) == 0, "-tw option does not match record.");
    infile_offlineParam.close();

    Array<int> snapshotSize(nset);
    Array<int> snapshotSizeFv(nset);
    Array<int> snapshotSizeFe(nset);
    int dimX, dimV, dimE, dimFv, dimFe;

    for (int sampleWindow = 0; sampleWindow < numWindows; ++sampleWindow)
    {
        GetSnapshotDim(0, outputPath, "X", sampleWindow, dimX, snapshotSize[0]);
        {
            int dummy = 0;
            GetSnapshotDim(0, outputPath, "V", sampleWindow, dimV, dummy);
            MFEM_VERIFY(dummy == snapshotSize[0], "Inconsistent snapshot sizes");
            GetSnapshotDim(0, outputPath, "E", sampleWindow, dimE, dummy);
            MFEM_VERIFY(dummy == snapshotSize[0], "Inconsistent snapshot sizes");

            if (rhsBasis)
            {
                GetSnapshotDim(0, outputPath, "Fv", sampleWindow, dimFv, snapshotSizeFv[0]);
                MFEM_VERIFY(snapshotSizeFv[0] >= snapshotSize[0], "Inconsistent snapshot sizes");
                GetSnapshotDim(0, outputPath, "Fe", sampleWindow, dimFe, snapshotSizeFe[0]);
                MFEM_VERIFY(dummy >= snapshotSize[0], "Inconsistent snapshot sizes");
            }
        }

        MFEM_VERIFY(dimX == dimV, "Different sizes for X and V");
        MFEM_VERIFY(dimFv == dimV, "Different sizes for V and Fv");
        MFEM_VERIFY(dimFe == dimE, "Different sizes for E and Fe");

        int totalSnapshotSize = snapshotSize[0];
        int totalSnapshotSizeFv = snapshotSizeFv[0];
        int totalSnapshotSizeFe = snapshotSizeFe[0];
        for (int paramID = 1; paramID < nset; ++paramID)
        {
            int dummy = 0;
            int dim = 0;
            GetSnapshotDim(paramID, outputPath, "X", sampleWindow, dim, snapshotSize[paramID]);
            MFEM_VERIFY(dim == dimX, "Inconsistent snapshot sizes");
            GetSnapshotDim(paramID, outputPath, "V", sampleWindow, dim, dummy);
            MFEM_VERIFY(dim == dimV && dummy == snapshotSize[paramID], "Inconsistent snapshot sizes");
            GetSnapshotDim(paramID, outputPath, "E", sampleWindow, dim, dummy);
            MFEM_VERIFY(dim == dimE && dummy == snapshotSize[paramID], "Inconsistent snapshot sizes");

            if (rhsBasis)
            {
                GetSnapshotDim(paramID, outputPath, "Fv", sampleWindow, dim, snapshotSizeFv[paramID]);
                MFEM_VERIFY(dim == dimV && snapshotSizeFv[paramID] >= snapshotSize[paramID], "Inconsistent snapshot sizes");
                GetSnapshotDim(paramID, outputPath, "Fe", sampleWindow, dim, snapshotSizeFe[paramID]);
                MFEM_VERIFY(dim == dimE && snapshotSizeFe[paramID] >= snapshotSize[paramID], "Inconsistent snapshot sizes");
            }

            totalSnapshotSize += snapshotSize[paramID];
            totalSnapshotSizeFv += snapshotSizeFv[paramID];
            totalSnapshotSizeFe += snapshotSizeFe[paramID];
        }

        int lastBasisWindow = (windowNumSamples > 0) ? numBasisWindows - 1 : sampleWindow;
        for (int basisWindow = sampleWindow; basisWindow <= lastBasisWindow; ++basisWindow)
        {
            LoadSampleSets(myid, energyFraction, nset, outputPath, VariableName::X, usingWindows, windowNumSamples, windowOverlapSamples, basisWindow, useOffset, trueOffsetType, dimX, totalSnapshotSize, offsetAllWindows, cutoff[0]);
            LoadSampleSets(myid, energyFraction, nset, outputPath, VariableName::V, usingWindows, windowNumSamples, windowOverlapSamples, basisWindow, useOffset, trueOffsetType, dimV, totalSnapshotSize, offsetAllWindows, cutoff[1]);
            LoadSampleSets(myid, energyFraction, nset, outputPath, VariableName::E, usingWindows, windowNumSamples, windowOverlapSamples, basisWindow, useOffset, trueOffsetType, dimE, totalSnapshotSize, offsetAllWindows, cutoff[2]);

            args.AddOption(&windowOverlapSamples, "-nwinover", "--numwindowoverlap", "Number of samples for ROM window overlap.");
            if (rhsBasis)
            {
                LoadSampleSets(myid, energyFraction, nset, outputPath, VariableName::Fv, usingWindows, windowNumSamples, windowOverlapSamples, basisWindow, useOffset, trueOffsetType, dimV, totalSnapshotSizeFv, offsetAllWindows, cutoff[3]);
                LoadSampleSets(myid, energyFraction, nset, outputPath, VariableName::Fe, usingWindows, windowNumSamples, windowOverlapSamples, basisWindow, useOffset, trueOffsetType, dimE, totalSnapshotSizeFe, offsetAllWindows, cutoff[4]);
            }

            if (myid == 0 && usingWindows)
            {
                outfile_twp << twep[basisWindow] << ", ";
                if (rhsBasis)
                    outfile_twp << cutoff[0] << ", " << cutoff[1] << ", " << cutoff[2] << ", "
                                << cutoff[3] << ", " << cutoff[4] << "\n";
                else
                    outfile_twp << cutoff[0] << ", " << cutoff[1] << ", " << cutoff[2] << "\n";
            }
        }
    }

    return 0;
}
