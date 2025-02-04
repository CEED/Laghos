#include "mfem.hpp"
#include "laghos_utils.hpp"

#include "linalg/BasisGenerator.h"
#include "linalg/BasisReader.h"

using namespace std;
using namespace mfem;


void MergePhysicalTimeWindow(const int rank, const int first_sv, const double energyFraction, const int nsets, const std::string& basename, const std::string& varName,
                             const std::string& basisIdentifier, const std::string& basis_filename, const bool usingWindows, const int basisWindow, const int dim,
                             const int totalSamples, const std::vector<std::vector<int>> &offsetAllWindows, int& cutoff,
                             bool squareSV)
{
    std::unique_ptr<CAROM::BasisGenerator> basis_generator;
    CAROM::Options static_svd_options(dim, totalSamples, 1);
    basis_generator.reset(new CAROM::BasisGenerator(static_svd_options, false, basis_filename));

    if (usingWindows)
    {
        cout << "Loading snapshots for " << varName << " in physical time window " << basisWindow << endl; // sampleWindow = basisWindow
    }
    else
    {
        cout << "Loading snapshots for " << varName << endl;
    }

    for (int i=0; i<nsets; ++i)
    {
        std::string filename = basename + "/param" + std::to_string(i) + "_var" + varName + std::to_string(basisWindow) + basisIdentifier + "_snapshot";
        basis_generator->loadSamples(filename,"snapshot");
    }

    if (usingWindows)
    {
        cout << "Computing SVD for " << varName << " in physical time window " << basisWindow << endl;
    }
    else
    {
        cout << "Computing SVD for " << varName << endl;
    }
    basis_generator->endSamples();  // save the basis file

    if (rank == 0)
    {
        cout << varName << " basis summary output: ";
        BasisGeneratorFinalSummary(basis_generator.get(), first_sv, energyFraction, cutoff, basename + "/" + "rdim" + varName + basisIdentifier, true, squareSV);
        PrintSingularValues(rank, basename, varName + basisIdentifier, basis_generator.get(), usingWindows, basisWindow);
    }
}

void MergeSamplingWindow(const int rank, const int first_sv,
                         const double energyFraction, const int nsets,
                         const std::string& basename, VariableName v,
                         const std::string& varName,
                         const std::string& basisIdentifier,
                         const std::string& basis_filename,
                         const int windowOverlapSamples, const int basisWindow,
                         const bool useOffset, const offsetStyle offsetType,
                         const int dim, const int totalSamples,
                         const std::vector<std::vector<int>> &offsetAllWindows,
                         int& cutoff, bool eqp, bool squareSV)
{
    bool offsetInit = (useOffset && offsetType != useInitialState && basisWindow > 0) && (v == X || v == V || v == E);
    std::unique_ptr<CAROM::BasisGenerator> window_basis_generator;
    CAROM::Options static_svd_options(dim, totalSamples, 1);

    int windowSamples = 0;
    for (int paramID = 0; paramID < nsets; ++paramID)
    {
        int num_snap = offsetAllWindows[offsetAllWindows.size()-1][paramID+nsets*v]+1;
        int col_lb = offsetAllWindows[basisWindow][paramID+nsets*v];
        int col_ub = std::min(offsetAllWindows[basisWindow+1][paramID+nsets*v]+windowOverlapSamples+1, num_snap);
        windowSamples += col_ub - col_lb - offsetInit;
    }

    CAROM::Options window_static_svd_options(dim, windowSamples);
    window_basis_generator.reset(new CAROM::BasisGenerator(window_static_svd_options, false, basis_filename));

    cout << "Loading snapshots for " << varName << " in basis window " << basisWindow << endl;

    for (int paramID = 0; paramID < nsets; ++paramID)
    {
        std::string snapshot_filename = basename + "/param" + std::to_string(paramID) + "_var" + varName + "0" + basisIdentifier + "_snapshot";
        std::unique_ptr<CAROM::BasisReader> basis_reader(new CAROM::BasisReader(snapshot_filename));

        // getSnapshotMatrix is 1-indexed, so we need to add 1.
        // TODO: why does this need to be different for EQP?
        const int add1 = eqp ? 0 : 1;
        const int num_snap = std::min(offsetAllWindows[offsetAllWindows.size()-1][paramID+nsets*v] + add1,
                                      basis_reader->getNumSamples("snapshot"));
        const int col_lb = offsetAllWindows[basisWindow][paramID+nsets*v] + 1;

        // getSnapshotMatrix includes the final column, so we don't add 1.
        const int col_ub = std::min(offsetAllWindows[basisWindow+1][paramID+nsets*v]+windowOverlapSamples+1, num_snap);

        int num_cols = col_ub - col_lb + 1;
        std::cout << num_cols << " columns read. Columns " << col_lb - 1
                  << " to " << col_ub - 1 << std::endl;
        CAROM::Matrix* mat = basis_reader->getSnapshotMatrix(col_lb, col_ub);
        MFEM_VERIFY(dim == mat->numRows(), "Inconsistent snapshot size");
        MFEM_VERIFY(num_cols == mat->numColumns(), "Inconsistent number of snapshots");

        CAROM::Vector *init = nullptr;

        if (offsetInit && offsetType == interpolateOffset)
        {
            std::string path_init = basename + "/ROMoffset" + basisIdentifier + "/param" + std::to_string(paramID) + "_init";
            init = new CAROM::Vector(dim, true);
            init->read(path_init + varName + "0");

            for (int i = 0; i < dim; ++i)
            {
                for (int j = 0; j < num_cols; ++j)
                    mat->item(i,j) += (*init)(i);

                (*init)(i) = mat->item(i,0);
            }
            init->write(path_init + varName + std::to_string(basisWindow));
        }

        Vector tmp;
        tmp.SetSize(dim);
        for (int j = 0; j < num_cols; ++j)
        {
            if (j == 0 && offsetInit)
                continue;

            for (int i = 0; i < dim; ++i)
            {
                tmp[i] = (offsetInit) ? mat->item(i,j) - mat->item(i,0) : mat->item(i,j);
            }

            window_basis_generator->takeSample(tmp.GetData());
        }

        if (offsetInit)
        {
            for (int j = 1; j < num_cols; ++j)
            {
                for (int i = 0; i < dim; ++i)
                {
                    mat->item(i,j) -= mat->item(i,0);
                }
            }

            for (int i = 0; i < dim; ++i)
            {
                mat->item(i,0) = 0.0;
            }
        }

        if (eqp)  // Write snapshots to be used in the EQP system
        {
            std::string m_snapshot_filename = basename + "/mparam" +
                                              std::to_string(paramID) + "_var"
                                              + varName +
                                              std::to_string(basisWindow)
                                              + basisIdentifier + "_snapshot";

            if (offsetInit)
            {
                // Omit first column, which is zero.
                // TODO: make optional input in Matrix::write for column range.
                CAROM::Matrix mat1(mat->numRows(), mat->numColumns() - 1, mat->distributed());
                for (int j = 1; j < num_cols; ++j)
                {
                    for (int i = 0; i < dim; ++i)
                    {
                        mat1(i,j-1) = (*mat)(i,j);
                    }
                }

                mat1.write(m_snapshot_filename);
            }
            else
                mat->write(m_snapshot_filename);
        }

        delete init;
        delete mat;
    }

    cout << "Computing SVD for " << varName << " in basis window " << basisWindow << endl;
    window_basis_generator->endSamples();  // save the basis file

    if (rank == 0)
    {
        cout << varName << " basis summary output: ";
        BasisGeneratorFinalSummary(window_basis_generator.get(), first_sv, energyFraction, cutoff, basename + "/" + "rdim" + varName + basisIdentifier, true, squareSV);
        PrintSingularValues(rank, basename, varName + basisIdentifier, window_basis_generator.get(), true, basisWindow);
    }
}

void LoadSampleSets(const int rank, const double energyFraction, const int sv_shift, const int nsets, const std::string& basename, VariableName v,
                    const std::string& basisIdentifier, const bool usingWindows, const int windowNumSamples, const int windowOverlapSamples, const int basisWindow,
                    const bool useOffset, const offsetStyle offsetType, const int dim, const int totalSamples,
                    const std::vector<std::vector<int>> &offsetAllWindows, int& cutoff, bool eqp, bool squareSV)
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
    std::string basis_filename = basename + "/basis" + varName + std::to_string(basisWindow) + basisIdentifier;

    int first_sv = (useOffset && offsetType == useInitialState && basisWindow > 0) && (v == X || v == V || v == E) ? sv_shift : 0;
    if (windowNumSamples > 0)
    {
        MergeSamplingWindow(rank, first_sv, energyFraction, nsets, basename, v, varName, basisIdentifier, basis_filename, windowOverlapSamples, basisWindow,
                            useOffset, offsetType, dim, totalSamples, offsetAllWindows, cutoff, eqp, squareSV);
    }
    else
    {
        MergePhysicalTimeWindow(rank, first_sv, energyFraction, nsets, basename, varName, basisIdentifier, basis_filename, usingWindows, basisWindow, dim, totalSamples, offsetAllWindows, cutoff, squareSV);
    }
}

// id is snapshot index, 0-based
void GetSnapshotDim(const int id, const std::string& basename, const std::string& varName, const std::string& basisIdentifier, const int window, int& varDim, int& numSnapshots)
{
    std::string filename = basename + "/param" + std::to_string(id) + "_var" + varName + std::to_string(window) + basisIdentifier + "_snapshot";

    CAROM::BasisReader reader(filename);
    const CAROM::Matrix *S = reader.getSnapshotMatrix();
    varDim = S->numRows();
    numSnapshots = S->numColumns();
}

void GetSnapshotTime(const int id, const std::string& basename, const std::string& varName, const std::string& basisIdentifier, std::vector<double> &tSnap)
{
    std::string filename = basename + "/param" + std::to_string(id) + "_tSnap" + varName + basisIdentifier;
    std::cout << filename << std::endl;
    std::ifstream infile_tSnap(filename);
    MFEM_VERIFY(infile_tSnap.is_open(), "Snapshot time input file does not exists.");

    tSnap.clear();
    double t = 0.0;
    while (infile_tSnap >> t)
    {
        tSnap.push_back(t);
    }
}

void GetSnapshotPenetrationDistance(const int id, const std::string& basename, std::vector<double> &pdSnap)
{
    std::string filename = basename + "/param" + std::to_string(id) + "_pdSnapX";
    std::ifstream infile_pdSnap(filename);
    MFEM_VERIFY(infile_pdSnap.is_open(), "Snapshot time input file does not exists.");

    pdSnap.clear();
    double pd = 0.0;
    while (infile_pdSnap >> pd)
    {
        pdSnap.push_back(pd);
    }
}

void GetParametricTimeWindows(const int nset, const bool SNS, const bool pd, const std::string& basename, const std::string& basisIdentifier,
                              const int windowNumSamples, int &numBasisWindows, Array<double> &twep, std::vector<std::vector<int>> &offsetAllWindows)
{
    std::vector<double> tVec;
    std::vector<std::vector<double>> tSnapX, tSnapV, tSnapE, tSnapFv, tSnapFe, pdSnap;
    const int numVar = (SNS) ? 3 : 5;
    std::vector<int> numSnap(nset*numVar);
    // The snapshot time vectors are placed in descending order, with the last element is when the first snapshot is taken
    for (int paramID = 0; paramID < nset; ++paramID)
    {
        GetSnapshotTime(paramID, basename, "X", basisIdentifier, tVec);
        reverse(tVec.begin(), tVec.end());
        tSnapX.push_back(tVec);
        numSnap[paramID+nset*VariableName::X] = tVec.size();
        GetSnapshotTime(paramID, basename, "V", basisIdentifier, tVec);
        reverse(tVec.begin(), tVec.end());
        tSnapV.push_back(tVec);
        numSnap[paramID+nset*VariableName::V] = tVec.size();
        GetSnapshotTime(paramID, basename, "E", basisIdentifier, tVec);
        reverse(tVec.begin(), tVec.end());
        tSnapE.push_back(tVec);
        numSnap[paramID+nset*VariableName::E] = tVec.size();

        if (!SNS)
        {
            GetSnapshotTime(paramID, basename, "Fv", basisIdentifier, tVec);
            reverse(tVec.begin(), tVec.end());
            tSnapFv.push_back(tVec);
            numSnap[paramID+nset*VariableName::Fv] = tVec.size();
            GetSnapshotTime(paramID, basename, "Fe", basisIdentifier, tVec);
            reverse(tVec.begin(), tVec.end());
            tSnapFe.push_back(tVec);
            numSnap[paramID+nset*VariableName::Fe] = tVec.size();
        }

        if (pd)
        {
            GetSnapshotPenetrationDistance(paramID, basename, tVec);
            reverse(tVec.begin(), tVec.end());
            pdSnap.push_back(tVec);
        }
    }

    bool lastBasisWindow = false;
    std::vector<bool> lastSnapshot(nset, false);
    std::vector<double> tTemp((pd) ? nset : nset*numVar, 0.0);
    std::vector<int> offsetCurrentWindow(nset*numVar, 0);

    offsetAllWindows.push_back(offsetCurrentWindow);
    numBasisWindows = 0;

    while (!lastBasisWindow)
    {
        // The snapshot time vectors are placed in descending order, with the last element is when the last snapshot in previous time window is taken
        // Find the smallest time, windowRight, such that at most windowNumSamples+1 new snapshots are counted for every variable and parameter
        for (int paramID = 0; paramID < nset; ++paramID)
        {
            if (!pd)
            {
                tTemp[paramID+nset*VariableName::X] = *(tSnapX[paramID].rbegin() + std::min(windowNumSamples + 1, static_cast<int>(tSnapX[paramID].size()) - 1));
                tTemp[paramID+nset*VariableName::V] = *(tSnapV[paramID].rbegin() + std::min(windowNumSamples + 1, static_cast<int>(tSnapV[paramID].size()) - 1));
                tTemp[paramID+nset*VariableName::E] = *(tSnapE[paramID].rbegin() + std::min(windowNumSamples + 1, static_cast<int>(tSnapE[paramID].size()) - 1));

                if (!SNS)
                {
                    tTemp[paramID+nset*VariableName::Fv] = *(tSnapFv[paramID].rbegin() + std::min(windowNumSamples + 1, static_cast<int>(tSnapFv[paramID].size()) - 1));
                    tTemp[paramID+nset*VariableName::Fe] = *(tSnapFe[paramID].rbegin() + std::min(windowNumSamples + 1, static_cast<int>(tSnapFe[paramID].size()) - 1));
                }
            }
            else
            {
                tTemp[paramID] = lastSnapshot[paramID] ? std::numeric_limits<double>::max() :
                                 *(pdSnap[paramID].rbegin() + std::min(windowNumSamples + 1, static_cast<int>(pdSnap[paramID].size()) - 1));
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
            double tMax = -1.0;
            if (pd)
            {
                if (lastSnapshot[paramID]) continue;
                MFEM_VERIFY(tSnapX[paramID].size() == pdSnap[paramID].size(), "pdSnap and tSnapX do not have the same number of elements");
                for (int t = 0; t < windowNumSamples + 2; ++t)
                {
                    if (pdSnap[paramID].back() < windowRight)
                    {
                        pdSnap[paramID].pop_back();
                    }
                    else
                    {
                        tMax = *(tSnapX[paramID].rbegin() + t);
                        break;
                    }
                }
                MFEM_VERIFY(tMax > 0.0, "Did not read tMax correctly");
                lastSnapshot[paramID] = (pdSnap[paramID].size() == 1);
            }
            else
                tMax = windowRight;

            constexpr double eps = 1.0e-8;

            double tLastX = -1.0e100;
            double tLastV = -1.0e100;
            double tLastE = -1.0e100;

            for (int t = 0; t < windowNumSamples + 2; ++t)
            {
                if (tSnapX[paramID].size() > 0) tLastX = tSnapX[paramID].back();
                if (tSnapV[paramID].size() > 0) tLastV = tSnapV[paramID].back();
                if (tSnapE[paramID].size() > 0) tLastE = tSnapE[paramID].back();

                if (tSnapX[paramID].size() > 0 && (tSnapX[paramID].back() < tMax || tSnapX[paramID][0] < tMax + eps))
                {
                    tSnapX[paramID].pop_back();
                    offsetCurrentWindow[paramID+nset*VariableName::X] += 1;
                }
                if (tSnapV[paramID].size() > 0 && (tSnapV[paramID].back() < tMax || tSnapV[paramID][0] < tMax + eps))
                {
                    tSnapV[paramID].pop_back();
                    offsetCurrentWindow[paramID+nset*VariableName::V] += 1;
                }
                if (tSnapE[paramID].size() > 0 && (tSnapE[paramID].back() < tMax || tSnapE[paramID][0] < tMax + eps))
                {
                    tSnapE[paramID].pop_back();
                    offsetCurrentWindow[paramID+nset*VariableName::E] += 1;
                }

                if (!SNS)
                {
                    if (tSnapFv[paramID].back() < tMax)
                    {
                        tSnapFv[paramID].pop_back();
                        offsetCurrentWindow[paramID+nset*VariableName::Fv] += 1;
                    }
                    if (tSnapFe[paramID].back() < tMax)
                    {
                        tSnapFe[paramID].pop_back();
                        offsetCurrentWindow[paramID+nset*VariableName::Fe] += 1;
                    }
                }
            }

            if (tSnapX[paramID].size() == 0)
                tSnapX[paramID].push_back(tLastX);
            if (tSnapV[paramID].size() == 0)
                tSnapV[paramID].push_back(tLastV);
            if (tSnapE[paramID].size() == 0)
                tSnapE[paramID].push_back(tLastE);
        }

        offsetAllWindows.push_back(offsetCurrentWindow);
        numBasisWindows += 1;

        for (int paramID = 0; paramID < nset; ++paramID)
        {
            if (!pd)
            {
                tTemp[paramID+nset*VariableName::X] = tSnapX[paramID].back();
                tTemp[paramID+nset*VariableName::V] = tSnapV[paramID].back();
                tTemp[paramID+nset*VariableName::E] = tSnapE[paramID].back();

                if (!SNS)
                {
                    tTemp[paramID+nset*VariableName::Fv] = tSnapFv[paramID].back();
                    tTemp[paramID+nset*VariableName::Fe] = tSnapFe[paramID].back();
                }
            }
            else
            {
                tTemp[paramID] = pdSnap[paramID].back();
            }
        }

        // Find the largest time, windowLeft, such that the last snapshot is counted for every variable and parameter
        // The next basis window takes this snapshot and is opened at the midpoint of windowLeft and windowRight.
        double windowLeft = *max_element(tTemp.begin(), tTemp.end());
        double overlapMidpoint = (windowLeft + windowRight) / 2;
        twep.Append(overlapMidpoint);

        lastBasisWindow = true;
        for (int i = 0; i < nset*numVar; ++i)
        {
            if (numSnap[i] > offsetCurrentWindow[i]+1)
            {
                lastBasisWindow = false;
            }
        }

        if (lastBasisWindow)
        {
            MFEM_VERIFY(windowLeft == windowRight, "Fail to merge since windowLeft is not equal to windowRight at the final time");
        }
    }
}

int main(int argc, char *argv[])
{
    // Initialize MPI.
    Mpi::Init();
    const int myid = Mpi::WorldRank();
    const bool root = myid == 0;

    // Parse command-line options.
    int nset = 0;
    int numWindows = 0;
    int windowNumSamples = 0;
    int windowOverlapSamples = 0;
    const char *offsetType = "initial";
    const char *indicatorType = "time";
    const char *basename = "";
    const char *twfile = "tw.csv";
    const char *twpfile = "twp.csv";
    const char *basisIdentifier = "";
    bool eqp = false;
    ROM_Options romOptions;

    OptionsParser args(argc, argv);
    args.AddOption(&nset, "-nset", "--numsets", "Number of sample sets to merge.");
    args.AddOption(&numWindows, "-nwin", "--numwindows", "Number of ROM time windows.");
    args.AddOption(&windowNumSamples, "-nwinsamp", "--numwindowsamples", "Number of samples in ROM windows.");
    args.AddOption(&windowOverlapSamples, "-nwinover", "--numwindowoverlap", "Number of samples for ROM window overlap.");
    args.AddOption(&basisIdentifier, "-bi", "--bi", "Basis identifier for the greedy algorithm.");
    args.AddOption(&romOptions.energyFraction, "-ef", "--rom-ef", "Energy fraction for recommended ROM basis sizes.");
    args.AddOption(&romOptions.sv_shift, "-sv-shift", "--sv-shift",
                   "Number of shifted singular values in energy fraction calculation when window-dependent offsets are not used.");
    args.AddOption(&romOptions.useOffset, "-romos", "--romoffset", "-no-romoffset", "--no-romoffset",
                   "Enable or disable initial state offset for ROM.");
    args.AddOption(&offsetType, "-rostype", "--romoffsettype",
                   "Offset type for initializing ROM windows.");
    args.AddOption(&indicatorType, "-loctype", "--romindicatortype",
                   "Indicator type for partitioning ROM windows.");
    args.AddOption(&romOptions.SNS, "-romsns", "--romsns", "-no-romsns", "--no-romsns",
                   "Enable or disable SNS in hyperreduction on Fv and Fe");
    args.AddOption(&basename, "-o", "--outputfilename",
                   "Name of the sub-folder to dump files within the run directory");
    args.AddOption(&twfile, "-tw", "--timewindowfilename",
                   "Name of the CSV file defining offline time windows");
    args.AddOption(&twpfile, "-twp", "--timewindowparamfilename",
                   "Name of the CSV file defining online time window parameters");
    args.AddOption(&eqp, "-eqp", "--eqp", "-no-eqp", "--no-eqp",
                   "Using EQP");
    args.AddOption(&romOptions.squareSV, "-sqsv", "--square-sv", "-no-sqsv", "--no-square-sv",
                   "Use singular values squared in energy fraction.");
    args.Parse();
    if (!args.Good())
    {
        if (root) {
            args.PrintUsage(cout);
        }
        return 1;
    }
    if (root) {
        args.PrintOptions(cout);
    }
    std::string outputPath = "run";
    if (std::string(basename) != "") {
        outputPath += "/" + std::string(basename);
    }

    romOptions.offsetType = getOffsetStyle(offsetType);
    romOptions.indicatorType = getlocalROMIndicator(indicatorType);
    const bool pd = (romOptions.indicatorType != physicalTime);

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
    std::string basisIdentifierString = std::string(basisIdentifier);
    std::string twp_file_path = std::string(twpfile) + basisIdentifierString;

    if (numWindows > 0) {
        numBasisWindows = numWindows;
        const int err = ReadTimeWindows(numWindows, twfile, twep, myid == 0);
        MFEM_VERIFY(err == 0, "Error in ReadTimeWindows");
        outfile_twp.open(outputPath + "/" + std::string(twp_file_path));
    }
    else if (windowNumSamples > 0) {
        numWindows = 1;
        GetParametricTimeWindows(nset, romOptions.SNS, pd, outputPath, basisIdentifierString, windowNumSamples, numBasisWindows, twep, offsetAllWindows);
        outfile_twp.open(outputPath + "/" + std::string(twp_file_path));
    }
    else {
        numWindows = 1;
        numBasisWindows = 1;
    }

    int dim; // dummy
    double dt; // dummy
    std::string offlineParam_outputPath = outputPath + "/offline_param" + basisIdentifierString + ".csv";
    VerifyOfflineParam(dim, dt, romOptions, numWindows, twfile, offlineParam_outputPath, true);

    Array<int> snapshotSize(nset);
    Array<int> snapshotSizeFv(nset);
    Array<int> snapshotSizeFe(nset);
    int dimX, dimV, dimE, dimFv, dimFe;

    StopWatch mergeTimer;
    mergeTimer.Start();

    for (int sampleWindow = 0; sampleWindow < numWindows; ++sampleWindow)
    {
        GetSnapshotDim(0, outputPath, "X", basisIdentifierString, sampleWindow, dimX, snapshotSize[0]);
        int extraV = 0;
        {
            int dummy = 0;
            GetSnapshotDim(0, outputPath, "V", basisIdentifierString, sampleWindow, dimV, dummy);
            extraV = (dummy == snapshotSize[0] + 1);  // 0 or 1
            MFEM_VERIFY(dummy == snapshotSize[0] + extraV, "Inconsistent snapshot sizes");

            GetSnapshotDim(0, outputPath, "E", basisIdentifierString, sampleWindow, dimE, dummy);
            MFEM_VERIFY(dummy == snapshotSize[0], "Inconsistent snapshot sizes");

            if (!romOptions.SNS)
            {
                GetSnapshotDim(0, outputPath, "Fv", basisIdentifierString, sampleWindow, dimFv, snapshotSizeFv[0]);
                MFEM_VERIFY(snapshotSizeFv[0] >= snapshotSize[0], "Inconsistent snapshot sizes");
                GetSnapshotDim(0, outputPath, "Fe", basisIdentifierString, sampleWindow, dimFe, snapshotSizeFe[0]);
                MFEM_VERIFY(dummy >= snapshotSize[0], "Inconsistent snapshot sizes");
            }
        }

        MFEM_VERIFY(dimX == dimV, "Different sizes for X and V");
        if (!romOptions.SNS)
        {
            MFEM_VERIFY(dimFv == dimV, "Different sizes for V and Fv");
            MFEM_VERIFY(dimFe == dimE, "Different sizes for E and Fe");
        }

        int totalSnapshotSize = snapshotSize[0];
        int totalSnapshotSizeFv = snapshotSizeFv[0];
        int totalSnapshotSizeFe = snapshotSizeFe[0];
        for (int paramID = 1; paramID < nset; ++paramID)
        {
            int dummy = 0;
            int dim = 0;
            GetSnapshotDim(paramID, outputPath, "X", basisIdentifierString, sampleWindow, dim, snapshotSize[paramID]);
            MFEM_VERIFY(dim == dimX, "Inconsistent snapshot sizes");
            GetSnapshotDim(paramID, outputPath, "V", basisIdentifierString, sampleWindow, dim, dummy);
            MFEM_VERIFY(dim == dimV && dummy == snapshotSize[paramID] + extraV, "Inconsistent snapshot sizes");
            GetSnapshotDim(paramID, outputPath, "E", basisIdentifierString, sampleWindow, dim, dummy);
            MFEM_VERIFY(dim == dimE && dummy == snapshotSize[paramID], "Inconsistent snapshot sizes");

            if (!romOptions.SNS)
            {
                GetSnapshotDim(paramID, outputPath, "Fv", basisIdentifierString, sampleWindow, dim, snapshotSizeFv[paramID]);
                MFEM_VERIFY(dim == dimV && snapshotSizeFv[paramID] >= snapshotSize[paramID], "Inconsistent snapshot sizes");
                GetSnapshotDim(paramID, outputPath, "Fe", basisIdentifierString, sampleWindow, dim, snapshotSizeFe[paramID]);
                MFEM_VERIFY(dim == dimE && snapshotSizeFe[paramID] >= snapshotSize[paramID], "Inconsistent snapshot sizes");
            }

            totalSnapshotSize += snapshotSize[paramID];
            totalSnapshotSizeFv += snapshotSizeFv[paramID];
            totalSnapshotSizeFe += snapshotSizeFe[paramID];
        }

        int lastBasisWindow = (windowNumSamples > 0) ? numBasisWindows - 1 : sampleWindow;
        for (int basisWindow = sampleWindow; basisWindow <= lastBasisWindow; ++basisWindow)
        {
            LoadSampleSets(myid, romOptions.energyFraction, romOptions.sv_shift, nset, outputPath, VariableName::X, basisIdentifierString, usingWindows, windowNumSamples, windowOverlapSamples,
                           basisWindow, romOptions.useOffset, romOptions.offsetType, dimX, totalSnapshotSize, offsetAllWindows, cutoff[0], eqp, romOptions.squareSV);
            LoadSampleSets(myid, romOptions.energyFraction, romOptions.sv_shift, nset, outputPath, VariableName::V, basisIdentifierString, usingWindows, windowNumSamples, windowOverlapSamples,
                           basisWindow, romOptions.useOffset, romOptions.offsetType, dimV, totalSnapshotSize + extraV, offsetAllWindows, cutoff[1], eqp, romOptions.squareSV);
            LoadSampleSets(myid, romOptions.energyFraction, romOptions.sv_shift, nset, outputPath, VariableName::E, basisIdentifierString, usingWindows, windowNumSamples, windowOverlapSamples,
                           basisWindow, romOptions.useOffset, romOptions.offsetType, dimE, totalSnapshotSize, offsetAllWindows, cutoff[2], eqp, romOptions.squareSV);

            if (!romOptions.SNS)
            {
                LoadSampleSets(myid, romOptions.energyFraction, romOptions.sv_shift, nset, outputPath, VariableName::Fv, basisIdentifierString, usingWindows, windowNumSamples, windowOverlapSamples,
                               basisWindow, romOptions.useOffset, romOptions.offsetType, dimV, totalSnapshotSizeFv, offsetAllWindows, cutoff[3], eqp, romOptions.squareSV);
                LoadSampleSets(myid, romOptions.energyFraction, romOptions.sv_shift, nset, outputPath, VariableName::Fe, basisIdentifierString, usingWindows, windowNumSamples, windowOverlapSamples,
                               basisWindow, romOptions.useOffset, romOptions.offsetType, dimE, totalSnapshotSizeFe, offsetAllWindows, cutoff[4], eqp, romOptions.squareSV);
            }

            if (myid == 0 && usingWindows)
            {
                outfile_twp << twep[basisWindow] << ", " << cutoff[0] << ", " << cutoff[1] << ", " << cutoff[2];
                if (romOptions.SNS)
                    outfile_twp << "\n";
                else
                    outfile_twp << ", " << cutoff[3] << ", " << cutoff[4] << "\n";
            }
        }
    }

    mergeTimer.Stop();
    if (myid == 0)
    {
        cout << "Elapsed time for merge: " << mergeTimer.RealTime() << " sec\n";
    }
    return 0;
}
