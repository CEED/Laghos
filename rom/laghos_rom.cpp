#include "laghos_rom.hpp"
#include "laghos_utils.hpp"

#include "BasisGenerator.h"
#include "BasisReader.h"
#include "GreedyParameterPointRandomSampler.h"

#include "DEIM.h"
#include "QDEIM.h"
#include "SampleMesh.hpp"
#include "STSampling.h"

using namespace std;

void ROM_Sampler::SampleSolution(const double t, const double dt, const double pd, Vector const& S)
{
    SetStateVariables(S);
    SetStateVariableRates(dt);

    const bool sampleX = generator_X->isNextSample(t);

    Vector dSdt;
    if (!sns)
    {
        dSdt.SetSize(S.Size());
        lhoper->Mult(S, dSdt);
    }

    if (sampleX && !useXV)
    {
        if (rank == 0)
        {
            cout << "X taking sample at t " << t << endl;
        }

        bool addSample;

        if (offsetInit)
        {
            for (int i=0; i<tH1size; ++i)
            {
                Xdiff[i] = X[i] - (*initX)(i);
            }

            addSample = generator_X->takeSample(Xdiff.GetData(), t, dt);
            generator_X->computeNextSampleTime(Xdiff.GetData(), dXdt.GetData(), t);
        }
        else
        {
            addSample = generator_X->takeSample(X.GetData(), t, dt);
            generator_X->computeNextSampleTime(X.GetData(), dXdt.GetData(), t);
        }

        if (writeSnapshots && addSample)
        {
            tSnapX.push_back(t);
            if (pd >= 0.0) pdSnap.push_back(pd);
        }
    }

    const bool sampleV = generator_V->isNextSample(t);

    //TODO: use this, plus generator_Fv->computeNextSampleTime? So far, it seems sampleV == true on every step.
    //const bool sampleFv = generator_Fv->isNextSample(t);

    if (sampleV)
    {
        if (!useVX)
        {
            if (rank == 0)
            {
                cout << "V taking sample at t " << t << endl;
            }

            bool addSample;

            if (offsetInit && Voffset)
            {
                for (int i=0; i<tH1size; ++i)
                {
                    Xdiff[i] = V[i] - (*initV)(i);
                }

                addSample = generator_V->takeSample(Xdiff.GetData(), t, dt);
                generator_V->computeNextSampleTime(Xdiff.GetData(), dVdt.GetData(), t);
            }
            else
            {
                addSample = generator_V->takeSample(V.GetData(), t, dt);
                generator_V->computeNextSampleTime(V.GetData(), dVdt.GetData(), t);
            }

            if (writeSnapshots && addSample)
            {
                tSnapV.push_back(t);
            }
        }

        if (!sns)
        {
            MFEM_VERIFY(gfH1.Size() == H1size, "");
            for (int i=0; i<H1size; ++i)
                gfH1[i] = dSdt[H1size + i];  // Fv

            gfH1.GetTrueDofs(Xdiff);
            bool addSampleF = generator_Fv->takeSample(Xdiff.GetData(), t, dt);

            if (writeSnapshots && addSampleF)
            {
                tSnapFv.push_back(t);
            }
        }
    }

    const bool sampleE = generator_E->isNextSample(t);

    if (sampleE)
    {
        if (rank == 0)
        {
            cout << "E taking sample at t " << t << endl;
        }

        bool addSample, addSampleF;

        if (offsetInit)
        {
            for (int i=0; i<tL2size; ++i)
            {
                Ediff[i] = E[i] - (*initE)(i);
            }

            addSample = generator_E->takeSample(Ediff.GetData(), t, dt);
            generator_E->computeNextSampleTime(Ediff.GetData(), dEdt.GetData(), t);

        }
        else
        {
            addSample = generator_E->takeSample(E.GetData(), t, dt);
            generator_E->computeNextSampleTime(E.GetData(), dEdt.GetData(), t);
        }

        if (!sns)
        {
            MFEM_VERIFY(gfL2.Size() == L2size, "");
            for (int i=0; i<L2size; ++i)
                gfL2[i] = dSdt[(2*H1size) + i];  // Fe

            gfL2.GetTrueDofs(Ediff);
            addSampleF = generator_Fe->takeSample(Ediff.GetData(), t, dt);

            if (writeSnapshots && addSampleF)
            {
                tSnapFe.push_back(t);
            }
        }

        if (writeSnapshots && addSample)
        {
            tSnapE.push_back(t);
        }
    }
}

void printSnapshotTime(std::vector<double> const &tSnap, std::string const path, std::string const var)
{
    cout << var << " snapshot size: " << tSnap.size() << endl;
    std::ofstream outfile_tSnap(path + var);

    outfile_tSnap.precision(std::numeric_limits<double>::max_digits10);
    for (auto const& i: tSnap)
    {
        outfile_tSnap << i << endl;
    }
}

void ROM_Sampler::Finalize(Array<int> &cutoff, ROM_Options& input)
{
    if (writeSnapshots)
    {
        if (!useXV) generator_X->writeSnapshot();
        if (!useVX) generator_V->writeSnapshot();
        generator_E->writeSnapshot();
        if (!sns)
        {
            generator_Fv->writeSnapshot();
            generator_Fe->writeSnapshot();
        }
    }
    else
    {
        if (!useXV) generator_X->endSamples();
        if (!useVX) generator_V->endSamples();
        generator_E->endSamples();
        if (!sns)
        {
            generator_Fv->endSamples();
            generator_Fe->endSamples();
        }
    }

    if (rank == 0 && !writeSnapshots)
    {
        if (!useXV)
        {
            cout << "X basis summary output: " << endl;
            BasisGeneratorFinalSummary(generator_X, energyFraction_X, cutoff[0], basename + "/" + "rdimx" + input.basisIdentifier);
            PrintSingularValues(rank, basename, "X" + input.basisIdentifier, generator_X);
        }

        if (!useVX)
        {
            cout << "V basis summary output: " << endl;
            BasisGeneratorFinalSummary(generator_V, energyFraction, cutoff[1], basename + "/" + "rdimv" + input.basisIdentifier);
            PrintSingularValues(rank, basename, "V" + input.basisIdentifier, generator_V);
        }

        cout << "E basis summary output: " << endl;
        BasisGeneratorFinalSummary(generator_E, energyFraction, cutoff[2], basename + "/" + "rdime" + input.basisIdentifier);
        PrintSingularValues(rank, basename, "E" + input.basisIdentifier, generator_E);

        if (!sns)
        {
            cout << "Fv basis summary output: " << endl;
            BasisGeneratorFinalSummary(generator_Fv, energyFraction, cutoff[3], basename + "/" + "rdimfv" + input.basisIdentifier);

            cout << "Fe basis summary output: " << endl;
            BasisGeneratorFinalSummary(generator_Fe, energyFraction, cutoff[4], basename + "/" + "rdimfe" + input.basisIdentifier);
        }
    }

    if (rank == 0 && writeSnapshots)
    {
        std::string path_tSnap = basename + "/param" + std::to_string(parameterID) + "_tSnap";

        printSnapshotTime(tSnapX, path_tSnap, "X");
        printSnapshotTime(tSnapV, path_tSnap, "V");
        printSnapshotTime(tSnapE, path_tSnap, "E");

        if (!sns)
        {
            printSnapshotTime(tSnapFv, path_tSnap, "Fv");
            printSnapshotTime(tSnapFe, path_tSnap, "Fe");
        }

        if (pdSnap.size() > 0)
        {
            std::string path_pdSnap = basename + "/param" + std::to_string(parameterID) + "_pdSnap";
            printSnapshotTime(pdSnap, path_pdSnap, "X");
        }
    }

    if (spaceTime)
    {
        finalNumSamples = generator_X->getTemporalBasis()->numRows();
        // TODO: this is a lot of checks, for debugging. Maybe these should be removed later.
        MFEM_VERIFY(finalNumSamples == MaxNumSamples(), "bug");
        MFEM_VERIFY(finalNumSamples == generator_V->getTemporalBasis()->numRows() + VTos, "bug");
        MFEM_VERIFY(finalNumSamples == generator_E->getTemporalBasis()->numRows(), "bug");
        MFEM_VERIFY(finalNumSamples == generator_Fv->getTemporalBasis()->numRows(), "bug");
        MFEM_VERIFY(finalNumSamples == generator_Fe->getTemporalBasis()->numRows() + VTos, "bug");
    }

    delete generator_X;
    delete generator_V;
    delete generator_E;

    if (!sns)
    {
        delete generator_Fv;
        delete generator_Fe;
    }

    finalized = true;
}

CAROM::Matrix* GetFirstColumns(const int N, const CAROM::Matrix* A, const int rowOS, const int numRows)
{
    CAROM::Matrix* S = new CAROM::Matrix(numRows, std::min(N, A->numColumns()), A->distributed());
    for (int i=0; i<S->numRows(); ++i)
    {
        for (int j=0; j<S->numColumns(); ++j)
            (*S)(i,j) = (*A)(rowOS + i, j);
    }

    return S;
}

CAROM::Matrix* ReadBasisROM(const int rank, const std::string filename, const int vectorSize, const int rowOS, int& dim)
{
    CAROM::BasisReader reader(filename);
    const CAROM::Matrix *basis = (CAROM::Matrix*) reader.getSpatialBasis(0.0);

    if (dim == -1)
        dim = basis->numColumns();

    // Make a deep copy of basis, which is inefficient but necessary since BasisReader owns the basis data and deletes it when BasisReader goes out of scope.
    // An alternative would be to keep all the BasisReader instances as long as each basis is kept, but that would be inconvenient.
    // Another alternative would be for BasisReader to allow the option to release ownership.
    // On the other hand, maybe it is best just to keep doing this copy, since it truncates the basis.
    CAROM::Matrix* basisCopy = GetFirstColumns(dim, basis, rowOS, vectorSize);

    MFEM_VERIFY(basisCopy->numRows() == vectorSize, "");

    if (rank == 0)
        cout << "Read basis " << filename << " of dimension " << basisCopy->numColumns() << endl;

    //delete basis;  // TODO: it seems this can be done safely.
    return basisCopy;
}

CAROM::Matrix* ReadTemporalBasisROM(const int rank, const std::string filename, int& temporalSize, int& dim)
{
    CAROM::BasisReader reader(filename);
    const CAROM::Matrix *basis = (CAROM::Matrix*) reader.getTemporalBasis(0.0);

    // The size of basis is (number of time samples) x (basis dimension), and it is a distributed matrix.
    // In libROM, a Matrix is always distributed row-wise. In this case, the global matrix is on each process.
    temporalSize = basis->numRows();
    if (dim == -1)
        dim = basis->numColumns();

    // Make a deep copy of basis, which is inefficient but necessary since BasisReader owns the basis data and deletes it when BasisReader goes out of scope.
    // An alternative would be to keep all the BasisReader instances as long as each basis is kept, but that would be inconvenient.
    // Another alternative would be for BasisReader to allow the option to release ownership.
    // On the other hand, maybe it is best just to keep doing this copy, since it truncates the basis.
    CAROM::Matrix* basisCopy = GetFirstColumns(dim, basis, 0, temporalSize);

    MFEM_VERIFY(basisCopy->numRows() == temporalSize, "");

    if (rank == 0)
        cout << "Read temporal basis " << filename << " of dimension " << basisCopy->numColumns() << endl;

    //delete basis;  // TODO: it seems this can be done safely.
    return basisCopy;
}

CAROM::Matrix* MultBasisROM(const int rank, const std::string filename, const int vectorSize, const int rowOS, int& dim,
                            hydrodynamics::LagrangianHydroOperator *lhoper, const int var)
{
    CAROM::Matrix* A = ReadBasisROM(rank, filename, vectorSize, rowOS, dim);
    CAROM::Matrix* S = new CAROM::Matrix(A->numRows(), A->numColumns(), A->distributed());
    Vector Bej(A->numRows());
    Vector MBej(A->numRows());

    for (int j=0; j<S->numColumns(); ++j)
    {
        for (int i=0; i<S->numRows(); ++i)
            Bej[i] = (*A)(i,j);

        if (var == 1)
            lhoper->MultMv(Bej, MBej);
        else if (var == 2)
            lhoper->MultMe(Bej, MBej);
        else
            MFEM_ABORT("Invalid input");

        for (int i=0; i<S->numRows(); ++i)
            (*S)(i,j) = MBej[i];
    }

    delete A;

    return S;
}

ROM_Basis::ROM_Basis(ROM_Options const& input, MPI_Comm comm_, MPI_Comm rom_com_, const double sFactorX, const double sFactorV,
                     const std::vector<double> *timesteps)
    : comm(comm_), rom_com(rom_com_), rdimx(input.dimX), rdimv(input.dimV), rdime(input.dimE), rdimfv(input.dimFv), rdimfe(input.dimFe),
      numSamplesX(input.sampX), numSamplesV(input.sampV), numSamplesE(input.sampE),
      numTimeSamplesV(input.tsampV), numTimeSamplesE(input.tsampE),
      use_sns(input.SNS),  offsetInit(input.useOffset),
      hyperreduce(input.hyperreduce), hyperreduce_prep(input.hyperreduce_prep),
      useGramSchmidt(input.GramSchmidt), lhoper(input.FOMoper),
      RK2AvgFormulation(input.RK2AvgSolver), basename(*input.basename), solution_basename(*input.solution_basename),
      mergeXV(input.mergeXV), useXV(input.useXV), useVX(input.useVX), Voffset(!input.useXV && !input.useVX && !input.mergeXV),
      energyFraction_X(input.energyFraction_X), use_qdeim(input.qdeim), basisIdentifier(input.basisIdentifier),
      spaceTimeMethod(input.spaceTimeMethod), spaceTime(input.spaceTimeMethod != no_space_time), VTos(input.VTos)
{
    MFEM_VERIFY(!(input.useXV && input.useVX) && !(input.useXV && input.mergeXV) && !(input.useVX && input.mergeXV), "");

    if (useXV) rdimx = rdimv;
    if (useVX) rdimv = rdimx;

    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    Array<int> nH1(nprocs);

    if (spaceTime || !hyperreduce)
    {
        tH1size = input.H1FESpace->GetTrueVSize();
        tL2size = input.L2FESpace->GetTrueVSize();
        H1size = input.H1FESpace->GetVSize();
        L2size = input.L2FESpace->GetVSize();
        gfH1 = new ParGridFunction(input.H1FESpace);
        gfL2 = new ParGridFunction(input.L2FESpace);

        Array<int> osH1(nprocs+1);
        Array<int> osL2(nprocs+1);
        MPI_Allgather(&tH1size, 1, MPI_INT, osH1.GetData(), 1, MPI_INT, comm);
        MPI_Allgather(&tL2size, 1, MPI_INT, osL2.GetData(), 1, MPI_INT, comm);

        for (int i=nprocs-1; i>=0; --i)
        {
            nH1[i] = osH1[i];
            osH1[i+1] = osH1[i];
            osL2[i+1] = osL2[i];
        }

        osH1[0] = 0;
        osL2[0] = 0;

        osH1.PartialSum();
        osL2.PartialSum();

        rowOffsetH1 = osH1[rank];
        rowOffsetL2 = osL2[rank];

        fH1 = new CAROM::Vector(tH1size, true);
        fL2 = new CAROM::Vector(tL2size, true);

        mfH1.SetSize(tH1size);
        mfL2.SetSize(tL2size);

        ReadSolutionBases(input.window);
        if (spaceTime)
        {
            ReadTemporalBases(input.window);
        }

        if (hyperreduce_prep && rank == 0)
        {
            writeNum(rdimx, basename + "/" + "rdimx" + "_" + to_string(input.window));
            writeNum(rdimv, basename + "/" + "rdimv" + "_" + to_string(input.window));
            writeNum(rdime, basename + "/" + "rdime" + "_" + to_string(input.window));
        }
    }
    else if (rank == 0 && !spaceTime)  // TODO: read/write this for spaceTime case?
    {
        readNum(rdimx, basename + "/" + "rdimx" + "_" + to_string(input.window));
        readNum(rdimv, basename + "/" + "rdimv" + "_" + to_string(input.window));
        readNum(rdime, basename + "/" + "rdime" + "_" + to_string(input.window));
    }

    if (mergeXV)
    {
        // Update number of samples based on changed rdimx and rdimv.
        numSamplesX = sFactorX * rdimx;
        numSamplesV = sFactorV * rdimv;
    }

    rX = new CAROM::Vector(rdimx, false);
    rV = new CAROM::Vector(rdimv, false);
    rE = new CAROM::Vector(rdime, false);

    if (RK2AvgFormulation)
    {
        rX2 = new CAROM::Vector(rdimx, false);
        rV2 = new CAROM::Vector(rdimv, false);
        rE2 = new CAROM::Vector(rdime, false);
    }

    if (hyperreduce)
    {
        if (rank == 0)
        {
            readSP(input, input.window); // TODO: in space-time case, hangs in parallel!
        }
        if (!spaceTime) return;
    }

    if (offsetInit || spaceTime)
    {
        initX = new CAROM::Vector(tH1size, true);
        initV = new CAROM::Vector(tH1size, true);
        initE = new CAROM::Vector(tL2size, true);
    }

    if (offsetInit)
    {
        std::string path_init = basename + "/ROMoffset/init";

        if (input.offsetType == useInitialState)
        {
            // Read offset in the online phase of initial mode
            initX->read(path_init + "X0");
            initV->read(path_init + "V0");
            initE->read(path_init + "E0");

            cout << "Read init vectors X, V, E with norms " << initX->norm() << ", " << initV->norm() << ", " << initE->norm() << endl;
        }
        else if (input.restore || input.offsetType == saveLoadOffset)
        {
            // Read offsets in the restore phase or in the online phase of non-parametric save-and-load mode
            initX->read(path_init + "X" + std::to_string(input.window));
            initV->read(path_init + "V" + std::to_string(input.window));
            initE->read(path_init + "E" + std::to_string(input.window));

            cout << "Read init vectors X, V, E with norms " << initX->norm() << ", " << initV->norm() << ", " << initE->norm() << endl;
        }
        else if (input.offsetType == interpolateOffset)
        {

            // Calculation of coefficients of offset data using inverse distance weighting interpolation
            std::ifstream infile_offlineParam(basename + "/offline_param.csv");
            MFEM_VERIFY(infile_offlineParam.is_open(), "Offline parameter record file does not exist.");
            std::string line;
            std::vector<std::string> words;
            std::getline(infile_offlineParam, line);
            int true_idx = -1;
            double coeff_sum = 0.0;
            // Compute the distances from online parameters to each offline parameter
            while (std::getline(infile_offlineParam, line))
            {
                split_line(line, words);
                paramID_list.push_back(std::stoi(words[0]));
                double coeff = 0.0;
                coeff += (input.rhoFactor - atof(words[1].c_str())) * (input.rhoFactor - atof(words[1].c_str()));
                coeff += (input.blast_energyFactor - atof(words[2].c_str())) * (input.blast_energyFactor - atof(words[2].c_str()));
                coeff += (input.atwoodFactor - atof(words[3].c_str())) * (input.atwoodFactor - atof(words[3].c_str()));
                if (coeff == 0.0)
                {
                    true_idx = coeff_list.size();
                }
                coeff = 1.0 / sqrt(coeff);
                coeff_sum += coeff;
                coeff_list.push_back(coeff);
            }
            // Determine the coefficients with respect to the offline parameters
            // The coefficients are inversely porportional to distances and form a convex combination of offset data
            for (int param=0; param<paramID_list.size(); ++param)
            {
                if (true_idx >= 0)
                {
                    coeff_list[param] = (param == true_idx) ? 1.0 : 0.0;
                }
                else
                {
                    coeff_list[param] /= coeff_sum;
                }
            }
            infile_offlineParam.close();

            // Interpolate and save offset in the online phase of parametric save-and-load mode
            for (int i=0; i<tH1size; ++i)
                (*initX)(i) = 0;
            for (int i=0; i<tH1size; ++i)
                (*initV)(i) = 0;
            for (int i=0; i<tL2size; ++i)
                (*initE)(i) = 0;

            for (int param=0; param<paramID_list.size(); ++param)
            {
                CAROM::Vector *initX_off = 0;
                CAROM::Vector *initV_off = 0;
                CAROM::Vector *initE_off = 0;

                initX_off = new CAROM::Vector(tH1size, true);
                initV_off = new CAROM::Vector(tH1size, true);
                initE_off = new CAROM::Vector(tL2size, true);

                int paramID_off = paramID_list[param];
                std::string path_init_off = basename + "/ROMoffset/param" + std::to_string(paramID_off) + "_init" ; // paramID_off = 0, 1, 2, ...

                initX_off->read(path_init_off + "X" + std::to_string(input.window));
                initV_off->read(path_init_off + "V" + std::to_string(input.window));
                initE_off->read(path_init_off + "E" + std::to_string(input.window));

                for (int i=0; i<tH1size; ++i)
                    (*initX)(i) += coeff_list[param] * (*initX_off)(i);
                for (int i=0; i<tH1size; ++i)
                    (*initV)(i) += coeff_list[param] * (*initV_off)(i);
                for (int i=0; i<tL2size; ++i)
                    (*initE)(i) += coeff_list[param] * (*initE_off)(i);
            }
            initX->write(path_init + "X" + std::to_string(input.window));
            initV->write(path_init + "V" + std::to_string(input.window));
            initE->write(path_init + "E" + std::to_string(input.window));

            cout << "Interpolated init vectors X, V, E with norms " << initX->norm() << ", " << initV->norm() << ", " << initE->norm() << endl;
        }
    }

    if (hyperreduce_prep)
    {
        if (rank == 0) cout << "start preprocessing hyper-reduction\n";
        StopWatch preprocessHyperreductionTimer;
        preprocessHyperreductionTimer.Start();
        SetupHyperreduction(input.H1FESpace, input.L2FESpace, nH1, input.window, timesteps);
        preprocessHyperreductionTimer.Stop();
        if (rank == 0) cout << "Elapsed time for hyper-reduction preprocessing: " << preprocessHyperreductionTimer.RealTime() << " sec\n";
    }

    if (spaceTime && hyperreduce)
    {
        // TODO: include in preprocessHyperreductionTimer?
        SetSpaceTimeInitialGuess(input);
    }
}

void ROM_Basis::ProjectFromPreviousWindow(ROM_Options const& input, Vector& romS, int window, int rdimxPrev, int rdimvPrev, int rdimePrev)
{
    MFEM_VERIFY(rank == 0 && window > 0, "");

    BwinX = new CAROM::Matrix(rdimx, rdimxPrev, false);
    BwinV = new CAROM::Matrix(rdimv, rdimvPrev, false);
    BwinE = new CAROM::Matrix(rdime, rdimePrev, false);

    BwinX->read(basename + "/" + "BwinX" + "_" + to_string(window));
    BwinV->read(basename + "/" + "BwinV" + "_" + to_string(window));
    BwinE->read(basename + "/" + "BwinE" + "_" + to_string(window));

    CAROM::Vector romS_oldX(rdimxPrev, false);
    CAROM::Vector romS_oldV(rdimvPrev, false);
    CAROM::Vector romS_oldE(rdimePrev, false);
    CAROM::Vector romS_X(rdimx, false);
    CAROM::Vector romS_V(rdimv, false);
    CAROM::Vector romS_E(rdime, false);

    for (int i=0; i<rdimxPrev; ++i)
        romS_oldX(i) = romS[i];

    for (int i=0; i<rdimvPrev; ++i)
        romS_oldV(i) = romS[rdimxPrev + i];

    for (int i=0; i<rdimePrev; ++i)
        romS_oldE(i) = romS[rdimxPrev + rdimvPrev + i];

    BwinX->mult(romS_oldX, romS_X);
    BwinV->mult(romS_oldV, romS_V);
    BwinE->mult(romS_oldE, romS_E);

    if (offsetInit && (input.offsetType == interpolateOffset || input.offsetType == saveLoadOffset))
    {
        BtInitDiffX = new CAROM::Vector(rdimx, false);
        BtInitDiffV = new CAROM::Vector(rdimv, false);
        BtInitDiffE = new CAROM::Vector(rdime, false);

        BtInitDiffX->read(basename + "/" + "BtInitDiffX" + "_" + to_string(window));
        BtInitDiffV->read(basename + "/" + "BtInitDiffV" + "_" + to_string(window));
        BtInitDiffE->read(basename + "/" + "BtInitDiffE" + "_" + to_string(window));

        romS_X += *BtInitDiffX;
        romS_V += *BtInitDiffV;
        romS_E += *BtInitDiffE;
    }

    romS.SetSize(rdimx + rdimv + rdime);

    for (int i=0; i<rdimx; ++i)
        romS[i] = romS_X(i);

    for (int i=0; i<rdimv; ++i)
        romS[rdimx + i] = romS_V(i);

    for (int i=0; i<rdime; ++i)
        romS[rdimx + rdimv + i] = romS_E(i);
}

void ROM_Basis::Init(ROM_Options const& input, Vector const& S)
{
    if ((offsetInit || spaceTime) && !input.restore && input.offsetType == useInitialState && input.window == 0)
    {
        std::string path_init = basename + "/ROMoffset/init";

        // Compute and save offset in the online phase for the initial window in the useInitialState mode
        Vector X, V, E;

        for (int i=0; i<H1size; ++i)
        {
            (*gfH1)(i) = S[i];
        }
        gfH1->GetTrueDofs(X);
        for (int i=0; i<tH1size; ++i)
        {
            (*initX)(i) = X[i];
        }

        for (int i=0; i<H1size; ++i)
        {
            (*gfH1)(i) = S[H1size+i];
        }
        gfH1->GetTrueDofs(V);
        for (int i=0; i<tH1size; ++i)
        {
            (*initV)(i) = V[i];
        }

        for (int i=0; i<L2size; ++i)
        {
            (*gfL2)(i) = S[2*H1size+i];
        }
        gfL2->GetTrueDofs(E);
        for (int i=0; i<tL2size; ++i)
        {
            (*initE)(i) = E[i];
        }

        if (!spaceTime)
        {
            initX->write(path_init + "X" + std::to_string(input.window));
            initV->write(path_init + "V" + std::to_string(input.window));
            initE->write(path_init + "E" + std::to_string(input.window));
        }
    }

    if ((offsetInit || spaceTime) && hyperreduce_prep)
    {
        CAROM::Matrix FOMX0(tH1size, 2, true);

        for (int i=0; i<tH1size; ++i)
        {
            FOMX0(i,0) = (*initX)(i);
            FOMX0(i,1) = (*initV)(i);
        }

        CAROM::Matrix FOME0(tL2size, 1, true);

        for (int i=0; i<tL2size; ++i)
        {
            FOME0(i,0) = (*initE)(i);
        }

        CAROM::Matrix spX0mat(rank == 0 ? size_H1_sp : 1, 2, false);
        CAROM::Matrix spE0mat(rank == 0 ? size_L2_sp : 1, 1, false);

#ifdef FULL_DOF_STENCIL
        const int NR = input.H1FESpace->GetVSize();
        GatherDistributedMatrixRows(FOMX0, FOME0, 2, 1, NR, *input.H1FESpace, *input.L2FESpace, st2sp, sprows, all_sprows, spX0mat, spE0mat);
#else
        GatherDistributedMatrixRows(FOMX0, FOME0, 2, 1, st2sp, sprows, all_sprows, spX0mat, spE0mat);
#endif

        if (rank == 0)
        {
            initXsp = new CAROM::Vector(size_H1_sp, false);
            initVsp = new CAROM::Vector(size_H1_sp, false);
            initEsp = new CAROM::Vector(size_L2_sp, false);
            for (int i=0; i<size_H1_sp; ++i)
            {
                (*initXsp)(i) = spX0mat(i,0);
                (*initVsp)(i) = spX0mat(i,1);
            }
            for (int i=0; i<size_L2_sp; ++i)
            {
                (*initEsp)(i) = spE0mat(i,0);
            }
        }
    }
}

// cp = a x b
void CrossProduct3D(Vector const& a, Vector const& b, Vector& cp)
{
    cp[0] = (a[1]*b[2]) - (a[2]*b[1]);
    cp[1] = (a[2]*b[0]) - (a[0]*b[2]);
    cp[2] = (a[0]*b[1]) - (a[1]*b[0]);
}

// Set attributes 1/2/3 corresponding to fixed-x/y/z boundaries.
void SetBdryAttrForVelocity(ParMesh *pmesh)
{
    for (int b=0; b<pmesh->GetNBE(); ++b)
    {
        Element *belem = pmesh->GetBdrElement(b);
        Array<int> vert;
        belem->GetVertices(vert);
        MFEM_VERIFY(vert.Size() > 2, "");

        Vector normal(3);

        Vector t1(3);
        Vector t2(3);

        for (int i=0; i<3; ++i)
        {
            t1[i] = pmesh->GetVertex(vert[0])[i] - pmesh->GetVertex(vert[1])[i];
            t2[i] = pmesh->GetVertex(vert[2])[i] - pmesh->GetVertex(vert[1])[i];
        }

        CrossProduct3D(t1, t2, normal);

        const double s = normal.Norml2();
        MFEM_VERIFY(s > 1.0e-8, "");

        normal /= s;

        int attr = -1;

        {   // Verify that the normal is in the direction of a Cartesian axis.
            const double tol = 1.0e-8;
            const double al1 = fabs(normal[0]) + fabs(normal[1]) + fabs(normal[2]);
            MFEM_VERIFY(fabs(al1 - 1.0) < tol, "");
            bool axisFound = false;

            if (fabs(1.0 - fabs(normal[0])) < tol)
                attr = 1;
            else if (fabs(1.0 - fabs(normal[1])) < tol)
                attr = 2;
            else if (fabs(1.0 - fabs(normal[2])) < tol)
                attr = 3;
        }

        MFEM_VERIFY(attr > 0, "");

        belem->SetAttribute(attr);
    }

    pmesh->SetAttributes();
}

// Set attributes 1/2/3 corresponding to fixed-x/y/z boundaries on a 2D or 3D ParMesh
// with boundaries aligned with Cartesian axes.
void SetBdryAttrForVelocity_Cartesian(ParMesh *pmesh)
{
    const int dim = pmesh->Dimension();

    // First set minimum and maximum coordinates, locally then globally.
    MFEM_VERIFY(pmesh->GetNV() > 0 && (dim == 2 || dim == 3), "");

    double xmin[dim], xmax[dim];
    for (int i=0; i<dim; ++i)
    {
        xmin[i] = pmesh->GetVertex(0)[i];
        xmax[i] = xmin[i];
    }

    for (int v=1; v<pmesh->GetNV(); ++v)
    {
        for (int i=0; i<dim; ++i)
        {
            xmin[i] = std::min(pmesh->GetVertex(v)[i], xmin[i]);
            xmax[i] = std::max(pmesh->GetVertex(v)[i], xmax[i]);
        }
    }

    {   // Globally reduce
        double local[dim];
        for (int i=0; i<dim; ++i)
            local[i] = xmin[i];

        MPI_Allreduce(local, xmin, dim, MPI_DOUBLE, MPI_MIN, pmesh->GetComm());

        for (int i=0; i<dim; ++i)
            local[i] = xmax[i];

        MPI_Allreduce(local, xmax, dim, MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
    }

    for (int b=0; b<pmesh->GetNBE(); ++b)
    {
        Element *belem = pmesh->GetBdrElement(b);
        Array<int> vert;
        belem->GetVertices(vert);
        MFEM_VERIFY(vert.Size() > 1, "");

        Vector normal(dim);

        if (dim == 3)
        {
            Vector t1(dim);
            Vector t2(dim);

            for (int i=0; i<dim; ++i)
            {
                t1[i] = pmesh->GetVertex(vert[0])[i] - pmesh->GetVertex(vert[1])[i];
                t2[i] = pmesh->GetVertex(vert[2])[i] - pmesh->GetVertex(vert[1])[i];
            }

            CrossProduct3D(t1, t2, normal);
        }
        else
        {
            normal[0] = -(pmesh->GetVertex(vert[0])[1] - pmesh->GetVertex(vert[1])[1]);  // -tangent_y
            normal[1] = (pmesh->GetVertex(vert[0])[0] - pmesh->GetVertex(vert[1])[0]);  // tangent_x
        }

        {
            const double s = normal.Norml2();
            MFEM_VERIFY(s > 1.0e-8, "");

            normal /= s;
        }

        int attr = -1;

        const double tol = 1.0e-8;

        {   // Verify that the normal is in the direction of a Cartesian axis.
            const double al1 = fabs(normal[0]) + fabs(normal[1]) + (dim == 3 ? fabs(normal[2]) : 0);
            MFEM_VERIFY(fabs(al1 - 1.0) < tol, "");
            bool axisFound = false;

            if (fabs(1.0 - fabs(normal[0])) < tol)
                attr = 1;
            else if (fabs(1.0 - fabs(normal[1])) < tol)
                attr = 2;
            else if (dim == 3 && fabs(1.0 - fabs(normal[2])) < tol)
                attr = 3;
        }

        MFEM_VERIFY(attr > 0 && attr < dim+1, "");

        bool onBoundary = true;
        {
            const double xd0 = pmesh->GetVertex(vert[0])[attr-1];

            for (int j=0; j<vert.Size(); ++j)
            {
                const double xd = pmesh->GetVertex(vert[j])[attr-1];
                if (fabs(xd - xmin[attr-1]) > tol && fabs(xmax[attr-1] - xd) > tol)  // specific to Cartesian-aligned domains
                    onBoundary = false;

                if (j > 0 && fabs(xd - xd0) > tol)
                    onBoundary = false;
            }
        }

        if (onBoundary)
            belem->SetAttribute(attr);
        else
            belem->SetAttribute(10);
    }

    pmesh->SetAttributes();
}

void ROM_Basis::SetupHyperreduction(ParFiniteElementSpace *H1FESpace, ParFiniteElementSpace *L2FESpace, Array<int>& nH1, const int window,
                                    const std::vector<double> *timesteps)
{
    ParMesh *pmesh = H1FESpace->GetParMesh();

    const int fomH1size = H1FESpace->GlobalTrueVSize();
    const int fomL2size = L2FESpace->GlobalTrueVSize();

    numSamplesX = 0;
    vector<int> sample_dofs_X(numSamplesX);
    vector<int> num_sample_dofs_per_procX(nprocs);
    BsinvX = NULL;

    numSamplesV = std::min(fomH1size, numSamplesV);
    vector<int> sample_dofs_V(numSamplesV);
    vector<int> num_sample_dofs_per_procV(nprocs);
    BsinvV = spaceTime ? NULL : new CAROM::Matrix(numSamplesV, rdimfv, false);

    numSamplesE = std::min(fomL2size, numSamplesE);
    vector<int> sample_dofs_E(numSamplesE);
    vector<int> num_sample_dofs_per_procE(nprocs);
    BsinvE = spaceTime ? NULL : new CAROM::Matrix(numSamplesE, rdimfe, false);

    if (rank == 0)
    {
        cout << "number of samples for velocity: " << numSamplesV << "\n";
        cout << "number of samples for energy  : " << numSamplesE << "\n";
    }

    // Perform DEIM, GNAT, or QDEIM to find sample DOF's.

    if (spaceTime)
    {
        MFEM_VERIFY(temporalSize == tbasisFv->numRows(), "");
        MFEM_VERIFY(temporalSize == tbasisFe->numRows() + VTos, "");

        MFEM_VERIFY(numTimeSamplesV < temporalSize, "");
        MFEM_VERIFY(numTimeSamplesE < temporalSize, "");

        std::vector<int> t_samples_V(numTimeSamplesV);
        std::vector<int> t_samples_E(numTimeSamplesE);

        const bool excludeFinalTimeSample = true;

        CAROM::Matrix sampledSpatialBasisV(numSamplesV, basisFv->numColumns(), false); // TODO: distributed
        CAROM::Matrix sampledSpatialBasisE(numSamplesE, basisFe->numColumns(), false); // TODO: distributed

        CAROM::SpaceTimeSampling(basisFv, tbasisFv, rdimfv, t_samples_V, sample_dofs_V.data(),
                                 num_sample_dofs_per_procV.data(), sampledSpatialBasisV, rank, nprocs,
                                 numTimeSamplesV, numSamplesV, excludeFinalTimeSample);

        CAROM::SpaceTimeSampling(basisFe, tbasisFe, rdimfe, t_samples_E, sample_dofs_E.data(),
                                 num_sample_dofs_per_procE.data(), sampledSpatialBasisE, rank, nprocs,
                                 numTimeSamplesE, numSamplesE, excludeFinalTimeSample);

        CAROM::Matrix *sampledSpatialBasisX = NULL;
        if (spaceTimeMethod == gnat_lspg)
        {
            sampledSpatialBasisX = new CAROM::Matrix(numSamplesV, basisV->numColumns(), false); // TODO: distributed
            std::vector<int> t_samples_X(numTimeSamplesV);
            sample_dofs_X.resize(numSamplesV);
            CAROM::SpaceTimeSampling(basisV, tbasisV, rdimv, t_samples_X, sample_dofs_X.data(),
                                     num_sample_dofs_per_procX.data(), *sampledSpatialBasisX, rank, nprocs,
                                     numTimeSamplesV, numSamplesV, excludeFinalTimeSample);

            for (int i=0; i<nprocs; ++i)
                num_sample_dofs_per_procX[i] = 0;
        }

        // Shift Fe time samples by VTos
        // TODO: should the shift be in t_samples_E or just mergedTimeSamples?
        for (int i=0; i<numTimeSamplesE; ++i)
            t_samples_E[i] = t_samples_E[i] + VTos;

        // Merge time samples for Fv and Fe
        std::set<int> mergedTimeSamples;
        for (int i=0; i<numTimeSamplesV; ++i)
            mergedTimeSamples.insert(t_samples_V[i]);

        for (int i=0; i<numTimeSamplesE; ++i)
            mergedTimeSamples.insert(t_samples_E[i]);

        timeSamples.resize(mergedTimeSamples.size());
        int cnt = 0;
        for (auto s : mergedTimeSamples)
        {
            mfem::out << rank << ": Time sample " << cnt << ": " << s << '\n';
            timeSamples[cnt++] = s;
        }

        {
            // Write timeSamples to file
            std::string filename = basename + "/timeSamples.csv";
            std::ofstream outfile(filename);
            for (int i=0; i<mergedTimeSamples.size(); ++i)
                outfile << timeSamples[i] << "\n";

            outfile.close();
        }

        BsinvV = new CAROM::Matrix(timeSamples.size() * numSamplesV, rdimfv, false);
        BsinvE = new CAROM::Matrix(timeSamples.size() * numSamplesE, rdimfe, false);

        GetSampledSpaceTimeBasis(timeSamples, tbasisFv, sampledSpatialBasisV, *BsinvV);
        GetSampledSpaceTimeBasis(timeSamples, tbasisFe, sampledSpatialBasisE, *BsinvE);

        if (spaceTimeMethod == gnat_lspg)
        {
            // Use V basis for hyperreduction of the RHS of the equation of motion dX/dt = V, although it is linear.
            // Also use V samples, which are actually for Fv.
            // TODO: can fewer samples be used here, or does it matter (would it improve speed)?
            BsinvX = new CAROM::Matrix(timeSamples.size() * numSamplesV, rdimv, false);
            GetSampledSpaceTimeBasis(timeSamples, tbasisV, *sampledSpatialBasisX, *BsinvX);
            delete sampledSpatialBasisX;
        }

        // TODO: BsinvV and BsinvE are already set by CAROM::SpaceTimeSampling for their own time samples, which is useless
        // after merging time samples. Now we have to construct them for the merged time samples, so the initial computation
        // is wasted and should be disabled by an input flag to CAROM::SpaceTimeSampling.

        MFEM_VERIFY(timesteps != NULL && timesteps->size() == temporalSize-1, "");
        {
            std::vector<double> RK4scaling(temporalSize-1);
            for (int i=0; i<temporalSize-1; ++i)
            {
                const double ti = (i == 0) ? t_initial : (*timesteps)[i-1];
                const double h = (*timesteps)[i] - ti;
                RK4scaling[i] = h * (1.0 + (0.5 * h) + (h * h / 6.0) + (h * h * h / 24.0));
            }

            if (!(spaceTimeMethod == gnat_lspg || spaceTimeMethod == coll_lspg))
            {
                // TODO: remove these SpaceTimeProduct calls?
                // TODO: set arguments based on whether VTos == 1
#ifdef STXV
                PiXtransPiV = SpaceTimeProduct(basisV, tbasisV, basisV, tbasisV, &RK4scaling, true, true, true);
#else
                PiXtransPiV = SpaceTimeProduct(basisX, tbasisX, basisV, tbasisV, &RK4scaling, false, true, true);
#endif
            }
        }

        if (!(spaceTimeMethod == gnat_lspg || spaceTimeMethod == coll_lspg))
        {
            // TODO: remove these SpaceTimeProduct calls?
            // TODO: set arguments based on whether VTos == 1
#ifdef STXV
            PiXtransPiX = SpaceTimeProduct(basisV, tbasisV, basisX, tbasisX, NULL, true, false, false, true);
            PiXtransPiXlag = SpaceTimeProduct(basisV, tbasisV, basisX, tbasisX, NULL, true, false, true, false);
#else
            PiXtransPiX = SpaceTimeProduct(basisX, tbasisX, basisX, tbasisX, NULL, false, false, false, true);
            PiXtransPiXlag = SpaceTimeProduct(basisX, tbasisX, basisX, tbasisX, NULL, false, false, true, false);
#endif

            PiVtransPiFv = SpaceTimeProduct(basisV, tbasisV, basisFv, tbasisFv, NULL, true, false);
            PiEtransPiFe = SpaceTimeProduct(basisE, tbasisE, basisFe, tbasisFe, NULL, false, true);

            MFEM_VERIFY(PiVtransPiFv->numRows() == rdimv && PiVtransPiFv->numColumns() == rdimfv, "");
            MFEM_VERIFY(PiEtransPiFe->numRows() == rdime && PiEtransPiFe->numColumns() == rdimfe, "");
        }
    }
    else // not spaceTime
    {
        if (use_qdeim)
        {
            CAROM::QDEIM(basisFv,
                         rdimfv,
                         sample_dofs_V.data(),
                         num_sample_dofs_per_procV.data(),
                         *BsinvV,
                         rank,
                         nprocs,
                         numSamplesV);

            CAROM::QDEIM(basisFe,
                         rdimfe,
                         sample_dofs_E.data(),
                         num_sample_dofs_per_procE.data(),
                         *BsinvE,
                         rank,
                         nprocs,
                         numSamplesE);
        }
        else
        {
            CAROM::GNAT(basisFv,
                        rdimfv,
                        sample_dofs_V.data(),
                        num_sample_dofs_per_procV.data(),
                        *BsinvV,
                        rank,
                        nprocs,
                        numSamplesV);

            CAROM::GNAT(basisFe,
                        rdimfe,
                        sample_dofs_E.data(),
                        num_sample_dofs_per_procE.data(),
                        *BsinvE,
                        rank,
                        nprocs,
                        numSamplesE);
        }
    }

    // We assume that the same H1 fespace is used for X and V, and a different L2 fespace is used for E.
    // We merge all sample DOF's for X, V, and E into one set for each process.
    // The pair of spaces (H1, L2) is used here.

    vector<int> sample_dofs_merged;
    vector<int> num_sample_dofs_per_proc_merged(nprocs);
    int os_merged = 0;
    for (int p=0; p<nprocs; ++p)
    {
        std::set<int> sample_dofs_H1, sample_dofs_L2;
        {
            int os = 0;
            for (int q=0; q<p; ++q)
            {
                os += num_sample_dofs_per_procX[q];
            }

            for (int j=0; j<num_sample_dofs_per_procX[p]; ++j)
            {
                sample_dofs_H1.insert(sample_dofs_X[os + j]);
            }

            os = 0;
            for (int q=0; q<p; ++q)
            {
                os += num_sample_dofs_per_procV[q];
            }

            for (int j=0; j<num_sample_dofs_per_procV[p]; ++j)
            {
                sample_dofs_H1.insert(sample_dofs_V[os + j]);
            }

            os = 0;
            for (int q=0; q<p; ++q)
            {
                os += num_sample_dofs_per_procE[q];
            }

            for (int j=0; j<num_sample_dofs_per_procE[p]; ++j)
            {
                sample_dofs_L2.insert(sample_dofs_E[os + j]);
            }
        }

        num_sample_dofs_per_proc_merged[p] = sample_dofs_H1.size() + sample_dofs_L2.size();

        for (std::set<int>::const_iterator it = sample_dofs_H1.begin(); it != sample_dofs_H1.end(); ++it)
        {
            sample_dofs_merged.push_back((*it));
        }

        for (std::set<int>::const_iterator it = sample_dofs_L2.begin(); it != sample_dofs_L2.end(); ++it)
        {
            sample_dofs_merged.push_back(nH1[p] + (*it));  // offset by nH1[p] for the mixed spaces (H1, L2)
        }

        // For each of the num_sample_dofs_per_procX[p] samples, set s2sp_X[] to be its index in sample_dofs_merged.
        {
            int os = 0;
            for (int q=0; q<p; ++q)
                os += num_sample_dofs_per_procX[q];

            s2sp_X.resize(numSamplesX);

            for (int j=0; j<num_sample_dofs_per_procX[p]; ++j)
            {
                const int sample = sample_dofs_X[os + j];

                // Note: this has quadratic complexity and could be improved with a std::map<int, int>, but it should not be a bottleneck.
                int k = -1;
                int cnt = 0;
                for (std::set<int>::const_iterator it = sample_dofs_H1.begin(); it != sample_dofs_H1.end(); ++it, ++cnt)
                {
                    if (*it == sample)
                    {
                        MFEM_VERIFY(k == -1, "");
                        k = cnt;
                    }
                }

                MFEM_VERIFY(k >= 0, "");
                s2sp_X[os + j] = os_merged + k;
            }
        }

        // For each of the num_sample_dofs_per_procV[p] samples, set s2sp_V[] to be its index in sample_dofs_merged.
        {
            int os = 0;
            for (int q=0; q<p; ++q)
                os += num_sample_dofs_per_procV[q];

            s2sp_V.resize(numSamplesV);

            for (int j=0; j<num_sample_dofs_per_procV[p]; ++j)
            {
                const int sample = sample_dofs_V[os + j];

                // Note: this has quadratic complexity and could be improved with a std::map<int, int>, but it should not be a bottleneck.
                int k = -1;
                int cnt = 0;
                for (std::set<int>::const_iterator it = sample_dofs_H1.begin(); it != sample_dofs_H1.end(); ++it, ++cnt)
                {
                    if (*it == sample)
                    {
                        MFEM_VERIFY(k == -1, "");
                        k = cnt;
                    }
                }

                MFEM_VERIFY(k >= 0, "");
                s2sp_V[os + j] = os_merged + k;
            }
        }

        // For each of the num_sample_dofs_per_procE[p] samples, set s2sp_E[] to be its index in sample_dofs_merged.
        {
            int os = 0;
            for (int q=0; q<p; ++q)
                os += num_sample_dofs_per_procE[q];

            s2sp_E.resize(numSamplesE);

            for (int j=0; j<num_sample_dofs_per_procE[p]; ++j)
            {
                const int sample = sample_dofs_E[os + j];

                // Note: this has quadratic complexity and could be improved with a std::map<int, int>, but it should not be a bottleneck.
                int k = -1;
                int cnt = 0;
                for (std::set<int>::const_iterator it = sample_dofs_L2.begin(); it != sample_dofs_L2.end(); ++it, ++cnt)
                {
                    if (*it == sample)
                    {
                        MFEM_VERIFY(k == -1, "");
                        k = cnt;
                    }
                }

                MFEM_VERIFY(k >= 0, "");
                s2sp_E[os + j] = os_merged + sample_dofs_H1.size() + k;
            }
        }

        os_merged += num_sample_dofs_per_proc_merged[p];
    }  // loop over p

    // Define a superfluous finite element space, merely to get global vertex indices for the sample mesh construction.
    const int dim = pmesh->Dimension();
    H1_FECollection h1_coll(1, dim);  // Must be first order, to get a bijection between vertices and DOF's.
    ParFiniteElementSpace H1_space(pmesh, &h1_coll);  // This constructor effectively sets vertex (DOF) global indices.

    ParFiniteElementSpace *sp_H1_space = NULL;
    ParFiniteElementSpace *sp_L2_space = NULL;

    // Construct sample mesh

    // This creates sample_pmesh, sp_H1_space, and sp_L2_space only on rank 0.
    CAROM::CreateSampleMesh(*pmesh, H1_space, *H1FESpace, *L2FESpace, *(H1FESpace->FEColl()),
                            *(L2FESpace->FEColl()), rom_com, sample_dofs_merged,
                            num_sample_dofs_per_proc_merged, sample_pmesh, sprows, all_sprows, s2sp, st2sp, sp_H1_space, sp_L2_space);

    if (rank == 0)
    {
        sample_pmesh->ReorientTetMesh();  // re-orient the mesh, required for tets, no-op for hex
        //SetBdryAttrForVelocity(sample_pmesh);
        SetBdryAttrForVelocity_Cartesian(sample_pmesh);
        sample_pmesh->EnsureNodes();

        const bool printSampleMesh = true;
        if (printSampleMesh)
        {
            ostringstream mesh_name;
            mesh_name << basename + "/smesh." << setfill('0') << setw(6) << rank;

            ofstream mesh_ofs(mesh_name.str().c_str());
            mesh_ofs.precision(8);
            sample_pmesh->Print(mesh_ofs);
        }
    }

    // Set s2sp_H1 and s2sp_L2 from s2sp

    const int NH1sp = (rank == 0) ? sp_H1_space->GetTrueVSize() : 0;

    if (rank == 0)
    {
        int offset = 0;
        for (int p=0; p<nprocs; ++p)
        {
            for (int i=0; i<num_sample_dofs_per_proc_merged[p]; ++i)
            {
                if (sample_dofs_merged[offset + i] >= nH1[p])
                    s2sp_L2.push_back(s2sp[offset + i] - NH1sp);
                else
                    s2sp_H1.push_back(s2sp[offset + i]);
            }

            offset += num_sample_dofs_per_proc_merged[p];
        }

        MFEM_VERIFY(s2sp.size() == offset, "");

        size_H1_sp = sp_H1_space->GetTrueVSize();
        size_L2_sp = sp_L2_space->GetTrueVSize();

        // Define the map s2sp_X from X samples to sample mesh X dofs.
        {
            int os_p = 0;
            for (int p=0; p<nprocs; ++p)
            {
                for (int j=0; j<num_sample_dofs_per_procX[p]; ++j)
                {
                    MFEM_VERIFY(sample_dofs_merged[s2sp_X[os_p + j]] < nH1[p], "");
                    const int spId = s2sp[s2sp_X[os_p + j]];
                    s2sp_X[os_p + j] = spId;
                }

                os_p += num_sample_dofs_per_procX[p];
            }

            MFEM_VERIFY(os_p == numSamplesX, "");
        }

        // Define the map s2sp_V from V samples to sample mesh V dofs.
        {
            int os_p = 0;
            for (int p=0; p<nprocs; ++p)
            {
                for (int j=0; j<num_sample_dofs_per_procV[p]; ++j)
                {
                    MFEM_VERIFY(sample_dofs_merged[s2sp_V[os_p + j]] < nH1[p], "");
                    const int spId = s2sp[s2sp_V[os_p + j]];
                    s2sp_V[os_p + j] = spId;
                }

                os_p += num_sample_dofs_per_procV[p];
            }

            MFEM_VERIFY(os_p == numSamplesV, "");
        }

        // Define the map s2sp_E from E samples to sample mesh E dofs.
        {
            int os_p = 0;
            for (int p=0; p<nprocs; ++p)
            {
                for (int j=0; j<num_sample_dofs_per_procE[p]; ++j)
                {
                    MFEM_VERIFY(sample_dofs_merged[s2sp_E[os_p + j]] >= nH1[p], "");
                    const int spId = s2sp[s2sp_E[os_p + j]];
                    s2sp_E[os_p + j] = spId - NH1sp;
                }

                os_p += num_sample_dofs_per_procE[p];
            }

            MFEM_VERIFY(os_p == numSamplesE, "");
        }

        BXsp = new CAROM::Matrix(size_H1_sp, rdimx, false);
        BVsp = new CAROM::Matrix(size_H1_sp, rdimv, false);
        BEsp = new CAROM::Matrix(size_L2_sp, rdime, false);

        spX = new CAROM::Vector(size_H1_sp, false);
        spV = new CAROM::Vector(size_H1_sp, false);
        spE = new CAROM::Vector(size_L2_sp, false);

        sX = numSamplesX == 0 ? NULL : new CAROM::Vector(numSamplesX, false);
        sV = new CAROM::Vector(numSamplesV, false);
        sE = new CAROM::Vector(numSamplesE, false);

        BFvsp = new CAROM::Matrix(size_H1_sp, rdimfv, false);
        BFesp = new CAROM::Matrix(size_L2_sp, rdimfe, false);
    }  // if (rank == 0)

    // This gathers only to rank 0.
#ifdef FULL_DOF_STENCIL
    const int NR = H1FESpace->GetVSize();
    GatherDistributedMatrixRows(*basisX, *basisE, rdimx, rdime, NR, *H1FESpace, *L2FESpace, st2sp, sprows, all_sprows, *BXsp, *BEsp);
    // TODO: this redundantly gathers BEsp again, but only once per simulation.
    GatherDistributedMatrixRows(*basisV, *basisE, rdimv, rdime, NR, *H1FESpace, *L2FESpace, st2sp, sprows, all_sprows, *BVsp, *BEsp);

    GatherDistributedMatrixRows(*basisFv, *basisFe, rdimfv, rdimfe, NR, *H1FESpace, *L2FESpace, st2sp, sprows, all_sprows, *BFvsp, *BFesp);
#else
    GatherDistributedMatrixRows(*basisX, *basisE, rdimx, rdime, st2sp, sprows, all_sprows, *BXsp, *BEsp);
    // TODO: this redundantly gathers BEsp again, but only once per simulation.
    GatherDistributedMatrixRows(*basisV, *basisE, rdimv, rdime, st2sp, sprows, all_sprows, *BVsp, *BEsp);
#endif

    delete sp_H1_space;
    delete sp_L2_space;
}

void ROM_Basis::ComputeReducedMatrices(bool sns1)
{
    if (!spaceTime && rank == 0)
    {
        // Compute reduced matrix BsinvX = BXsp^T BVsp
        BsinvX = BXsp->transposeMult(BVsp);

        if (offsetInit)
        {
            BX0 = BXsp->transposeMult(initVsp);
            MFEM_VERIFY(BX0->dim() == rdimx, "");
        }

        if (!sns1)
        {
            // Compute reduced matrix BsinvV = (BVsp^T BFvsp BsinvV^T)^T = BsinvV BFvsp^T BVsp
            CAROM::Matrix *prod1 = BFvsp->transposeMult(BVsp);
            CAROM::Matrix *prod2 = BsinvV->mult(prod1);

            delete prod1;
            delete BsinvV;

            BsinvV = prod2;

            // Compute reduced matrix BsinvE = (BEsp^T BFesp BsinvE^T)^T = BsinvE BFesp^T BEsp
            prod1 = BFesp->transposeMult(BEsp);
            prod2 = BsinvE->mult(prod1);

            delete prod1;
            delete BsinvE;

            BsinvE = prod2;
        }

        if (RK2AvgFormulation)
        {
            const CAROM::Matrix *prodX = BXsp->transposeMult(BXsp);
            BXXinv = prodX->inverse();
            delete prodX;

            const CAROM::Matrix *prodV = BVsp->transposeMult(BVsp);
            BVVinv = prodV->inverse();
            delete prodV;

            const CAROM::Matrix *prodE = BEsp->transposeMult(BEsp);
            BEEinv = prodE->inverse();
            delete prodE;
        }
    }
}

void ROM_Basis::ApplyEssentialBCtoInitXsp(Array<int> const& ess_tdofs)
{
    if (rank != 0 || !offsetInit)
        return;

    for (int i=0; i<ess_tdofs.Size(); ++i)
    {
        (*initXsp)(ess_tdofs[i]) = 0.0;
    }
}

int ROM_Basis::SolutionSize() const
{
    return rdimx + rdimv + rdime;
}

int ROM_Basis::SolutionSizeSP() const
{
    return (2*size_H1_sp) + size_L2_sp;
}

int ROM_Basis::SolutionSizeFOM() const
{
    return (2*H1size) + L2size;  // full size, not true DOF size
}

void ROM_Basis::ReadSolutionBases(const int window)
{
    if (!useVX)
        basisV = ReadBasisROM(rank, basename + "/" + ROMBasisName::V + std::to_string(window) + basisIdentifier, tH1size, 0, rdimv);

    basisE = ReadBasisROM(rank, basename + "/" + ROMBasisName::E + std::to_string(window) + basisIdentifier, tL2size, 0, rdime);

    if (useXV)
        basisX = basisV;
    else
        basisX = ReadBasisROM(rank, basename + "/" + ROMBasisName::X + std::to_string(window) + basisIdentifier, tH1size, 0, rdimx);

    if (useVX)
        basisV = basisX;

    if (mergeXV)
    {
        const int max_model_dim = basisX->numColumns() + basisV->numColumns();
        CAROM::Options static_x_options(tH1size, max_model_dim, 1);
        CAROM::BasisGenerator generator_XV(static_x_options, false);

        Vector Bej(basisX->numRows());  // owns its data
        MFEM_VERIFY(Bej.Size() == tH1size, "");

        CAROM::Vector ej(basisX->numColumns(), false);
        CAROM::Vector CBej(Bej.GetData(), basisX->numRows(), true, false);  // data owned by Bej
        ej = 0.0;
        for (int j=0; j<basisX->numColumns(); ++j)
        {
            ej(j) = 1.0;
            basisX->mult(ej, CBej);

            const bool addSample = generator_XV.takeSample(Bej.GetData(), 0.0, 1.0);  // artificial time and timestep
            MFEM_VERIFY(addSample, "Sample not added");

            ej(j) = 0.0;
        }

        ej.setSize(basisV->numColumns());
        ej = 0.0;

        for (int j=0; j<basisV->numColumns(); ++j)
        {
            ej(j) = 1.0;
            basisV->mult(ej, CBej);

            const bool addSample = generator_XV.takeSample(Bej.GetData(), 0.0, 1.0);  // artificial time and timestep
            MFEM_VERIFY(addSample, "Sample not added");

            ej(j) = 0.0;
        }

        BasisGeneratorFinalSummary(&generator_XV, energyFraction_X, rdimx, "", false);
        rdimv = rdimx;

        cout << rank << ": ROM_Basis used energy fraction " << energyFraction_X
             << " and merged X-X0 and V bases with resulting dimension " << rdimx << endl;

        delete basisX;
        delete basisV;

        const CAROM::Matrix* basisX_full = generator_XV.getSpatialBasis();

        // Make a deep copy first rdimx columns of basisX_full, which is inefficient.
        basisX = GetFirstColumns(rdimx, basisX_full, 0, tH1size);
        MFEM_VERIFY(basisX->numRows() == tH1size, "");
        basisV = basisX;
    }

    if (use_sns)
    {
        basisFv = MultBasisROM(rank, basename + "/" + ROMBasisName::V + std::to_string(window) + basisIdentifier, tH1size, 0, rdimfv, lhoper, 1);
        basisFe = MultBasisROM(rank, basename + "/" + ROMBasisName::E + std::to_string(window) + basisIdentifier, tL2size, 0, rdimfe, lhoper, 2);
    }
    else
    {
        basisFv = ReadBasisROM(rank, basename + "/" + ROMBasisName::Fv + std::to_string(window) + basisIdentifier, tH1size, 0, rdimfv);
        basisFe = ReadBasisROM(rank, basename + "/" + ROMBasisName::Fe + std::to_string(window) + basisIdentifier, tL2size, 0, rdimfe);
    }
}

void ROM_Basis::ReadTemporalBases(const int window)
{
    MFEM_VERIFY(!useXV && !useVX && !mergeXV, "Case not implemented");

    int tsize = 0;
    tbasisX = ReadTemporalBasisROM(rank, basename + "/" + ROMBasisName::X + std::to_string(window), tsize, rdimx);
    temporalSize = tsize;
    MFEM_VERIFY(temporalSize > 0, "");

    tbasisV = ReadTemporalBasisROM(rank, basename + "/" + ROMBasisName::V + std::to_string(window), tsize, rdimv);
    MFEM_VERIFY(temporalSize == tsize + VTos, "");
    tbasisE = ReadTemporalBasisROM(rank, basename + "/" + ROMBasisName::E + std::to_string(window), tsize, rdime);
    MFEM_VERIFY(temporalSize == tsize, "");

    // TODO: SNS
    tbasisFv = ReadTemporalBasisROM(rank, basename + "/" + ROMBasisName::Fv + std::to_string(window), tsize, rdimfv);
    MFEM_VERIFY(temporalSize == tsize, "");

    tbasisFe = ReadTemporalBasisROM(rank, basename + "/" + ROMBasisName::Fe + std::to_string(window), tsize, rdimfe);
    MFEM_VERIFY(temporalSize == tsize + VTos, "");
}

// f is a full vector, not a true vector
void ROM_Basis::ProjectFOMtoROM(Vector const& f, Vector & r, const bool timeDerivative)
{
    MFEM_VERIFY(r.Size() == rdimx + rdimv + rdime, "");
    MFEM_VERIFY(f.Size() == (2*H1size) + L2size, "");

    const bool useOffset = offsetInit && (!timeDerivative);

    for (int i=0; i<H1size; ++i)
        (*gfH1)(i) = f[i];

    gfH1->GetTrueDofs(mfH1);

    for (int i=0; i<tH1size; ++i)
        (*fH1)(i) = useOffset ? mfH1[i] - (*initX)(i) : mfH1[i];

    basisX->transposeMult(*fH1, *rX);

    for (int i=0; i<H1size; ++i)
        (*gfH1)(i) = f[H1size + i];

    gfH1->GetTrueDofs(mfH1);

    for (int i=0; i<tH1size; ++i)
        (*fH1)(i) = (useOffset && Voffset) ? mfH1[i] - (*initV)(i) : mfH1[i];

    basisV->transposeMult(*fH1, *rV);

    for (int i=0; i<L2size; ++i)
        (*gfL2)(i) = f[(2*H1size) + i];

    gfL2->GetTrueDofs(mfL2);

    for (int i=0; i<tL2size; ++i)
        (*fL2)(i) = useOffset ? mfL2[i] - (*initE)(i) : mfL2[i];

    basisE->transposeMult(*fL2, *rE);

    for (int i=0; i<rdimx; ++i)
        r[i] = (*rX)(i);

    for (int i=0; i<rdimv; ++i)
        r[rdimx + i] = (*rV)(i);

    for (int i=0; i<rdime; ++i)
        r[rdimx + rdimv + i] = (*rE)(i);
}

// f is a full vector, not a true vector
void ROM_Basis::LiftROMtoFOM(Vector const& r, Vector & f)
{
    MFEM_VERIFY(r.Size() == rdimx + rdimv + rdime, "");
    MFEM_VERIFY(f.Size() == (2*H1size) + L2size, "");

    for (int i=0; i<rdimx; ++i)
        (*rX)(i) = r[i];

    for (int i=0; i<rdimv; ++i)
        (*rV)(i) = r[rdimx + i];

    for (int i=0; i<rdime; ++i)
        (*rE)(i) = r[rdimx + rdimv + i];

    basisX->mult(*rX, *fH1);

    for (int i=0; i<tH1size; ++i)
        mfH1[i] = offsetInit ? (*initX)(i) + (*fH1)(i) : (*fH1)(i);

    gfH1->SetFromTrueDofs(mfH1);

    for (int i=0; i<H1size; ++i)
        f[i] = (*gfH1)(i);

    basisV->mult(*rV, *fH1);

    for (int i=0; i<tH1size; ++i)
        mfH1[i] = (offsetInit && Voffset) ? (*initV)(i) + (*fH1)(i) : (*fH1)(i);

    gfH1->SetFromTrueDofs(mfH1);

    for (int i=0; i<H1size; ++i)
        f[H1size + i] = (*gfH1)(i);

    basisE->mult(*rE, *fL2);

    for (int i=0; i<tL2size; ++i)
        mfL2[i] = offsetInit ? (*initE)(i) + (*fL2)(i) : (*fL2)(i);

    gfL2->SetFromTrueDofs(mfL2);

    for (int i=0; i<L2size; ++i)
        f[(2*H1size) + i] = (*gfL2)(i);
}

void ROM_Basis::LiftToSampleMesh(const Vector &u, Vector &usp) const
{
    MFEM_VERIFY(u.Size() == SolutionSize(), "");  // rdimx + rdimv + rdime
    MFEM_VERIFY(usp.Size() == SolutionSizeSP(), "");  // (2*size_H1_sp) + size_L2_sp

    if (rank == 0)
    {
        for (int i=0; i<rdimx; ++i)
            (*rX)(i) = u[i];

        for (int i=0; i<rdimv; ++i)
            (*rV)(i) = u[rdimx + i];

        for (int i=0; i<rdime; ++i)
            (*rE)(i) = u[rdimx + rdimv + i];

        BXsp->mult(*rX, *spX);
        BVsp->mult(*rV, *spV);
        BEsp->mult(*rE, *spE);

        for (int i=0; i<size_H1_sp; ++i)
        {
            usp[i] = offsetInit ? (*initXsp)(i) + (*spX)(i) : (*spX)(i);
            usp[size_H1_sp + i] = (offsetInit && Voffset) ? (*initVsp)(i) + (*spV)(i) : (*spV)(i);
        }

        for (int i=0; i<size_L2_sp; ++i)
        {
            //usp[(2*size_H1_sp) + i] = std::max((*spE)(i), 0.0);
            usp[(2*size_H1_sp) + i] = offsetInit ? (*initEsp)(i) + (*spE)(i) : (*spE)(i);
        }
    }
}

void ROM_Basis::SampleMeshAddInitialState(Vector &usp) const
{
    MFEM_VERIFY(usp.Size() == SolutionSizeSP(), "");  // (2*size_H1_sp) + size_L2_sp

    if (rank == 0)
    {
        for (int i=0; i<size_H1_sp; ++i)
        {
            usp[i] += (*initXsp)(i);
            usp[size_H1_sp + i] += (*initVsp)(i);
        }

        for (int i=0; i<size_L2_sp; ++i)
        {
            usp[(2*size_H1_sp) + i] += (*initEsp)(i);
        }
    }
}

void ROM_Basis::RestrictFromSampleMesh(const Vector &usp, Vector &u, const bool timeDerivative,
                                       const bool rhs_without_mass_matrix, const DenseMatrix *invMvROM,
                                       const DenseMatrix *invMeROM) const
{
    MFEM_VERIFY(u.Size() == SolutionSize(), "");  // rdimx + rdimv + rdime
    MFEM_VERIFY(usp.Size() == SolutionSizeSP(), "");  // (2*size_H1_sp) + size_L2_sp

    if (RK2AvgFormulation)
    {
        ProjectFromSampleMesh(usp, u, timeDerivative);
        return;
    }

    const bool useOffset = offsetInit && (!timeDerivative);

    // Select entries out of usp on the sample mesh.

    // Note that s2sp_X maps from X samples to sample mesh H1 dofs, and similarly for V and E.

    for (int i=0; i<numSamplesX; ++i)
        (*sX)(i) = useOffset ? usp[s2sp_X[i]] - (*initXsp)(s2sp_X[i]) : usp[s2sp_X[i]];

    for (int i=0; i<numSamplesV; ++i)
        (*sV)(i) = (useOffset && Voffset) ? usp[size_H1_sp + s2sp_V[i]] - (*initVsp)(s2sp_V[i]) : usp[size_H1_sp + s2sp_V[i]];

    for (int i=0; i<numSamplesE; ++i)
        (*sE)(i) = useOffset ? usp[(2*size_H1_sp) + s2sp_E[i]] - (*initEsp)(s2sp_E[i]) : usp[(2*size_H1_sp) + s2sp_E[i]];

    // ROM operation on source: map sample mesh evaluation to reduced coefficients with respect to solution bases
    BsinvV->transposeMult(*sV, *rV);
    BsinvE->transposeMult(*sE, *rE);

    if (rhs_without_mass_matrix && timeDerivative)
    {
        Vector rhs(rdimv);
        Vector invMrhs(rdimv);
        for (int i=0; i<rdimv; ++i)
            rhs[i] = (*rV)(i);

        // TODO: multiply BsinvV by invMvROM and store that rather than doing 2 mults, in the case of no Gram-Schmidt (will that version be maintained?).
        invMvROM->Mult(rhs, invMrhs);
        for (int i=0; i<rdimv; ++i)
            u[rdimx + i] = invMrhs[i];
    }
    else
    {
        for (int i=0; i<rdimv; ++i)
            u[rdimx + i] = (*rV)(i);
    }

    if (rhs_without_mass_matrix && timeDerivative)
    {
        Vector rhs(rdime);
        Vector invMrhs(rdime);
        for (int i=0; i<rdime; ++i)
            rhs[i] = (*rE)(i);

        invMeROM->Mult(rhs, invMrhs);
        for (int i=0; i<rdime; ++i)
            u[rdimx + rdimv + i] = invMrhs[i];
    }
    else
    {
        for (int i=0; i<rdime; ++i)
            u[rdimx + rdimv + i] = (*rE)(i);
    }
}

void ROM_Basis::RestrictFromSampleMesh_V(const Vector &usp, Vector &u) const
{
    MFEM_VERIFY(u.Size() == rdimv, "");
    MFEM_VERIFY(usp.Size() == size_H1_sp, "");

    for (int i=0; i<size_H1_sp; ++i)
        (*spV)(i) = usp[i];

    BVsp->transposeMult(*spV, *rV);

    for (int i=0; i<rdimv; ++i)
        u[i] = (*rV)(i);
}

void ROM_Basis::RestrictFromSampleMesh_E(const Vector &usp, Vector &u) const
{
    MFEM_VERIFY(u.Size() == rdime, "");
    MFEM_VERIFY(usp.Size() == size_L2_sp, "");

    for (int i=0; i<size_L2_sp; ++i)
        (*spE)(i) = usp[i];

    BEsp->transposeMult(*spE, *rE);

    for (int i=0; i<rdime; ++i)
        u[i] = (*rE)(i);
}

void ROM_Basis::ProjectFromSampleMesh(const Vector &usp, Vector &u,
                                      const bool timeDerivative) const
{
    MFEM_VERIFY(u.Size() == SolutionSize(), "");
    MFEM_VERIFY(usp.Size() == (2*size_H1_sp) + size_L2_sp, "");

    const bool useOffset = offsetInit && (!timeDerivative);

    int ossp = 0;
    int osrom = 0;

    // X
    for (int i=0; i<size_H1_sp; ++i)
        (*spX)(i) = useOffset ? usp[ossp + i] - (*initXsp)(i) : usp[ossp + i];

    BXsp->transposeMult(*spX, *rX);
    BXXinv->mult(*rX, *rX2);

    for (int i=0; i<rdimx; ++i)
        u[osrom + i] = (*rX2)(i);

    osrom += rdimx;
    ossp += size_H1_sp;

    // V
    for (int i=0; i<size_H1_sp; ++i)
        (*spV)(i) = (useOffset && Voffset) ? usp[ossp + i] - (*initVsp)(i) : usp[ossp + i];

    BVsp->transposeMult(*spV, *rV);
    BVVinv->mult(*rV, *rV2);

    for (int i=0; i<rdimv; ++i)
        u[osrom + i] = (*rV2)(i);

    osrom += rdimv;
    ossp += size_H1_sp;

    // E
    for (int i=0; i<size_L2_sp; ++i)
        (*spE)(i) = useOffset ? usp[ossp + i] - (*initEsp)(i) : usp[ossp + i];

    BEsp->transposeMult(*spE, *rE);
    BEEinv->mult(*rE, *rE2);

    for (int i=0; i<rdime; ++i)
        u[osrom + i] = (*rE2)(i);
}

ROM_Operator::ROM_Operator(ROM_Options const& input, ROM_Basis *b,
                           Coefficient& rho_coeff, FunctionCoefficient& mat_coeff,
                           const int order_e, const int source,
                           const bool visc, const bool vort, const double cfl,
                           const bool p_assembly, const double cg_tol, const int cg_max_iter,
                           const double ftz_tol, H1_FECollection *H1fec,
                           FiniteElementCollection *L2fec, std::vector<double> *timesteps)
    : TimeDependentOperator(b->SolutionSize()), operFOM(input.FOMoper), basis(b),
      rank(b->GetRank()), hyperreduce(input.hyperreduce), useGramSchmidt(input.GramSchmidt),
      spaceTimeMethod(input.spaceTimeMethod)
{
    if (hyperreduce && rank == 0)
    {
        const int spsize = basis->SolutionSizeSP();

        fx.SetSize(spsize);
        fy.SetSize(spsize);

        spmesh = b->GetSampleMesh();

        // The following code is copied from laghos.cpp to define a LagrangianHydroOperator on spmesh.

        L2FESpaceSP = new ParFiniteElementSpace(spmesh, L2fec);
        H1FESpaceSP = new ParFiniteElementSpace(spmesh, H1fec, spmesh->Dimension());

        xsp_gf = new ParGridFunction(H1FESpaceSP);
        spmesh->SetNodalGridFunction(xsp_gf);

        Vsize_l2sp = L2FESpaceSP->GetVSize();
        Vsize_h1sp = H1FESpaceSP->GetVSize();

        MFEM_VERIFY(((2*Vsize_h1sp) + Vsize_l2sp) == spsize, "");

        Array<int> ossp(4);
        ossp[0] = 0;
        ossp[1] = ossp[0] + Vsize_h1sp;
        ossp[2] = ossp[1] + Vsize_h1sp;
        ossp[3] = ossp[2] + Vsize_l2sp;
        BlockVector S(ossp);

        // On the sample mesh, we impose no essential DOF's. The reason is that it does not
        // make sense to set boundary conditions on the sample mesh boundary, which may lie
        // in the interior of the domain. Also, the boundary conditions on the domain
        // boundary are enforced due to the fact that BVsp is defined as a submatrix of
        // basisV, which has boundary conditions applied in the full-order discretization.

        //cout << "Sample mesh bdr att max " << spmesh->bdr_attributes.Max() << endl;

        // Boundary conditions: all tests use v.n = 0 on the boundary, and we assume
        // that the boundaries are straight.
        {
            Array<int> ess_bdr(spmesh->bdr_attributes.Max()), tdofs1d;
            for (int d = 0; d < spmesh->Dimension(); d++)
            {
                // Attributes 1/2/3 correspond to fixed-x/y/z boundaries, i.e., we must
                // enforce v_x/y/z = 0 for the velocity components.
                ess_bdr = 0;
                ess_bdr[d] = 1;
                H1FESpaceSP->GetEssentialTrueDofs(ess_bdr, tdofs1d, d);
                ess_tdofs.Append(tdofs1d);
            }
        }

        //basis->ApplyEssentialBCtoInitXsp(ess_tdofs);  // TODO: this should be only for v?

        ParGridFunction rho(L2FESpaceSP);
        L2_FECollection l2_fec(order_e, spmesh->Dimension());
        ParFiniteElementSpace l2_fes(spmesh, &l2_fec);
        ParGridFunction l2_rho(&l2_fes), l2_e(&l2_fes);
        l2_rho.ProjectCoefficient(rho_coeff);
        rho.ProjectGridFunction(l2_rho);

        // Piecewise constant ideal gas coefficient over the Lagrangian mesh. The
        // gamma values are projected on a function that stays constant on the moving
        // mesh.
        mat_fec = new L2_FECollection(0, spmesh->Dimension());
        mat_fes = new ParFiniteElementSpace(spmesh, mat_fec);
        mat_gf = new ParGridFunction(mat_fes);
        mat_gf->ProjectCoefficient(mat_coeff);
        mat_gf_coeff = new GridFunctionCoefficient(mat_gf);

        sns1 = (input.SNS && input.dimV == input.dimFv && input.dimE == input.dimFe); // SNS type I
        noMsolve = (useReducedM || useGramSchmidt || sns1);

        operSP = new hydrodynamics::LagrangianHydroOperator(S.Size(), *H1FESpaceSP, *L2FESpaceSP,
                ess_tdofs, rho, source, cfl, mat_gf_coeff,
                visc, vort, p_assembly, cg_tol, cg_max_iter, ftz_tol,
                H1fec->GetBasisType(), noMsolve, noMsolve);

        if (input.spaceTimeMethod != no_space_time)
        {
            ST_ode_solver = new RK4Solver; // TODO: allow for other types of solvers
            ST_ode_solver->Init(*operSP);

            STbasis = new STROM_Basis(input, b, timesteps);
            Sr.SetSize(STbasis->GetTotalNumSamples());
        }
    }
    else if (!hyperreduce)
    {
        MFEM_VERIFY(input.FOMoper->Height() == input.FOMoper->Width(), "");
        fx.SetSize(input.FOMoper->Height());
        fy.SetSize(input.FOMoper->Height());
    }

    if (useReducedM)
    {
        ComputeReducedMv();
        ComputeReducedMe();
    }
}

void ROM_Basis::GetBasisVectorV(const bool sp, const int id, Vector &v) const
{
    if (sp)  // Sample mesh version
    {
        MFEM_VERIFY(v.Size() == size_H1_sp, "");

        for (int i=0; i<size_H1_sp; ++i)
            v[i] = (*BVsp)(i,id);
    }
    else  // FOM version
    {
        MFEM_VERIFY(v.Size() == tH1size, "");
        CAROM::Vector ej(rdimv, false);
        CAROM::Vector CBej(tH1size, true);
        ej = 0.0;
        ej(id) = 1.0;
        basisV->mult(ej, CBej);
        for (int i=0; i<tH1size; ++i)
            v[i] = CBej(i);
    }
}

void ROM_Basis::GetBasisVectorE(const bool sp, const int id, Vector &v) const
{
    if (sp)  // Sample mesh version
    {
        MFEM_VERIFY(v.Size() == size_L2_sp, "");

        for (int i=0; i<size_L2_sp; ++i)
            v[i] = (*BEsp)(i,id);
    }
    else  // FOM version
    {
        MFEM_VERIFY(v.Size() == tL2size, "");
        CAROM::Vector ej(rdime, false);
        CAROM::Vector CBej(tL2size, true);
        ej = 0.0;
        ej(id) = 1.0;
        basisE->mult(ej, CBej);
        for (int i=0; i<tL2size; ++i)
            v[i] = CBej(i);
    }
}

void ROM_Basis::Set_dxdt_Reduced(const Vector &x, Vector &y) const
{
    for (int i=0; i<rdimv; ++i)
        (*rV)(i) = x[rdimx + i];

    BsinvX->mult(*rV, *rX);
    for (int i=0; i<rdimx; ++i)
        y[i] = offsetInit ? (*rX)(i) + (*BX0)(i) : (*rX)(i);
}

void ROM_Basis::HyperreduceRHS_V(Vector &v) const
{
    MFEM_VERIFY(useGramSchmidt, "apply reduced mass matrix inverse");
    MFEM_VERIFY(v.Size() == size_H1_sp, "");

    for (int i=0; i<numSamplesV; ++i)
        (*sV)(i) = v[s2sp_V[i]];

    BsinvV->transposeMult(*sV, *rV);

    // Lift from rV to v
    // Note that here there is the product BVsp BVsp^T, which cannot be simplified and should not be stored.
    BVsp->mult(*rV, *spV);
    for (int i=0; i<size_H1_sp; ++i)
        v[i] = (*spV)(i);
}

void ROM_Basis::HyperreduceRHS_E(Vector &e) const
{
    MFEM_VERIFY(useGramSchmidt, "apply reduced mass matrix inverse");
    MFEM_VERIFY(e.Size() == size_L2_sp, "");

    for (int i=0; i<numSamplesE; ++i)
        (*sE)(i) = e[s2sp_E[i]];

    BsinvE->transposeMult(*sE, *rE);

    // Lift from rE to e
    // Note that here there is the product BEsp BEsp^T, which cannot be simplified and should not be stored.
    BEsp->mult(*rE, *spE);
    for (int i=0; i<size_L2_sp; ++i)
        e[i] = (*spE)(i);
}

// t is a time sample, ranging from 0 to temporalSize-1.
void ROM_Basis::ScaleByTemporalBasis(const int t, Vector const& u, Vector &ut)
{
    MFEM_VERIFY(u.Size() == SolutionSize() && ut.Size() == u.Size(), "");
    MFEM_VERIFY(tbasisX->numColumns() == rdimx && basisX->numColumns() == rdimx, "");
    MFEM_VERIFY(tbasisV->numColumns() == rdimv && basisV->numColumns() == rdimv, "");
    MFEM_VERIFY(tbasisE->numColumns() == rdime && basisE->numColumns() == rdime, "");

    MFEM_VERIFY(tbasisX->numRows() == temporalSize, "");  // TODO: remove?
    MFEM_VERIFY(tbasisV->numRows() + VTos == temporalSize, "");  // TODO: remove?
    MFEM_VERIFY(tbasisE->numRows() == temporalSize, "");  // TODO: remove?

    MFEM_VERIFY(0 <= t && t < temporalSize, "");

    for (int i=0; i<rdimx; ++i)
        ut[i] = tbasisX->item(t, i) * u[i];

    int os = rdimx;
    for (int i=0; i<rdimv; ++i)
        ut[os + i] = (t == 0) ? 0.0 : tbasisV->item(t - VTos, i) * u[os + i];  // Assuming v=0 at t=0, which is not sampled.

    os += rdimv;
    for (int i=0; i<rdime; ++i)
        ut[os + i] = tbasisE->item(t, i) * u[os + i];
}

void ROM_Basis::computeWindowProjection(const ROM_Basis& basisPrev, ROM_Options const& input, const int window)
{
    BwinX = basisX->transposeMult(basisPrev.basisX);
    BwinV = basisV->transposeMult(basisPrev.basisV);
    BwinE = basisE->transposeMult(basisPrev.basisE);

    if (offsetInit && (input.offsetType == interpolateOffset || input.offsetType == saveLoadOffset))
    {
        CAROM::Vector dX(tH1size, true);
        CAROM::Vector dE(tL2size, true);

        CAROM::Vector Btd(basisX->numColumns(), false);

        basisPrev.initX->minus(*initX, dX);  // dX = basisPrev.initX - initX
        basisX->transposeMult(dX, Btd);
        Btd.write(basename + "/" + "BtInitDiffX" + "_" + to_string(window));

        Btd.setSize(basisV->numColumns());
        basisPrev.initV->minus(*initV, dX);  // dX = basisPrev.initV - initV
        basisV->transposeMult(dX, Btd);
        Btd.write(basename + "/" + "BtInitDiffV" + "_" + to_string(window));

        Btd.setSize(basisE->numColumns());
        basisPrev.initE->minus(*initE, dE);  // dE = basisPrev.initE - initE
        basisE->transposeMult(dE, Btd);
        Btd.write(basename + "/" + "BtInitDiffE" + "_" + to_string(window));
    }
}

void ROM_Basis::writeSP(ROM_Options const& input, const int window) const
{
    writeNum(numSamplesX, basename + "/" + "numSamplesX" + "_" + to_string(window));
    writeNum(numSamplesV, basename + "/" + "numSamplesV" + "_" + to_string(window));
    writeNum(numSamplesE, basename + "/" + "numSamplesE" + "_" + to_string(window));

    writeVec(s2sp_X, basename + "/" + "s2sp_X" + "_" + to_string(window));
    writeVec(s2sp_V, basename + "/" + "s2sp_V" + "_" + to_string(window));
    writeVec(s2sp_E, basename + "/" + "s2sp_E" + "_" + to_string(window));

    std::string outfile_string = basename + "/" + "sample_pmesh" + "_" + to_string(window);
    std::ofstream outfile_romS(outfile_string.c_str());
    sample_pmesh->ParPrint(outfile_romS);

    writeNum(size_H1_sp, basename + "/" + "size_H1_sp" + "_" + to_string(window));
    writeNum(size_L2_sp, basename + "/" + "size_L2_sp" + "_" + to_string(window));

    if (spaceTimeMethod == gnat_lspg) BsinvX->write(basename + "/" + "BsinvX" + "_" + to_string(window));
    BsinvV->write(basename + "/" + "BsinvV" + "_" + to_string(window));
    BsinvE->write(basename + "/" + "BsinvE" + "_" + to_string(window));
    BXsp->write(basename + "/" + "BXsp" + "_" + to_string(window));
    BVsp->write(basename + "/" + "BVsp" + "_" + to_string(window));
    BEsp->write(basename + "/" + "BEsp" + "_" + to_string(window));

    BFvsp->write(basename + "/" + "BFvsp" + "_" + to_string(window));
    BFesp->write(basename + "/" + "BFesp" + "_" + to_string(window));

    spX->write(basename + "/" + "spX" + "_" + to_string(window));
    spV->write(basename + "/" + "spV" + "_" + to_string(window));
    spE->write(basename + "/" + "spE" + "_" + to_string(window));
    if (offsetInit || spaceTime) // TODO: why is this necessary for spaceTime case? See SampleMeshAddInitialState
    {
        initXsp->write(basename + "/" + "initXsp" + "_" + to_string(window));
        initVsp->write(basename + "/" + "initVsp" + "_" + to_string(window));
        initEsp->write(basename + "/" + "initEsp" + "_" + to_string(window));
    }

    if (window > 0)
    {
        BwinX->write(basename + "/" + "BwinX" + "_" + to_string(window));
        BwinV->write(basename + "/" + "BwinV" + "_" + to_string(window));
        BwinE->write(basename + "/" + "BwinE" + "_" + to_string(window));
    }
}

void ROM_Basis::readSP(ROM_Options const& input, const int window)
{
    if (spaceTime)
    {
        // Read timeSamples from file
        std::string filename = basename + "/timeSamples.csv";
        std::ifstream infile(filename);

        MFEM_VERIFY(infile.is_open(), "Time sample file does not exist.");
        std::string line;
        std::vector<std::string> words;
        while (std::getline(infile, line))
        {
            split_line(line, words);
            timeSamples.push_back(std::stoi(words[0]));
        }

        infile.close();
    }

    readNum(numSamplesX, basename + "/" + "numSamplesX" + "_" + to_string(window));
    readNum(numSamplesV, basename + "/" + "numSamplesV" + "_" + to_string(window));
    readNum(numSamplesE, basename + "/" + "numSamplesE" + "_" + to_string(window));

    readVec(s2sp_X, basename + "/" + "s2sp_X" + "_" + to_string(window));
    readVec(s2sp_V, basename + "/" + "s2sp_V" + "_" + to_string(window));
    readVec(s2sp_E, basename + "/" + "s2sp_E" + "_" + to_string(window));

    std::string outfile_string = basename + "/" + "sample_pmesh" + "_" + to_string(window);
    std::ifstream outfile_romS(outfile_string.c_str());
    sample_pmesh = new ParMesh(comm, outfile_romS);

    readNum(size_H1_sp, basename + "/" + "size_H1_sp" + "_" + to_string(window));
    readNum(size_L2_sp, basename + "/" + "size_L2_sp" + "_" + to_string(window));

    const int ntsamp = spaceTime ? timeSamples.size() : 1;

    BsinvX = NULL;
    BsinvV = new CAROM::Matrix(ntsamp * numSamplesV, rdimfv, false);
    BsinvE = new CAROM::Matrix(ntsamp * numSamplesE, rdimfe, false);

    BXsp = new CAROM::Matrix(size_H1_sp, rdimx, false);
    BVsp = new CAROM::Matrix(size_H1_sp, rdimv, false);
    BEsp = new CAROM::Matrix(size_L2_sp, rdime, false);

    spX = new CAROM::Vector(size_H1_sp, false);
    spV = new CAROM::Vector(size_H1_sp, false);
    spE = new CAROM::Vector(size_L2_sp, false);

    sX = numSamplesX == 0 ? NULL : new CAROM::Vector(numSamplesX, false);
    sV = new CAROM::Vector(numSamplesV, false);
    sE = new CAROM::Vector(numSamplesE, false);

    if (spaceTimeMethod == gnat_lspg)
    {
        BsinvX = new CAROM::Matrix(timeSamples.size() * numSamplesV, rdimv, false);
        BsinvX->read(basename + "/" + "BsinvX" + "_" + to_string(window));
    }

    BsinvV->read(basename + "/" + "BsinvV" + "_" + to_string(window));
    BsinvE->read(basename + "/" + "BsinvE" + "_" + to_string(window));

    BXsp->read(basename + "/" + "BXsp" + "_" + to_string(window));
    BVsp->read(basename + "/" + "BVsp" + "_" + to_string(window));
    BEsp->read(basename + "/" + "BEsp" + "_" + to_string(window));

    BFvsp = new CAROM::Matrix(size_H1_sp, rdimfv, false);
    BFesp = new CAROM::Matrix(size_L2_sp, rdimfe, false);
    BFvsp->read(basename + "/" + "BFvsp" + "_" + to_string(window));
    BFesp->read(basename + "/" + "BFesp" + "_" + to_string(window));

    spX->read(basename + "/" + "spX" + "_" + to_string(window));
    spV->read(basename + "/" + "spV" + "_" + to_string(window));
    spE->read(basename + "/" + "spE" + "_" + to_string(window));

    if (offsetInit || spaceTime) // TODO: why is this necessary for spaceTime case? See SampleMeshAddInitialState
    {
        initXsp = new CAROM::Vector(size_H1_sp, false);
        initVsp = new CAROM::Vector(size_H1_sp, false);
        initEsp = new CAROM::Vector(size_L2_sp, false);

        initXsp->read(basename + "/" + "initXsp" + "_" + to_string(window));
        initVsp->read(basename + "/" + "initVsp" + "_" + to_string(window));
        initEsp->read(basename + "/" + "initEsp" + "_" + to_string(window));
    }
}

void ROM_Basis::writePDweights(const int id, const int window) const
{
    std::string pd_weight_outPath = basename + "/pd_weight" + to_string(window);
    std::ofstream outfile_pd_weight(pd_weight_outPath.c_str());
    if (id >= 0)
    {
        if (offsetInit)
        {
            outfile_pd_weight << (*initX)(id) << endl;
        }
        else
        {
            outfile_pd_weight << "0.0" << endl;
        }
        for (int i=0; i < rdimx; ++i)
        {
            outfile_pd_weight << (*basisX)(id,i) << endl;
        }
    }
    outfile_pd_weight.close();
}

void ROM_Operator::ComputeReducedMv()
{
    const int nv = basis->GetDimV();

    if (hyperreduce && rank == 0)
    {
        invMvROM.SetSize(nv);
        const int size_H1_sp = basis->SolutionSizeH1SP();

        Vector vj_sp(size_H1_sp);
        Vector Mvj_sp(size_H1_sp);
        Vector Mvj(nv);
        for (int j=0; j<nv; ++j)
        {
            basis->GetBasisVectorV(hyperreduce, j, vj_sp);
            operSP->MultMv(vj_sp, Mvj_sp);
            basis->RestrictFromSampleMesh_V(Mvj_sp, Mvj);

            for (int i=0; i<nv; ++i)
                invMvROM(i,j) = Mvj[i];
        }

        invMvROM.Invert();
    }
    else if (!hyperreduce)
    {
        MFEM_ABORT("TODO");
    }
}

void ROM_Operator::ComputeReducedMe()
{
    const int ne = basis->GetDimE();

    if (hyperreduce && rank == 0)
    {
        invMeROM.SetSize(ne);
        const int size_L2_sp = basis->SolutionSizeL2SP();

        Vector ej_sp(size_L2_sp);
        Vector Mej_sp(size_L2_sp);
        Vector Mej(ne);
        for (int j=0; j<ne; ++j)
        {
            basis->GetBasisVectorE(hyperreduce, j, ej_sp);
            operSP->MultMe(ej_sp, Mej_sp);
            basis->RestrictFromSampleMesh_E(Mej_sp, Mej);

            for (int i=0; i<ne; ++i)
                invMeROM(i,j) = Mej[i];
        }

        invMeROM.Invert();
    }
    else if (!hyperreduce)
    {
        MFEM_ABORT("TODO");
    }
}

void ROM_Operator::UpdateSampleMeshNodes(Vector const& romSol)
{
    if (!hyperreduce || rank != 0)
        return;

    // Lift romSol to the sample mesh space to get X.
    basis->LiftToSampleMesh(romSol, fx);

    MFEM_VERIFY(xsp_gf->Size() == Vsize_h1sp, "");  // Since the sample mesh is serial (only on rank 0).

    for (int i=0; i<Vsize_h1sp; ++i)
        (*xsp_gf)[i] = fx[i];

    spmesh->NewNodes(*xsp_gf, false);
}

void ROM_Operator::Mult(const Vector &x, Vector &y) const
{
    MFEM_VERIFY(x.Size() == basis->SolutionSize(), "");  // rdimx + rdimv + rdime
    MFEM_VERIFY(x.Size() == y.Size(), "");

    if (hyperreduce)
    {
        if (rank == 0)
        {
            basis->LiftToSampleMesh(x, fx);

            operSP->Mult(fx, fy);
            basis->RestrictFromSampleMesh(fy, y, true, (useReducedM && !useGramSchmidt && !sns1), &invMvROM, &invMeROM);
            basis->Set_dxdt_Reduced(x, y);

            operSP->ResetQuadratureData();
        }

        MPI_Bcast(y.GetData(), y.Size(), MPI_DOUBLE, 0, basis->comm);
    }
    else
    {
        basis->LiftROMtoFOM(x, fx);
        operFOM->Mult(fx, fy);
        basis->ProjectFOMtoROM(fy, y, true);
    }
}

void ROM_Operator::InducedInnerProduct(const int id1, const int id2, const int var, const int dim, double &ip)
{
    ip = 0.0;
    if (hyperreduce)
    {
        Vector xj_sp(dim);
        Vector xi_sp(dim);
        Vector Mxj_sp(dim);

        if (var == 1) // velocity
        {
            basis->GetBasisVectorV(hyperreduce, id1, xj_sp);
            basis->GetBasisVectorV(hyperreduce, id2, xi_sp);
            operSP->MultMv(xj_sp, Mxj_sp);
        }
        else if (var == 2) // energy
        {
            basis->GetBasisVectorE(hyperreduce, id1, xj_sp);
            basis->GetBasisVectorE(hyperreduce, id2, xi_sp);
            operSP->MultMe(xj_sp, Mxj_sp);
        }
        else
            MFEM_ABORT("Invalid input");

        for (int k=0; k<dim; ++k)
        {
            ip += Mxj_sp[k]*xi_sp[k];
        }
    }
    else if (!hyperreduce)
    {
        MFEM_ABORT("TODO");
    }
}

void ROM_Operator::InducedGramSchmidt(const int var, Vector &S)
{
    if (hyperreduce && rank == 0)
    {
        // Induced Gram Schmidt normalization is equivalent to
        // factorizing the basis into X = QR,
        // where size(Q) = size(X), Q is M-orthonormal,
        // and R is square and upper triangular.
        // Matrix X will be substituted by matrix Q.
        int spdim, rdim, offset;
        CAROM::Matrix *X;
        DenseMatrix *R;
        double factor;

        if (var == 1) // velocity
        {
            spdim = basis->SolutionSizeH1SP();
            rdim = basis->GetDimV();
            offset = basis->GetDimX();
            CoordinateBVsp.SetSize(rdim);
            X = basis->GetBVsp();
            R = &CoordinateBVsp;
        }
        else if (var == 2) // energy
        {
            spdim = basis->SolutionSizeL2SP();
            rdim = basis->GetDimE();
            offset = basis->GetDimX() + basis->GetDimV();
            CoordinateBEsp.SetSize(rdim);
            X = basis->GetBEsp();
            R = &CoordinateBEsp;
        }
        else
        {
            MFEM_ABORT("Invalid variable index");
        }

        InducedInnerProduct(0, 0, var, spdim, factor);
        (*R)(0,0) = sqrt(factor);
        for (int k=0; k<spdim; ++k)
        {
            (*X)(k,0) /= (*R)(0,0); // normalize
        }

        for (int j=1; j<rdim; ++j)
        {
            for (int i=0; i<j; ++i)
            {
                InducedInnerProduct(j, i, var, spdim, factor);
                (*R)(i,j) = factor;
                for (int k=0; k<spdim; ++k)
                {
                    (*X)(k,j) -= (*R)(i,j)*(*X)(k,i); // orthogonalize
                }
            }
            InducedInnerProduct(j, j, var, spdim, factor);
            (*R)(j,j) = sqrt(factor);
            for (int k=0; k<spdim; ++k)
            {
                (*X)(k,j) /= (*R)(j,j); // normalize
            }
        }

        // With solution representation by s = Xc = Qd,
        // the coefficients of s with respect to Q is
        // obtained by d = Rc.
        for (int i=0; i<rdim; ++i)
        {
            S[offset+i] *= (*R)(i,i);
            for (int j=i+1; j<rdim; ++j)
            {
                S[offset+i] += (*R)(i,j)*S[offset+j]; // triangular update
            }
        }
    }
    else if (!hyperreduce)
    {
        MFEM_ABORT("TODO");
    }
}

void ROM_Operator::UndoInducedGramSchmidt(const int var, Vector &S, bool keep_data)
{
    if (hyperreduce && rank == 0)
    {
        // Get back the original matrix X from matrix Q by undoing all the operations
        // in the induced Gram Schmidt normalization process.
        // See ROM_Operator::InducedGramSchmidt
        int spdim, rdim, offset;
        CAROM::Matrix *X;
        DenseMatrix *R;

        if (var == 1) // velocity
        {
            spdim = basis->SolutionSizeH1SP();
            rdim = basis->GetDimV();
            offset = basis->GetDimX();
            X = keep_data ? new CAROM::Matrix(*basis->GetBVsp()) : basis->GetBVsp();
            R = &CoordinateBVsp;
        }
        else if (var == 2) // energy
        {
            spdim = basis->SolutionSizeL2SP();
            rdim = basis->GetDimE();
            offset = basis->GetDimX() + basis->GetDimV();
            X = keep_data ? new CAROM::Matrix(*basis->GetBEsp()) : basis->GetBEsp();
            R = &CoordinateBEsp;
        }
        else
        {
            MFEM_ABORT("Invalid variable index");
        }

        for (int j=rdim-1; j>-1; --j)
        {
            for (int k=0; k<spdim; ++k)
            {
                (*X)(k,j) *= (*R)(j,j);
            }
            for (int i=0; i<j; ++i)
            {
                for (int k=0; k<spdim; ++k)
                {
                    (*X)(k,j) += (*R)(i,j)*(*X)(k,i);
                }
            }
        }

        // With solution representation by s = Xc = Qd,
        // the coefficients of s with respect to X is
        // obtained from c = R\d.
        for (int i=rdim-1; i>-1; --i)
        {
            for (int j = rdim-1; j>i; --j)
            {
                S[offset+i] -= (*R)(i,j)*S[offset+j]; // backward substitution
            }
            S[offset+i] /= (*R)(i,i);
        }

        if (keep_data)
            delete X;
        else
            (*R).Clear();
    }
    else if (!hyperreduce)
    {
        MFEM_ABORT("TODO");
    }
}

void ROM_Operator::ApplyHyperreduction(Vector &S)
{
    if (useGramSchmidt && !sns1)
    {
        InducedGramSchmidt(1, S); // velocity
        InducedGramSchmidt(2, S); // energy
        MPI_Bcast(S.GetData(), S.Size(), MPI_DOUBLE, 0, basis->comm);
    }
    basis->ComputeReducedMatrices(sns1);
}

void ROM_Operator::PostprocessHyperreduction(Vector &S, bool keep_data)
{
    if (useGramSchmidt && !sns1)
    {
        UndoInducedGramSchmidt(1, S, keep_data); // velocity
        UndoInducedGramSchmidt(2, S, keep_data); // energy
        MPI_Bcast(S.GetData(), S.Size(), MPI_DOUBLE, 0, basis->comm);
    }
}

void STROM_Basis::LiftToSampleMesh(const int ti, Vector const& u, Vector &usp) const
{
    // Get the basis at timestep ti by scaling by the corresponding temporal basis entries.
    MFEM_VERIFY(u.Size() == SolutionSizeST(), "");
    MFEM_VERIFY(u.Size() == u_ti.Size(), "");
    b->ScaleByTemporalBasis(ti, u, u_ti);
    b->LiftToSampleMesh(u_ti, usp);

    if (ti == 0)
    {
        // At initial time, replace basis approximation with exact initial state on the sample mesh.
        usp = 0.0;
        b->SampleMeshAddInitialState(usp);
    }
}

void STROM_Basis::ApplySpaceTimeHyperreductionInverses(Vector const& u, Vector &w) const
{
    MFEM_VERIFY(u.Size() == GetTotalNumSamples(), "");
    if (GaussNewton)
    {
#ifdef STXV
        MFEM_VERIFY(w.Size() == b->rdimv + b->rdimfv + b->rdimfe, "");
#else
        MFEM_VERIFY(w.Size() == (spaceTimeMethod == gnat_lspg) ? b->rdimv + b->rdimfv + b->rdimfe : b->rdimx + b->rdimfv + b->rdimfe, "");
#endif
    }
    else
    {
        MFEM_VERIFY(w.Size() == b->SolutionSize(), "");
    }

    MFEM_VERIFY(b->numSamplesX == 0, "");

    int os = 0;

    const int ntsamp = GetNumSampledTimes();
    CAROM::Vector uV(GetNumSamplesV(), false);  // TODO: make this a member variable?
    CAROM::Vector wV(b->rdimv, false);  // TODO: make this a member variable?

    if (spaceTimeMethod == gnat_lspg)
    {
        // The X RHS has hyperreduced term V.

        // BsinvX stores the transpose of the pseudo-inverse.
        CAROM::Vector BuX(b->rdimv, false);  // TODO: make this a member variable?

        // Set uV from u

        for (int ti=0; ti<ntsamp; ++ti)
        {
            const int offset = ti * GetNumSpatialSamples();
            for (int i=0; i<b->numSamplesV; ++i)
                uV.item(ti + (i*ntsamp)) = u[offset + i];

            // Note that the ordering of uV must match that in GetSampledSpaceTimeBasis, since it will be multiplied by BsinvV^T.
        }

        // Multiply uV by BsinvX^T

        b->BsinvX->transposeMult(uV, BuX);
        for (int i=0; i<b->rdimv; ++i)
            w[os + i] = BuX.item(i);

        os += b->rdimv;
    }
    else
    {
        // The X equation is linear and has no hyperreduction, just a linear operator multiplication against the vector of V ROM coefficients.
#ifdef STXV
        os += b->rdimv;
#else
        os += b->rdimx;
#endif
    }

    // The V RHS has hyperreduced term Fv.

    // BsinvV stores the transpose of the pseudo-inverse.
    CAROM::Vector BuV(b->rdimfv, false);  // TODO: make this a member variable?
    //CAROM::Vector wV(b->rdimv, false);  // TODO: make this a member variable?

    // Set uV from u

    for (int ti=0; ti<ntsamp; ++ti)
    {
        const int offset = (ti * GetNumSpatialSamples()) + ((spaceTimeMethod == gnat_lspg) ? b->numSamplesV : 0);
        for (int i=0; i<b->numSamplesV; ++i)
            uV.item(ti + (i*ntsamp)) = u[offset + i];

        // Note that the ordering of uV must match that in GetSampledSpaceTimeBasis, since it will be multiplied by BsinvV^T.
    }

    // Multiply uV by STbasisV^T STbasis_Fv BsinvV^T
    // TODO: store the product STbasisV^T STbasis_Fv BsinvV^T

    b->BsinvV->transposeMult(uV, BuV);
    if (GaussNewton)
    {
        for (int i=0; i<b->rdimfv; ++i)
            w[os + i] = BuV.item(i);

        os += b->rdimfv;
    }
    else
    {
        b->PiVtransPiFv->mult(BuV, wV);

        for (int i=0; i<b->rdimv; ++i)
            w[os + i] = wV.item(i);

        os += b->rdimv;
    }

    // The E RHS has hyperreduced term Fe.

    // BsinvE stores the transpose of the pseudo-inverse.
    CAROM::Vector uE(GetNumSamplesE(), false);  // TODO: make this a member variable?
    CAROM::Vector BuE(b->rdimfe, false);  // TODO: make this a member variable?
    CAROM::Vector wE(b->rdime, false);  // TODO: make this a member variable?

    // Set uE from u

    for (int ti=0; ti<ntsamp; ++ti)
    {
        const int offset = (ti * GetNumSpatialSamples()) + ((spaceTimeMethod == gnat_lspg) ? (2*b->numSamplesV) : b->numSamplesV);
        for (int i=0; i<b->numSamplesE; ++i)
            uE.item(ti + (i*ntsamp)) = u[offset + i];

        // Note that the ordering of uE must match that in GetSampledSpaceTimeBasis, since it will be multiplied by BsinvE^T.
    }

    // Multiply uE by STbasisE^T STbasis_Fe BsinvE^T
    // TODO: store the product STbasisE^T STbasis_Fe BsinvE^T

    b->BsinvE->transposeMult(uE, BuE);

    if (GaussNewton)
    {
        for (int i=0; i<b->rdimfe; ++i)
            w[os + i] = BuE.item(i);

        MFEM_VERIFY(w.Size() == os + b->rdimfe, "");
    }
    else
    {
        b->PiEtransPiFe->mult(BuE, wE);

        for (int i=0; i<b->rdime; ++i)
            w[os + i] = wE.item(i);

        MFEM_VERIFY(w.Size() == os + b->rdime, "");
    }
}

// Select sample indices from xsp and set the corresponding values in x.
// ti is the index of the time sample, ranging from 0 to GetNumSampledTimes()-1.
void STROM_Basis::RestrictFromSampleMesh(const int ti, Vector const& usp, Vector &u) const
{
    MFEM_VERIFY(ti < GetNumSampledTimes(), "");
    MFEM_VERIFY(u.Size() == GetTotalNumSamples(), "");
    MFEM_VERIFY(usp.Size() == b->SolutionSizeSP(), "");  // (2*size_H1_sp) + size_L2_sp

    int offset = ti * GetNumSpatialSamples();

    // Select entries out of usp on the sample mesh.
    // Note that s2sp_X maps from X samples to sample mesh H1 dofs, and similarly for V and E.

    // TODO: since the X RHS is linear, there should be no sampling of X! A linear operator (stored as a matrix) should simply be applied to the ROM coefficients, not the samples.

    if (spaceTimeMethod == gnat_lspg || spaceTimeMethod == coll_lspg) // use V samples for X
    {
        for (int i=0; i<b->numSamplesV; ++i)
            u[offset + i] = usp[b->s2sp_V[i]];

        offset += b->numSamplesV;
    }
    else
    {
        MFEM_VERIFY(b->numSamplesX == 0, "");
        //for (int i=0; i<b->numSamplesX; ++i)
        //u[offset + i] = usp[b->s2sp_X[i]];
        offset += b->numSamplesX;
    }

    for (int i=0; i<b->numSamplesV; ++i)
        u[offset + i] = usp[b->size_H1_sp + b->s2sp_V[i]];

    offset += b->numSamplesV;
    for (int i=0; i<b->numSamplesE; ++i)
        u[offset + i] = usp[(2*b->size_H1_sp) + b->s2sp_E[i]];
}

// TODO: remove argument rdim?
// TODO: remove argument nt?
void ROM_Basis::SetSpaceTimeInitialGuessComponent(Vector& st, std::string const& name,
        ParFiniteElementSpace *fespace,
        const CAROM::Matrix* basis,
        const CAROM::Matrix* tbasis,
        const int nt,
        const int rdim) const
{
    MFEM_VERIFY(rdim == st.Size(), "");

    Vector b(st.Size());

    char fileExtension[100];
    sprintf(fileExtension, ".%06d", rank);

    std::string fullname = solution_basename + "/ST_Sol_" + name + fileExtension;
    std::ifstream ifs(fullname.c_str());

    const int tvsize = fespace->GetTrueVSize();
    //Vector s(tvsize);
    MFEM_VERIFY(tvsize == basis->numRows() && nt == tbasis->numRows(), "");
    MFEM_VERIFY(rdim == basis->numColumns() && rdim == tbasis->numColumns(), "");

    // Compute the inner product of the input space-time vector against each
    // space-time basis vector, which for the j-th basis vector is
    // \sum_{t=0}^{nt-1} \sum_{i=0}^{tvsize-1} sol(t,i) basis(i,j) tbasis(t,j)
    // Store the result in the vector b.
    // Also, form the mass matrix for the space-time basis.

    DenseMatrix M(rdim);

    b = 0.0;
    M = 0.0;

    // TODO: this is a full-order computation. Should it be hyperreduced? In any case, the FOM solution will need to be read, since the hyperreduction samples are unknown when the FOM solution is written to file, so there does not seem to be potential savings.

    for (int t=0; t<nt; ++t)
    {
        for (int i=0; i<tvsize; ++i)
        {
            double d;
            ifs >> d;
            for (int j=0; j<rdim; ++j)
            {
                b[j] += d * basis->item(i, j) * tbasis->item(t, j);
                //s[i] = d;

                for (int k=j; k<rdim; ++k)  // Upper triangular part only
                    M(j,k) += basis->item(i, j) * tbasis->item(t, j) * basis->item(i, k) * tbasis->item(t, k);
            }
        }
    }

    ifs.close();

    // Assert that the strictly upper triangular part of the mass matrix is zero
    for (int i=0; i<rdim-1; ++i)
        for (int j=i+1; j<rdim; ++j)
        {
            if (fabs(M(i,j)) >= 1.0e-13)
                mfem::out << "M " << M(i,j) << endl;

            MFEM_VERIFY(fabs(M(i,j)) < 1.0e-13, "");
        }

    // TODO: remove the assertion that the mass matrix is diagonal?

    // TODO: is M = I in general?

    // Solve for st
    for (int i=0; i<rdim; ++i)
        st[i] = b[i] / M(i,i);
}

void ROM_Basis::SetSpaceTimeInitialGuess(ROM_Options const& input)
{
    // TODO: this assumes 1 temporal basis vector for each spatial vector. Generalize to allow for multiple temporal basis vectors per spatial vector.

    //st0.SetSize(SolutionSizeST());
    st0.SetSize(SolutionSize());
    st0 = 0.0; // TODO

    // For now, we simply test the reproductive case by projecting the known FOM solution.
    // With hyperreduction, this projection will not be the exact solution of the ROM system,
    // but it should be close.
    // We use the entire FOM space-time solution read from a file in order to compute this projection.

    Vector st0X, st0V, st0E;
    st0X.MakeRef(st0, 0, rdimx);
    st0V.MakeRef(st0, rdimx, rdimv);
    st0E.MakeRef(st0, rdimx + rdimv, rdime);
    SetSpaceTimeInitialGuessComponent(st0X, "Position", input.H1FESpace, basisX, tbasisX, temporalSize, rdimx);
    SetSpaceTimeInitialGuessComponent(st0V, "Velocity", input.H1FESpace, basisV, tbasisV, temporalSize - VTos, rdimv);
    SetSpaceTimeInitialGuessComponent(st0E, "Energy", input.L2FESpace, basisE, tbasisE, temporalSize, rdime);
}

void ROM_Basis::GetSpaceTimeInitialGuess(Vector& st) const
{
    st = st0;
}

void ROM_Operator::SolveSpaceTime(Vector &S)
{
    Vector x;
    basis->GetSpaceTimeInitialGuess(x);

    MFEM_VERIFY(S.Size() == x.Size(), "");

    if (useGramSchmidt)
    {
        //InducedGramSchmidtInitialize(x); // TODO: this assumes 1 temporal basis vector per spatial basis vector and needs to be generalized.
        ApplyHyperreduction(x); // TODO: this assumes 1 temporal basis vector per spatial basis vector and needs to be generalized.
    }

    Vector c(x.Size());
    Vector r(x.Size());

    const int n = basis->SolutionSize();
    const int m = GaussNewton ? basis->GetDimX() + basis->GetDimFv() + basis->GetDimFe() : n;

    MFEM_VERIFY(n == x.Size(), "");

    DenseMatrix jac(m, n);

    MFEM_VERIFY(rank == 0, "Space-time solver is serial");

    // Newton's method, with zero RHS.
    int it;
    double norm0, norm, norm_goal;
    const double rel_tol = 1.0e-8;
    const double abs_tol = 1.0e-12;
    const int print_level = 0;
    const int max_iter = 12;

    EvalSpaceTimeResidual_RK4(x, r);

    // TODO: in parallel, replace Norml2 with sqrt(InnerProduct(comm,...), see vector.hpp.
    norm0 = norm = r.Norml2();
    norm_goal = std::max(rel_tol*norm, abs_tol);

    mfem::out << "Newton initial norm " << norm << '\n';
    x.Print();

    // x_{i+1} = x_i - [DF(x_i)]^{-1} [F(x_i)-b]
    for (it = 0; true; it++)
    {
        MFEM_VERIFY(IsFinite(norm), "norm = " << norm);
        cout << "Newton iteration " << it << endl;  // TODO: remove
        if (print_level >= 0)
        {
            mfem::out << "Newton iteration " << setw(2) << it
                      << " : ||r|| = " << norm;
            if (it > 0)
            {
                mfem::out << ", ||r||/||r_0|| = " << norm/norm0;
            }
            mfem::out << '\n';
        }

        if (norm <= norm_goal)
            break;

        if (it >= max_iter)
            break;

        EvalSpaceTimeJacobian_RK4(x, jac);

        c = r;  // TODO: eliminate c?
        LinearSolve(jac, c.GetData());

        const double c_scale = 0.25; //ComputeScalingFactor(x, b);
        if (c_scale == 0.0)
            break;

        add(x, -c_scale, c, x);

        x.Print();

        EvalSpaceTimeResidual_RK4(x, r);

        norm = r.Norml2();
    }  // end of Newton iteration

    if (it >= max_iter)
        mfem::out << "ERROR: Newton failed to converge" << endl;

    if (useGramSchmidt)
    {
        //InducedGramSchmidtFinalize(x);
        PostprocessHyperreduction(x);
    }

    // Scale by the temporal basis at the final time.
    basis->ScaleByTemporalBasis(basis->GetTemporalSize() - 1, x, S);
}

void ROM_Operator::SolveSpaceTimeGN(Vector &S)
{
    MFEM_VERIFY(rank == 0, "Space-time solver is serial");
    MFEM_VERIFY(GaussNewton, "");
    Vector x;
    basis->GetSpaceTimeInitialGuess(x);

    MFEM_VERIFY(S.Size() == x.Size(), "");

    if (useGramSchmidt)
    {
        //InducedGramSchmidtInitialize(x); // TODO: this assumes 1 temporal basis vector per spatial basis vector and needs to be generalized.
        ApplyHyperreduction(x); // TODO: this assumes 1 temporal basis vector per spatial basis vector and needs to be generalized.
    }

    const int n = (spaceTimeMethod == coll_lspg) ? STbasis->SolutionSizeST() : basis->SolutionSize();
#ifdef STXV
    const int m = GaussNewton ? basis->GetDimV() + basis->GetDimFv() + basis->GetDimFe() : n;
#else
    // TODO: simplify this
    const int m = (spaceTimeMethod == coll_lspg) ? STbasis->GetTotalNumSamples() :
                  (GaussNewton ? ((spaceTimeMethod == gnat_lspg) ? basis->GetDimV() + basis->GetDimFv() + basis->GetDimFe()
                                  : basis->GetDimX() + basis->GetDimFv() + basis->GetDimFe()) : n);
#endif

    Vector c(n);
    Vector r(m);

    MFEM_VERIFY(n == x.Size(), "");

    DenseMatrix jac(m, n);
    DenseMatrix jacNormal(n, n);

    // Newton's method, with zero RHS.
    int it;
    double norm0, norm, norm_goal;
    const double rel_tol = 1.0e-8;
    const double abs_tol = 1.0e-12;
    const int print_level = 0;
    const int max_iter = 2;

    EvalSpaceTimeResidual_RK4(x, r);

    // TODO: in parallel, replace Norml2 with sqrt(InnerProduct(comm,...), see vector.hpp.
    norm0 = norm = r.Norml2();
    norm_goal = std::max(rel_tol*norm, abs_tol);

    mfem::out << "Gauss-Newton initial norm " << norm << '\n';
    x.Print();

    // x_{i+1} = x_i - [DF(x_i)]^{-1} [F(x_i)-b]
    for (it = 0; true; it++)
    {
        MFEM_VERIFY(IsFinite(norm), "norm = " << norm);
        cout << "Newton iteration " << it << endl;  // TODO: remove
        if (print_level >= 0)
        {
            mfem::out << "Newton iteration " << setw(2) << it
                      << " : ||r|| = " << norm;
            if (it > 0)
            {
                mfem::out << ", ||r||/||r_0|| = " << norm/norm0;
            }
            mfem::out << '\n';
        }

        if (norm <= norm_goal)
            break;

        if (it >= max_iter)
            break;

        EvalSpaceTimeJacobian_RK4(x, jac);

        //c = r;  // TODO: eliminate c?
        jac.MultTranspose(r, c);
        MultAtB(jac, jac, jacNormal);

        LinearSolve(jacNormal, c.GetData());  // TODO: use a more stable least-squares solver?

        const double c_scale = 1.0; //ComputeScalingFactor(x, b);
        //if (c_scale == 0.0)
        //break;

        add(x, -c_scale, c, x);

        x.Print();

        EvalSpaceTimeResidual_RK4(x, r);

        norm = r.Norml2();
    }  // end of Newton iteration

    if (it >= max_iter)
        mfem::out << "ERROR: Newton failed to converge" << endl;

    if (useGramSchmidt)
    {
        //InducedGramSchmidtFinalize(x);
        PostprocessHyperreduction(x);
    }

    // Scale by the temporal basis at the final time.
    basis->ScaleByTemporalBasis(basis->GetTemporalSize() - 1, x, S);
}

void ROM_Operator::EvalSpaceTimeJacobian_RK4(Vector const& S, DenseMatrix &J) const
{
    const int n = (spaceTimeMethod == coll_lspg) ? STbasis->SolutionSizeST() : basis->SolutionSize();
#ifdef STXV
    const int m = GaussNewton ? basis->GetDimV() + basis->GetDimFv() + basis->GetDimFe() : n;
#else
    // TODO: simplify this
    const int m = (spaceTimeMethod == coll_lspg) ? STbasis->GetTotalNumSamples() :
                  (GaussNewton ? ((spaceTimeMethod == gnat_lspg) ? basis->GetDimV() + basis->GetDimFv() + basis->GetDimFe()
                                  : basis->GetDimX() + basis->GetDimFv() + basis->GetDimFe()) : n);
#endif

    MFEM_VERIFY(J.Height() == m && J.Width() == n, "");

    J = 0.0;

    Vector r(m);
    Vector rp(m);
    Vector Sp(n);

    EvalSpaceTimeResidual_RK4(S, r);

    const double eps = 1.0e-8;

    for (int j=0; j<n; ++j)
    {
        Sp = S;
        const double eps_j = std::max(eps, eps * fabs(Sp[j]));
        Sp[j] += eps_j;
        EvalSpaceTimeResidual_RK4(Sp, rp);
        rp -= r;
        rp /= eps_j;

        for (int i=0; i<m; ++i)
            J(i,j) = rp[i];
    }

    /*
    ofstream jfile("stjac.txt");
    J.Print(jfile);
    jfile.close();
    */
}

void ROM_Operator::EvalSpaceTimeResidual_RK4(Vector const& S, Vector &f) const
{
    const int rdimx = basis->GetDimX();
    const int rdimv = basis->GetDimV();
    const int rdimfv = basis->GetDimFv();
    const int rdimfe = basis->GetDimFe();

    if (spaceTimeMethod == coll_lspg)
    {
        MFEM_VERIFY(S.Size() == STbasis->SolutionSizeST() && f.Size() == STbasis->GetTotalNumSamples(), "");
    }
    else
    {
        if (GaussNewton)
        {
#ifdef STXV
            MFEM_VERIFY(S.Size() == STbasis->SolutionSizeST() && f.Size() == rdimfv + rdimfe + rdimv, "");
#else
            if (spaceTimeMethod == gnat_lspg)
            {
                MFEM_VERIFY(S.Size() == STbasis->SolutionSizeST() && f.Size() == rdimfv + rdimfe + rdimv, "");
            }
            else
            {
                MFEM_VERIFY(S.Size() == STbasis->SolutionSizeST() && f.Size() == rdimfv + rdimfe + rdimx, "");
            }
#endif
        }
        else
        {
            MFEM_VERIFY(S.Size() == STbasis->SolutionSizeST() && S.Size() == f.Size(), "");
        }
    }

    MFEM_VERIFY(hyperreduce, "");

    Sr = 0.0;

    // Loop over sampled timesteps
    for (int i=0; i<STbasis->GetNumSampledTimes(); ++i)
    {
        const int ti = STbasis->GetTimeSampleIndex(i);
        double t = STbasis->GetTimeSample(i);  // t_n
        double dt = STbasis->GetTimestep(i);
        const double t0 = t;

        // Note: the time index ti corresponds to a time t_n, and we now compute the RK4 residual
        // w(t_{n+1}) - w(t_n) - dt/6 (k1 + 2k2 + 2k3 + k4)
        STbasis->LiftToSampleMesh(ti, S, fx);  // Set fx = w(t_n)

        {   // Update sample mesh nodes
            // TODO: remove this? Is it redundant?
            MFEM_VERIFY(xsp_gf->Size() == Vsize_h1sp, "");  // Since the sample mesh is serial (only on rank 0).

            for (int j=0; j<Vsize_h1sp; ++j)
                (*xsp_gf)[j] = fx[j];

            spmesh->NewNodes(*xsp_gf, false);
        }

        ST_ode_solver->Step(fx, t, dt);

        MFEM_VERIFY(fabs(t - t0 - dt) < 1.0e-12, "");

        // Now fx = w(t_n) + dt/6 (k1 + 2k2 + 2k3 + k4)

        STbasis->LiftToSampleMesh(ti+1, S, fy);  // Set fy = w(t_{n+1})
        fy -= fx;

        // Now fy is the residual w(t_{n+1}) - w(t_n) - dt/6 (k1 + 2k2 + 2k3 + k4)

        STbasis->RestrictFromSampleMesh(i, fy, Sr);
    }

    if (spaceTimeMethod == coll_lspg)
    {
        f = Sr;
        return;
    }

    STbasis->ApplySpaceTimeHyperreductionInverses(Sr, f);

    if (spaceTimeMethod == gnat_lspg)
        return;

    // Evaluate X equations without sampling
    {
        CAROM::Vector v(basis->GetDimV(), false);
#ifdef STXV
        CAROM::Vector xv(basis->GetDimV(), false);
#else
        CAROM::Vector xv(basis->GetDimX(), false);
#endif
        CAROM::Vector x(basis->GetDimX(), false);

        for (int i=0; i<basis->GetDimX(); ++i)
            x.item(i) = S[i];

        for (int i=0; i<basis->GetDimV(); ++i)
            v.item(i) = S[basis->GetDimX() + i];

        // TODO: since the V basis is larger than the X, would it be more accurate to multiply the X equation by Pi_V^T rather than Pi_X^T?
        // Or should the X and V bases be merged into one XV basis? Maybe the V basis would be sufficient.

        // Note that PiXtransPiV contains the RK4 scaling.

#ifdef STXV
        basis->PiXtransPiV->mult(v, xv);
        for (int i=0; i<basis->GetDimV(); ++i)
            f[i] = -xv.item(i);

        // TODO: store PiXtransPiX - PiXtransPiXlag?

        basis->PiXtransPiX->mult(x, xv);
        for (int i=0; i<basis->GetDimV(); ++i)
            f[i] += xv.item(i);

        basis->PiXtransPiXlag->mult(x, xv);
        for (int i=0; i<basis->GetDimV(); ++i)
            f[i] -= xv.item(i);
#else
        basis->PiXtransPiV->mult(v, xv);
        for (int i=0; i<basis->GetDimX(); ++i)
            f[i] = -xv.item(i);

        // TODO: store PiXtransPiX - PiXtransPiXlag?

        basis->PiXtransPiX->mult(x, xv);
        for (int i=0; i<basis->GetDimX(); ++i)
            f[i] += xv.item(i);

        basis->PiXtransPiXlag->mult(x, xv);
        for (int i=0; i<basis->GetDimX(); ++i)
            f[i] -= xv.item(i);
#endif
    }
}

CAROM::GreedyParameterPointSampler* BuildROMDatabase(ROM_Options& romOptions, double& t_final, const int myid, const std::string outputPath,
        bool& rom_offline, bool& rom_online, bool& rom_calc_rel_error, const char* greedyParamString, const char* greedyErrorIndicatorType, const char* greedySamplingType)
{
    CAROM::GreedyParameterPointSampler* parameterPointGreedySampler = NULL;
    samplingType sampleType = getSamplingType(greedySamplingType);

    romOptions.greedyErrorIndicatorType = getErrorIndicatorType(greedyErrorIndicatorType);

    char tmp[100];
    sprintf(tmp, ".%06d", myid);
    ifstream f(outputPath + "/greedy_algorithm_data" + tmp);
    if (f.good())
    {
        parameterPointGreedySampler = new CAROM::GreedyParameterPointRandomSampler(
            outputPath + "/greedy_algorithm_data",
            outputPath + "/greedy_algorithm_log.txt");
    }
    else
    {
        bool latin_hypercube = sampleType == latinHypercubeSampling;
        parameterPointGreedySampler = new CAROM::GreedyParameterPointRandomSampler(
            romOptions.greedyParamSpaceMin, romOptions.greedyParamSpaceMax,
            romOptions.greedyParamSpaceSize, true, romOptions.greedyTol, romOptions.greedyAlpha,
            romOptions.greedyMaxClamp, romOptions.greedySubsetSize, romOptions.greedyConvergenceSubsetSize,
            latin_hypercube, outputPath + "/greedy_algorithm_log.txt");

        ofstream o(outputPath + "/greedy_algorithm_log.txt", std::ios::app);
        o << "Parameter considered: " << greedyParamString << std::endl;
        o << "Error indicator: " << greedyErrorIndicatorType << std::endl;
        o.close();
    }

    // First check if we need to compute another error indicator
    struct CAROM::GreedyErrorIndicatorPoint pointRequiringErrorIndicator = parameterPointGreedySampler->getNextPointRequiringErrorIndicator();
    CAROM::Vector* errorIndicatorPointData = pointRequiringErrorIndicator.point.get();
    struct CAROM::GreedyErrorIndicatorPoint pointRequiringRelativeError = parameterPointGreedySampler->getNextPointRequiringRelativeError();
    CAROM::Vector* samplePointData = pointRequiringRelativeError.point.get();
    double* greedyParam = getGreedyParam(romOptions, greedyParamString);

    if (errorIndicatorPointData != NULL)
    {
        CAROM::Vector* localROM = pointRequiringErrorIndicator.localROM.get();
        std::string localROMString = "";
        for (int i = 0; i < localROM->dim(); i++)
        {
            localROMString += "_" + to_string(localROM->item(i));
        }
        romOptions.basisIdentifier = localROMString;
        *greedyParam = errorIndicatorPointData->item(0);

        double errorIndicatorEnergyFraction = 0.9999;

        char tmp[100];
        sprintf(tmp, ".%06d", myid);

        std::string fullname = outputPath + "/" + std::string("errorIndicatorVec") + tmp;

        if (romOptions.greedyErrorIndicatorType == varyBasisSize)
        {
            t_final = 0.001;
        }

        std::ifstream checkfile(fullname);
        if (!checkfile.good())
        {
            if (romOptions.greedyErrorIndicatorType == varyBasisSize)
            {
                errorIndicatorEnergyFraction = 0.99;
            }
        }

        // Get the rdim for the basis used.
        readNum(romOptions.dimX, outputPath + "/" + "rdimx" + romOptions.basisIdentifier + "_" + to_string(errorIndicatorEnergyFraction));
        readNum(romOptions.dimV, outputPath + "/" + "rdimv" + romOptions.basisIdentifier + "_" + to_string(errorIndicatorEnergyFraction));
        readNum(romOptions.dimE, outputPath + "/" + "rdime" + romOptions.basisIdentifier + "_" + to_string(errorIndicatorEnergyFraction));
        if (!romOptions.SNS)
        {
            readNum(romOptions.dimFv, outputPath + "/" + "rdimfv" + romOptions.basisIdentifier + "_" + to_string(errorIndicatorEnergyFraction));
            readNum(romOptions.dimFe, outputPath + "/" + "rdimfe" + romOptions.basisIdentifier + "_" + to_string(errorIndicatorEnergyFraction));
        }

        rom_online = true;
    }
    else if (samplePointData != NULL)
    {
        CAROM::Vector* localROM = pointRequiringRelativeError.localROM.get();
        std::string localROMString = "";
        for (int i = 0; i < localROM->dim(); i++)
        {
            localROMString += "_" + to_string(localROM->item(i));
        }
        romOptions.basisIdentifier = localROMString;
        *greedyParam = samplePointData->item(0);

        double errorIndicatorEnergyFraction = 0.9999;

        // Get the rdim for the basis used.
        readNum(romOptions.dimX, outputPath + "/" + "rdimx" + romOptions.basisIdentifier + "_" + to_string(errorIndicatorEnergyFraction));
        readNum(romOptions.dimV, outputPath + "/" + "rdimv" + romOptions.basisIdentifier + "_" + to_string(errorIndicatorEnergyFraction));
        readNum(romOptions.dimE, outputPath + "/" + "rdime" + romOptions.basisIdentifier + "_" + to_string(errorIndicatorEnergyFraction));
        if (!romOptions.SNS)
        {
            readNum(romOptions.dimFv, outputPath + "/" + "rdimfv" + romOptions.basisIdentifier + "_" + to_string(errorIndicatorEnergyFraction));
            readNum(romOptions.dimFe, outputPath + "/" + "rdimfe" + romOptions.basisIdentifier + "_" + to_string(errorIndicatorEnergyFraction));
        }

        rom_online = true;
        rom_calc_rel_error = true;
    }
    else
    {
        // Next check if we need to run FOM for another parameter point
        std::shared_ptr<CAROM::Vector> nextSampleParameterPoint = parameterPointGreedySampler->getNextParameterPoint();
        samplePointData = nextSampleParameterPoint.get();
        if (samplePointData != NULL)
        {
            std::string samplePointDataString = "";
            for (int i = 0; i < samplePointData->dim(); i++)
            {
                samplePointDataString += "_" + to_string(samplePointData->item(i));
            }
            romOptions.basisIdentifier = samplePointDataString;
            *greedyParam = samplePointData->item(0);
            rom_offline = true;
        }
        else
        {
            // The greedy algorithm procedure has ended
            MFEM_ABORT("The greedy algorithm procedure has failed!");
        }
    }

    return parameterPointGreedySampler;
}

CAROM::GreedyParameterPointSampler* UseROMDatabase(ROM_Options& romOptions, const int myid, const std::string outputPath, const char* greedyParamString)
{

    CAROM::GreedyParameterPointSampler* parameterPointGreedySampler = NULL;

    char tmp[100];
    sprintf(tmp, ".%06d", myid);
    ifstream f(outputPath + "/greedy_algorithm_data" + tmp);
    MFEM_VERIFY(f.good(), "The greedy algorithm has not been run yet.")

    parameterPointGreedySampler = new CAROM::GreedyParameterPointRandomSampler(
        outputPath + "/greedy_algorithm_data");
    double* greedyParam = getGreedyParam(romOptions,greedyParamString);

    CAROM::Vector parameter_point(1, false);
    parameter_point.item(0) = *greedyParam;

    std::shared_ptr<CAROM::Vector> nearestROM = parameterPointGreedySampler->getNearestROM(parameter_point);
    CAROM::Vector* pointData = nearestROM.get();

    MFEM_VERIFY(pointData != NULL, "No parameter points were found");
    std::string pointDataString = "";
    for (int i = 0; i < pointData->dim(); i++)
    {
        pointDataString += "_" + to_string(pointData->item(i));
    }
    romOptions.basisIdentifier = pointDataString;

    return parameterPointGreedySampler;
}

void SaveROMDatabase(CAROM::GreedyParameterPointSampler* parameterPointGreedySampler, ROM_Options& romOptions, const bool rom_online, const double errorIndicator,
                     const int errorIndicatorVecSize, const std::string outputPath)
{
    if (rom_online)
    {
        parameterPointGreedySampler->setPointErrorIndicator(errorIndicator, errorIndicatorVecSize);
    }

    parameterPointGreedySampler->save(outputPath + "/greedy_algorithm_data");
}

void SaveROMDatabase(CAROM::GreedyParameterPointSampler* parameterPointGreedySampler, ROM_Options& romOptions, const bool rom_online, const double relative_error,
                     const std::string outputPath)
{
    if (rom_online)
    {
        parameterPointGreedySampler->setPointRelativeError(relative_error);
    }

    parameterPointGreedySampler->save(outputPath + "/greedy_algorithm_data");
}

void ROM_Operator::StepRK2Avg(Vector &S, double &t, double &dt) const
{
    MFEM_VERIFY(S.Size() == basis->SolutionSize(), "");  // rdimx + rdimv + rdime

    hydrodynamics::LagrangianHydroOperator *hydro_oper = hyperreduce ? operSP : operFOM;

    if (!hyperreduce || rank == 0)
    {
        if (hyperreduce)
            basis->LiftToSampleMesh(S, fx);
        else
            basis->LiftROMtoFOM(S, fx);

        const int Vsize = hyperreduce ? basis->SolutionSizeH1SP() : basis->SolutionSizeH1FOM();
        const int Esize = basis->SolutionSizeL2SP();
        Vector V(Vsize), dS_dt(fx.Size()), S0(fx);

        // The monolithic BlockVector stores the unknown fields as follows:
        // (Position, Velocity, Specific Internal Energy).
        Vector dv_dt, v0, dx_dt, de_dt;
        v0.SetDataAndSize(S0.GetData() + Vsize, Vsize);
        de_dt.SetDataAndSize(dS_dt.GetData() + (2*Vsize), Esize);
        dv_dt.SetDataAndSize(dS_dt.GetData() + Vsize, Vsize);
        dx_dt.SetDataAndSize(dS_dt.GetData(), Vsize);

        // In each sub-step:
        // - Update the global state Vector S.
        // - Compute dv_dt using S.
        // - Update V using dv_dt.
        // - Compute de_dt and dx_dt using S and V.

        // -- 1.
        // S is S0.
        hydro_oper->UpdateMesh(fx);
        hydro_oper->SolveVelocity(fx, dS_dt);
        if (hyperreduce) basis->HyperreduceRHS_V(dv_dt); // Set dv_dt based on RHS computed by SolveVelocity
        // V = v0 + 0.5 * dt * dv_dt;
        add(v0, 0.5 * dt, dv_dt, V);
        hydro_oper->SolveEnergy(fx, V, dS_dt);
        if (hyperreduce) basis->HyperreduceRHS_E(de_dt); // Set de_dt based on RHS computed by SolveEnergy
        dx_dt = V;

        // -- 2.
        // S = S0 + 0.5 * dt * dS_dt;
        add(S0, 0.5 * dt, dS_dt, fx);
        hydro_oper->ResetQuadratureData();
        hydro_oper->UpdateMesh(fx);
        hydro_oper->SolveVelocity(fx, dS_dt);
        if (hyperreduce) basis->HyperreduceRHS_V(dv_dt); // Set dv_dt based on RHS computed by SolveVelocity
        // V = v0 + 0.5 * dt * dv_dt;
        add(v0, 0.5 * dt, dv_dt, V);
        hydro_oper->SolveEnergy(fx, V, dS_dt);
        if (hyperreduce) basis->HyperreduceRHS_E(de_dt); // Set de_dt based on RHS computed by SolveEnergy
        dx_dt = V;

        // -- 3.
        // S = S0 + dt * dS_dt.
        add(S0, dt, dS_dt, fx);
        hydro_oper->ResetQuadratureData();

        MFEM_VERIFY(!useReducedM, "TODO");

        if (hyperreduce)
            basis->RestrictFromSampleMesh(fx, S, false);
        else
            basis->ProjectFOMtoROM(fx, S);
    }

    if (hyperreduce)
    {
        MPI_Bcast(S.GetData(), S.Size(), MPI_DOUBLE, 0, basis->comm);
    }

    t += dt;
}
