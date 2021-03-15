#include "laghos_rom.hpp"
#include "laghos_utils.hpp"

#include "DEIM.h"
#include "QDEIM.h"
#include "SampleMesh.hpp"


using namespace std;

void ROM_Sampler::SampleSolution(const double t, const double dt, Vector const& S)
{
    SetStateVariables(S);
    SetStateVariableRates(dt);

    const bool sampleX = generator_X->isNextSample(t);

    Vector dSdt;
    if (sampleF)
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

        if (sampleF)
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

        if (sampleF)
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
    for (auto const& i: tSnap)
    {
        outfile_tSnap << i << endl;
    }
}

void ROM_Sampler::Finalize(const double t, const double dt, Vector const& S, Array<int> &cutoff)
{
    if (writeSnapshots)
    {
        if (!useXV) generator_X->writeSnapshot();
        if (!useVX) generator_V->writeSnapshot();
        generator_E->writeSnapshot();
        if (sampleF)
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
        if (sampleF)
        {
            generator_Fv->endSamples();
            generator_Fe->endSamples();
        }
    }

    if (rank == 0 && !writeSnapshots)
    {
        if (!useXV)
        {
            cout << "X basis summary output: ";
            BasisGeneratorFinalSummary(generator_X, energyFraction_X, cutoff[0]);
            PrintSingularValues(rank, basename, "X", generator_X);
        }

        if (!useVX)
        {
            cout << "V basis summary output: ";
            BasisGeneratorFinalSummary(generator_V, energyFraction, cutoff[1]);
            PrintSingularValues(rank, basename, "V", generator_V);
        }

        cout << "E basis summary output: ";
        BasisGeneratorFinalSummary(generator_E, energyFraction, cutoff[2]);
        PrintSingularValues(rank, basename, "E", generator_E);

        if (sampleF)
        {
            cout << "Fv basis summary output: ";
            BasisGeneratorFinalSummary(generator_Fv, energyFraction, cutoff[3]);

            cout << "Fe basis summary output: ";
            BasisGeneratorFinalSummary(generator_Fe, energyFraction, cutoff[4]);
        }
    }

    if (rank == 0 && writeSnapshots)
    {
        std::string path_tSnap = basename + "/param" + std::to_string(parameterID) + "_tSnap";

        printSnapshotTime(tSnapX, path_tSnap, "X");
        printSnapshotTime(tSnapV, path_tSnap, "V");
        printSnapshotTime(tSnapE, path_tSnap, "E");

        if (sampleF)
        {
            printSnapshotTime(tSnapFv, path_tSnap, "Fv");
            printSnapshotTime(tSnapFe, path_tSnap, "Fe");
        }
    }

    delete generator_X;
    delete generator_V;
    delete generator_E;

    if (sampleF)
    {
        delete generator_Fv;
        delete generator_Fe;
    }
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
    CAROM::Matrix* basisCopy = GetFirstColumns(dim, basis, rowOS, vectorSize);

    MFEM_VERIFY(basisCopy->numRows() == vectorSize, "");

    if (rank == 0)
        cout << "Read basis " << filename << " of dimension " << basisCopy->numColumns() << endl;

    //delete basis;
    return basisCopy;
}

CAROM::Matrix* MultBasisROM(const int rank, const std::string filename, const int vectorSize, const int rowOS, int& dim, hydrodynamics::LagrangianHydroOperator *lhoper, const int var)
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
            MFEM_VERIFY(false, "Invalid input");

        for (int i=0; i<S->numRows(); ++i)
            (*S)(i,j) = MBej[i];
    }

    return S;
}

ROM_Basis::ROM_Basis(ROM_Options const& input, MPI_Comm comm_, const double sFactorX, const double sFactorV)
    : comm(comm_), rdimx(input.dimX), rdimv(input.dimV), rdime(input.dimE), rdimfv(input.dimFv), rdimfe(input.dimFe),
      numSamplesX(input.sampX), numSamplesV(input.sampV), numSamplesE(input.sampE), use_sns(input.SNS),
      hyperreduce(input.hyperreduce), hyperreduce_prep(input.hyperreduce_prep), offsetInit(input.useOffset),
      RHSbasis(input.RHSbasis), useGramSchmidt(input.GramSchmidt), lhoper(input.FOMoper),
      RK2AvgFormulation(input.RK2AvgSolver), basename(*input.basename),
      mergeXV(input.mergeXV), useXV(input.useXV), useVX(input.useVX), Voffset(!input.useXV && !input.useVX && !input.mergeXV),
      energyFraction_X(input.energyFraction_X), use_qdeim(input.qdeim)
{
    MFEM_VERIFY(!(input.useXV && input.useVX) && !(input.useXV && input.mergeXV) && !(input.useVX && input.mergeXV), "");

    if (useXV) rdimx = rdimv;
    if (useVX) rdimv = rdimx;

    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    Array<int> nH1(nprocs);

    if (!hyperreduce)
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
        if (hyperreduce_prep && rank == 0)
        {
            writeNum(rdimx, basename + "/" + "rdimx" + "_" + to_string(input.window));
            writeNum(rdimv, basename + "/" + "rdimv" + "_" + to_string(input.window));
            writeNum(rdime, basename + "/" + "rdime" + "_" + to_string(input.window));
        }
    }
    else if (rank == 0)
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
            readSP(input, input.window);
        }
        return;
    }

    if (offsetInit)
    {
        initX = new CAROM::Vector(tH1size, true);
        initV = new CAROM::Vector(tH1size, true);
        initE = new CAROM::Vector(tL2size, true);

        std::string path_init = basename + "/ROMoffset/init";

        if (input.restore || input.offsetType == saveLoadOffset)
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
                if (coeff == 0.0)
                {
                    true_idx = coeff_list.size();
                }
                coeff = 1 / sqrt(coeff);
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
        else if (input.offsetType == useInitialState && input.window > 0)
        {
            // Read offset in the online phase of initial mode
            initX->read(path_init + "X0");
            initV->read(path_init + "V0");
            initE->read(path_init + "E0");

            cout << "Read init vectors X, V, E with norms " << initX->norm() << ", " << initV->norm() << ", " << initE->norm() << endl;
        }
    }

    if (hyperreduce_prep)
    {
        if (rank == 0) cout << "start preprocessing hyper-reduction\n";
        StopWatch preprocessHyperreductionTymer;
        preprocessHyperreductionTymer.Start();
        SetupHyperreduction(input.H1FESpace, input.L2FESpace, nH1, input.window);
        preprocessHyperreductionTymer.Stop();
        if (rank == 0) cout << "Elapsed time for hyper-reduction preprocessing: " << preprocessHyperreductionTymer.RealTime() << " sec\n";
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
    if (offsetInit && !input.restore && input.offsetType == useInitialState && input.window == 0)
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

        initX->write(path_init + "X" + std::to_string(input.window));
        initV->write(path_init + "V" + std::to_string(input.window));
        initE->write(path_init + "E" + std::to_string(input.window));
    }

    if (offsetInit && hyperreduce_prep)
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

void ROM_Basis::SetupHyperreduction(ParFiniteElementSpace *H1FESpace, ParFiniteElementSpace *L2FESpace, Array<int>& nH1, const int window)
{
    ParMesh *pmesh = H1FESpace->GetParMesh();

    const int fomH1size = H1FESpace->GlobalTrueVSize();
    const int fomL2size = L2FESpace->GlobalTrueVSize();

    numSamplesX = RHSbasis ? 0 : std::min(fomH1size, numSamplesX);
    vector<int> sample_dofs_X(numSamplesX);
    vector<int> num_sample_dofs_per_procX(nprocs);
    BsinvX = RHSbasis ? NULL : new CAROM::Matrix(numSamplesX, rdimx, false);

    numSamplesV = std::min(fomH1size, numSamplesV);
    vector<int> sample_dofs_V(numSamplesV);
    vector<int> num_sample_dofs_per_procV(nprocs);
    BsinvV = new CAROM::Matrix(numSamplesV, RHSbasis ? rdimfv : rdimv, false);

    numSamplesE = std::min(fomL2size, numSamplesE);
    vector<int> sample_dofs_E(numSamplesE);
    vector<int> num_sample_dofs_per_procE(nprocs);
    BsinvE = new CAROM::Matrix(numSamplesE, RHSbasis ? rdimfe : rdime, false);
    if (rank == 0)
    {
        cout << "number of samples for position: " << numSamplesX << "\n";
        cout << "number of samples for velocity: " << numSamplesV << "\n";
        cout << "number of samples for energy  : " << numSamplesE << "\n";
    }

    // Perform DEIM, GNAT, or QDEIM to find sample DOF's.
    if (RHSbasis)
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
    else
    {
        CAROM::GNAT(basisX,
                    rdimx,
                    sample_dofs_X.data(),
                    num_sample_dofs_per_procX.data(),
                    *BsinvX,
                    rank,
                    nprocs,
                    numSamplesX);

        CAROM::GNAT(basisV,
                    rdimv,
                    sample_dofs_V.data(),
                    num_sample_dofs_per_procV.data(),
                    *BsinvV,
                    rank,
                    nprocs,
                    numSamplesV);

        CAROM::GNAT(basisE,
                    rdime,
                    sample_dofs_E.data(),
                    num_sample_dofs_per_procE.data(),
                    *BsinvE,
                    rank,
                    nprocs,
                    numSamplesE);
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

    MPI_Comm rom_com;
    int color = (rank != 0);
    const int status = MPI_Comm_split(MPI_COMM_WORLD, color, rank, &rom_com);
    MFEM_VERIFY(status == MPI_SUCCESS, "Construction of hyperreduction comm failed");

    // Construct sample mesh

    // This creates sample_pmesh, sp_H1_space, and sp_L2_space only on rank 0.
    CreateSampleMesh(*pmesh, H1_space, *H1FESpace, *L2FESpace, *(H1FESpace->FEColl()),
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

        if (RHSbasis)
        {
            BFvsp = new CAROM::Matrix(size_H1_sp, rdimfv, false);
            BFesp = new CAROM::Matrix(size_L2_sp, rdimfe, false);
        }
    }  // if (rank == 0)

    // This gathers only to rank 0.
#ifdef FULL_DOF_STENCIL
    const int NR = H1FESpace->GetVSize();
    GatherDistributedMatrixRows(*basisX, *basisE, rdimx, rdime, NR, *H1FESpace, *L2FESpace, st2sp, sprows, all_sprows, *BXsp, *BEsp);
    // TODO: this redundantly gathers BEsp again, but only once per simulation.
    GatherDistributedMatrixRows(*basisV, *basisE, rdimv, rdime, NR, *H1FESpace, *L2FESpace, st2sp, sprows, all_sprows, *BVsp, *BEsp);

    if (RHSbasis)
        GatherDistributedMatrixRows(*basisFv, *basisFe, rdimfv, rdimfe, NR, *H1FESpace, *L2FESpace, st2sp, sprows, all_sprows, *BFvsp, *BFesp);
#else
    GatherDistributedMatrixRows(*basisX, *basisE, rdimx, rdime, st2sp, sprows, all_sprows, *BXsp, *BEsp);
    // TODO: this redundantly gathers BEsp again, but only once per simulation.
    GatherDistributedMatrixRows(*basisV, *basisE, rdimv, rdime, st2sp, sprows, all_sprows, *BVsp, *BEsp);
    MFEM_VERIFY(!RHSbasis, "");
#endif

    delete sp_H1_space;
    delete sp_L2_space;
}

void ROM_Basis::ComputeReducedMatrices(bool sns1)
{
    if (RHSbasis && rank == 0)
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
        basisV = ReadBasisROM(rank, basename + "/" + ROMBasisName::V + std::to_string(window), tH1size, 0, rdimv);

    basisE = ReadBasisROM(rank, basename + "/" + ROMBasisName::E + std::to_string(window), tL2size, 0, rdime);

    if (useXV)
        basisX = basisV;
    else
        basisX = ReadBasisROM(rank, basename + "/" + ROMBasisName::X + std::to_string(window), tH1size, 0, rdimx);

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

        BasisGeneratorFinalSummary(&generator_XV, energyFraction_X, rdimx, false);
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

    if (!use_sns)
    {
        basisFv = ReadBasisROM(rank, basename + "/" + ROMBasisName::Fv + std::to_string(window), tH1size, 0, rdimfv);
        basisFe = ReadBasisROM(rank, basename + "/" + ROMBasisName::Fe + std::to_string(window), tL2size, 0, rdimfe);
    }
    else
    {
        basisFv = MultBasisROM(rank, basename + "/" + ROMBasisName::V + std::to_string(window), tH1size, 0, rdimfv, lhoper, 1);
        basisFe = MultBasisROM(rank, basename + "/" + ROMBasisName::E + std::to_string(window), tL2size, 0, rdimfe, lhoper, 2);
    }
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

void ROM_Basis::RestrictFromSampleMesh(const Vector &usp, Vector &u, const bool timeDerivative,
                                       const bool rhs_without_mass_matrix, const DenseMatrix *invMvROM,
                                       const DenseMatrix *invMeROM) const
{
    MFEM_VERIFY(u.Size() == SolutionSize(), "");  // rdimx + rdimv + rdime
    MFEM_VERIFY(usp.Size() == SolutionSizeSP(), "");  // (2*size_H1_sp) + size_L2_sp

    if (RK2AvgFormulation && RHSbasis)
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

    if (!RHSbasis)
    {
        BsinvX->transposeMult(*sX, *rX);
        for (int i=0; i<rdimx; ++i)
            u[i] = (*rX)(i);
    }

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
    MFEM_VERIFY(u.Size() == TotalSize(), "");
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
                           const int order_e, const int source, const bool visc, const double cfl,
                           const bool p_assembly, const double cg_tol, const int cg_max_iter,
                           const double ftz_tol, H1_FECollection *H1fec,
                           FiniteElementCollection *L2fec)
    : TimeDependentOperator(b->TotalSize()), operFOM(input.FOMoper), basis(b),
      rank(b->GetRank()), hyperreduce(input.hyperreduce), useGramSchmidt(input.GramSchmidt)
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
                visc, p_assembly, cg_tol, cg_max_iter, ftz_tol,
                H1fec->GetBasisType(), noMsolve, noMsolve);
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
    if (RHSbasis)
    {
        for (int i=0; i<rdimv; ++i)
            (*rV)(i) = x[rdimx + i];

        BsinvX->mult(*rV, *rX);
        for (int i=0; i<rdimx; ++i)
            y[i] = offsetInit ? (*rX)(i) + (*BX0)(i) : (*rX)(i);
    }
}

void ROM_Basis::HyperreduceRHS_V(Vector &v) const
{
    if (!RHSbasis) return;

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
    if (!RHSbasis) return;

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

    if (!RHSbasis)
    {
        BsinvX->write(basename + "/" + "BsinvX" + "_" + to_string(window));
    }
    BsinvV->write(basename + "/" + "BsinvV" + "_" + to_string(window));
    BsinvE->write(basename + "/" + "BsinvE" + "_" + to_string(window));
    BXsp->write(basename + "/" + "BXsp" + "_" + to_string(window));
    BVsp->write(basename + "/" + "BVsp" + "_" + to_string(window));
    BEsp->write(basename + "/" + "BEsp" + "_" + to_string(window));

    if (RHSbasis)
    {
        BFvsp->write(basename + "/" + "BFvsp" + "_" + to_string(window));
        BFesp->write(basename + "/" + "BFesp" + "_" + to_string(window));
    }

    spX->write(basename + "/" + "spX" + "_" + to_string(window));
    spV->write(basename + "/" + "spV" + "_" + to_string(window));
    spE->write(basename + "/" + "spE" + "_" + to_string(window));
    if (offsetInit)
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

    BsinvX = RHSbasis ? NULL : new CAROM::Matrix(numSamplesX, rdimx, false);
    BsinvV = new CAROM::Matrix(numSamplesV, RHSbasis ? rdimfv : rdimv, false);
    BsinvE = new CAROM::Matrix(numSamplesE, RHSbasis ? rdimfe : rdime, false);

    BXsp = new CAROM::Matrix(size_H1_sp, rdimx, false);
    BVsp = new CAROM::Matrix(size_H1_sp, rdimv, false);
    BEsp = new CAROM::Matrix(size_L2_sp, rdime, false);

    spX = new CAROM::Vector(size_H1_sp, false);
    spV = new CAROM::Vector(size_H1_sp, false);
    spE = new CAROM::Vector(size_L2_sp, false);

    sX = numSamplesX == 0 ? NULL : new CAROM::Vector(numSamplesX, false);
    sV = new CAROM::Vector(numSamplesV, false);
    sE = new CAROM::Vector(numSamplesE, false);

    if (!RHSbasis)
    {
        BsinvX->read(basename + "/" + "BsinvX" + "_" + to_string(window));
    }
    BsinvV->read(basename + "/" + "BsinvV" + "_" + to_string(window));
    BsinvE->read(basename + "/" + "BsinvE" + "_" + to_string(window));

    BXsp->read(basename + "/" + "BXsp" + "_" + to_string(window));
    BVsp->read(basename + "/" + "BVsp" + "_" + to_string(window));
    BEsp->read(basename + "/" + "BEsp" + "_" + to_string(window));

    if (RHSbasis)
    {
        BFvsp = new CAROM::Matrix(size_H1_sp, rdimfv, false);
        BFesp = new CAROM::Matrix(size_L2_sp, rdimfe, false);
        BFvsp->read(basename + "/" + "BFvsp" + "_" + to_string(window));
        BFesp->read(basename + "/" + "BFesp" + "_" + to_string(window));
    }

    spX->read(basename + "/" + "spX" + "_" + to_string(window));
    spV->read(basename + "/" + "spV" + "_" + to_string(window));
    spE->read(basename + "/" + "spE" + "_" + to_string(window));

    if (offsetInit)
    {
        initXsp = new CAROM::Vector(size_H1_sp, false);
        initVsp = new CAROM::Vector(size_H1_sp, false);
        initEsp = new CAROM::Vector(size_L2_sp, false);

        initXsp->read(basename + "/" + "initXsp" + "_" + to_string(window));
        initVsp->read(basename + "/" + "initVsp" + "_" + to_string(window));
        initEsp->read(basename + "/" + "initEsp" + "_" + to_string(window));
    }
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
        MFEM_VERIFY(false, "TODO");
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
        MFEM_VERIFY(false, "TODO");
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
            MFEM_VERIFY(false, "Invalid input");

        for (int k=0; k<dim; ++k)
        {
            ip += Mxj_sp[k]*xi_sp[k];
        }
    }
    else if (!hyperreduce)
    {
        MFEM_VERIFY(false, "TODO");
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
            MFEM_VERIFY(false, "Invalid input");

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
        MFEM_VERIFY(false, "TODO");
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
            MFEM_VERIFY(false, "Invalid input");

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
        MFEM_VERIFY(false, "TODO");
    }
}

void ROM_Operator::InducedGramSchmidtInitialize(Vector &S)
{
    if (useGramSchmidt && !sns1)
    {
        InducedGramSchmidt(1, S); // velocity
        InducedGramSchmidt(2, S); // energy
        MPI_Bcast(S.GetData(), S.Size(), MPI_DOUBLE, 0, basis->comm);
    }
    basis->ComputeReducedMatrices(sns1);
}

void ROM_Operator::InducedGramSchmidtFinalize(Vector &S, bool keep_data)
{
    if (useGramSchmidt && !sns1)
    {
        UndoInducedGramSchmidt(1, S, keep_data); // velocity
        UndoInducedGramSchmidt(2, S, keep_data); // energy
        MPI_Bcast(S.GetData(), S.Size(), MPI_DOUBLE, 0, basis->comm);
    }
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

void PrintNormsOfParGridFunctions(NormType normtype, const int rank, const std::string& name, ParGridFunction *f1, ParGridFunction *f2,
                                  const bool scalar)
{
    ConstantCoefficient zero(0.0);
    Vector zerov(3);
    zerov = 0.0;
    VectorConstantCoefficient vzero(zerov);

    double fomloc, romloc, diffloc;

    // TODO: why does ComputeL2Error call the local GridFunction version rather than the global ParGridFunction version?
    // Only f2->ComputeL2Error calls the ParGridFunction version.
    if (scalar)
    {
        switch(normtype)
        {
        case l1norm:
            fomloc = f1->ComputeL1Error(zero);
            romloc = f2->ComputeL1Error(zero);
            break;
        case l2norm:
            fomloc = f1->ComputeL2Error(zero);
            romloc = f2->ComputeL2Error(zero);
            break;
        case maxnorm:
            fomloc = f1->ComputeMaxError(zero);
            romloc = f2->ComputeMaxError(zero);
            break;
        }
    }
    else
    {
        switch(normtype)
        {
        case l1norm:
            fomloc = f1->ComputeL1Error(vzero);
            romloc = f2->ComputeL1Error(vzero);
            break;
        case l2norm:
            fomloc = f1->ComputeL2Error(vzero);
            romloc = f2->ComputeL2Error(vzero);
            break;
        case maxnorm:
            fomloc = f1->ComputeMaxError(vzero);
            romloc = f2->ComputeMaxError(vzero);
            break;
        }
    }

    *f1 -= *f2;  // works because GridFunction is derived from Vector

    if (scalar)
    {
        switch(normtype)
        {
        case l1norm:
            diffloc = f1->ComputeL1Error(zero);
            break;
        case l2norm:
            diffloc = f1->ComputeL2Error(zero);
            break;
        case maxnorm:
            diffloc = f1->ComputeMaxError(zero);
            break;
        }
    }
    else
    {
        switch(normtype)
        {
        case l1norm:
            diffloc = f1->ComputeL1Error(vzero);
            break;
        case l2norm:
            diffloc = f1->ComputeL2Error(vzero);
            break;
        case maxnorm:
            diffloc = f1->ComputeMaxError(vzero);
            break;
        }
    }

    double fomloc2 = fomloc*fomloc;
    double romloc2 = romloc*romloc;
    double diffloc2 = diffloc*diffloc;

    double fomglob2, romglob2, diffglob2;

    // TODO: is this right? The "loc" norms should be global, but they are not.
    MPI_Allreduce(&fomloc2, &fomglob2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&romloc2, &romglob2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&diffloc2, &diffglob2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    /*
    fomglob2 = fomloc2;
    romglob2 = romloc2;
    diffglob2 = diffloc2;
    */

    double FOMnorm = sqrt(fomglob2);
    double ROMnorm = sqrt(romglob2);
    double DIFFnorm = sqrt(diffglob2);
    double relDIFFnorm = sqrt(diffglob2)/sqrt(fomglob2);

    if (rank == 0)
    {
        switch(normtype)
        {
        case l1norm:
            cout << "L1 norm error:" << endl;
            break;
        case l2norm:
            cout << "L2 norm error:" << endl;
            break;
        case maxnorm:
            cout << "MAX norm error:" << endl;
            break;
        }

        cout << rank << ": " << name << " FOM norm " << FOMnorm << endl;
        cout << rank << ": " << name << " ROM norm " << ROMnorm << endl;
        cout << rank << ": " << name << " DIFF norm " << DIFFnorm << endl;
        cout << rank << ": " << name << " Rel. DIFF norm " << relDIFFnorm << endl;
    }

    char tmp[100];
    sprintf(tmp, ".%06d", rank);

    std::string fullname = name + "_norms" + tmp;

    std::ofstream ofs(fullname.c_str(), std::ofstream::out);
    ofs.precision(16);

    ofs << "FOM norm " << FOMnorm << endl;
    ofs << "ROM norm " << ROMnorm << endl;
    ofs << "DIFF norm " << DIFFnorm << endl;
    ofs << "Rel. DIFF norm " << relDIFFnorm << endl;

    ofs.close();

}

void PrintL2NormsOfParGridFunctions(const int rank, const std::string& name, ParGridFunction *f1, ParGridFunction *f2,
                                    const bool scalar)
{
    ConstantCoefficient zero(0.0);
    Vector zerov(3);
    zerov = 0.0;
    VectorConstantCoefficient vzero(zerov);

    double fomloc, romloc, diffloc;

    // TODO: why does ComputeL2Error call the local GridFunction version rather than the global ParGridFunction version?
    // Only f2->ComputeL2Error calls the ParGridFunction version.
    if (scalar)
    {
        fomloc = f1->ComputeL2Error(zero);
        romloc = f2->ComputeL2Error(zero);
    }
    else
    {
        fomloc = f1->ComputeL2Error(vzero);
        romloc = f2->ComputeL2Error(vzero);
    }

    *f1 -= *f2;  // works because GridFunction is derived from Vector

    if (scalar)
    {
        diffloc = f1->ComputeL2Error(zero);
    }
    else
    {
        diffloc = f1->ComputeL2Error(vzero);
    }

    double fomloc2 = fomloc*fomloc;
    double romloc2 = romloc*romloc;
    double diffloc2 = diffloc*diffloc;

    double fomglob2, romglob2, diffglob2;

    // TODO: is this right? The "loc" norms should be global, but they are not.
    MPI_Allreduce(&fomloc2, &fomglob2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&romloc2, &romglob2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&diffloc2, &diffglob2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    /*
    fomglob2 = fomloc2;
    romglob2 = romloc2;
    diffglob2 = diffloc2;
    */

    cout << rank << ": " << name << " FOM norm " << sqrt(fomglob2) << endl;
    cout << rank << ": " << name << " ROM norm " << sqrt(romglob2) << endl;
    cout << rank << ": " << name << " DIFF norm " << sqrt(diffglob2) << endl;
    cout << rank << ": " << name << " Rel. DIFF norm " << sqrt(diffglob2)/sqrt(fomglob2) << endl;
}
