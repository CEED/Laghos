#include "laghos_rom.hpp"

#include "DEIM.h"
#include "SampleMesh.hpp"


using namespace std;

void ROM_Sampler::SampleSolution(const double t, const double dt, Vector const& S)
{
    SetStateVariables(S);
    SetStateVariableRates(dt);

    const bool sampleX = generator_X->isNextSample(t);

    if (sampleX)
    {
        if (rank == 0)
        {
            cout << "X taking sample at t " << t << endl;
        }

        if (offsetXinit)
        {
            for (int i=0; i<tH1size; ++i)
            {
                Xdiff[i] = X[i] - (*initX)(i);
            }

            generator_X->takeSample(Xdiff.GetData(), t, dt);
            generator_X->computeNextSampleTime(Xdiff.GetData(), dXdt.GetData(), t);
        }
        else
        {
            generator_X->takeSample(X.GetData(), t, dt);
            generator_X->computeNextSampleTime(X.GetData(), dXdt.GetData(), t);
        }
    }

    const bool sampleV = generator_V->isNextSample(t);

    if (sampleV)
    {
        if (rank == 0)
        {
            cout << "V taking sample at t " << t << endl;
        }

        generator_V->takeSample(V.GetData(), t, dt);
        generator_V->computeNextSampleTime(V.GetData(), dVdt.GetData(), t);
    }

    const bool sampleE = generator_E->isNextSample(t);

    if (sampleE)
    {
        if (rank == 0)
        {
            cout << "E taking sample at t " << t << endl;
        }

        generator_E->takeSample(E.GetData(), t, dt);
        generator_E->computeNextSampleTime(E.GetData(), dEdt.GetData(), t);
    }
}

void BasisGeneratorFinalSummary(CAROM::SVDBasisGenerator* bg, const double energyFraction)
{
    const int rom_dim = bg->getSpatialBasis()->numColumns();
    cout << "ROM dimension = " << rom_dim << endl;

    const CAROM::Matrix* sing_vals = bg->getSingularValues();

    MFEM_VERIFY(rom_dim == sing_vals->numColumns(), "");

    cout << "Singular Values:" << endl;

    double sum = 0.0;
    for (int sv = 0; sv < sing_vals->numColumns(); ++sv) {
        cout << (*sing_vals)(sv, sv) << endl;
        sum += (*sing_vals)(sv, sv);
    }

    double partialSum = 0.0;
    int cutoff = 0;
    for (int sv = 0; sv < sing_vals->numColumns(); ++sv) {
        partialSum += (*sing_vals)(sv, sv);
        if (partialSum / sum > energyFraction)
        {
            cutoff = sv;
            break;
        }
    }

    cout << "Take first " << cutoff+1 << " of " << sing_vals->numColumns() << " basis vectors" << endl;
}

void ROM_Sampler::Finalize(const double t, const double dt, Vector const& S)
{
    SetStateVariables(S);

    if (offsetXinit)
    {
        for (int i=0; i<tH1size; ++i)
        {
            Xdiff[i] = X[i] - (*initX)(i);
        }

        generator_X->takeSample(Xdiff.GetData(), t, dt);
    }
    else
        generator_X->takeSample(X.GetData(), t, dt);

    generator_X->endSamples();

    generator_V->takeSample(V.GetData(), t, dt);
    generator_V->endSamples();

    generator_E->takeSample(E.GetData(), t, dt);
    generator_E->endSamples();

    if (rank == 0)
    {
        cout << "X basis summary output" << endl;
        BasisGeneratorFinalSummary(generator_X, energyFraction);

        cout << "V basis summary output" << endl;
        BasisGeneratorFinalSummary(generator_V, energyFraction);

        cout << "E basis summary output" << endl;
        BasisGeneratorFinalSummary(generator_E, energyFraction);
    }

    delete generator_X;
    delete generator_V;
    delete generator_E;
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

ROM_Basis::ROM_Basis(MPI_Comm comm_, ParFiniteElementSpace *H1FESpace, ParFiniteElementSpace *L2FESpace,
                     int & dimX, int & dimV, int & dimE, int nsamx, int nsamv, int nsame,
                     const bool staticSVD_, const bool hyperreduce_, const bool useXoffset, const int window)
    : comm(comm_), tH1size(H1FESpace->GetTrueVSize()), tL2size(L2FESpace->GetTrueVSize()),
      H1size(H1FESpace->GetVSize()), L2size(L2FESpace->GetVSize()),
      gfH1(H1FESpace), gfL2(L2FESpace),
      rdimx(dimX), rdimv(dimV), rdime(dimE),
      numSamplesX(nsamx), numSamplesV(nsamv), numSamplesE(nsame),
      staticSVD(staticSVD_), hyperreduce(hyperreduce_), offsetXinit(useXoffset)
{
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    Array<int> osH1(nprocs+1);
    Array<int> nH1(nprocs);
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

    ReadSolutionBases(window);

    rX = new CAROM::Vector(rdimx, false);
    rV = new CAROM::Vector(rdimv, false);
    rE = new CAROM::Vector(rdime, false);

    dimX = rdimx;
    dimV = rdimv;
    dimE = rdime;

    if (hyperreduce)
    {
        if(rank == 0) cout << "start preprocessing hyper-reduction\n";
        StopWatch preprocessHyperreductionTymer;
        preprocessHyperreductionTymer.Start();
        SetupHyperreduction(H1FESpace, L2FESpace, nH1);
        preprocessHyperreductionTymer.Stop();
        if(rank == 0) cout << "Elapsed time for hyper-reduction preprocessing: " << preprocessHyperreductionTymer.RealTime() << " sec\n";
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

void ROM_Basis::SetupHyperreduction(ParFiniteElementSpace *H1FESpace, ParFiniteElementSpace *L2FESpace, Array<int>& nH1)
{
    ParMesh *pmesh = H1FESpace->GetParMesh();

    const int fomH1size = H1FESpace->GlobalTrueVSize();
    const int fomL2size = L2FESpace->GlobalTrueVSize();

    const int nsamp = 35;
    const int overSample = 100;

    numSamplesX = std::min(fomH1size, numSamplesX);
    vector<int> sample_dofs_X(numSamplesX);
    vector<int> num_sample_dofs_per_procX(nprocs);
    BsinvX = new CAROM::Matrix(numSamplesX, rdimx, false);

    numSamplesV = std::min(fomH1size, numSamplesV);
    vector<int> sample_dofs_V(numSamplesV);
    vector<int> num_sample_dofs_per_procV(nprocs);
    BsinvV = new CAROM::Matrix(numSamplesV, rdimv, false);

    numSamplesE = std::min(fomL2size, numSamplesE);
    vector<int> sample_dofs_E(numSamplesE);
    vector<int> num_sample_dofs_per_procE(nprocs);
    BsinvE = new CAROM::Matrix(numSamplesE, rdime, false);
    if(rank == 0)
    {
        cout << "number of samples for position: " << numSamplesX << "\n";
        cout << "number of samples for velocity: " << numSamplesV << "\n";
        cout << "number of samples for energy  : " << numSamplesE << "\n";
    }

    // Perform DEIM or GNAT to find sample DOF's.
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

    vector<int> sprows;
    vector<int> all_sprows;

    vector<int> s2sp;   // mapping from sample dofs in original mesh (s) to stencil dofs in sample mesh (s+), for both F and E

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
            mesh_name << "smesh." << setfill('0') << setw(6) << rank;

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

        sX = new CAROM::Vector(numSamplesX, false);
        sV = new CAROM::Vector(numSamplesV, false);
        sE = new CAROM::Vector(numSamplesE, false);
    }  // if (rank == 0)

    // This gathers only to rank 0.
#ifdef FULL_DOF_STENCIL
    const int NR = H1FESpace->GetVSize();
    GatherDistributedMatrixRows(*basisX, *basisE, rdimx, rdime, NR, *H1FESpace, *L2FESpace, st2sp, sprows, all_sprows, *BXsp, *BEsp);
    // TODO: this redundantly gathers BEsp again, but only once per simulation.
    GatherDistributedMatrixRows(*basisV, *basisE, rdimv, rdime, NR, *H1FESpace, *L2FESpace, st2sp, sprows, all_sprows, *BVsp, *BEsp);
#else
    GatherDistributedMatrixRows(*basisX, *basisE, rdimx, rdime, st2sp, sprows, all_sprows, *BXsp, *BEsp);
    // TODO: this redundantly gathers BEsp again, but only once per simulation.
    GatherDistributedMatrixRows(*basisV, *basisE, rdimv, rdime, st2sp, sprows, all_sprows, *BVsp, *BEsp);
#endif

    if (offsetXinit)
    {
        MFEM_VERIFY(false, "TODO: E needs an offset as well");

        initX = new CAROM::Vector(tH1size, true);
        initX->read("initX");

        CAROM::Matrix FOMX0(tH1size, 1, true);

        for (int i=0; i<tH1size; ++i)
        {
            FOMX0(i,0) = (*initX)(i);
        }

        CAROM::Matrix FOMzero(tH1size, 1, true);
        FOMzero = 0.0;

        CAROM::Matrix spX0mat(rank == 0 ? size_H1_sp : 1, 1, false);
        CAROM::Matrix spzero(rank == 0 ? size_H1_sp : 1, 1, false);

#ifdef FULL_DOF_STENCIL
        GatherDistributedMatrixRows(FOMX0, FOMzero, 1, 1, NR, *H1FESpace, *L2FESpace, st2sp, sprows, all_sprows, spX0mat, spzero);
#else
        GatherDistributedMatrixRows(FOMX0, FOMzero, 1, 1, st2sp, sprows, all_sprows, spX0mat, spzero);
#endif

        if (rank == 0)
        {
            initXsp = new CAROM::Vector(size_H1_sp, false);
            for (int i=0; i<size_H1_sp; ++i)
            {
                (*initXsp)(i) = spX0mat(i,0);
            }
        }
    }

    delete sp_H1_space;
    delete sp_L2_space;
}

void ROM_Basis::ApplyEssentialBCtoInitXsp(Array<int> const& ess_tdofs)
{
    if (rank != 0 || !offsetXinit)
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
    /*
    basisX = ReadBasisROM(rank, ROMBasisName::X, H1size, (staticSVD ? rowOffsetH1 : 0), rdimx);
    basisV = ReadBasisROM(rank, ROMBasisName::V, H1size, (staticSVD ? rowOffsetH1 : 0), rdimv);
    basisE = ReadBasisROM(rank, ROMBasisName::E, L2size, (staticSVD ? rowOffsetL2 : 0), rdime);
    */

    basisX = ReadBasisROM(rank, ROMBasisName::X + std::to_string(window), tH1size, 0, rdimx);
    basisV = ReadBasisROM(rank, ROMBasisName::V + std::to_string(window), tH1size, 0, rdimv);
    basisE = ReadBasisROM(rank, ROMBasisName::E + std::to_string(window), tL2size, 0, rdime);
}

// f is a full vector, not a true vector
void ROM_Basis::ProjectFOMtoROM(Vector const& f, Vector & r)
{
    MFEM_VERIFY(r.Size() == rdimx + rdimv + rdime, "");
    MFEM_VERIFY(f.Size() == (2*H1size) + L2size, "");

    for (int i=0; i<H1size; ++i)
        gfH1[i] = f[i];

    gfH1.GetTrueDofs(mfH1);

    for (int i=0; i<tH1size; ++i)
        (*fH1)(i) = mfH1[i];

    basisX->transposeMult(*fH1, *rX);

    for (int i=0; i<H1size; ++i)
        gfH1[i] = f[H1size + i];

    gfH1.GetTrueDofs(mfH1);

    for (int i=0; i<tH1size; ++i)
        (*fH1)(i) = mfH1[i];

    basisV->transposeMult(*fH1, *rV);

    for (int i=0; i<L2size; ++i)
        gfL2[i] = f[(2*H1size) + i];

    gfL2.GetTrueDofs(mfL2);

    for (int i=0; i<tL2size; ++i)
        (*fL2)(i) = mfL2[i];

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
        mfH1[i] = offsetXinit ? (*initX)(i) + (*fH1)(i) : (*fH1)(i);

    gfH1.SetFromTrueDofs(mfH1);

    for (int i=0; i<H1size; ++i)
        f[i] = gfH1[i];

    basisV->mult(*rV, *fH1);

    for (int i=0; i<tH1size; ++i)
        mfH1[i] = (*fH1)(i);

    gfH1.SetFromTrueDofs(mfH1);

    for (int i=0; i<H1size; ++i)
        f[H1size + i] = gfH1[i];

    basisE->mult(*rE, *fL2);

    for (int i=0; i<tL2size; ++i)
        mfL2[i] = (*fL2)(i);

    gfL2.SetFromTrueDofs(mfL2);

    for (int i=0; i<L2size; ++i)
        f[(2*H1size) + i] = gfL2[i];
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
            usp[i] = offsetXinit ? (*initXsp)(i) + (*spX)(i) : (*spX)(i);
            usp[size_H1_sp + i] = (*spV)(i);
        }

        for (int i=0; i<size_L2_sp; ++i)
        {
            //usp[(2*size_H1_sp) + i] = std::max((*spE)(i), 0.0);
            usp[(2*size_H1_sp) + i] = (*spE)(i);
        }
    }
}

void ROM_Basis::RestrictFromSampleMesh(const Vector &usp, Vector &u) const
{
    MFEM_VERIFY(u.Size() == SolutionSize(), "");  // rdimx + rdimv + rdime
    MFEM_VERIFY(usp.Size() == SolutionSizeSP(), "");  // (2*size_H1_sp) + size_L2_sp

    // Select entries out of usp on the sample mesh.

    // Note that s2sp_X maps from X samples to sample mesh H1 dofs, and similarly for V and E.

    for (int i=0; i<numSamplesX; ++i)
        (*sX)(i) = usp[s2sp_X[i]];

    for (int i=0; i<numSamplesV; ++i)
        (*sV)(i) = usp[size_H1_sp + s2sp_V[i]];

    for (int i=0; i<numSamplesE; ++i)
        (*sE)(i) = usp[(2*size_H1_sp) + s2sp_E[i]];

    BsinvX->transposeMult(*sX, *rX);
    BsinvV->transposeMult(*sV, *rV);
    BsinvE->transposeMult(*sE, *rE);

    for (int i=0; i<rdimx; ++i)
        u[i] = (*rX)(i);

    for (int i=0; i<rdimv; ++i)
        u[rdimx + i] = (*rV)(i);

    for (int i=0; i<rdime; ++i)
        u[rdimx + rdimv + i] = (*rE)(i);
}

ROM_Operator::ROM_Operator(hydrodynamics::LagrangianHydroOperator *lhoper, ROM_Basis *b,
                           FunctionCoefficient& rho_coeff, FunctionCoefficient& mat_coeff,
                           const int order_e, const int source, const bool visc, const double cfl,
                           const bool p_assembly, const double cg_tol, const int cg_max_iter,
                           const double ftz_tol, const bool hyperreduce_, H1_FECollection *H1fec,
                           FiniteElementCollection *L2fec)
    : TimeDependentOperator(b->TotalSize()), operFOM(lhoper), basis(b),
      rank(b->GetRank()), hyperreduce(hyperreduce_)
{
    MFEM_VERIFY(lhoper->Height() == lhoper->Width(), "");

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

        operSP = new hydrodynamics::LagrangianHydroOperator(S.Size(), *H1FESpaceSP, *L2FESpaceSP,
                ess_tdofs, rho, source, cfl, mat_gf_coeff,
                visc, p_assembly, cg_tol, cg_max_iter, ftz_tol,
                H1fec->GetBasisType());
    }
    else if (!hyperreduce)
    {
        fx.SetSize(lhoper->Height());
        fy.SetSize(lhoper->Height());
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

            // TODO: is this necessary? Does the call to UpdateMesh in operSP->Mult accomplish this anyway?
            {   // update mesh
                for (int i=0; i<Vsize_h1sp; ++i)
                    (*xsp_gf)[i] = fx[i];

                spmesh->NewNodes(*xsp_gf, false);
            }

            operSP->Mult(fx, fy);
            basis->RestrictFromSampleMesh(fy, y);

            operSP->ResetQuadratureData();
        }

        MPI_Bcast(y.GetData(), y.Size(), MPI_DOUBLE, 0, basis->comm);
        MPI_Bcast(&dt_est_SP, 1, MPI_DOUBLE, 0, basis->comm);
    }
    else
    {
        basis->LiftROMtoFOM(x, fx);
        operFOM->Mult(fx, fy);
        basis->ProjectFOMtoROM(fy, y);
    }
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

    cout << rank << ": " << name << " FOM norm " << sqrt(fomglob2) << endl;
    cout << rank << ": " << name << " ROM norm " << sqrt(romglob2) << endl;
    cout << rank << ": " << name << " DIFF norm " << sqrt(diffglob2) << endl;
    cout << rank << ": " << name << " Rel. DIFF norm " << sqrt(diffglob2)/sqrt(fomglob2) << endl;
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
