#include "laghos_rom.hpp"

using namespace std;

void ROM_Sampler::SampleSolution(const double t, const double dt, Vector const& S)
{
  SetStateVariables(S);
  SetStateVariableRates(dt);
  
  const int tmp = generator_X->getNumBasisTimeIntervals();
  
  const bool sampleX = generator_X->isNextSample(t);

  if (sampleX)
    {
      if (rank == 0)
	{
	  cout << "X taking sample at t " << t << endl;
	}
      
      generator_X->takeSample(X.GetData(), t, dt);
      generator_X->computeNextSampleTime(X.GetData(), dXdt.GetData(), t);
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

void BasisGeneratorFinalSummary(CAROM::SVDBasisGenerator* bg)
{
  const int rom_dim = bg->getSpatialBasis()->numColumns();
  cout << "ROM dimension = " << rom_dim << endl;

  const CAROM::Matrix* sing_vals = bg->getSingularValues();
            
  cout << "Singular Values:" << endl;
  for (int sv = 0; sv < sing_vals->numColumns(); ++sv) {
    cout << (*sing_vals)(sv, sv) << endl;
  }
}

void ROM_Sampler::Finalize(const double t, const double dt, Vector const& S)
{
  SetStateVariables(S);
  
  generator_X->takeSample(X.GetData(), t, dt);
  generator_X->endSamples();

  generator_V->takeSample(V.GetData(), t, dt);
  generator_V->endSamples();

  generator_E->takeSample(E.GetData(), t, dt);
  generator_E->endSamples();

  if (rank == 0)
    {
      cout << "X basis summary output" << endl;
      BasisGeneratorFinalSummary(generator_X);

      cout << "V basis summary output" << endl;
      BasisGeneratorFinalSummary(generator_V);

      cout << "E basis summary output" << endl;
      BasisGeneratorFinalSummary(generator_E);
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
		     int & dimX, int & dimV, int & dimE,
		     const bool staticSVD_)
  : comm(comm_), tH1size(H1FESpace->GetTrueVSize()), tL2size(L2FESpace->GetTrueVSize()),
    H1size(H1FESpace->GetVSize()), L2size(L2FESpace->GetVSize()),
    gfH1(H1FESpace), gfL2(L2FESpace), 
    rdimx(dimX), rdimv(dimV), rdime(dimE), staticSVD(staticSVD_)
{
  MPI_Comm_size(comm, &nprocs);
  MPI_Comm_rank(comm, &rank);

  Array<int> osH1(nprocs+1);
  Array<int> osL2(nprocs+1);
  MPI_Allgather(&tH1size, 1, MPI_INT, osH1.GetData(), 1, MPI_INT, comm);
  MPI_Allgather(&tL2size, 1, MPI_INT, osL2.GetData(), 1, MPI_INT, comm);

  for (int i=nprocs-1; i>=0; --i)
    {
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
  
  ReadSolutionBases();

  rX = new CAROM::Vector(rdimx, false);
  rV = new CAROM::Vector(rdimv, false);
  rE = new CAROM::Vector(rdime, false);

  dimX = rdimx;
  dimV = rdimv;
  dimE = rdime;
}

void ROM_Basis::ReadSolutionBases()
{
  /*
  basisX = ReadBasisROM(rank, ROMBasisName::X, H1size, (staticSVD ? rowOffsetH1 : 0), rdimx);
  basisV = ReadBasisROM(rank, ROMBasisName::V, H1size, (staticSVD ? rowOffsetH1 : 0), rdimv);
  basisE = ReadBasisROM(rank, ROMBasisName::E, L2size, (staticSVD ? rowOffsetL2 : 0), rdime);
  */
  
  basisX = ReadBasisROM(rank, ROMBasisName::X, tH1size, 0, rdimx);
  basisV = ReadBasisROM(rank, ROMBasisName::V, tH1size, 0, rdimv);
  basisE = ReadBasisROM(rank, ROMBasisName::E, tL2size, 0, rdime);
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
    mfH1[i] = (*fH1)(i);

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

void ROM_Operator::Mult(const Vector &x, Vector &y) const
{
  basis->LiftROMtoFOM(x, fx);
  operFOM->Mult(fx, fy);
  basis->ProjectFOMtoROM(fy, y);
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
}
