// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
#ifndef LAGHOS_CUDA_BILINEARFORM
#define LAGHOS_CUDA_BILINEARFORM

namespace mfem
{

// ***************************************************************************
// * CudaIntegratorType
// ***************************************************************************
enum CudaIntegratorType
{
   DomainIntegrator       = 0,
   BoundaryIntegrator     = 1,
   InteriorFaceIntegrator = 2,
   BoundaryFaceIntegrator = 3,
};

class CudaIntegrator;

// ***************************************************************************
// * CudaBilinearForm
// ***************************************************************************
class CudaBilinearForm : public CudaOperator
{
   friend class CudaIntegrator;
protected:
   typedef std::vector<CudaIntegrator*> IntegratorVector;
   mutable Mesh* mesh;
   mutable CudaFiniteElementSpace* trialFes;
   mutable CudaFiniteElementSpace* testFes;
   IntegratorVector integrators;
   mutable CudaVector localX, localY;
public:
   CudaBilinearForm(CudaFiniteElementSpace*);
   ~CudaBilinearForm();
   Mesh& GetMesh() const { return *mesh; }
   CudaFiniteElementSpace& GetTrialFESpace() const { return *trialFes;}
   CudaFiniteElementSpace& GetTestFESpace() const { return *testFes;}
   // *************************************************************************
   void AddDomainIntegrator(CudaIntegrator*);
   void AddBoundaryIntegrator(CudaIntegrator*);
   void AddInteriorFaceIntegrator(CudaIntegrator*);
   void AddBoundaryFaceIntegrator(CudaIntegrator*);
   void AddIntegrator(CudaIntegrator*, const CudaIntegratorType);
   // *************************************************************************
   virtual void Assemble();
   void FormLinearSystem(const Array<int>& constraintList,
                         CudaVector& x, CudaVector& b,
                         CudaOperator*& Aout,
                         CudaVector& X, CudaVector& B,
                         int copy_interior = 0);
   void FormOperator(const Array<int>& constraintList, CudaOperator*& Aout);
   void InitRHS(const Array<int>& constraintList,
                const CudaVector& x, const CudaVector& b,
                CudaOperator* Aout,
                CudaVector& X, CudaVector& B,
                int copy_interior = 0);
   virtual void Mult(const CudaVector& x, CudaVector& y) const;
   virtual void MultTranspose(const CudaVector& x, CudaVector& y) const;
   void RecoverFEMSolution(const CudaVector&, const CudaVector&, CudaVector&);
};


// ***************************************************************************
// * Constrained Operator
// ***************************************************************************
class CudaConstrainedOperator : public CudaOperator
{
protected:
   CudaOperator *A;
   bool own_A;
   CudaArray<int> constraintList;
   int constraintIndices;
   mutable CudaVector z, w;
public:
   CudaConstrainedOperator(CudaOperator*, const Array<int>&, bool = false);
   void Setup(CudaOperator*, const Array<int>&, bool = false);
   void EliminateRHS(const CudaVector&, CudaVector&) const;
   virtual void Mult(const CudaVector&, CudaVector&) const;
   virtual ~CudaConstrainedOperator() {}
};

} // mfem

#endif // LAGHOS_CUDA_BILINEARFORM
