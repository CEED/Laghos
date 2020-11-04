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
#ifndef LAGHOS_HIP_BILINEARFORM
#define LAGHOS_HIP_BILINEARFORM

namespace mfem
{

// ***************************************************************************
// * HipIntegratorType
// ***************************************************************************
enum HipIntegratorType
{
   DomainIntegrator       = 0,
   BoundaryIntegrator     = 1,
   InteriorFaceIntegrator = 2,
   BoundaryFaceIntegrator = 3,
};

class HipIntegrator;

// ***************************************************************************
// * HipBilinearForm
// ***************************************************************************
class HipBilinearForm : public HipOperator
{
   friend class HipIntegrator;
protected:
   typedef std::vector<HipIntegrator*> IntegratorVector;
   mutable Mesh* mesh;
   mutable HipFiniteElementSpace* trialFes;
   mutable HipFiniteElementSpace* testFes;
   IntegratorVector integrators;
   mutable HipVector localX, localY;
public:
   HipBilinearForm(HipFiniteElementSpace*);
   ~HipBilinearForm();
   Mesh& GetMesh() const { return *mesh; }
   HipFiniteElementSpace& GetTrialFESpace() const { return *trialFes;}
   HipFiniteElementSpace& GetTestFESpace() const { return *testFes;}
   // *************************************************************************
   void AddDomainIntegrator(HipIntegrator*);
   void AddBoundaryIntegrator(HipIntegrator*);
   void AddInteriorFaceIntegrator(HipIntegrator*);
   void AddBoundaryFaceIntegrator(HipIntegrator*);
   void AddIntegrator(HipIntegrator*, const HipIntegratorType);
   // *************************************************************************
   virtual void Assemble();
   void FormLinearSystem(const Array<int>& constraintList,
                         HipVector& x, HipVector& b,
                         HipOperator*& Aout,
                         HipVector& X, HipVector& B,
                         int copy_interior = 0);
   void FormOperator(const Array<int>& constraintList, HipOperator*& Aout);
   void InitRHS(const Array<int>& constraintList,
                const HipVector& x, const HipVector& b,
                HipOperator* Aout,
                HipVector& X, HipVector& B,
                int copy_interior = 0);
   virtual void Mult(const HipVector& x, HipVector& y) const;
   virtual void MultTranspose(const HipVector& x, HipVector& y) const;
   void RecoverFEMSolution(const HipVector&, const HipVector&, HipVector&);
};


// ***************************************************************************
// * Constrained Operator
// ***************************************************************************
class HipConstrainedOperator : public HipOperator
{
protected:
   HipOperator *A;
   bool own_A;
   HipArray<int> constraintList;
   int constraintIndices;
   mutable HipVector z, w;
public:
   HipConstrainedOperator(HipOperator*, const Array<int>&, bool = false);
   void Setup(HipOperator*, const Array<int>&, bool = false);
   void EliminateRHS(const HipVector&, HipVector&) const;
   virtual void Mult(const HipVector&, HipVector&) const;
   virtual ~HipConstrainedOperator() {}
};

} // mfem

#endif // LAGHOS_HIP_BILINEARFORM
