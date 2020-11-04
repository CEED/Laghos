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
/////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018,2019 Advanced Micro Devices, Inc.
/////////////////////////////////////////////////////////////////////////////////
#ifndef LAGHOS_HIP_FESPACE
#define LAGHOS_HIP_FESPACE

namespace mfem
{

// ***************************************************************************
// * HipFiniteElementSpace
//  **************************************************************************
class HipFiniteElementSpace : public ParFiniteElementSpace
{
private:
   int globalDofs, localDofs;
   HipArray<int> offsets;
   HipArray<int> indices, *reorderIndices;
   HipArray<int> map;
   HipOperator *restrictionOp, *prolongationOp;
public:
   HipFiniteElementSpace(Mesh* mesh,
                          const FiniteElementCollection* fec,
                          const int vdim_ = 1,
                          Ordering::Type ordering_ = Ordering::byNODES);
   ~HipFiniteElementSpace();
   // *************************************************************************
   bool hasTensorBasis() const;
   int GetLocalDofs() const { return localDofs; }
   const HipOperator* GetRestrictionOperator() { return restrictionOp; }
   const HipOperator* GetProlongationOperator() { return prolongationOp; }
   const HipArray<int>& GetLocalToGlobalMap() const { return map; }
   // *************************************************************************
   void GlobalToLocal(const HipVector&, HipVector&) const;
   void LocalToGlobal(const HipVector&, HipVector&) const;
};

} // mfem

#endif // LAGHOS_HIP_FESPACE
