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
#ifndef LAGHOS_RAJA_FESPACE
#define LAGHOS_RAJA_FESPACE

namespace mfem {
  
// ***************************************************************************
// * RajaRestrictionOperator
// ***************************************************************************
class RajaRestrictionOperator : public RajaOperator {
 protected:
  int entries;
  const RajaArray<int> *indices;
 public:
  RajaRestrictionOperator(const int h, const int w,
                          const RajaArray<int> *idx):
    RajaOperator(h,w),
    entries(idx->size()>>1),
    indices(idx){} 
  void Mult(const RajaVector& x, RajaVector& y) const ;
};


// *****************************************************************************
// * RajaConformingProlongationOperator
// *****************************************************************************
class RajaConformingProlongationOperator : public RajaOperator{
protected:
  Array<int> external_ldofs;
  GroupCommunicator &gc;
public:
  RajaConformingProlongationOperator(ParFiniteElementSpace &pfes);
  //void Mult(const RajaVector &x, RajaVector &y) const;
  //void MultTranspose(const RajaVector &x, RajaVector &y) const;  
  void h_Mult(const Vector &x, Vector &y) const;
  void h_MultTranspose(const Vector &x, Vector &y) const;
};

  
// ***************************************************************************
// * RajaProlongationOperator
// ***************************************************************************
class RajaProlongationOperator : public RajaOperator {
 protected:
  const RajaConformingProlongationOperator* pmat = NULL;
 public:
  RajaProlongationOperator(const RajaConformingProlongationOperator* Op);
  void Mult(const RajaVector& x, RajaVector& y) const;
  void MultTranspose(const RajaVector& x, RajaVector& y) const ;
};
  
  
// ***************************************************************************
// * RajaFiniteElementSpace
// ***************************************************************************
class RajaFiniteElementSpace : public ParFiniteElementSpace {
 private:
  int globalDofs, localDofs;
  RajaArray<int> offsets;
  RajaArray<int> indices, *reorderIndices;
  RajaArray<int> map;
  RajaOperator *restrictionOp, *prolongationOp;
 public:
  RajaFiniteElementSpace(Mesh* mesh,
                         const FiniteElementCollection* fec,
                         const int vdim_ = 1,
                         Ordering::Type ordering_ = Ordering::byNODES);
  ~RajaFiniteElementSpace();
  // *************************************************************************
  bool hasTensorBasis() const;
  int GetLocalDofs() const { return localDofs; }
  const RajaOperator* GetRestrictionOperator() { return restrictionOp; }
  const RajaOperator* GetProlongationOperator() { return prolongationOp; }
  const RajaArray<int>& GetLocalToGlobalMap() const { return map; }
  // *************************************************************************
  void GlobalToLocal(const RajaVector&, RajaVector&) const;
  void LocalToGlobal(const RajaVector&, RajaVector&) const;
};

}

#endif // LAGHOS_RAJA_FESPACE
