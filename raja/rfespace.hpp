// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
#ifndef MFEM_RAJA_FESPACE
#define MFEM_RAJA_FESPACE

namespace mfem {

// ***************************************************************************
// * RajaRestrictionOperator
// ***************************************************************************
class RajaRestrictionOperator : public Operator {
 protected:
  int entries;
  RajaArray<int> indices;
 public:
  RajaRestrictionOperator(const int h, const int w, RajaArray<int> i):
    Operator(h,w) {
    entries = i.size()>>1;
    indices = i;
  }
  virtual void Mult(const RajaVector& x, RajaVector& y) const {
    kExtractSubVector(entries, indices.ptr(), x, y);
  }
};

// ***************************************************************************
// * RajaProlongationOperator
// ***************************************************************************
class RajaProlongationOperator : public Operator {
 protected:
  const Operator* pmat = NULL;
 public:
  RajaProlongationOperator(const Operator* Op):
    Operator(Op->Height(), Op->Width()), pmat(Op) {}
  virtual void Mult(const RajaVector& x, RajaVector& y) const {
    const Vector hostX(x.ptr(), x.Size());
    Vector hostY(y.ptr(), y.Size());
    pmat->Mult(hostX, hostY);
  }
  virtual void MultTranspose(const RajaVector& x, RajaVector& y) const {
    const Vector hostX(x.ptr(), x.Size());
    Vector hostY(y.ptr(), y.Size());
    //mfem::ConformingProlongationOperator::MultTranspose
    pmat->MultTranspose(hostX, hostY);
  }
};

// ***************************************************************************
// * RajaFiniteElementSpace
// ***************************************************************************
class RajaFiniteElementSpace : public ParFiniteElementSpace {
 private:
  int globalDofs, localDofs;
  RajaArray<int> offsets;
  RajaArray<int> indices;
  RajaArray<int> map;
  Operator* restrictionOp, *prolongationOp;
 public:
  RajaFiniteElementSpace(Mesh* mesh,
                         const FiniteElementCollection* fec,
                         const int vdim_ = 1,
                         Ordering::Type ordering_ = Ordering::byNODES);
  ~RajaFiniteElementSpace();
  // *************************************************************************
  bool hasTensorBasis() const;
  int GetLocalDofs() const { return localDofs; }
  const Operator* GetRestrictionOperator() { return restrictionOp; }
  const Operator* GetProlongationOperator() { return prolongationOp; }
  const RajaArray<int> GetLocalToGlobalMap() const { return map; }
  // *************************************************************************
  void GlobalToLocal(const RajaVector&, RajaVector&) const;
  void LocalToGlobal(const RajaVector&, RajaVector&) const;
};
}

#endif
