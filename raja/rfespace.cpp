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
#include "raja.hpp"

namespace mfem {

// ***************************************************************************
// * RajaFiniteElementSpace
// ***************************************************************************
RajaFiniteElementSpace::RajaFiniteElementSpace(Mesh* mesh,
                                               const FiniteElementCollection* fec,
                                               const int vdim_,
                                               Ordering::Type ordering_)
  :ParFiniteElementSpace(dynamic_cast<ParMesh*>(mesh),fec,vdim_,ordering_),
   globalDofs(GetNDofs()),
   localDofs(GetFE(0)->GetDof()),
   offsets(globalDofs+1),
   indices(localDofs, GetNE()),  
   map(localDofs, GetNE()) {
  const FiniteElement& fe = *GetFE(0);
  const TensorBasisElement* el = dynamic_cast<const TensorBasisElement*>(&fe);
  const Table& e2dTable = GetElementToDofTable();
  const int* elementMap = e2dTable.GetJ();
  const int elements = GetNE();

  int* elementDofMap = new int[localDofs];
  ::memcpy(elementDofMap,el->GetDofMap().GetData(),localDofs * sizeof(int));

  // We'll be keeping a count of how many local nodes point to its global dof
  for (int i = 0; i <= globalDofs; ++i) {
    offsets[i] = 0;
  }
  for (int e = 0; e < elements; ++e) {
    for (int d = 0; d < localDofs; ++d) {
      const int gid = elementMap[localDofs*e + d];
      ++offsets[gid + 1];
    }
  }
  // Aggregate to find offsets for each global dof
  for (int i = 1; i <= globalDofs; ++i) {
    offsets[i] += offsets[i - 1];
  }

  // For each global dof, fill in all local nodes that point   to it
  for (int e = 0; e < elements; ++e) {
    for (int d = 0; d < localDofs; ++d) {
      const int gid = elementMap[localDofs*e + elementDofMap[d]];
      const int lid = localDofs*e + d;
      indices[offsets[gid]++] = lid;
      map[lid] = gid;
    }
  }
  ::delete [] elementDofMap;

  // We shifted the offsets vector by 1 by using it as a counter
  // Now we shift it back.
  for (int i = globalDofs; i > 0; --i) {
    offsets[i] = offsets[i - 1];
  }
  offsets[0] = 0;

  const SparseMatrix* R = GetRestrictionMatrix(); assert(R);
  const Operator* P = GetProlongationMatrix(); assert(P);
  
  const int mHeight = R->Height();
  const int* I = R->GetI();
  const int* J = R->GetJ();
  int trueCount = 0;
  for (int i = 0; i < mHeight; ++i) {
    trueCount += ((I[i + 1] - I[i]) == 1);
  }
  
  reorderIndices = ::new RajaArray<int>(2*trueCount);
  for (int i = 0, trueIdx=0; i < mHeight; ++i) {
    if ((I[i + 1] - I[i]) == 1) {
      reorderIndices->operator[](trueIdx++) = J[I[i]];
      reorderIndices->operator[](trueIdx++) = i;
    }
  }
  restrictionOp = new RajaRestrictionOperator(R->Height(),
                                              R->Width(),
                                              reorderIndices);
  prolongationOp = new RajaProlongationOperator(P);
}

// ***************************************************************************
RajaFiniteElementSpace::~RajaFiniteElementSpace() {
  ::delete restrictionOp;
  ::delete prolongationOp;
  ::delete reorderIndices;
}

// ***************************************************************************
bool RajaFiniteElementSpace::hasTensorBasis() const {
  assert(dynamic_cast<const TensorBasisElement*>(GetFE(0)));
  return true;
}

// ***************************************************************************
void RajaFiniteElementSpace::GlobalToLocal(const RajaVector& globalVec,
    RajaVector& localVec) const {
  const int vdim = GetVDim();
  const int localEntries = localDofs * GetNE();
  const bool vdim_ordering = ordering == Ordering::byVDIM;
  rGlobalToLocal(vdim,
                 vdim_ordering,
                 globalDofs,
                 localEntries,
                 offsets,
                 indices,
                 globalVec,
                 localVec);
}

// ***************************************************************************
// Aggregate local node values to their respective global dofs
void RajaFiniteElementSpace::LocalToGlobal(const RajaVector& localVec,
    RajaVector& globalVec) const {
  const int vdim = GetVDim();
  const int localEntries = localDofs * GetNE();
  const bool vdim_ordering = ordering == Ordering::byVDIM;
  rLocalToGlobal(vdim,
                 vdim_ordering,
                 globalDofs,
                 localEntries,
                 offsets,
                 indices,
                 localVec,
                 globalVec);
}
  
} // namespace mfem
