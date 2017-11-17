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
#include "defines.h"

// *****************************************************************************
extern "C" void kLocalToGlobal(const int NUM_VDIM,
                               const bool VDIM_ORDERING,
                               const int globalEntries,
                               const int localEntries,
                               const int* offsets,
                               const int* indices,
                               const double* localX,
                               double* __restrict globalX) {
  for (int i = 0; i < globalEntries; ++i) {
    const int offset = offsets[i];
    const int nextOffset = offsets[i + 1];
    for (int v = 0; v < NUM_VDIM; ++v) {
      double dofValue = 0;
      for (int j = offset; j < nextOffset; ++j) {
        const int l_offset = ijNMt(v,indices[j],NUM_VDIM,localEntries,VDIM_ORDERING);
        dofValue += localX[l_offset];
      }
      const int g_offset = ijNMt(v,i,NUM_VDIM,globalEntries,VDIM_ORDERING);
      globalX[g_offset] = dofValue;
    }
  }
}
