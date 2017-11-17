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

extern "C" void kExtractSubVector(const int entries,
                                  const int* indices,
                                  const double* in,
                                  double* __restrict out) {
  for (int i = 0; i < entries; ++i) {
    out[i] = in[indices[i]];
  }
}

extern "C"  void kSetSubVector(const int entries,
                               const int* indices,
                               const double*  in,
                               double* __restrict out) {
  for (int i = 0; i < entries; ++i) {
    out[indices[i]] = in[i];
  }
}

extern "C" void kMapSubVector(const int entries,
                              const int* indices,
                              const double* in,
                              double* __restrict out) {
  for (int i = 0; i < entries; ++i) {
    const int fromIdx = indices[2*i + 0];
    const int toIdx   = indices[2*i + 1];
    out[toIdx] = in[fromIdx];
  }
}
