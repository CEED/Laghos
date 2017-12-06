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

extern "C"
void kExtractSubVector(const int entries,
                       const int* indices,
                       const double* in,
                       double* __restrict out) {
  for (int i = 0; i < entries; ++i) {
    out[i] = in[indices[i]];
  }
}

extern "C"
void kSetSubVector(const int entries,
                   const int* indices,
                   const double*  in,
                   double* __restrict out) {
  for (int i = 0; i < entries; ++i) {
    out[indices[i]] = in[i];
  }
}

extern "C"
void kMapSubVector(const int entries,
                   const int* indices,
                   const double* in,
                   double* __restrict out) {
  for (int i = 0; i < entries; ++i) {
    const int fromIdx = indices[2*i + 0];
    const int toIdx   = indices[2*i + 1];
    out[toIdx] = in[fromIdx];
  }
}
