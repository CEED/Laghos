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
#include "../cuda.hpp"

// *****************************************************************************
extern "C" kernel
void rSetSubVector0(const int N,
                    const int* indices,
                    const double* in,
                    double* __restrict out)
{
   const int i = blockDim.x * blockIdx.x + threadIdx.x;
   if (i < N) { out[indices[i]] = in[i]; }
}

// *****************************************************************************
void rSetSubVector(const int N,
                   const int* indices,
                   const double* in,
                   double* __restrict out)
{
   cuKer(rSetSubVector,N,indices,in,out);
}

// *****************************************************************************
extern "C" kernel
void rMapSubVector0(const int N,
                    const int* indices,
                    const double* in,
                    double* __restrict out)
{
   const int i = blockDim.x * blockIdx.x + threadIdx.x;
   if (i < N)
   {
      const int fromIdx = indices[2*i + 0];
      const int toIdx   = indices[2*i + 1];
      out[toIdx] = in[fromIdx];
   }
}

// *****************************************************************************
void rMapSubVector(const int N,
                   const int* indices,
                   const double* in,
                   double* __restrict out)
{
   cuKer(rMapSubVector,N,indices,in,out);
}

// *****************************************************************************
extern "C" kernel
void rExtractSubVector0(const int N,
                        const int* indices,
                        const double* in,
                        double* __restrict out)
{
   const int i = blockDim.x * blockIdx.x + threadIdx.x;
   if (i < N) { out[i] = in[indices[i]]; }
}

// *****************************************************************************
void rExtractSubVector(const int N,
                       const int* indices,
                       const double* in,
                       double* __restrict out)
{
   cuKer(rExtractSubVector,N,indices,in,out);
}
