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
#include "raja.hpp"


// *****************************************************************************
template<const int NUM_QUAD> __global__
void rInitQuadratureData(const int nzones,
                         const double* restrict rho0,
                         const double* restrict detJ,
                         const double* restrict quadWeights,
                         double* restrict rho0DetJ0w) {
  const int el = blockDim.x * blockIdx.x + threadIdx.x;
  if (el < nzones){
    for (int q = 0; q < NUM_QUAD; ++q){
      rho0DetJ0w[ijN(q,el,NUM_QUAD)] =
        rho0[ijN(q,el,NUM_QUAD)]*detJ[ijN(q,el,NUM_QUAD)]*quadWeights[q];
    }
  }
}
typedef void (*fInitQuadratureData)(const int,const double*,const double*,const double*,double*);
void rInitQuadratureData(const int NUM_QUAD,
                         const int numElements,
                         const double* restrict rho0,
                         const double* restrict detJ,
                         const double* restrict quadWeights,
                         double* restrict rho0DetJ0w) {
  const unsigned int id = NUM_QUAD;
  static std::unordered_map<unsigned int, fInitQuadratureData> call = {
    {2,&rInitQuadratureData<2>},
    {4,&rInitQuadratureData<4>},
    {8,&rInitQuadratureData<8>},
    {16,&rInitQuadratureData<16>},
    {25,&rInitQuadratureData<25>},
    {36,&rInitQuadratureData<36>},
    {49,&rInitQuadratureData<49>},
    {64,&rInitQuadratureData<64>},
    {81,&rInitQuadratureData<81>},
    {100,&rInitQuadratureData<100>},
    {121,&rInitQuadratureData<121>},
    {144,&rInitQuadratureData<144>},
    {0xD8,&rInitQuadratureData<0xD8>},
  };
  if (!call[id]){
    printf("\n[rInitQuadratureData] id \033[33m0x%X\033[m ",id);
    fflush(stdout);
  }
  assert(call[id]);
  const int grid = numElements;
  const int blck = NUM_QUAD;
  call0(rInitQuadratureData,id,grid,blck,numElements,rho0,detJ,quadWeights,rho0DetJ0w);
}
