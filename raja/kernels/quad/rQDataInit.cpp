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
#include "../raja.hpp"


// *****************************************************************************
#ifdef __TEMPLATES__
template<const int NUM_QUAD> kernel
#endif
void rInitQuadData(
#ifndef __TEMPLATES__
                         const int NUM_QUAD,
#endif
                         const int nzones,
                         const double* restrict rho0,
                         const double* restrict detJ,
                         const double* restrict quadWeights,
                         double* restrict rho0DetJ0w) {
#ifndef __LAMBDA__
  const int el = blockDim.x * blockIdx.x + threadIdx.x;
  if (el < nzones)
#else
  forall(el,nzones,
#endif
  {
    for (int q = 0; q < NUM_QUAD; ++q){
      rho0DetJ0w[ijN(q,el,NUM_QUAD)] =
        rho0[ijN(q,el,NUM_QUAD)]*detJ[ijN(q,el,NUM_QUAD)]*quadWeights[q];
    }
  }
#ifdef __LAMBDA__
          );
#endif
}
typedef void (*fInitQuadratureData)(const int,const double*,const double*,const double*,double*);
void rInitQuadratureData(const int NUM_QUAD,
                         const int numElements,
                         const double* restrict rho0,
                         const double* restrict detJ,
                         const double* restrict quadWeights,
                         double* restrict rho0DetJ0w) {
  push(Lime);
  const unsigned int id = NUM_QUAD;
#ifndef __LAMBDA__
  const int grid = numElements;
  const int blck = NUM_QUAD;
#endif
#ifdef __TEMPLATES__
  static std::unordered_map<unsigned int, fInitQuadratureData> call = {
    {2,&rInitQuadData<2>},
    {4,&rInitQuadData<4>},
    {8,&rInitQuadData<8>},
    {16,&rInitQuadData<16>},
    {25,&rInitQuadData<25>},
    {36,&rInitQuadData<36>},
    {49,&rInitQuadData<49>},
    {64,&rInitQuadData<64>},
    {81,&rInitQuadData<81>},
    {100,&rInitQuadData<100>},
    {121,&rInitQuadData<121>},
    {144,&rInitQuadData<144>},
    {0xD8,&rInitQuadData<0xD8>},
  };
  if (!call[id]){
    printf("\n[rInitQuadratureData] id \033[33m0x%X\033[m ",id);
    fflush(stdout);
  }
  assert(call[id]);
  call0(rInitQuadData,id,grid,blck,
        numElements,rho0,detJ,quadWeights,rho0DetJ0w);
#else
  call0(rInitQuadData,id,grid,blck,
        NUM_QUAD,
        numElements,rho0,detJ,quadWeights,rho0DetJ0w);
#endif
  pop();
}
