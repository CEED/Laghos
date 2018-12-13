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
#include "../kernels.hpp"

// *****************************************************************************
template<const int NUM_DIM,
         const int NUM_DOFS_1D,
         const int NUM_QUAD_1D,
         const int L2_DOFS_1D,
         const int H1_DOFS_1D> kernel
void rForceMultTranspose2D(const int numElements,
                           const double* L2QuadToDof,
                           const double* H1DofToQuad,
                           const double* H1DofToQuadD,
                           const double* stressJinvT,
                           const double* v,
                           double* e) ;
// *****************************************************************************
template<const int NUM_DIM,
         const int NUM_DOFS_1D,
         const int NUM_QUAD_1D,
         const int L2_DOFS_1D,
         const int H1_DOFS_1D> kernel
void rForceMultTranspose3D(const int numElements,
                           const double* L2QuadToDof,
                           const double* H1DofToQuad,
                           const double* H1DofToQuadD,
                           const double* stressJinvT,
                           const double* v,
                           double* e);

// *****************************************************************************
typedef void (*fForceMultTranspose)(const int numElements,
                                    const double* L2QuadToDof,
                                    const double* H1DofToQuad,
                                    const double* H1DofToQuadD,
                                    const double* stressJinvT,
                                    const double* v,
                                    double* e);

// *****************************************************************************
void rForceMultTranspose(const int NUM_DIM,
                         const int NUM_DOFS_1D,
                         const int NUM_QUAD_1D,
                         const int L2_DOFS_1D,
                         const int H1_DOFS_1D,
                         const int nzones,
                         const double* L2QuadToDof,
                         const double* H1DofToQuad,
                         const double* H1DofToQuadD,
                         const double* stressJinvT,
                         const double* v,
                         double* e)
{
   assert(NUM_DOFS_1D==H1_DOFS_1D);
   assert(L2_DOFS_1D==NUM_DOFS_1D-1);
   assert(NUM_QUAD_1D==2*(NUM_DOFS_1D-1));
   assert(NUM_QUAD_1D==2*(NUM_DOFS_1D-1));
   const unsigned int id = ((NUM_DIM)<<4)|(NUM_DOFS_1D-2);
   static std::unordered_map<unsigned long long, fForceMultTranspose> call = {
      // 2D
      {0x20,&rForceMultTranspose2D<2,2,2,1,2>},
      {0x21,&rForceMultTranspose2D<2,3,4,2,3>},
      {0x22,&rForceMultTranspose2D<2,4,6,3,4>},
      {0x23,&rForceMultTranspose2D<2,5,8,4,5>},
      {0x24,&rForceMultTranspose2D<2,6,10,5,6>},
      {0x25,&rForceMultTranspose2D<2,7,12,6,7>},
      {0x26,&rForceMultTranspose2D<2,8,14,7,8>},
      {0x27,&rForceMultTranspose2D<2,9,16,8,9>},
      {0x28,&rForceMultTranspose2D<2,10,18,9,10>},
      {0x29,&rForceMultTranspose2D<2,11,20,10,11>},
      {0x2A,&rForceMultTranspose2D<2,12,22,11,12>},
      {0x2B,&rForceMultTranspose2D<2,13,24,12,13>},
      {0x2C,&rForceMultTranspose2D<2,14,26,13,14>},
      {0x2D,&rForceMultTranspose2D<2,15,28,14,15>},
      {0x2E,&rForceMultTranspose2D<2,16,30,15,16>},
      {0x2F,&rForceMultTranspose2D<2,17,32,16,17>},
      // 3D
      {0x30,&rForceMultTranspose3D<3,2,2,1,2>},
      {0x31,&rForceMultTranspose3D<3,3,4,2,3>},
      {0x32,&rForceMultTranspose3D<3,4,6,3,4>},
      {0x33,&rForceMultTranspose3D<3,5,8,4,5>},
      {0x34,&rForceMultTranspose3D<3,6,10,5,6>},
      {0x35,&rForceMultTranspose3D<3,7,12,6,7>},
      {0x36,&rForceMultTranspose3D<3,8,14,7,8>},
      {0x37,&rForceMultTranspose3D<3,9,16,8,9>},
      {0x38,&rForceMultTranspose3D<3,10,18,9,10>},
      {0x39,&rForceMultTranspose3D<3,11,20,10,11>},
      {0x3A,&rForceMultTranspose3D<3,12,22,11,12>},
      {0x3B,&rForceMultTranspose3D<3,13,24,12,13>},
      {0x3C,&rForceMultTranspose3D<3,14,26,13,14>},
      {0x3D,&rForceMultTranspose3D<3,15,28,14,15>},
      {0x3E,&rForceMultTranspose3D<3,16,30,15,16>},
      {0x3F,&rForceMultTranspose3D<3,17,32,16,17>},
   };
   if (!call[id]) {
      printf("\n[rForceMultTranspose] id \033[33m0x%X\033[m ",id);
      fflush(stdout);
   }
   assert(call[id]);
   
   GET_CONST_ADRS(L2QuadToDof);
   GET_CONST_ADRS(H1DofToQuad);
   GET_CONST_ADRS(H1DofToQuadD);
   GET_CONST_ADRS(stressJinvT);
   GET_CONST_ADRS(v);
   GET_ADRS(e);
   
   call[id](nzones,
            d_L2QuadToDof,
            d_H1DofToQuad,
            d_H1DofToQuadD,
            d_stressJinvT,
            d_v,
            d_e);
}
