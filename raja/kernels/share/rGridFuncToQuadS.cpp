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
template<const int NUM_VDIM,
         const int NUM_DOFS_1D,
         const int NUM_QUAD_1D> kernel
#endif
void rGridFuncToQuad2S(
#ifndef __TEMPLATES__
                       const int NUM_VDIM,
                       const int NUM_DOFS_1D,
                       const int NUM_QUAD_1D,
#endif
                       const int numElements,
                       const double* restrict dofToQuad,
                       const int* restrict l2gMap,
                       const double * restrict gf,
                       double* restrict out) {
  const int NUM_QUAD_DOFS_1D = (NUM_QUAD_1D * NUM_DOFS_1D);
  const int NUM_MAX_1D = (NUM_QUAD_1D<NUM_DOFS_1D)?NUM_DOFS_1D:NUM_QUAD_1D;
  // Iterate over elements
#ifdef __LAMBDA__
  forallS(eOff,numElements,M2_ELEMENT_BATCH,
#else
  const int idx = blockDim.x*blockIdx.x + threadIdx.x;
  const int eOff = idx * M2_ELEMENT_BATCH;
  if (eOff < numElements)
#endif
  {
    // Store dof <--> quad mappings
    share double s_dofToQuad[NUM_QUAD_DOFS_1D];//@dim(NUM_QUAD_1D, NUM_DOFS_1D);

    // Store xy planes in shared memory
    share double s_xy[NUM_QUAD_DOFS_1D];//@dim(NUM_DOFS_1D, NUM_QUAD_1D);

    for (int x = 0; x < NUM_MAX_1D; ++x) {
      for (int id = x; id < NUM_QUAD_DOFS_1D; id += NUM_MAX_1D) {
        s_dofToQuad[id] = dofToQuad[id];
      }
    }

    for (int e = eOff; e < (eOff + M2_ELEMENT_BATCH); ++e) {
      if (e < numElements) {
        sync;
        for (int dx = 0; dx < NUM_MAX_1D; ++dx) {
          if (dx < NUM_DOFS_1D) {
            double r_x[NUM_DOFS_1D];
            for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
              r_x[dy] = gf[l2gMap[ijkN(dx, dy, e,NUM_DOFS_1D)]];
            }
            for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
              double xy = 0;
              for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
                xy += r_x[dy] * s_dofToQuad[ijN(qy, dy,NUM_QUAD_1D)];
              }
              s_xy[ijN(dx, qy,NUM_DOFS_1D)] = xy;
            }
          }
        }
        sync;
        for (int qy = 0; qy < NUM_MAX_1D; ++qy) {
          if (qy < NUM_QUAD_1D) {
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
              double val = 0;
              for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
                val += s_xy[ijN(dx, qy,NUM_DOFS_1D)] * s_dofToQuad[ijN(qx, dx,NUM_QUAD_1D)];
              }
              out[ijkN(qx, qy, e,NUM_QUAD_1D)] = val;
            }
          }
        }
      }
    }
  }
#ifdef __LAMBDA__
         );
#endif
}
// *****************************************************************************
typedef void (*fGridFuncToQuad)(const int numElements,
                                const double* restrict dofToQuad,
                                const int* restrict l2gMap,
                                const double* gf,
                                double* restrict out);
// *****************************************************************************
void rGridFuncToQuadS(const int DIM,
                     const int NUM_VDIM,
                     const int NUM_DOFS_1D,
                     const int NUM_QUAD_1D,
                     const int numElements,
                     const double* dofToQuad,
                     const int* l2gMap,
                     const double* gf,
                     double* __restrict out) {
  push(Green);
#ifndef __LAMBDA__
  const int grid = ((numElements+M2_ELEMENT_BATCH-1)/M2_ELEMENT_BATCH);
  const int NUM_MAX_1D = (NUM_QUAD_1D<NUM_DOFS_1D)?NUM_DOFS_1D:NUM_QUAD_1D;
  const int blck = NUM_MAX_1D;
#endif
#ifdef __TEMPLATES__
  const unsigned int id = (DIM<<24)|(NUM_VDIM<<16)|(NUM_DOFS_1D<<8)|(NUM_QUAD_1D);
  assert(LOG2(DIM)<=8);//printf("DIM:%d ",DIM);
  assert(LOG2(NUM_VDIM)<=8);//printf("NUM_VDIM:%d ",NUM_VDIM);
  assert(LOG2(NUM_DOFS_1D)<=8);//printf("NUM_DOFS_1D:%d ",NUM_DOFS_1D);
  assert(LOG2(NUM_QUAD_1D)<=8);//printf("NUM_QUAD_1D:%d ",NUM_QUAD_1D);
  static std::unordered_map<unsigned int, fGridFuncToQuad> call = {
    // 2D
    {0x2010102,&rGridFuncToQuad2S<1,1,2>},
    {0x2010202,&rGridFuncToQuad2S<1,2,2>},
    {0x2010203,&rGridFuncToQuad2S<1,2,3>},
    {0x2010204,&rGridFuncToQuad2S<1,2,4>},
    {0x2010205,&rGridFuncToQuad2S<1,2,5>},
    {0x2010206,&rGridFuncToQuad2S<1,2,6>},
    {0x2010207,&rGridFuncToQuad2S<1,2,7>},
    {0x2010208,&rGridFuncToQuad2S<1,2,8>},
    {0x2010209,&rGridFuncToQuad2S<1,2,9>},
    {0x201020A,&rGridFuncToQuad2S<1,2,10>},
    
    {0x2010302,&rGridFuncToQuad2S<1,3,2>},
    {0x2010303,&rGridFuncToQuad2S<1,3,3>},
    {0x2010304,&rGridFuncToQuad2S<1,3,4>},
    {0x2010305,&rGridFuncToQuad2S<1,3,5>},
    {0x2010306,&rGridFuncToQuad2S<1,3,6>},
    {0x2010307,&rGridFuncToQuad2S<1,3,7>},
    {0x2010308,&rGridFuncToQuad2S<1,3,8>},
    {0x2010309,&rGridFuncToQuad2S<1,3,9>},
    {0x201030A,&rGridFuncToQuad2S<1,3,10>},

    {0x2010402,&rGridFuncToQuad2S<1,4,2>},
    {0x2010403,&rGridFuncToQuad2S<1,4,3>},
    {0x2010404,&rGridFuncToQuad2S<1,4,4>},
    {0x2010405,&rGridFuncToQuad2S<1,4,5>},
    {0x2010406,&rGridFuncToQuad2S<1,4,6>},
    {0x2010407,&rGridFuncToQuad2S<1,4,7>},
    {0x2010408,&rGridFuncToQuad2S<1,4,8>},
    {0x2010409,&rGridFuncToQuad2S<1,4,9>},
    {0x201040A,&rGridFuncToQuad2S<1,4,10>},
    
    {0x2010502,&rGridFuncToQuad2S<1,5,2>},
    {0x2010503,&rGridFuncToQuad2S<1,5,3>},
    {0x2010504,&rGridFuncToQuad2S<1,5,4>},
    {0x2010505,&rGridFuncToQuad2S<1,5,5>},
    {0x2010506,&rGridFuncToQuad2S<1,5,6>},
    {0x2010507,&rGridFuncToQuad2S<1,5,7>},
    {0x2010508,&rGridFuncToQuad2S<1,5,8>},
    {0x2010509,&rGridFuncToQuad2S<1,5,9>},
    {0x201050A,&rGridFuncToQuad2S<1,5,10>},
    
    {0x2010602,&rGridFuncToQuad2S<1,6,2>},
    {0x2010603,&rGridFuncToQuad2S<1,6,3>},
    {0x2010604,&rGridFuncToQuad2S<1,6,4>},
    {0x2010605,&rGridFuncToQuad2S<1,6,5>},
    {0x2010606,&rGridFuncToQuad2S<1,6,6>},
    {0x2010607,&rGridFuncToQuad2S<1,6,7>},
    {0x2010608,&rGridFuncToQuad2S<1,6,8>},
    {0x2010609,&rGridFuncToQuad2S<1,6,9>},
    {0x201060A,&rGridFuncToQuad2S<1,6,10>},
    {0x201060C,&rGridFuncToQuad2S<1,6,12>},
    
    {0x2010702,&rGridFuncToQuad2S<1,7,2>},
    {0x2010703,&rGridFuncToQuad2S<1,7,3>},
    {0x2010704,&rGridFuncToQuad2S<1,7,4>},
    {0x2010705,&rGridFuncToQuad2S<1,7,5>},
    {0x2010706,&rGridFuncToQuad2S<1,7,6>},
    {0x2010707,&rGridFuncToQuad2S<1,7,7>},
    {0x2010708,&rGridFuncToQuad2S<1,7,8>},
    {0x2010709,&rGridFuncToQuad2S<1,7,9>},
    {0x201070A,&rGridFuncToQuad2S<1,7,10>},

    // 3D
/*    {0x3010202,&rGridFuncToQuad3D<1,2,2>},
    {0x3010203,&rGridFuncToQuad3D<1,2,3>},
    {0x3010204,&rGridFuncToQuad3D<1,2,4>},
    {0x3010205,&rGridFuncToQuad3D<1,2,5>},
    {0x3010206,&rGridFuncToQuad3D<1,2,6>},
    
    {0x3010302,&rGridFuncToQuad3D<1,3,2>},
    {0x3010303,&rGridFuncToQuad3D<1,3,3>},
    {0x3010304,&rGridFuncToQuad3D<1,3,4>},
    {0x3010305,&rGridFuncToQuad3D<1,3,5>},
    {0x3010306,&rGridFuncToQuad3D<1,3,6>},
    
    {0x3010402,&rGridFuncToQuad3D<1,4,2>},
    {0x3010403,&rGridFuncToQuad3D<1,4,3>},
    {0x3010404,&rGridFuncToQuad3D<1,4,4>},
    {0x3010405,&rGridFuncToQuad3D<1,4,5>},
    {0x3010406,&rGridFuncToQuad3D<1,4,6>},*/
  };
  if (!call[id]){
    printf("\n[rGridFuncToQuad] id \033[33m0x%X\033[m ",id);
    fflush(stdout);
  }
  assert(call[id]);
  call0(rGridFuncToQuadS,id,grid,blck,
        numElements,dofToQuad,l2gMap,gf,out);
#else
  if (DIM==1) assert(false);
  if (DIM==2)
    call0(rGridFuncToQuad2S,id,grid,blck,
          NUM_VDIM,NUM_DOFS_1D,NUM_QUAD_1D,
          numElements,dofToQuad,l2gMap,gf,out);
  if (DIM==3) assert(false);
#endif
  pop();
}
