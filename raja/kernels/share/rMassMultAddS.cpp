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
template<const int NUM_DOFS_1D,
         const int NUM_QUAD_1D> kernel
#endif
void rMassMultAdd2S(
#ifndef __TEMPLATES__
                    const int NUM_DOFS_1D,
                    const int NUM_QUAD_1D,
#endif
                    const int numElements,
                    const double* restrict dofToQuad,
                    const double* restrict dofToQuadD,
                    const double* restrict quadToDof,
                    const double* restrict quadToDofD,
                    const double* restrict oper,
                    const double* restrict solIn,
                    double* restrict solOut) {
  const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
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
    share double s_quadToDof[NUM_QUAD_DOFS_1D];//@dim(NUM_DOFS_1D, NUM_QUAD_1D);

    // Store xy planes in shared memory
    share double s_xy[NUM_QUAD_DOFS_1D];//@dim(NUM_DOFS_1D, NUM_QUAD_1D);
    share double s_xy2[NUM_QUAD_2D];//@dim(NUM_QUAD_1D, NUM_QUAD_1D);

    double r_x[NUM_MAX_1D];

    for (int x = 0; x < NUM_MAX_1D; ++x) {
      for (int id = x; id < NUM_QUAD_DOFS_1D; id += NUM_MAX_1D) {
        s_dofToQuad[id]  = dofToQuad[id];
        s_quadToDof[id]  = quadToDof[id];
      }
    }

    for (int e = eOff; e < (eOff + M2_ELEMENT_BATCH); ++e) {
      if (e < numElements) {
        sync;
        for (int dx = 0; dx < NUM_MAX_1D; ++dx) {
          if (dx < NUM_DOFS_1D) {
            for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
              s_xy[ijN(dx, qy,NUM_DOFS_1D)] = 0;
            }
            for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
              r_x[dy] = solIn[ijkN(dx, dy, e,NUM_DOFS_1D)];
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
              double s = 0;
              for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
                s += s_xy[ijN(dx, qy,NUM_DOFS_1D)] * s_dofToQuad[ijN(qx, dx,NUM_QUAD_1D)];
              }
              s_xy2[ijN(qx, qy,NUM_QUAD_1D)] = s * oper[ijkN(qx, qy, e,NUM_QUAD_1D)];
            }
          }
        }
        sync;
        
        for (int qx = 0; qx < NUM_MAX_1D; ++qx) {
          if (qx < NUM_QUAD_1D) {
            for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
              s_xy[ijN(dy, qx,NUM_DOFS_1D)] = 0;
            }
            for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
              r_x[qy] = s_xy2[ijN(qx, qy,NUM_QUAD_1D)];
            }
            for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
              double s = 0;
              for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
                s += r_x[qy] * s_quadToDof[ijN(dy, qy,NUM_DOFS_1D)];
              }
              s_xy[ijN(dy, qx,NUM_DOFS_1D)] = s;
            }
          }
        }
        sync;
        for (int dx = 0; dx < NUM_MAX_1D; ++dx) {
          if (dx < NUM_DOFS_1D) {
            for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
              double s = 0;
              for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
                s += (s_xy[ijN(dy, qx,NUM_DOFS_1D)] * s_quadToDof[ijN(dx, qx,NUM_DOFS_1D)]);
              }
              solOut[ijkN(dx, dy, e,NUM_DOFS_1D)] += s;
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
typedef void (*fMassMultAdd)(const int numElements,
                             const double* dofToQuad,
                             const double* dofToQuadD,
                             const double* quadToDof,
                             const double* quadToDofD,
                             const double* oper,
                             const double* solIn,
                             double* __restrict solOut);

// *****************************************************************************
void rMassMultAddS(const int DIM,
                   const int NUM_DOFS_1D,
                   const int NUM_QUAD_1D,
                   const int numElements,
                   const double* dofToQuad,
                   const double* dofToQuadD,
                   const double* quadToDof,
                   const double* quadToDofD,
                   const double* op,
                   const double* x,
                   double* __restrict y) {
  push(Green);
#ifndef __LAMBDA__
  const int grid = ((numElements+M2_ELEMENT_BATCH-1)/M2_ELEMENT_BATCH);
  const int blck = NUM_QUAD_1D;
#endif
#ifdef __TEMPLATES__
  const unsigned int id = (DIM<<16)|(NUM_DOFS_1D<<8)|(NUM_QUAD_1D);
  assert(LOG2(DIM)<=8);
  assert(LOG2(NUM_DOFS_1D)<=8);
  assert(LOG2(NUM_QUAD_1D)<=8);
  static std::unordered_map<unsigned int, fMassMultAdd> call = {
    // 2D
    {0x20102,&rMassMultAdd2S<1,2>},
    {0x20202,&rMassMultAdd2S<2,2>},
    {0x20203,&rMassMultAdd2S<2,3>},
    {0x20204,&rMassMultAdd2S<2,4>},
    {0x20205,&rMassMultAdd2S<2,5>},
    {0x20206,&rMassMultAdd2S<2,6>},
    {0x20207,&rMassMultAdd2S<2,7>},
    {0x20208,&rMassMultAdd2S<2,8>},
    
    {0x20302,&rMassMultAdd2S<3,2>},
    {0x20303,&rMassMultAdd2S<3,3>},
    {0x20304,&rMassMultAdd2S<3,4>},
    {0x20305,&rMassMultAdd2S<3,5>},
    {0x20306,&rMassMultAdd2S<3,6>},
    {0x20307,&rMassMultAdd2S<3,7>},
    {0x20308,&rMassMultAdd2S<3,8>},
    
    {0x20402,&rMassMultAdd2S<4,2>},
    {0x20403,&rMassMultAdd2S<4,3>},
    {0x20404,&rMassMultAdd2S<4,4>},
    {0x20405,&rMassMultAdd2S<4,5>},
    {0x20406,&rMassMultAdd2S<4,6>},
    {0x20407,&rMassMultAdd2S<4,7>},
    {0x20408,&rMassMultAdd2S<4,8>},

    {0x20502,&rMassMultAdd2S<5,2>},
    {0x20503,&rMassMultAdd2S<5,3>},
    {0x20504,&rMassMultAdd2S<5,4>},
    {0x20505,&rMassMultAdd2S<5,5>},
    {0x20506,&rMassMultAdd2S<5,6>},
    {0x20507,&rMassMultAdd2S<5,7>},
    {0x20508,&rMassMultAdd2S<5,8>},
    {0x20509,&rMassMultAdd2S<5,9>},
    {0x2050A,&rMassMultAdd2S<5,10>},

    {0x20602,&rMassMultAdd2S<6,2>},
    {0x20603,&rMassMultAdd2S<6,3>},
    {0x20604,&rMassMultAdd2S<6,4>},
    {0x20605,&rMassMultAdd2S<6,5>},
    {0x20606,&rMassMultAdd2S<6,6>},
    {0x20607,&rMassMultAdd2S<6,7>},
    {0x20608,&rMassMultAdd2S<6,8>},
    {0x20609,&rMassMultAdd2S<6,9>},
    {0x2060A,&rMassMultAdd2S<6,10>},
    {0x2060C,&rMassMultAdd2S<6,12>},
    
    {0x20702,&rMassMultAdd2S<7,2>},
    {0x20703,&rMassMultAdd2S<7,3>},
    {0x20704,&rMassMultAdd2S<7,4>},
    {0x20705,&rMassMultAdd2S<7,5>},
    {0x20706,&rMassMultAdd2S<7,6>},
    {0x20707,&rMassMultAdd2S<7,7>},
    {0x20708,&rMassMultAdd2S<7,8>},
    {0x20709,&rMassMultAdd2S<7,9>},
    {0x2070A,&rMassMultAdd2S<7,10>},
    {0x2070C,&rMassMultAdd2S<7,12>},

    // 3D
/*    {0x30202,&rMassMultAdd3S<2,2>},
    {0x30203,&rMassMultAdd3S<2,3>},
    {0x30204,&rMassMultAdd3S<2,4>},
    {0x30205,&rMassMultAdd3S<2,5>},
    {0x30206,&rMassMultAdd3S<2,6>},
    {0x30207,&rMassMultAdd3S<2,7>},
    {0x30208,&rMassMultAdd3S<2,8>},
    {0x30209,&rMassMultAdd3S<2,9>},
    
    {0x30302,&rMassMultAdd3S<3,2>},
    {0x30303,&rMassMultAdd3S<3,3>},
    {0x30304,&rMassMultAdd3S<3,4>},
    {0x30305,&rMassMultAdd3S<3,5>},
    {0x30306,&rMassMultAdd3S<3,6>},
    {0x30307,&rMassMultAdd3S<3,7>},
    {0x30308,&rMassMultAdd3S<3,8>},
    
    {0x30402,&rMassMultAdd3S<4,2>},
    {0x30403,&rMassMultAdd3S<4,3>},
    {0x30404,&rMassMultAdd3S<4,4>},
    {0x30405,&rMassMultAdd3S<4,5>},
    {0x30406,&rMassMultAdd3S<4,6>},
    {0x30407,&rMassMultAdd3S<4,7>},
    {0x30408,&rMassMultAdd3S<4,8>},
*/
  };
  if(!call[id]){
    printf("\n[rMassMultAddS] id \033[33m0x%X\033[m ",id);
    fflush(stdout);
  }
  assert(call[id]);
  call0(rMassMultAdd2S,id,grid,blck,
        numElements,dofToQuad,dofToQuadD,quadToDof,quadToDofD,op,x,y);
#else
  if (DIM==1) assert(false);
  if (DIM==2)
    call0(rMassMultAdd2S,id,grid,blck,
          NUM_DOFS_1D,NUM_QUAD_1D,
          numElements,dofToQuad,dofToQuadD,quadToDof,quadToDofD,op,x,y); 
  if (DIM==3) assert(false);
#endif
  pop();
}
