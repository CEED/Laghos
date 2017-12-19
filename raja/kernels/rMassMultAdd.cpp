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
template<const int NUM_DOFS_1D,
         const int NUM_QUAD_1D>
static void rMassMultAdd2D(const int numElements,
                           const double* restrict dofToQuad,
                           const double* restrict dofToQuadD,
                           const double* restrict quadToDof,
                           const double* restrict quadToDofD,
                           const double* restrict oper,
                           const double* restrict solIn,
                           double* restrict solOut) {
  forall(e,numElements,{
      double sol_xy[NUM_QUAD_1D][NUM_QUAD_1D];
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          sol_xy[qy][qx] = 0.0;
        }
      }
      for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
        double sol_x[NUM_QUAD_1D];
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          sol_x[qy] = 0.0;
        }
        for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
          const double s = solIn[ijkN(dx,dy,e,NUM_DOFS_1D)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            sol_x[qx] += dofToQuad[ijN(qx,dx,NUM_QUAD_1D)]* s;
          }
        }
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          const double d2q = dofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            sol_xy[qy][qx] += d2q * sol_x[qx];
          }
        }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          sol_xy[qy][qx] *= oper[ijkN(qx,qy,e,NUM_QUAD_1D)];
        }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        double sol_x[NUM_DOFS_1D];
        for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
          sol_x[dx] = 0.0;
        }
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          const double s = sol_xy[qy][qx];
          for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
            sol_x[dx] += quadToDof[ijN(dx,qx,NUM_DOFS_1D)] * s;
          }
        }
        for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
          const double q2d = quadToDof[ijN(dy,qy,NUM_DOFS_1D)];
          for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
            solOut[ijkN(dx,dy,e,NUM_DOFS_1D)] += q2d * sol_x[dx];
          }
        }
      }
    });
}

// *****************************************************************************
template<const int NUM_DOFS_1D,
         const int NUM_QUAD_1D>
static void rMassMultAdd3D(const int numElements,
                           const double* dofToQuad,
                           const double* dofToQuadD,
                           const double* quadToDof,
                           const double* quadToDofD,
                           const double* oper,
                           const double* solIn,
                           double* __restrict solOut) {
  forall(e,numElements,{
    double sol_xyz[NUM_QUAD_1D][NUM_QUAD_1D][NUM_QUAD_1D];
    for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          sol_xyz[qz][qy][qx] = 0;
        }
      }
    }
    for (int dz = 0; dz < NUM_DOFS_1D; ++dz) {
      double sol_xy[NUM_QUAD_1D][NUM_QUAD_1D];
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          sol_xy[qy][qx] = 0;
        }
      }
      for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
        double sol_x[NUM_QUAD_1D];
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          sol_x[qx] = 0;
        }
        for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
          const double s = solIn[ijklN(dx,dy,dz,e,NUM_DOFS_1D)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            sol_x[qx] += dofToQuad[ijN(qx,dx,NUM_QUAD_1D)] * s;
          }
        }
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          const double wy = dofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            sol_xy[qy][qx] += wy * sol_x[qx];
          }
        }
      }
      for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
        const double wz = dofToQuad[ijN(qz,dz,NUM_QUAD_1D)];
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            sol_xyz[qz][qy][qx] += wz * sol_xy[qy][qx];
          }
        }
      }
    }
    for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          sol_xyz[qz][qy][qx] *= oper[ijklN(qx,qy,qz,e,NUM_QUAD_1D)];
        }
      }
    }
    for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
      double sol_xy[NUM_DOFS_1D][NUM_DOFS_1D];
      for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
        for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
          sol_xy[dy][dx] = 0;
        }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        double sol_x[NUM_DOFS_1D];
        for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
          sol_x[dx] = 0;
        }
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          const double s = sol_xyz[qz][qy][qx];
          for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
            sol_x[dx] += quadToDof[ijN(dx,qx,NUM_DOFS_1D)] * s;
          }
        }
        for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
          const double wy = quadToDof[ijN(dy,qy,NUM_DOFS_1D)];
          for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
            sol_xy[dy][dx] += wy * sol_x[dx];
          }
        }
      }
      for (int dz = 0; dz < NUM_DOFS_1D; ++dz) {
        const double wz = quadToDof[ijN(dz,qz,NUM_DOFS_1D)];
        for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
          for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
            solOut[ijklN(dx,dy,dz,e,NUM_DOFS_1D)] += wz * sol_xy[dy][dx];
          }
        }
      }
    }
  });
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
void rMassMultAdd(const int DIM,
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
  const unsigned int id = (DIM<<16)|(NUM_DOFS_1D<<8)|(NUM_QUAD_1D);
  assert(LOG2(DIM)<8);
  assert(LOG2(NUM_DOFS_1D)<8);
  assert(LOG2(NUM_QUAD_1D)<8);
  static std::unordered_map<unsigned int, fMassMultAdd> call = {
    {0x20202,&rMassMultAdd2D<2,2>},
    {0x20203,&rMassMultAdd2D<2,3>},
    {0x20204,&rMassMultAdd2D<2,4>},
    {0x20205,&rMassMultAdd2D<2,5>},
    {0x20206,&rMassMultAdd2D<2,6>},
    {0x20207,&rMassMultAdd2D<2,7>},
    {0x20208,&rMassMultAdd2D<2,8>},
    
    {0x20302,&rMassMultAdd2D<3,2>},
    {0x20303,&rMassMultAdd2D<3,3>},
    {0x20304,&rMassMultAdd2D<3,4>},
    {0x20305,&rMassMultAdd2D<3,5>},
    {0x20306,&rMassMultAdd2D<3,6>},
    {0x20307,&rMassMultAdd2D<3,7>},
    {0x20308,&rMassMultAdd2D<3,8>},
    
    {0x20402,&rMassMultAdd2D<4,2>},
    {0x20403,&rMassMultAdd2D<4,3>},
    {0x20404,&rMassMultAdd2D<4,4>},
    {0x20405,&rMassMultAdd2D<4,5>},
    {0x20406,&rMassMultAdd2D<4,6>},
    {0x20407,&rMassMultAdd2D<4,7>},
    {0x20408,&rMassMultAdd2D<4,8>},

    {0x20502,&rMassMultAdd2D<5,2>},
    {0x20503,&rMassMultAdd2D<5,3>},
    {0x20504,&rMassMultAdd2D<5,4>},
    {0x20505,&rMassMultAdd2D<5,5>},
    {0x20506,&rMassMultAdd2D<5,6>},
    {0x20507,&rMassMultAdd2D<5,7>},
    {0x20508,&rMassMultAdd2D<5,8>},
    {0x20509,&rMassMultAdd2D<5,9>},
    {0x2050A,&rMassMultAdd2D<5,10>},

    {0x20602,&rMassMultAdd2D<6,2>},
    {0x20603,&rMassMultAdd2D<6,3>},
    {0x20604,&rMassMultAdd2D<6,4>},
    {0x20605,&rMassMultAdd2D<6,5>},
    {0x20606,&rMassMultAdd2D<6,6>},
    {0x20607,&rMassMultAdd2D<6,7>},
    {0x20608,&rMassMultAdd2D<6,8>},
    {0x20609,&rMassMultAdd2D<6,9>},
    {0x2060A,&rMassMultAdd2D<6,10>},
    
    {0x20702,&rMassMultAdd2D<7,2>},
    {0x20703,&rMassMultAdd2D<7,3>},
    {0x20704,&rMassMultAdd2D<7,4>},
    {0x20705,&rMassMultAdd2D<7,5>},
    {0x20706,&rMassMultAdd2D<7,6>},
    {0x20707,&rMassMultAdd2D<7,7>},
    {0x20708,&rMassMultAdd2D<7,8>},
    {0x20709,&rMassMultAdd2D<7,9>},
    
    {0x30202,&rMassMultAdd3D<2,2>},
    {0x30203,&rMassMultAdd3D<2,3>},
    {0x30204,&rMassMultAdd3D<2,4>},
    {0x30205,&rMassMultAdd3D<2,5>},
    {0x30206,&rMassMultAdd3D<2,6>},
    {0x30207,&rMassMultAdd3D<2,7>},
    {0x30208,&rMassMultAdd3D<2,8>},
    {0x30209,&rMassMultAdd3D<2,9>},
    
    {0x30302,&rMassMultAdd3D<3,2>},
    {0x30303,&rMassMultAdd3D<3,3>},
    {0x30304,&rMassMultAdd3D<3,4>},
    {0x30305,&rMassMultAdd3D<3,5>},
    {0x30306,&rMassMultAdd3D<3,6>},
    {0x30307,&rMassMultAdd3D<3,7>},
    {0x30308,&rMassMultAdd3D<3,8>},
    
    {0x30402,&rMassMultAdd3D<4,2>},
    {0x30403,&rMassMultAdd3D<4,3>},
    {0x30404,&rMassMultAdd3D<4,4>},
    {0x30405,&rMassMultAdd3D<4,5>},
    {0x30406,&rMassMultAdd3D<4,6>},
    {0x30407,&rMassMultAdd3D<4,7>},
    {0x30408,&rMassMultAdd3D<4,8>},
  };
  if(!call[id]){
    printf("\n[rMassMultAdd] id \033[33m0x%X\033[m ",id);
    fflush(stdout);
  }
  assert(call[id]);
  call[id](numElements,dofToQuad,dofToQuadD,quadToDof,quadToDofD,op,x,y);
}
