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
template<const int NUM_VDIM,
         const int NUM_DOFS_1D,
         const int NUM_QUAD_1D>
static void rGridFuncToQuad1D(const int numElements,
                              const double* restrict dofToQuad,
                              const int* restrict l2gMap,
                              const double* restrict gf,
                              double* restrict out) {  
  forall(e,numElements,{
      double r_out[NUM_VDIM][NUM_QUAD_1D];
      for (int v = 0; v < NUM_VDIM; ++v) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          r_out[v][qx] = 0;
        }
      }
      for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
        const int gid = l2gMap[(dx) + (NUM_DOFS_1D) * (e)];
        for (int v = 0; v < NUM_VDIM; ++v) {
          const double r_gf = gf[v + gid * NUM_VDIM];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            r_out[v][qx] += r_gf * dofToQuad[(qx) + (NUM_QUAD_1D) * (dx)];
          }
        }
      }
      for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
        for (int v = 0; v < NUM_VDIM; ++v) {
          out[(qx) + (NUM_QUAD_1D) * ((e) + (numElements) * (v))] = r_out[v][qx];
        }
      }
    });
}

// *****************************************************************************
template<const int NUM_VDIM,
         const int NUM_DOFS_1D,
         const int NUM_QUAD_1D>
static void rGridFuncToQuad2D(const int numElements,
                              const double* restrict dofToQuad,
                              const int* restrict l2gMap,
                              const double* restrict gf,
                              double* restrict out) {
  forall(e,numElements,{
      double out_xy[NUM_VDIM][NUM_QUAD_1D][NUM_QUAD_1D];
      for (int v = 0; v < NUM_VDIM; ++v) {
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            out_xy[v][qy][qx] = 0;
          }
        }
      }

      for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
        double out_x[NUM_VDIM][NUM_QUAD_1D];
        for (int v = 0; v < NUM_VDIM; ++v) {
          for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
            out_x[v][qy] = 0;
          }
        }

        for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
          const int gid = l2gMap[ijkN(dx, dy, e,NUM_DOFS_1D)];
          for (int v = 0; v < NUM_VDIM; ++v) {
            const double r_gf = gf[v + gid*NUM_VDIM];
            for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
              out_x[v][qy] += r_gf * dofToQuad[ijN(qy, dx,NUM_QUAD_1D)];
            }
          }
        }

        for (int v = 0; v < NUM_VDIM; ++v) {
          for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
            const double d2q = dofToQuad[ijN(qy, dy,NUM_QUAD_1D)];
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
              out_xy[v][qy][qx] += d2q * out_x[v][qx];
            }
          }
        }
      }

      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          for (int v = 0; v < NUM_VDIM; ++v) {
            out[_ijklNM(v, qx, qy, e,NUM_QUAD_1D,numElements)] = out_xy[v][qy][qx];
          }
        }
      }
    });
}

// *****************************************************************************
template<const int NUM_VDIM,
         const int NUM_DOFS_1D,
         const int NUM_QUAD_1D>
static void rGridFuncToQuad3D(const int numElements,
                              const double* restrict dofToQuad,
                              const int* restrict l2gMap,
                              const double* restrict gf,
                              double* restrict out) {
  forall(e,numElements,{
    double out_xyz[NUM_VDIM][NUM_QUAD_1D][NUM_QUAD_1D][NUM_QUAD_1D];
    for (int v = 0; v < NUM_VDIM; ++v) {
      for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            out_xyz[v][qz][qy][qx] = 0;
          }
        }
      }
    }
    
    for (int dz = 0; dz < NUM_DOFS_1D; ++dz) {
      double out_xy[NUM_VDIM][NUM_QUAD_1D][NUM_QUAD_1D];
      for (int v = 0; v < NUM_VDIM; ++v) {
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            out_xy[v][qy][qx] = 0;
          }
        }
      }

      for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
        double out_x[NUM_VDIM][NUM_QUAD_1D];
        for (int v = 0; v < NUM_VDIM; ++v) {
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            out_x[v][qx] = 0;
          }
        }

        for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
          const int gid = l2gMap[ijklN(dx, dy, dz, e,NUM_DOFS_1D)];
          for (int v = 0; v < NUM_VDIM; ++v) {
            const double r_gf = gf[v + gid*NUM_VDIM];
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
              out_x[v][qx] += r_gf * dofToQuad[ijN(qx, dx, NUM_QUAD_1D)];
            }
          }
        }

        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          const double wy = dofToQuad[ijN(qy, dy, NUM_QUAD_1D)];
          for (int v = 0; v < NUM_VDIM; ++v) {
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
              out_xy[v][qy][qx] += wy * out_x[v][qx];
            }
          }
        }
      }

      for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
        const double wz = dofToQuad[ijN(qz, dz, NUM_QUAD_1D)];
        for (int v = 0; v < NUM_VDIM; ++v) {
          for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
              out_xyz[v][qz][qy][qx] += wz * out_xy[v][qy][qx];
            }
          }
        }
      }
    }

    for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          for (int v = 0; v < NUM_VDIM; ++v) {
            out[_ijklmNM(v, qx, qy, qz, e,NUM_QUAD_1D,numElements)] = out_xyz[v][qz][qy][qx];
          }
        }
      }
    }
    });
}

// *****************************************************************************
typedef void (*fGridFuncToQuad)(const int numElements,
                                const double* restrict dofToQuad,
                                const int* restrict l2gMap,
                                const double* gf,
                                double* restrict out);

// *****************************************************************************
void rGridFuncToQuad(const int DIM,
                     const int NUM_VDIM,
                     const int NUM_DOFS_1D,
                     const int NUM_QUAD_1D,
                     const int numElements,
                     const double* dofToQuad,
                     const int* l2gMap,
                     const double* gf,
                     double* __restrict out) {
  const unsigned int id = (DIM<<24)|(NUM_VDIM<<16)|(NUM_DOFS_1D<<8)|(NUM_QUAD_1D);
  assert(LOG2(DIM)<=8);//printf("DIM:%d ",DIM);
  assert(LOG2(NUM_VDIM)<=8);//printf("NUM_VDIM:%d ",NUM_VDIM);
  assert(LOG2(NUM_DOFS_1D)<=8);//printf("NUM_DOFS_1D:%d ",NUM_DOFS_1D);
  assert(LOG2(NUM_QUAD_1D)<=8);//printf("NUM_QUAD_1D:%d ",NUM_QUAD_1D);
  static std::unordered_map<unsigned int, fGridFuncToQuad> call = {
    // 2D
    {0x2010102,&rGridFuncToQuad2D<1,1,2>},
    {0x2010202,&rGridFuncToQuad2D<1,2,2>},
    {0x2010203,&rGridFuncToQuad2D<1,2,3>},
    {0x2010204,&rGridFuncToQuad2D<1,2,4>},
    {0x2010205,&rGridFuncToQuad2D<1,2,5>},
    {0x2010206,&rGridFuncToQuad2D<1,2,6>},
    {0x2010207,&rGridFuncToQuad2D<1,2,7>},
    {0x2010208,&rGridFuncToQuad2D<1,2,8>},
    {0x2010209,&rGridFuncToQuad2D<1,2,9>},
    {0x201020A,&rGridFuncToQuad2D<1,2,10>},
    
    {0x2010302,&rGridFuncToQuad2D<1,3,2>},
    {0x2010303,&rGridFuncToQuad2D<1,3,3>},
    {0x2010304,&rGridFuncToQuad2D<1,3,4>},
    {0x2010305,&rGridFuncToQuad2D<1,3,5>},
    {0x2010306,&rGridFuncToQuad2D<1,3,6>},
    {0x2010307,&rGridFuncToQuad2D<1,3,7>},
    {0x2010308,&rGridFuncToQuad2D<1,3,8>},
    {0x2010309,&rGridFuncToQuad2D<1,3,9>},
    {0x201030A,&rGridFuncToQuad2D<1,3,10>},

    {0x2010402,&rGridFuncToQuad2D<1,4,2>},
    {0x2010403,&rGridFuncToQuad2D<1,4,3>},
    {0x2010404,&rGridFuncToQuad2D<1,4,4>},
    {0x2010405,&rGridFuncToQuad2D<1,4,5>},
    {0x2010406,&rGridFuncToQuad2D<1,4,6>},
    {0x2010407,&rGridFuncToQuad2D<1,4,7>},
    {0x2010408,&rGridFuncToQuad2D<1,4,8>},
    {0x2010409,&rGridFuncToQuad2D<1,4,9>},
    {0x201040A,&rGridFuncToQuad2D<1,4,10>},
    
    {0x2010502,&rGridFuncToQuad2D<1,5,2>},
    {0x2010503,&rGridFuncToQuad2D<1,5,3>},
    {0x2010504,&rGridFuncToQuad2D<1,5,4>},
    {0x2010505,&rGridFuncToQuad2D<1,5,5>},
    {0x2010506,&rGridFuncToQuad2D<1,5,6>},
    {0x2010507,&rGridFuncToQuad2D<1,5,7>},
    {0x2010508,&rGridFuncToQuad2D<1,5,8>},
    {0x2010509,&rGridFuncToQuad2D<1,5,9>},
    {0x201050A,&rGridFuncToQuad2D<1,5,10>},
    
    {0x2010602,&rGridFuncToQuad2D<1,6,2>},
    {0x2010603,&rGridFuncToQuad2D<1,6,3>},
    {0x2010604,&rGridFuncToQuad2D<1,6,4>},
    {0x2010605,&rGridFuncToQuad2D<1,6,5>},
    {0x2010606,&rGridFuncToQuad2D<1,6,6>},
    {0x2010607,&rGridFuncToQuad2D<1,6,7>},
    {0x2010608,&rGridFuncToQuad2D<1,6,8>},
    {0x2010609,&rGridFuncToQuad2D<1,6,9>},
    {0x201060A,&rGridFuncToQuad2D<1,6,10>},
    {0x201060C,&rGridFuncToQuad2D<1,6,12>},
    
    {0x2010702,&rGridFuncToQuad2D<1,7,2>},
    {0x2010703,&rGridFuncToQuad2D<1,7,3>},
    {0x2010704,&rGridFuncToQuad2D<1,7,4>},
    {0x2010705,&rGridFuncToQuad2D<1,7,5>},
    {0x2010706,&rGridFuncToQuad2D<1,7,6>},
    {0x2010707,&rGridFuncToQuad2D<1,7,7>},
    {0x2010708,&rGridFuncToQuad2D<1,7,8>},
    {0x2010709,&rGridFuncToQuad2D<1,7,9>},
    {0x201070A,&rGridFuncToQuad2D<1,7,10>},

    // 3D
    {0x3010202,&rGridFuncToQuad3D<1,2,2>},
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
    {0x3010406,&rGridFuncToQuad3D<1,4,6>},
  };
  if (!call[id]){
    printf("\n[rGridFuncToQuad] id \033[33m0x%X\033[m ",id);
    fflush(stdout);
  }
  assert(call[id]);
  call[id](numElements,dofToQuad,l2gMap,gf,out);
}
