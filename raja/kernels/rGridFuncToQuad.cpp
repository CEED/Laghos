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
#include "defines.hpp"

// *****************************************************************************
static void rGridFuncToQuad1D(const int NUM_VDIM,
                              const int NUM_DOFS_1D,
                              const int NUM_QUAD_1D,
                              const int numElements,
                              const double* dofToQuad,
                              const int* l2gMap,
                              const double* gf,
                              double* __restrict out) {
  assert(NUM_VDIM==1); const int v1 = 1;
  assert(NUM_QUAD_1D==1); const int q1 = 1;
  
  forall(numElements,[=]_device_(int e){
    double r_out[v1][q1];
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
static void rGridFuncToQuad2D(const int NUM_VDIM,
                              const int NUM_DOFS_1D,
                              const int NUM_QUAD_1D,
                              const int numElements,
                              const double* __restrict dofToQuad,
                              const int* __restrict l2gMap,
                              const double* __restrict gf,
                              double* __restrict out) {
  //printf("\033[31m[NUM_VDIM=%d]\033[m\n",NUM_VDIM);
  //printf("\033[31m[NUM_QUAD_1D=%d]\033[m\n",NUM_QUAD_1D);
  assert(NUM_VDIM==1); const int v1 = 1;
  assert(NUM_QUAD_1D==4); const int q1 = 4;

  forall(numElements,[=]_device_(int e){//for (int e = 0; e < numElements; ++e) {
    double out_xy[v1][q1][q1];
    for (int v = 0; v < NUM_VDIM; ++v) {
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          out_xy[v][qy][qx] = 0;
        }
      }
    }

    for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
      double out_x[v1][q1];
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
static void rGridFuncToQuad3D(const int NUM_VDIM,
                              const int NUM_DOFS_1D,
                              const int NUM_QUAD_1D,
                              const int numElements,
                              const double* __restrict dofToQuad,
                              const int* __restrict l2gMap,
                              const double* gf,
                              double* __restrict out) {
  assert(NUM_VDIM==1); const int v1 = 1;
  assert(NUM_QUAD_1D==1); const int q1 = 1;

  forall(numElements,[=]_device_(int e){//for (int e = 0; e < numElements; ++e) {
    double out_xyz[v1][q1][q1][q1];
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
      double out_xy[v1][q1][q1];
      for (int v = 0; v < NUM_VDIM; ++v) {
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            out_xy[v][qy][qx] = 0;
          }
        }
      }

      for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
        double out_x[v1][q1];
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
void rGridFuncToQuad(const int dim,
                     const int NUM_VDIM,
                     const int NUM_DOFS_1D,
                     const int NUM_QUAD_1D,
                     const int numElements,
                     const double* dofToQuad,
                     const int* l2gMap,
                     const double* gf,
                     double* __restrict out) {
  switch (dim) {
    case 1: rGridFuncToQuad1D(NUM_VDIM,NUM_DOFS_1D,NUM_QUAD_1D,numElements,dofToQuad,l2gMap,gf,out); break;
    case 2: rGridFuncToQuad2D(NUM_VDIM,NUM_DOFS_1D,NUM_QUAD_1D,numElements,dofToQuad,l2gMap,gf,out); break;
    case 3: rGridFuncToQuad3D(NUM_VDIM,NUM_DOFS_1D,NUM_QUAD_1D,numElements,dofToQuad,l2gMap,gf,out); break;
    default:assert(false);
  }
}
