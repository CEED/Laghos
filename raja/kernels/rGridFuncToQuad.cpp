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

  forall(numElements,[&](int e){
    double r_out[NUM_VDIM*NUM_QUAD_1D];
    forall(NUM_VDIM,[&](int v){
      forall(NUM_QUAD_1D,[&](int qx){
          r_out[ijN(v,qx,NUM_VDIM)] = 0;
        });
      });
    forall(NUM_DOFS_1D,[&](int dx){
      const int gid = l2gMap[(dx) + (NUM_DOFS_1D) * (e)];
      forall(NUM_VDIM,[&](int v){
        const double r_gf = gf[v + gid * NUM_VDIM];
        forall(NUM_QUAD_1D,[&](int qx){
            r_out[ijN(v,qx,NUM_VDIM)] += r_gf * dofToQuad[(qx) + (NUM_QUAD_1D) * (dx)];
          });
        });
      });
    forall(NUM_QUAD_1D,[&](int qx){
        forall(NUM_VDIM,[&](int v){
          out[(qx) + (NUM_QUAD_1D) * ((e) + (numElements) * (v))] = r_out[ijN(v,qx,NUM_VDIM)];
        });
      });
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
  forall(numElements,[=](int e){
      double out_xy[NUM_VDIM*NUM_QUAD_1D*NUM_QUAD_1D];
      forall(NUM_VDIM,[&](int v){
          forall(NUM_QUAD_1D,[&](int qy){
              forall(NUM_QUAD_1D,[&](int qx){
                  out_xy[ijkNM(v,qy,qx,NUM_VDIM,NUM_QUAD_1D)] = 0;
                });
            });
        });
      forall(NUM_DOFS_1D,[&](int dy){
          double out_x[NUM_VDIM*NUM_QUAD_1D];
          forall(NUM_VDIM,[&](int v){
              forall(NUM_QUAD_1D,[&](int qy){
                  out_x[ijN(v,qy,NUM_VDIM)] = 0;
                });
            });
          forall(NUM_DOFS_1D,[&](int dx){
              const int gid = l2gMap[ijkN(dx, dy, e,NUM_DOFS_1D)];
              forall(NUM_VDIM,[&](int v){
                  const double r_gf = gf[v + gid*NUM_VDIM];
                  forall(NUM_QUAD_1D,[&](int qy){
                      out_x[ijN(v,qy,NUM_VDIM)] += r_gf * dofToQuad[ijN(qy, dx,NUM_QUAD_1D)];
                    });
                });
            });
          forall(NUM_VDIM,[&](int v){
              forall(NUM_QUAD_1D,[&](int qy){
                  const double d2q = dofToQuad[ijN(qy, dy,NUM_QUAD_1D)];
                  forall(NUM_QUAD_1D,[&](int qx){
                      out_xy[ijkNM(v,qy,qx,NUM_VDIM,NUM_QUAD_1D)] += d2q * out_x[ijN(v,qx,NUM_VDIM)];
                    });
                });
            });
        });
      forall(NUM_QUAD_1D,[&](int qy){
          forall(NUM_QUAD_1D,[&](int qx){
              forall(NUM_VDIM,[&](int v){
                  out[_ijklNM(v, qx, qy, e,NUM_QUAD_1D,numElements)] = out_xy[ijkNM(v,qy,qx,NUM_VDIM,NUM_QUAD_1D)];
                });
            });
        });
    });
}

// *****************************************************************************
static void rGridFuncToQuad3D(const int NUM_VDIM,
                              const int NUM_DOFS_1D,
                              const int NUM_QUAD_1D,
                              const int numElements,
                              const double*__restrict dofToQuad,
                              const int* __restrict l2gMap,
                              const double* gf,
                              double* __restrict out) {
  forall(numElements,[&](int e){
      double out_xyz[NUM_VDIM*NUM_QUAD_1D*NUM_QUAD_1D*NUM_QUAD_1D];
      forall(NUM_VDIM,[&](int v){
          forall(NUM_QUAD_1D,[&](int qz){
              forall(NUM_QUAD_1D,[&](int qy){
                  forall(NUM_QUAD_1D,[&](int qx){
                      out_xyz[_ijklNM(v,qz,qy,qx,NUM_QUAD_1D,NUM_VDIM)] = 0;
                    });
                });
            });
        });

      forall(NUM_DOFS_1D,[&](int dz){
          double out_xy[NUM_VDIM*NUM_QUAD_1D*NUM_QUAD_1D];
          forall(NUM_VDIM,[&](int v){
              forall(NUM_QUAD_1D,[&](int qy){
                  forall(NUM_QUAD_1D,[&](int qx){
                      out_xy[_ijkNM(v,qy,qx,NUM_QUAD_1D,NUM_VDIM)] = 0;
                    });
                });
            });

          forall(NUM_DOFS_1D,[&](int dy){
              double out_x[NUM_VDIM*NUM_QUAD_1D];
              forall(NUM_VDIM,[&](int v){
                  forall(NUM_QUAD_1D,[&](int qx){
                      out_x[ijN(v,qx,NUM_VDIM)] = 0;
                    });
                });

              forall(NUM_DOFS_1D,[&](int dx){
                  const int gid = l2gMap[ijklN(dx, dy, dz, e,NUM_DOFS_1D)];
                  forall(NUM_VDIM,[&](int v){
                      const double r_gf = gf[v + gid*NUM_VDIM];
                      forall(NUM_QUAD_1D,[&](int qx){
                          out_x[ijN(v,qx,NUM_VDIM)] += r_gf * dofToQuad[ijN(qx, dx, NUM_QUAD_1D)];
                        });
                    });
                });

              forall(NUM_QUAD_1D,[&](int qy){
                  const double wy = dofToQuad[ijN(qy, dy, NUM_QUAD_1D)];
                  forall(NUM_VDIM,[&](int v){
                      forall(NUM_QUAD_1D,[&](int qx){
                          out_xy[_ijkNM(v,qy,qx,NUM_QUAD_1D,NUM_VDIM)] += wy * out_x[ijN(v,qx,NUM_VDIM)];
                        });
                    });
                });
            });

          forall(NUM_QUAD_1D,[&](int qz){
              const double wz = dofToQuad[ijN(qz, dz, NUM_QUAD_1D)];
              forall(NUM_VDIM,[&](int v){
                  forall(NUM_QUAD_1D,[&](int qy){
                      forall(NUM_QUAD_1D,[&](int qx){
                          out_xyz[_ijklNM(v,qz,qy,qx,NUM_QUAD_1D,NUM_VDIM)] += wz * out_xy[_ijkNM(v,qy,qx,NUM_QUAD_1D,NUM_VDIM)];
                        });
                    });
                });
            });
        });

      forall(NUM_QUAD_1D,[&](int qz){
          forall(NUM_QUAD_1D,[&](int qy){
              forall(NUM_QUAD_1D,[&](int qx){
                  forall(NUM_VDIM,[&](int v){
                      out[_ijklmNM(v, qx, qy, qz, e,NUM_QUAD_1D,numElements)] = out_xyz[_ijklNM(v,qz,qy,qx,NUM_QUAD_1D,NUM_VDIM)];
                    });
                });
            });
        });
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
                     double* __restrict out){
  switch(dim){
  case 1: rGridFuncToQuad1D(NUM_VDIM,NUM_DOFS_1D,NUM_QUAD_1D,numElements,dofToQuad,l2gMap,gf,out);break;
  case 2: rGridFuncToQuad2D(NUM_VDIM,NUM_DOFS_1D,NUM_QUAD_1D,numElements,dofToQuad,l2gMap,gf,out);break;
  case 3: rGridFuncToQuad3D(NUM_VDIM,NUM_DOFS_1D,NUM_QUAD_1D,numElements,dofToQuad,l2gMap,gf,out);break;
  default:assert(false);
  }
}
