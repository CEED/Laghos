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
static void rForceMult2D(const int NUM_DIM,
                         const int NUM_DOFS_1D,
                         const int NUM_QUAD_1D,
                         const int NUM_QUAD_2D,
                         const int L2_DOFS_1D,
                         const int H1_DOFS_1D,
                         const int numElements,
                         const double* L2DofToQuad,
                         const double* H1QuadToDof,
                         const double* H1QuadToDofD,
                         const double* stressJinvT,
                         const double* e,
                         double* __restrict v) {
  //printf("\033[31m[%d]\033[m\n",NUM_QUAD_1D);
  //printf("\033[31m[%d]\033[m\n",NUM_QUAD_2D);
  //printf("\033[31m[%d]\033[m\n",H1_DOFS_1D);

  assert(NUM_QUAD_1D==4); const int q1 = 4;
  assert(NUM_QUAD_2D==16); const int q2 = 16;
  assert(H1_DOFS_1D==3);  const int h1 = 3;
  
  forall(numElements,[=]_device_(int el){
    double e_xy[q2];
    for (int i = 0; i < NUM_QUAD_2D; ++i) {
      e_xy[i] = 0;
    }
    for (int dy = 0; dy < L2_DOFS_1D; ++dy) {
      double e_x[q1];
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        e_x[qy] = 0;
      }
      for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
        const double r_e = e[ijkN(dx,dy,el,L2_DOFS_1D)];
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          e_x[qx] += L2DofToQuad[ijN(qx,dx,NUM_QUAD_1D)] * r_e;
        }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        const double wy = L2DofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          e_xy[ijN(qx,qy,NUM_QUAD_1D)] += wy * e_x[qx];
        }
      }
    }
    for (int c = 0; c < 2; ++c) {
      for (int dy = 0; dy < H1_DOFS_1D; ++dy) {
        for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
          v[_ijklNM(c,dx,dy,el,NUM_DOFS_1D,numElements)] = 0.0;
        }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        double Dxy[h1];
        double xy[h1];
        for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
          Dxy[dx] = 0.0;
          xy[dx]  = 0.0;
        }
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          const double esx = e_xy[ijN(qx,qy,NUM_QUAD_1D)] * stressJinvT[ijklmNM(0,c,qx,qy,el,NUM_DIM,NUM_QUAD_1D)];
          const double esy = e_xy[ijN(qx,qy,NUM_QUAD_1D)] * stressJinvT[ijklmNM(1,c,qx,qy,el,NUM_DIM,NUM_QUAD_1D)];
          for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
            Dxy[dx] += esx * H1QuadToDofD[ijN(dx,qx,H1_DOFS_1D)];
            xy[dx]  += esy * H1QuadToDof[ijN(dx,qx,H1_DOFS_1D)];
          }
        }
        for (int dy = 0; dy < H1_DOFS_1D; ++dy) {
          const double wy  = H1QuadToDof[ijN(dy,qy,H1_DOFS_1D)];
          const double wDy = H1QuadToDofD[ijN(dy,qy,H1_DOFS_1D)];
          for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
            v[_ijklNM(c,dx,dy,el,NUM_DOFS_1D,numElements)] += wy* Dxy[dx] + wDy*xy[dx];
          }
        }
      }
    }
    });
}

// *****************************************************************************
static void rForceMultTranspose2D(const int NUM_DIM,
                                  const int NUM_DOFS_1D,
                                  const int NUM_QUAD_1D,
                                  const int NUM_QUAD_2D,
                                  const int L2_DOFS_1D,
                                  const int H1_DOFS_1D,
                                  const int numElements,
                                  const double* L2QuadToDof,
                                  const double* H1DofToQuad,
                                  const double* H1DofToQuadD,
                                  const double* stressJinvT,
                                  const double* v,
                                  double* __restrict e) {
  //printf("\033[31m[%d]\033[m\n",NUM_QUAD_1D);
  //printf("\033[31m[%d]\033[m\n",NUM_QUAD_2D);
  //printf("\033[31m[%d]\033[m\n",L2_DOFS_1D);

  assert(NUM_QUAD_1D==4); const int q1 = 4;
  assert(NUM_QUAD_2D==16); const int q2 = 16;
  assert(L2_DOFS_1D==2);  const int l1 = 2;

  forall(numElements,[=]_device_(int el){
    double vStress[q2];
    for (int i = 0; i < NUM_QUAD_2D; ++i) {
      vStress[i] = 0;
    }
    for (int c = 0; c < NUM_DIM; ++c) {
      double v_Dxy[q2];
      double v_xDy[q2];
      for (int i = 0; i < NUM_QUAD_2D; ++i) {
        v_Dxy[i] = v_xDy[i] = 0;
      }
      for (int dy = 0; dy < H1_DOFS_1D; ++dy) {
        double v_x[q1];
        double v_Dx[q1];
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          v_x[qx] = v_Dx[qx] = 0;
        }

        for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
          const double r_v = v[_ijklNM(c,dx,dy,el,NUM_DOFS_1D,numElements)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            v_x[qx]  += r_v * H1DofToQuad[ijN(qx,dx,NUM_QUAD_1D)];
            v_Dx[qx] += r_v * H1DofToQuadD[ijN(qx,dx,NUM_QUAD_1D)];
          }
        }
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          const double wy  = H1DofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
          const double wDy = H1DofToQuadD[ijN(qy,dy,NUM_QUAD_1D)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            v_Dxy[ijN(qx,qy,NUM_QUAD_1D)] += v_Dx[qx] * wy;
            v_xDy[ijN(qx,qy,NUM_QUAD_1D)] += v_x[qx]  * wDy;
          }
        }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          vStress[ijN(qx,qy,NUM_QUAD_1D)] +=
            ((v_Dxy[ijN(qx,qy,NUM_QUAD_1D)] * stressJinvT[ijklmNM(0,c,qx,qy,el,NUM_DIM,NUM_QUAD_1D)]) +
             (v_xDy[ijN(qx,qy,NUM_QUAD_1D)] * stressJinvT[ijklmNM(1,c,qx,qy,el,NUM_DIM,NUM_QUAD_1D)]));
        }
      }
    }
    for (int dy = 0; dy < L2_DOFS_1D; ++dy) {
      for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
        e[ijkN(dx,dy,el,L2_DOFS_1D)] = 0;
      }
    }
    for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
      double e_x[l1];
      for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
        e_x[dx] = 0;
      }
      for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
        const double r_v = vStress[ijN(qx,qy,NUM_QUAD_1D)];
        for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
          e_x[dx] += r_v * L2QuadToDof[ijN(dx,qx,L2_DOFS_1D)];
        }
      }
      for (int dy = 0; dy < L2_DOFS_1D; ++dy) {
        const double w = L2QuadToDof[ijN(dy,qy,L2_DOFS_1D)];
        for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
          e[ijkN(dx,dy,el,L2_DOFS_1D)] += e_x[dx] * w;
        }
      }
    }
    });
}

// *****************************************************************************
static void rForceMult3D(const int NUM_DIM,
                         const int NUM_DOFS_1D,
                         const int NUM_QUAD_1D,
                         const int NUM_QUAD_2D,
                         const int NUM_QUAD_3D,
                         const int L2_DOFS_1D,
                         const int H1_DOFS_1D,
                         const int numElements,
                         const double* L2DofToQuad,
                         const double* H1QuadToDof,
                         const double* H1QuadToDofD,
                         const double* stressJinvT,
                         const double* e,
                         double* __restrict v) {
  
  assert(NUM_QUAD_1D==2); const int q1 = 2;
  assert(NUM_QUAD_2D==4); const int q2 = 4;
  assert(NUM_QUAD_3D==8); const int q3 = 8;
  assert(H1_DOFS_1D==2);  const int h1 = 2;
  
  forall(numElements,[=]_device_(int el){
    double e_xyz[q3];
    for (int i = 0; i < NUM_QUAD_3D; ++i) {
      e_xyz[i] = 0;
    }
    for (int dz = 0; dz < L2_DOFS_1D; ++dz) {
      double e_xy[q2];
      for (int i = 0; i < NUM_QUAD_2D; ++i) {
        e_xy[i] = 0;
      }
      for (int dy = 0; dy < L2_DOFS_1D; ++dy) {
        double e_x[q1];
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          e_x[qy] = 0;
        }
        for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
          const double r_e = e[ijklN(dx,dy,dz,el,L2_DOFS_1D)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            e_x[qx] += L2DofToQuad[ijN(qx,dx,NUM_QUAD_1D)] * r_e;
          }
        }
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          const double wy = L2DofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            e_xy[ijN(qx,qy,NUM_QUAD_1D)] += wy * e_x[qx];
          }
        }
      }
      for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
        const double wz = L2DofToQuad[ijN(qz,dz,NUM_QUAD_1D)];
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            e_xyz[ijkN(qx,qy,qz,NUM_QUAD_1D)] += wz * e_xy[ijN(qx,qy,NUM_QUAD_1D)];
          }
        }
      }
    }
    for (int c = 0; c < 3; ++c) {
      for (int dz = 0; dz < H1_DOFS_1D; ++dz) {
        for (int dy = 0; dy < H1_DOFS_1D; ++dy) {
          for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
            v[_ijklmNM(c,dx,dy,dz,el,NUM_DOFS_1D,numElements)] = 0;
          }
        }
      }
      for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
        double Dxy_x[h1 * h1];
        double xDy_y[h1 * h1];
        double xy_z[h1 * h1] ;
        for (int d = 0; d < (H1_DOFS_1D * H1_DOFS_1D); ++d) {
          Dxy_x[d] = xDy_y[d] = xy_z[d] = 0;
        }
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          double Dx_x[h1];
          double x_y[h1];
          double x_z[h1];
          for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
            Dx_x[dx] = x_y[dx] = x_z[dx] = 0;
          }
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            const double r_e = e_xyz[ijkN(qx,qy,qz,NUM_QUAD_1D)];
            const double esx = r_e * stressJinvT[ijklmnNM(0,c,qx,qy,qz,el,NUM_DIM,NUM_QUAD_1D)];
            const double esy = r_e * stressJinvT[ijklmnNM(1,c,qx,qy,qz,el,NUM_DIM,NUM_QUAD_1D)];
            const double esz = r_e * stressJinvT[ijklmnNM(2,c,qx,qy,qz,el,NUM_DIM,NUM_QUAD_1D)];
            for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
              Dx_x[dx] += esx * H1QuadToDofD[ijN(dx,qx,H1_DOFS_1D)];
              x_y[dx]  += esy * H1QuadToDof[ijN(dx,qx,H1_DOFS_1D)];
              x_z[dx]  += esz * H1QuadToDof[ijN(dx,qx,H1_DOFS_1D)];
            }
          }
          for (int dy = 0; dy < H1_DOFS_1D; ++dy) {
            const double wy  = H1QuadToDof[ijN(dy,qy,H1_DOFS_1D)];
            const double wDy = H1QuadToDofD[ijN(dy,qy,H1_DOFS_1D)];
            for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
              Dxy_x[ijN(dx,dy,H1_DOFS_1D)] += Dx_x[dx] * wy;
              xDy_y[ijN(dx,dy,H1_DOFS_1D)] += x_y[dx]  * wDy;
              xy_z[ijN(dx,dy,H1_DOFS_1D)]  += x_z[dx]  * wy;
            }
          }
        }
        for (int dz = 0; dz < H1_DOFS_1D; ++dz) {
          const double wz  = H1QuadToDof[ijN(dz,qz,H1_DOFS_1D)];
          const double wDz = H1QuadToDofD[ijN(dz,qz,H1_DOFS_1D)];
          for (int dy = 0; dy < H1_DOFS_1D; ++dy) {
            for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
              v[_ijklmNM(c,dx,dy,dz,el,NUM_DOFS_1D,numElements)] +=
                ((Dxy_x[ijN(dx,dy,H1_DOFS_1D)] * wz) +
                 (xDy_y[ijN(dx,dy,H1_DOFS_1D)] * wz) +
                 (xy_z[ijN(dx,dy,H1_DOFS_1D)]  * wDz));
            }
          }
        }
      }
    }
    });
}

// *****************************************************************************
static void rForceMultTranspose3D(const int NUM_DIM,
                                  const int NUM_DOFS_1D,
                                  const int NUM_QUAD_1D,
                                  const int NUM_QUAD_2D,
                                  const int NUM_QUAD_3D,
                                  const int L2_DOFS_1D,
                                  const int H1_DOFS_1D,
                                  const int numElements,
                                  const double* L2QuadToDof,
                                  const double* H1DofToQuad,
                                  const double* H1DofToQuadD,
                                  const double* stressJinvT,
                                  const double* v,
                                  double* __restrict e) {
  assert(NUM_QUAD_1D==2); const int q1 = 2;
  assert(NUM_QUAD_2D==4); const int q2 = 4;
  assert(NUM_QUAD_3D==8); const int q3 = 8;
  assert(L2_DOFS_1D==2);  const int l1 = 2;

  forall(numElements,[=]_device_(int el){
    double vStress[q3];
    for (int i = 0; i < NUM_QUAD_3D; ++i) {
      vStress[i] = 0;
    }
    for (int c = 0; c < NUM_DIM; ++c) {
      for (int dz = 0; dz < H1_DOFS_1D; ++dz) {
        double Dxy_x[q2];
        double xDy_y[q2];
        double xy_z[q2] ;
        for (int i = 0; i < NUM_QUAD_2D; ++i) {
          Dxy_x[i] = xDy_y[i] = xy_z[i] = 0;
        }
        for (int dy = 0; dy < H1_DOFS_1D; ++dy) {
          double Dx_x[q1];
          double x_y[q1];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            Dx_x[qx] = x_y[qx] = 0;
          }
          for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
            const double r_v = v[_ijklmNM(c,dx,dy,dz,el,NUM_DOFS_1D,numElements)];
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
              Dx_x[qx] += r_v * H1DofToQuadD[ijN(qx,dx,NUM_QUAD_1D)];
              x_y[qx]  += r_v * H1DofToQuad[ijN(qx,dx,NUM_QUAD_1D)];
            }
          }
          for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
            const double wy  = H1DofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
            const double wDy = H1DofToQuadD[ijN(qy,dy,NUM_QUAD_1D)];
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
              Dxy_x[ijN(qx,qy,NUM_QUAD_1D)] += Dx_x[qx] * wy;
              xDy_y[ijN(qx,qy,NUM_QUAD_1D)] += x_y[qx]  * wDy;
              xy_z[ijN(qx,qy,NUM_QUAD_1D)]  += x_y[qx]  * wy;
            }
          }
        }
        for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
          const double wz  = H1DofToQuad[ijN(qz,dz,NUM_QUAD_1D)];
          const double wDz = H1DofToQuadD[ijN(qz,dz,NUM_QUAD_1D)];
          for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
              vStress[ijkN(qx,qy,qz,NUM_QUAD_1D)] +=
                ((Dxy_x[ijN(qx,qy,NUM_QUAD_1D)]*wz *stressJinvT[ijklmnNM(0,c,qx,qy,qz,el,NUM_DIM,NUM_QUAD_1D)]) +
                 (xDy_y[ijN(qx,qy,NUM_QUAD_1D)]*wz *stressJinvT[ijklmnNM(1,c,qx,qy,qz,el,NUM_DIM,NUM_QUAD_1D)]) +
                 (xy_z[ijN(qx,qy,NUM_QUAD_1D)] *wDz*stressJinvT[ijklmnNM(2,c,qx,qy,qz,el,NUM_DIM,NUM_QUAD_1D)]));
            }
          }
        }
      }
    }
    for (int dz = 0; dz < L2_DOFS_1D; ++dz) {
      for (int dy = 0; dy < L2_DOFS_1D; ++dy) {
        for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
          e[ijklN(dx,dy,dz,el,L2_DOFS_1D)] = 0;
        }
      }
    }
    for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
      double e_xy[l1 * l1];
      for (int d = 0; d < (L2_DOFS_1D * L2_DOFS_1D); ++d) {
        e_xy[d] = 0;
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        double e_x[l1];
        for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
          e_x[dx] = 0;
        }
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          const double r_v = vStress[ijkN(qx,qy,qz,NUM_QUAD_1D)];
          for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
            e_x[dx] += r_v * L2QuadToDof[ijN(dx,qx,L2_DOFS_1D)];
          }
        }
        for (int dy = 0; dy < L2_DOFS_1D; ++dy) {
          const double w = L2QuadToDof[ijN(dy,qy,L2_DOFS_1D)];
          for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
            e_xy[ijN(dx,dy,L2_DOFS_1D)] += e_x[dx] * w;
          }
        }
      }
      for (int dz = 0; dz < L2_DOFS_1D; ++dz) {
        const double w = L2QuadToDof[ijN(dz,qz,L2_DOFS_1D)];
        for (int dy = 0; dy < L2_DOFS_1D; ++dy) {
          for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
            e[ijklN(dx,dy,dz,el,L2_DOFS_1D)] += w * e_xy[ijN(dx,dy,L2_DOFS_1D)];
          }
        }
      }
    }
    });
}

// *****************************************************************************
void rForceMult(const int NUM_DIM,
                const int NUM_DOFS_1D,
                const int NUM_QUAD_1D,
                const int L2_DOFS_1D,
                const int H1_DOFS_1D,
                const int nzones,
                const double* L2QuadToDof,
                const double* H1DofToQuad,
                const double* H1DofToQuadD,
                const double* stressJinvT,
                const double* e,
                double* __restrict v) {
  if (NUM_DIM==1) assert(false);
  if (NUM_DIM==2)
    rForceMult2D(NUM_DIM,
                 NUM_DOFS_1D,
                 NUM_QUAD_1D,
                 NUM_QUAD_1D*NUM_QUAD_1D,
                 L2_DOFS_1D,
                 H1_DOFS_1D,
                 nzones,
                 L2QuadToDof,
                 H1DofToQuad,
                 H1DofToQuadD,
                 stressJinvT,
                 e,v);

  if (NUM_DIM==3)
    rForceMult3D(NUM_DIM,
                 NUM_DOFS_1D,
                 NUM_QUAD_1D,
                 NUM_QUAD_1D*NUM_QUAD_1D,
                 NUM_QUAD_1D*NUM_QUAD_1D*NUM_QUAD_1D,
                 L2_DOFS_1D,
                 H1_DOFS_1D,
                 nzones,
                 L2QuadToDof,
                 H1DofToQuad,
                 H1DofToQuadD,
                 stressJinvT,
                 e,v);
}

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
                         double* __restrict e) {
  if (NUM_DIM==1) assert(false);

  if (NUM_DIM==2)
    rForceMultTranspose2D(NUM_DIM,
                          NUM_DOFS_1D,
                          NUM_QUAD_1D,
                          NUM_QUAD_1D*NUM_QUAD_1D,
                          L2_DOFS_1D,
                          H1_DOFS_1D,
                          nzones,
                          L2QuadToDof,
                          H1DofToQuad,
                          H1DofToQuadD,
                          stressJinvT,
                          v,e);
  if (NUM_DIM==3)
    rForceMultTranspose3D(NUM_DIM,
                          NUM_DOFS_1D,
                          NUM_QUAD_1D,
                          NUM_QUAD_1D*NUM_QUAD_1D,
                          NUM_QUAD_1D*NUM_QUAD_1D*NUM_QUAD_1D,
                          L2_DOFS_1D,
                          H1_DOFS_1D,
                          nzones,
                          L2QuadToDof,
                          H1DofToQuad,
                          H1DofToQuadD,
                          stressJinvT,
                          v,e);
}
