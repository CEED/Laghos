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
#ifdef __TEMPLATES__
template<const int NUM_DIM,
         const int NUM_DOFS_1D,
         const int NUM_QUAD_1D,
         const int L2_DOFS_1D,
         const int H1_DOFS_1D> kernel
#endif
void rForceMultTranspose3D(
#ifndef __TEMPLATES__
                           const int NUM_DIM,
                           const int NUM_DOFS_1D,
                           const int NUM_QUAD_1D,
                           const int L2_DOFS_1D,
                           const int H1_DOFS_1D,
#endif
                           const int numElements,
                           const double* L2QuadToDof,
                           const double* H1DofToQuad,
                           const double* H1DofToQuadD,
                           const double* stressJinvT,
                           const double* v,
                                  double* e) {
  const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
  const int NUM_QUAD_3D = NUM_QUAD_1D*NUM_QUAD_1D*NUM_QUAD_1D;
#ifndef __NVCC__
  forall(el,numElements,
#else
  const int el = blockDim.x * blockIdx.x + threadIdx.x;
  if (el < numElements)
#endif
  {
    double vStress[NUM_QUAD_3D];
    for (int i = 0; i < NUM_QUAD_3D; ++i) {
      vStress[i] = 0;
    }
    for (int c = 0; c < NUM_DIM; ++c) {
      for (int dz = 0; dz < H1_DOFS_1D; ++dz) {
        double Dxy_x[NUM_QUAD_2D];
        double xDy_y[NUM_QUAD_2D];
        double xy_z[NUM_QUAD_2D] ;
        for (int i = 0; i < NUM_QUAD_2D; ++i) {
          Dxy_x[i] = xDy_y[i] = xy_z[i] = 0;
        }
        for (int dy = 0; dy < H1_DOFS_1D; ++dy) {
          double Dx_x[NUM_QUAD_1D];
          double x_y[NUM_QUAD_1D];
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
                 ((Dxy_x[ijN(qx,qy,NUM_QUAD_1D)]*wz *stressJinvT[__ijxyzeDQE(0,c,qx,qy,qz,el,NUM_DIM,NUM_QUAD_1D,numElements)]) +
                  (xDy_y[ijN(qx,qy,NUM_QUAD_1D)]*wz *stressJinvT[__ijxyzeDQE(1,c,qx,qy,qz,el,NUM_DIM,NUM_QUAD_1D,numElements)]) +
                  (xy_z[ijN(qx,qy,NUM_QUAD_1D)] *wDz*stressJinvT[__ijxyzeDQE(2,c,qx,qy,qz,el,NUM_DIM,NUM_QUAD_1D,numElements)]));
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
      double e_xy[L2_DOFS_1D * L2_DOFS_1D];
      for (int d = 0; d < (L2_DOFS_1D * L2_DOFS_1D); ++d) {
        e_xy[d] = 0;
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        double e_x[L2_DOFS_1D];
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
  }
#ifndef __NVCC__
         );
#endif
}

template kernel void rForceMultTranspose3D<3,2,2,1,2>(int,double const*,double const*,double const*,double const*,double const*,double*);
template kernel void rForceMultTranspose3D<3,3,4,2,3>(int,double const*,double const*,double const*,double const*,double const*,double*);
template kernel void rForceMultTranspose3D<3,4,6,3,4>(int,double const*,double const*,double const*,double const*,double const*,double*);
template kernel void rForceMultTranspose3D<3,5,8,4,5>(int,double const*,double const*,double const*,double const*,double const*,double*);
template kernel void rForceMultTranspose3D<3,6,10,5,6>(int,double const*,double const*,double const*,double const*,double const*,double*);
template kernel void rForceMultTranspose3D<3,7,12,6,7>(int,double const*,double const*,double const*,double const*,double const*,double*);
template kernel void rForceMultTranspose3D<3,8,14,7,8>(int,double const*,double const*,double const*,double const*,double const*,double*);
template kernel void rForceMultTranspose3D<3,9,16,8,9>(int,double const*,double const*,double const*,double const*,double const*,double*);
template kernel void rForceMultTranspose3D<3,10,18,9,10>(int,double const*,double const*,double const*,double const*,double const*,double*);
template kernel void rForceMultTranspose3D<3,11,20,10,11>(int,double const*,double const*,double const*,double const*,double const*,double*);
template kernel void rForceMultTranspose3D<3,12,22,11,12>(int,double const*,double const*,double const*,double const*,double const*,double*);
template kernel void rForceMultTranspose3D<3,13,24,12,13>(int,double const*,double const*,double const*,double const*,double const*,double*);
template kernel void rForceMultTranspose3D<3,14,26,13,14>(int,double const*,double const*,double const*,double const*,double const*,double*);
template kernel void rForceMultTranspose3D<3,15,28,14,15>(int,double const*,double const*,double const*,double const*,double const*,double*);
template kernel void rForceMultTranspose3D<3,16,30,15,16>(int,double const*,double const*,double const*,double const*,double const*,double*);
template kernel void rForceMultTranspose3D<3,17,32,16,17>(int,double const*,double const*,double const*,double const*,double const*,double*);
