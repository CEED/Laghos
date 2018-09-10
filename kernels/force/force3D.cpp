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
void rForceMult3D(
#ifndef __TEMPLATES__
                  const int NUM_DIM,
                  const int NUM_DOFS_1D,
                  const int NUM_QUAD_1D,
                  const int L2_DOFS_1D,
                  const int H1_DOFS_1D,
#endif
                  const int numElements,
                  const double* L2DofToQuad,
                  const double* H1QuadToDof,
                  const double* H1QuadToDofD,
                  const double* stressJinvT,
                  const double* e,
                  double* v) {
  const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
  const int NUM_QUAD_3D = NUM_QUAD_1D*NUM_QUAD_1D*NUM_QUAD_1D;
#ifndef __NVCC__
  forall(el,numElements,
#else
  const int el = blockDim.x * blockIdx.x + threadIdx.x;
  if (el < numElements)
#endif
  {
    double e_xyz[NUM_QUAD_3D];
    for (int i = 0; i < NUM_QUAD_3D; ++i) {
      e_xyz[i] = 0;
    }
    for (int dz = 0; dz < L2_DOFS_1D; ++dz) {
      double e_xy[NUM_QUAD_2D];
      for (int i = 0; i < NUM_QUAD_2D; ++i) {
        e_xy[i] = 0;
      }
      for (int dy = 0; dy < L2_DOFS_1D; ++dy) {
        double e_x[NUM_QUAD_1D];
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
        double Dxy_x[H1_DOFS_1D * H1_DOFS_1D];
        double xDy_y[H1_DOFS_1D * H1_DOFS_1D];
        double xy_z[H1_DOFS_1D * H1_DOFS_1D] ;
        for (int d = 0; d < (H1_DOFS_1D * H1_DOFS_1D); ++d) {
          Dxy_x[d] = xDy_y[d] = xy_z[d] = 0;
        }
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          double Dx_x[H1_DOFS_1D];
          double x_y[H1_DOFS_1D];
          double x_z[H1_DOFS_1D];
          for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
            Dx_x[dx] = x_y[dx] = x_z[dx] = 0;
          }
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            const double r_e = e_xyz[ijkN(qx,qy,qz,NUM_QUAD_1D)];
            const double esx = r_e * stressJinvT[__ijxyzeDQE(0,c,qx,qy,qz,el,NUM_DIM,NUM_QUAD_1D,numElements)];
            const double esy = r_e * stressJinvT[__ijxyzeDQE(1,c,qx,qy,qz,el,NUM_DIM,NUM_QUAD_1D,numElements)];
            const double esz = r_e * stressJinvT[__ijxyzeDQE(2,c,qx,qy,qz,el,NUM_DIM,NUM_QUAD_1D,numElements)];
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
  }
#ifndef __NVCC__
         );
#endif
}

template kernel void rForceMult3D<3,2,2,1,2>(int,double const*,double const*,double const*,double const*,double const*,double*);
template kernel void rForceMult3D<3,3,4,2,3>(int,double const*,double const*,double const*,double const*,double const*,double*);
template kernel void rForceMult3D<3,4,6,3,4>(int,double const*,double const*,double const*,double const*,double const*,double*);
template kernel void rForceMult3D<3,5,8,4,5>(int,double const*,double const*,double const*,double const*,double const*,double*);
template kernel void rForceMult3D<3,6,10,5,6>(int,double const*,double const*,double const*,double const*,double const*,double*);
template kernel void rForceMult3D<3,7,12,6,7>(int,double const*,double const*,double const*,double const*,double const*,double*);
template kernel void rForceMult3D<3,8,14,7,8>(int,double const*,double const*,double const*,double const*,double const*,double*);
template kernel void rForceMult3D<3,9,16,8,9>(int,double const*,double const*,double const*,double const*,double const*,double*);
template kernel void rForceMult3D<3,10,18,9,10>(int,double const*,double const*,double const*,double const*,double const*,double*);
template kernel void rForceMult3D<3,11,20,10,11>(int,double const*,double const*,double const*,double const*,double const*,double*);
template kernel void rForceMult3D<3,12,22,11,12>(int,double const*,double const*,double const*,double const*,double const*,double*);
template kernel void rForceMult3D<3,13,24,12,13>(int,double const*,double const*,double const*,double const*,double const*,double*);
template kernel void rForceMult3D<3,14,26,13,14>(int,double const*,double const*,double const*,double const*,double const*,double*);
template kernel void rForceMult3D<3,15,28,14,15>(int,double const*,double const*,double const*,double const*,double const*,double*);
template kernel void rForceMult3D<3,16,30,15,16>(int,double const*,double const*,double const*,double const*,double const*,double*);
template kernel void rForceMult3D<3,17,32,16,17>(int,double const*,double const*,double const*,double const*,double const*,double*);
