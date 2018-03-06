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
template<const int NUM_DIM,
         const int NUM_DOFS_1D,
         const int NUM_QUAD_1D,
         const int L2_DOFS_1D,
         const int H1_DOFS_1D> kernel
#endif
void rForceMult2S(
#ifndef __TEMPLATES__
                  const int NUM_DIM,
                  const int NUM_DOFS_1D,
                  const int NUM_QUAD_1D,
                  const int L2_DOFS_1D,
                  const int H1_DOFS_1D,
#endif
                  const int numElements,
                  const double* restrict L2DofToQuad,
                  const double* restrict H1QuadToDof,
                  const double* restrict H1QuadToDofD,
                  const double* restrict stressJinvT,
                  const double* restrict e,
                  double* restrict v) {
  const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
  //const int NUM_QUAD = NUM_QUAD_2D;
  const int MAX_DOFS_1D = (L2_DOFS_1D > H1_DOFS_1D)?L2_DOFS_1D:H1_DOFS_1D;
  const int H1_MAX_1D = (H1_DOFS_1D > NUM_QUAD_1D)?H1_DOFS_1D:NUM_QUAD_1D;
  const int L2_MAX_1D = (L2_DOFS_1D > NUM_QUAD_1D)?L2_DOFS_1D:NUM_QUAD_1D;
  const int INNER_SIZE = (H1_MAX_1D > L2_MAX_1D)?H1_MAX_1D:L2_MAX_1D;
  //printf("INNER_SIZE=%d",INNER_SIZE);
#ifdef __LAMBDA__
  forallS(elBlock,numElements,ELEMENT_BATCH,
#else
  const int idx = blockIdx.x;
  const int elBlock = idx * ELEMENT_BATCH;
  if (elBlock < numElements)
#endif
  {
    share double s_L2DofToQuad[NUM_QUAD_1D * L2_DOFS_1D];
    share double s_H1QuadToDof[H1_DOFS_1D  * NUM_QUAD_1D];
    share double s_H1QuadToDofD[H1_DOFS_1D * NUM_QUAD_1D];
    
    share double s_xy[MAX_DOFS_1D * NUM_QUAD_1D];
    share double s_xDy[H1_DOFS_1D * NUM_QUAD_1D];
    share double s_e[NUM_QUAD_2D];

#ifdef __LAMBDA__
    for (int idBlock = 0; idBlock < INNER_SIZE; ++idBlock/*;inner*/)
#else
    const int idBlock = 0 + threadIdx.x;
#endif
    {
      for (int id = idBlock; id < (L2_DOFS_1D * NUM_QUAD_1D); id += INNER_SIZE) {
        s_L2DofToQuad[id] = L2DofToQuad[id];
      }
      for (int id = idBlock; id < (H1_DOFS_1D * NUM_QUAD_1D); id += INNER_SIZE) {
        s_H1QuadToDof[id]  = H1QuadToDof[id];
        s_H1QuadToDofD[id] = H1QuadToDofD[id];
      }
    }
    for (int el = elBlock; el < (elBlock + ELEMENT_BATCH); ++el) {
      if (el < numElements) {
        sync;
#ifdef __LAMBDA__
        for (int dx = 0; dx < INNER_SIZE; ++dx/*;inner*/) 
#else
        const int dx = 0 + threadIdx.x;
#endif
        {
          if (dx < L2_DOFS_1D) {
            double r_x[L2_DOFS_1D];
            for (int dy = 0; dy < L2_DOFS_1D; ++dy) {
              r_x[dy] = e[ijkN(dx,dy,el,L2_DOFS_1D)];
            }
            for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
              double xy = 0;
              for (int dy = 0; dy < L2_DOFS_1D; ++dy) {
                xy += r_x[dy]*s_L2DofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
              }
              s_xy[ijN(dx,qy,MAX_DOFS_1D)] = xy;
            }
          }
        }
        sync;
#ifdef __LAMBDA__
        for (int qy = 0; qy < INNER_SIZE; ++qy/*;inner*/) 
#else
        const int qy = 0 + threadIdx.x;
#endif
        {
          if (qy < NUM_QUAD_1D) {
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
              double r_e = 0;
              for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
                r_e += s_xy[ijN(dx,qy,MAX_DOFS_1D)]*s_L2DofToQuad[ijN(qx,dx,NUM_QUAD_1D)];
              }
              s_e[ijN(qx,qy,NUM_QUAD_1D)] = r_e;
            }
          }
        }

        for (int c = 0; c < NUM_DIM; ++c) {
          sync;
#ifdef __LAMBDA__
          for (int qx = 0; qx < INNER_SIZE; ++qx/*;inner*/)
#else
          const int qx = 0 + threadIdx.x;
#endif
          {
            if (qx < NUM_QUAD_1D) {
              double r_x[NUM_QUAD_1D];
              double r_y[NUM_QUAD_1D];

              for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
                const double r_e = s_e[(qx) + (NUM_QUAD_1D) * (qy)];
                r_x[qy] = r_e * stressJinvT[ijklmNM(0,c,qx,qy,el,NUM_DIM,NUM_QUAD_1D)];
                r_y[qy] = r_e * stressJinvT[ijklmNM(1,c,qx,qy,el,NUM_DIM,NUM_QUAD_1D)];
              }
              for (int dy = 0; dy < H1_DOFS_1D; ++dy) {
                double xy  = 0;
                double xDy = 0;
                for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
                  xy  += r_x[qy] * s_H1QuadToDof[ijN(dy,qy,H1_DOFS_1D)];
                  xDy += r_y[qy] * s_H1QuadToDofD[ijN(dy,qy,H1_DOFS_1D)];
                }
                s_xy[ijN(dy,qx,MAX_DOFS_1D)] = xy;
                s_xDy[ijN(dy,qx,H1_DOFS_1D)] = xDy;
              }
            }
          }
          sync;
#ifdef __LAMBDA__
          for (int dx = 0; dx < INNER_SIZE; ++dx/*;inner*/) 
#else
          const int dx = 0 + threadIdx.x;
#endif
          {
            if (dx < H1_DOFS_1D) {
              for (int dy = 0; dy < H1_DOFS_1D; ++dy) {
                double r_v = 0;
                for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
                  r_v += ((s_xy[ijN(dy,qx,MAX_DOFS_1D)] * s_H1QuadToDofD[ijN(dx,qx,H1_DOFS_1D)]) +
                          (s_xDy[ijN(dy,qx,H1_DOFS_1D)] * s_H1QuadToDof[ijN(dx,qx,H1_DOFS_1D)]));
                }
                v[ijklNM(dx,dy,el,c,NUM_DOFS_1D,numElements)] = r_v;
              }
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
#ifdef __TEMPLATES__
template<const int NUM_DIM,
         const int NUM_DOFS_1D,
         const int NUM_QUAD_1D,
         const int L2_DOFS_1D,
         const int H1_DOFS_1D> kernel
#endif
void rForceMultTranspose2S(
#ifndef __TEMPLATES__
                           const int NUM_DIM,
                           const int NUM_DOFS_1D,
                           const int NUM_QUAD_1D,
                           const int L2_DOFS_1D,
                           const int H1_DOFS_1D,
#endif
                           const int numElements,
                           const double* restrict L2QuadToDof,
                           const double* restrict H1DofToQuad,
                           const double* restrict H1DofToQuadD,
                           const double* restrict stressJinvT,
                           const double* restrict v,
                           double* restrict e) {
  const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
  const int NUM_QUAD = NUM_QUAD_2D;
  const int MAX_DOFS_1D = (L2_DOFS_1D > H1_DOFS_1D)?L2_DOFS_1D:H1_DOFS_1D;
  const int H1_MAX_1D = (H1_DOFS_1D > NUM_QUAD_1D)?H1_DOFS_1D:NUM_QUAD_1D;
  const int L2_MAX_1D = (L2_DOFS_1D > NUM_QUAD_1D)?L2_DOFS_1D:NUM_QUAD_1D;
  const int INNER_SIZE = (H1_MAX_1D > L2_MAX_1D)?H1_MAX_1D:L2_MAX_1D;
#ifdef __LAMBDA__
  forallS(elBlock,numElements,ELEMENT_BATCH,
#else
  const int idx = blockIdx.x;
  const int elBlock = idx * ELEMENT_BATCH;
  if (elBlock < numElements)
#endif
  {
    share double s_L2QuadToDof[NUM_QUAD_1D * L2_DOFS_1D];
    share double s_H1DofToQuad[H1_DOFS_1D  * NUM_QUAD_1D];
    share double s_H1DofToQuadD[H1_DOFS_1D * NUM_QUAD_1D];

    share double s_xy[MAX_DOFS_1D * NUM_QUAD_1D];
    share double s_xDy[H1_DOFS_1D * NUM_QUAD_1D];
    share double s_v[NUM_QUAD_1D  * NUM_QUAD_1D];

#ifdef __LAMBDA__
    for (int idBlock = 0; idBlock < INNER_SIZE; ++idBlock/*; inner*/) 
#else
    const int idBlock = 0 + threadIdx.x;
#endif
    {
      for (int id = idBlock; id < (L2_DOFS_1D * NUM_QUAD_1D); id += INNER_SIZE) {
        s_L2QuadToDof[id] = L2QuadToDof[id];
      }
      for (int id = idBlock; id < (H1_DOFS_1D * NUM_QUAD_1D); id += INNER_SIZE) {
        s_H1DofToQuad[id]  = H1DofToQuad[id];
        s_H1DofToQuadD[id] = H1DofToQuadD[id];
      }
    }

    for (int el = elBlock; el < (elBlock + ELEMENT_BATCH); ++el) {
      if (el < numElements) {
        sync;
#ifdef __LAMBDA__
        for (int qBlock = 0; qBlock < INNER_SIZE; ++qBlock/*; inner*/) 
#else
        const int qBlock = threadIdx.x;
#endif
        {
          for (int q = qBlock; q < NUM_QUAD; ++q) {
            s_v[q] = 0;
          }
        }
        for (int c = 0; c < NUM_DIM; ++c) {
          sync;
#ifdef __LAMBDA__
          for (int dx = 0; dx < INNER_SIZE; ++dx/*; inner*/)
#else
          const int dx = threadIdx.x;
#endif
          {
            if (dx < H1_DOFS_1D) {
              double r_v[H1_DOFS_1D];

              for (int dy = 0; dy < H1_DOFS_1D; ++dy) {
                r_v[dy] = v[ijklNM(dx,dy,el,c,H1_DOFS_1D,numElements)];
              }
              for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
                double xy  = 0;
                double xDy = 0;
                for (int dy = 0; dy < H1_DOFS_1D; ++dy) {
                  xy  += r_v[dy] * s_H1DofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
                  xDy += r_v[dy] * s_H1DofToQuadD[ijN(qy,dy,NUM_QUAD_1D)];
                }
                s_xy[ijN(qy,dx,NUM_QUAD_1D)]  = xy;
                s_xDy[ijN(qy,dx,NUM_QUAD_1D)] = xDy;
              }
            }
          }
          sync;
#ifdef __LAMBDA__
          for (int qx = 0; qx < INNER_SIZE; ++qx/*; inner*/)
#else
            const int qx = threadIdx.x;
#endif
          {
            if (qx < NUM_QUAD_1D) {
              for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
                double Dxy = 0;
                double xDy = 0;
                for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
                  Dxy += (s_xy[ijN(qy,dx,NUM_QUAD_1D)] * s_H1DofToQuadD[ijN(qx,dx,NUM_QUAD_1D)]);
                  xDy += (s_xDy[ijN(qy,dx,NUM_QUAD_1D)] * s_H1DofToQuad[ijN(qx,dx,NUM_QUAD_1D)]);
                }
                s_v[ijN(qx,qy,NUM_QUAD_1D)] += ((Dxy * stressJinvT[ijklmNM(0,c,qx,qy,el,NUM_DIM,NUM_QUAD_1D)]) +
                                                (xDy * stressJinvT[ijklmNM(1,c,qx,qy,el,NUM_DIM,NUM_QUAD_1D)]));
              }
            }
          }
        }
        sync;
#ifdef __LAMBDA__
        for (int qx = 0; qx < INNER_SIZE; ++qx/*; inner*/) 
#else
          const int qx = threadIdx.x;
#endif
        {
          if (qx < NUM_QUAD_1D) {
            double r_x[NUM_QUAD_1D];
            for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
              r_x[qy] = s_v[ijN(qx,qy,NUM_QUAD_1D)];
            }
            for (int dy = 0; dy < L2_DOFS_1D; ++dy) {
              double xy = 0;
              for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
                xy += r_x[qy] * s_L2QuadToDof[ijN(dy,qy,L2_DOFS_1D)];
              }
              s_xy[ijN(qx,dy,NUM_QUAD_1D)] = xy;
            }
          }
        }
        sync;
#ifdef __LAMBDA__
        for (int dy = 0; dy < INNER_SIZE; ++dy/*; inner*/)
#else
          const int dy = threadIdx.x;
#endif
        {
          if (dy < L2_DOFS_1D) {
            for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
              double r_e = 0;
              for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
                r_e += s_xy[ijN(qx,dy,NUM_QUAD_1D)] * s_L2QuadToDof[ijN(dx,qx,L2_DOFS_1D)];
              }
              e[ijkN(dx,dy,el,L2_DOFS_1D)] = r_e;
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
typedef void (*fForceMult2S)(const int numElements,
                             const double* restrict L2DofToQuad,
                             const double* restrict H1QuadToDof,
                             const double* restrict H1QuadToDofD,
                             const double* restrict stressJinvT,
                             const double* restrict e,
                             double* restrict v);

// *****************************************************************************
void rForceMultS(const int NUM_DIM,
                 const int NUM_DOFS_1D,
                 const int NUM_QUAD_1D,
                 const int L2_DOFS_1D,
                 const int H1_DOFS_1D,
                 const int nzones,
                 const double* restrict L2QuadToDof,
                 const double* restrict H1DofToQuad,
                 const double* restrict H1DofToQuadD,
                 const double* restrict stressJinvT,
                 const double* restrict e,
                 double* restrict v) {
  push(Green);
#ifndef __LAMBDA__
  const int H1_MAX_1D = (H1_DOFS_1D > NUM_QUAD_1D)?H1_DOFS_1D:NUM_QUAD_1D;
  const int L2_MAX_1D = (L2_DOFS_1D > NUM_QUAD_1D)?L2_DOFS_1D:NUM_QUAD_1D;
  const int INNER_SIZE = (H1_MAX_1D > L2_MAX_1D)?H1_MAX_1D:L2_MAX_1D;
  const int grid = ((nzones+ELEMENT_BATCH-1)/ELEMENT_BATCH);
  const int blck = INNER_SIZE;
#endif
#ifdef __TEMPLATES__
  assert(NUM_QUAD_1D==2*(NUM_DOFS_1D-1));
  assert(NUM_DOFS_1D==H1_DOFS_1D);
  assert(L2_DOFS_1D==NUM_DOFS_1D-1);
  const unsigned long long id =((NUM_DIM)<<4)|(NUM_DOFS_1D-2);
  assert(LOG2(NUM_DIM)<=4);
  assert(LOG2(NUM_DOFS_1D-2)<=4);
  static std::unordered_map<unsigned long long, fForceMult2S> call = {
    {0x20,&rForceMult2S<2,2,2,1,2>},
    {0x21,&rForceMult2S<2,3,4,2,3>},
    {0x22,&rForceMult2S<2,4,6,3,4>},
    {0x23,&rForceMult2S<2,5,8,4,5>},
    {0x24,&rForceMult2S<2,6,10,5,6>},
    {0x25,&rForceMult2S<2,7,12,6,7>},
    {0x26,&rForceMult2S<2,8,14,7,8>},
    {0x27,&rForceMult2S<2,9,16,8,9>},
    {0x28,&rForceMult2S<2,10,18,9,10>},
    {0x29,&rForceMult2S<2,11,20,10,11>},
    {0x2A,&rForceMult2S<2,12,22,11,12>},
    {0x2B,&rForceMult2S<2,13,24,12,13>},
    {0x2C,&rForceMult2S<2,14,26,13,14>},
    {0x2D,&rForceMult2S<2,15,28,14,15>},
    {0x2E,&rForceMult2S<2,16,30,15,16>},
    {0x2F,&rForceMult2S<2,17,32,16,17>},
    // 3D
/*    {0x30,&rForceMult3S<3,2,2,1,2>},
    {0x31,&rForceMult3S<3,3,4,2,3>},
    {0x32,&rForceMult3S<3,4,6,3,4>},
    {0x33,&rForceMult3S<3,5,8,4,5>},
    {0x34,&rForceMult3S<3,6,10,5,6>},
    {0x35,&rForceMult3S<3,7,12,6,7>},
    {0x36,&rForceMult3S<3,8,14,7,8>},
    {0x37,&rForceMult3S<3,9,16,8,9>},
    {0x38,&rForceMult3S<3,10,18,9,10>},
    {0x39,&rForceMult3S<3,11,20,10,11>},
    {0x3A,&rForceMult3S<3,12,22,11,12>},
    {0x3B,&rForceMult3S<3,13,24,12,13>},
    {0x3C,&rForceMult3S<3,14,26,13,14>},
    {0x3D,&rForceMult3S<3,15,28,14,15>},
    {0x3E,&rForceMult3S<3,16,30,15,16>},
    {0x3F,&rForceMult3S<3,17,32,16,17>},
*/  };
  if (!call[id]){
    printf("\n[rForceMult] id \033[33m0x%llX\033[m ",id);
    fflush(stdout);
  }
  assert(call[id]);
  call0(rForceMult2S,id,grid,blck,
        nzones,L2QuadToDof,H1DofToQuad,H1DofToQuadD,stressJinvT,e,v);
#else
  if (NUM_DIM==2)
    call0(rForceMult2S,id,grid,blck,
          NUM_DIM,NUM_DOFS_1D,NUM_QUAD_1D,L2_DOFS_1D,H1_DOFS_1D,
          nzones,L2QuadToDof,H1DofToQuad,H1DofToQuadD,stressJinvT,e,v);
  if (NUM_DIM!=2)
    exit(printf("\n[rForceMultS] NUM_DIM!=2 ERROR"));    
#endif
  pop(); 
}


// *****************************************************************************
typedef void (*fForceMultTransposeS)(const int numElements,
                                     const double* restrict L2QuadToDof,
                                     const double* restrict H1DofToQuad,
                                     const double* restrict H1DofToQuadD,
                                     const double* restrict stressJinvT,
                                     const double* restrict v,
                                     double* restrict e);

// *****************************************************************************
void rForceMultTransposeS(const int NUM_DIM,
                          const int NUM_DOFS_1D,
                          const int NUM_QUAD_1D,
                          const int L2_DOFS_1D,
                          const int H1_DOFS_1D,
                          const int nzones,
                          const double* restrict L2QuadToDof,
                          const double* restrict H1DofToQuad,
                          const double* restrict H1DofToQuadD,
                          const double* restrict stressJinvT,
                          const double* restrict v,
                          double* restrict e) {
  push(Green);
#ifndef __LAMBDA__
  const int H1_MAX_1D = (H1_DOFS_1D > NUM_QUAD_1D)?H1_DOFS_1D:NUM_QUAD_1D;
  const int L2_MAX_1D = (L2_DOFS_1D > NUM_QUAD_1D)?L2_DOFS_1D:NUM_QUAD_1D;
  const int INNER_SIZE = (H1_MAX_1D > L2_MAX_1D)?H1_MAX_1D:L2_MAX_1D;
  const int grid = ((nzones+ELEMENT_BATCH-1)/ELEMENT_BATCH);
  const int blck = INNER_SIZE;
#endif
#ifdef __TEMPLATES__
  assert(NUM_DOFS_1D==H1_DOFS_1D);
  assert(L2_DOFS_1D==NUM_DOFS_1D-1);
  assert(NUM_QUAD_1D==2*(NUM_DOFS_1D-1));
  assert(NUM_QUAD_1D==2*(NUM_DOFS_1D-1));
  const unsigned int id = ((NUM_DIM)<<4)|(NUM_DOFS_1D-2);
  static std::unordered_map<unsigned long long, fForceMultTransposeS> call = {
    // 2D
    {0x20,&rForceMultTranspose2S<2,2,2,1,2>},
    {0x21,&rForceMultTranspose2S<2,3,4,2,3>},
    {0x22,&rForceMultTranspose2S<2,4,6,3,4>},
    {0x23,&rForceMultTranspose2S<2,5,8,4,5>},
    {0x24,&rForceMultTranspose2S<2,6,10,5,6>},
    {0x25,&rForceMultTranspose2S<2,7,12,6,7>},
    {0x26,&rForceMultTranspose2S<2,8,14,7,8>},
    {0x27,&rForceMultTranspose2S<2,9,16,8,9>},
    {0x28,&rForceMultTranspose2S<2,10,18,9,10>},
    {0x29,&rForceMultTranspose2S<2,11,20,10,11>},
    {0x2A,&rForceMultTranspose2S<2,12,22,11,12>},
    {0x2B,&rForceMultTranspose2S<2,13,24,12,13>},
    {0x2C,&rForceMultTranspose2S<2,14,26,13,14>},
    {0x2D,&rForceMultTranspose2S<2,15,28,14,15>},
    {0x2E,&rForceMultTranspose2S<2,16,30,15,16>},
    {0x2F,&rForceMultTranspose2S<2,17,32,16,17>},
    // 3D
/*    {0x30,&rForceMultTranspose3S<3,2,2,1,2>},
    {0x31,&rForceMultTranspose3S<3,3,4,2,3>},
    {0x32,&rForceMultTranspose3S<3,4,6,3,4>},
    {0x33,&rForceMultTranspose3S<3,5,8,4,5>},
    {0x34,&rForceMultTranspose3S<3,6,10,5,6>},
    {0x35,&rForceMultTranspose3S<3,7,12,6,7>},
    {0x36,&rForceMultTranspose3S<3,8,14,7,8>},
    {0x37,&rForceMultTranspose3S<3,9,16,8,9>},
    {0x38,&rForceMultTranspose3S<3,10,18,9,10>},
    {0x39,&rForceMultTranspose3S<3,11,20,10,11>},
    {0x3A,&rForceMultTranspose3S<3,12,22,11,12>},
    {0x3B,&rForceMultTranspose3S<3,13,24,12,13>},
    {0x3C,&rForceMultTranspose3S<3,14,26,13,14>},
    {0x3D,&rForceMultTranspose3S<3,15,28,14,15>},
    {0x3E,&rForceMultTranspose3S<3,16,30,15,16>},
    {0x3F,&rForceMultTranspose3S<3,17,32,16,17>},
*/
  };
  if (!call[id]) {
    printf("\n[rForceMultTranspose] id \033[33m0x%llX\033[m ",id);
    fflush(stdout);
  }
  assert(call[id]);
  call0(rForceMultTranspose2S,id,grid,blck,
        nzones,L2QuadToDof,H1DofToQuad,H1DofToQuadD,stressJinvT,v,e);
#else
  if (NUM_DIM==2)
    call0(rForceMultTranspose2S,id,grid,blck,
          NUM_DIM,NUM_DOFS_1D,NUM_QUAD_1D,L2_DOFS_1D,H1_DOFS_1D,
          nzones,L2QuadToDof,H1DofToQuad,H1DofToQuadD,stressJinvT,v,e);
  if (NUM_DIM!=2)
    exit(printf("\n[rForceMultTransposeS] NUM_DIM!=2 ERROR"));   
  pop();
#endif
}
