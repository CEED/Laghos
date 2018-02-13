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
// * /home/camier1/.occa/libraries/laghos/19ef990e7ee5e602/deviceSource.cpp
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
  const int NUM_QUAD = NUM_QUAD_2D;

  const int MAX_DOFS_1D = (L2_DOFS_1D > H1_DOFS_1D)?L2_DOFS_1D:H1_DOFS_1D;
  const int H1_MAX_1D = (H1_DOFS_1D > NUM_QUAD_1D)?H1_DOFS_1D:NUM_QUAD;
  const int L2_MAX_1D = (L2_DOFS_1D > NUM_QUAD_1D)?L2_DOFS_1D:NUM_QUAD_1D;
  const int INNER_SIZE = (H1_MAX_1D > L2_MAX_1D)?H1_MAX_1D:L2_MAX_1D;

#ifdef __LAMBDA__
  forallS(elBlock,numElements,ELEMENT_BATCH,
#else
  const int idx = blockDim.x*blockIdx.x + threadIdx.x;
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

    for (int idBlock = 0; idBlock < INNER_SIZE; ++idBlock/*;inner*/) {
      for (int id = idBlock; id < (L2_DOFS_1D * NUM_QUAD_1D); id += INNER_SIZE) {
        s_L2DofToQuad[id] = L2DofToQuad[id];
      }
      for (int id = idBlock; id < (H1_DOFS_1D * NUM_QUAD_1D); id += INNER_SIZE) {
        s_H1QuadToDof[id]  = H1QuadToDof[id];
        s_H1QuadToDofD[id] = H1QuadToDofD[id];
      }
    }sync;

    for (int el = elBlock; el < (elBlock + ELEMENT_BATCH); ++el) {
      if (el < numElements) {
        for (int dx = 0; dx < INNER_SIZE; ++dx) {
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
        for (int qy = 0; qy < INNER_SIZE; ++qy/*;inner*/) {
          if (qy < NUM_QUAD_1D) {
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
              double r_e = 0;
              for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
                r_e += s_xy[ijN(dx,qy,MAX_DOFS_1D)]*s_L2DofToQuad[ijN(qx,dx,NUM_QUAD_1D)];
              }
              s_e[ijN(qx,qy,NUM_QUAD_1D)] = r_e;
            }
          }
        }sync;

        for (int c = 0; c < NUM_DIM; ++c) {
          for (int qx = 0; qx < INNER_SIZE; ++qx/*;inner*/) {
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
          }sync;
          for (int dx = 0; dx < INNER_SIZE; ++dx/*;inner*/) {
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
          }sync;
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
  const int H1_MAX_1D = (H1_DOFS_1D > NUM_QUAD_1D)?H1_DOFS_1D:NUM_QUAD;
  const int L2_MAX_1D = (L2_DOFS_1D > NUM_QUAD_1D)?L2_DOFS_1D:NUM_QUAD_1D;
  const int INNER_SIZE = (H1_MAX_1D > L2_MAX_1D)?H1_MAX_1D:L2_MAX_1D;

#ifdef __LAMBDA__
  forallS(elBlock,numElements,ELEMENT_BATCH,
#else
  const int idx = blockDim.x*blockIdx.x + threadIdx.x;
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

    for (int idBlock = 0; idBlock < INNER_SIZE; ++idBlock/*; inner*/) {
      for (int id = idBlock; id < (L2_DOFS_1D * NUM_QUAD_1D); id += INNER_SIZE) {
        s_L2QuadToDof[id] = L2QuadToDof[id];
      }
      for (int id = idBlock; id < (H1_DOFS_1D * NUM_QUAD_1D); id += INNER_SIZE) {
        s_H1DofToQuad[id]  = H1DofToQuad[id];
        s_H1DofToQuadD[id] = H1DofToQuadD[id];
      }
    }sync;

    for (int el = elBlock; el < (elBlock + ELEMENT_BATCH); ++el) {
      if (el < numElements) {
        for (int qBlock = 0; qBlock < INNER_SIZE; ++qBlock/*; inner*/) {
          for (int q = qBlock; q < NUM_QUAD; ++q) {
            s_v[q] = 0;
          }
        }sync;
        for (int c = 0; c < NUM_DIM; ++c) {
          for (int dx = 0; dx < INNER_SIZE; ++dx/*; inner*/) {
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
          }sync;
          for (int qx = 0; qx < INNER_SIZE; ++qx/*; inner*/) {
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
          }sync;
        }
        for (int qx = 0; qx < INNER_SIZE; ++qx/*; inner*/) {
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
        }sync;
        for (int dy = 0; dy < INNER_SIZE; ++dy/*; inner*/) {
          if (dy < L2_DOFS_1D) {
            for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
              double r_e = 0;
              for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
                r_e += s_xy[ijN(qx,dy,NUM_QUAD_1D)] * s_L2QuadToDof[ijN(dx,qx,L2_DOFS_1D)];
              }
              e[ijkN(dx,dy,el,L2_DOFS_1D)] = r_e;
            }
          }
        }sync;
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
                 const int L2_DOFS_1D, // order-thermo
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
  const int grid = ((nzones+ELEMENT_BATCH-1)/ELEMENT_BATCH);
  //const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
  //const int NUM_QUAD = NUM_QUAD_2D;
  //const int H1_MAX_1D = (H1_DOFS_1D > NUM_QUAD_1D)?H1_DOFS_1D:NUM_QUAD;
  //const int L2_MAX_1D = (L2_DOFS_1D > NUM_QUAD_1D)?L2_DOFS_1D:NUM_QUAD_1D;
  //const int INNER_SIZE = (H1_MAX_1D > L2_MAX_1D)?H1_MAX_1D:L2_MAX_1D;
  const int blck = NUM_QUAD_1D;
#endif
#ifdef __TEMPLATES__
  const unsigned long long id =
    (((unsigned long long)NUM_DIM)<<32)|
    (NUM_DOFS_1D<<24)|
    (NUM_QUAD_1D<<16)
    |(L2_DOFS_1D<<8)|
    (H1_DOFS_1D);
  assert(LOG2(NUM_DIM)<=8);//printf("NUM_DIM:%d ",(NUM_DIM));
  assert(LOG2(NUM_DOFS_1D)<=8);//printf("NUM_DOFS_1D:%d ",(NUM_DOFS_1D));
  assert(LOG2(NUM_QUAD_1D)<=8);//printf("NUM_QUAD_1D:%d ",(NUM_QUAD_1D));
  assert(LOG2(L2_DOFS_1D)<=8);//printf("L2_DOFS_1D:%d ",(L2_DOFS_1D));
  assert(LOG2(H1_DOFS_1D)<=8);//printf("H1_DOFS_1D:%d\n",(H1_DOFS_1D));
  static std::unordered_map<unsigned long long, fForceMult2S> call = {
    // 2D
    {0x202020102ull,&rForceMult2S<2,2,2,1,2>},
    {0x203040203ull,&rForceMult2S<2,3,4,2,3>},// default 2D
    {0x203040303ull,&rForceMult2S<2,3,4,3,3>},
    
    {0x203050403ull,&rForceMult2S<2,3,5,4,3>},
    {0x203050503ull,&rForceMult2S<2,3,5,5,3>},
    
    {0x203060603ull,&rForceMult2S<2,3,6,6,3>},
    {0x203060703ull,&rForceMult2S<2,3,6,7,3>},
    
    {0x204060304ull,&rForceMult2S<2,4,6,3,4>},
    {0x204060404ull,&rForceMult2S<2,4,6,4,4>},
    {0x204070504ull,&rForceMult2S<2,4,7,5,4>},
    {0x204070604ull,&rForceMult2S<2,4,7,6,4>},
    
    {0x205080405ull,&rForceMult2S<2,5,8,4,5>},
    {0x205080505ull,&rForceMult2S<2,5,8,5,5>},
    {0x205090605ull,&rForceMult2S<2,5,9,6,5>},
    
    {0x2060A0506ull,&rForceMult2S<2,6,10,5,6>},
    {0x2070C0607ull,&rForceMult2S<2,7,12,6,7>},
  };
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
  else assert(false);
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
  const int grid = ((nzones+ELEMENT_BATCH-1)/ELEMENT_BATCH);
  const int blck = NUM_QUAD_1D;
#endif
#ifdef __TEMPLATES__
  const unsigned long long id =
    (((unsigned long long)NUM_DIM)<<32)|
    (NUM_DOFS_1D<<24)|
    (NUM_QUAD_1D<<16)|
    (L2_DOFS_1D<<8)|
    (H1_DOFS_1D);
  static std::unordered_map<unsigned long long, fForceMultTransposeS> call = {
    // 2D
    {0x202020102ull,&rForceMultTranspose2S<2,2,2,1,2>},
    {0x203040203ull,&rForceMultTranspose2S<2,3,4,2,3>},
    {0x203040303ull,&rForceMultTranspose2S<2,3,4,3,3>},
    
    {0x203050403ull,&rForceMultTranspose2S<2,3,5,4,3>},
    {0x203050503ull,&rForceMultTranspose2S<2,3,5,5,3>},
    
    {0x203060603ull,&rForceMultTranspose2S<2,3,6,6,3>},
    {0x203060703ull,&rForceMultTranspose2S<2,3,6,7,3>},
    
    {0x204060304ull,&rForceMultTranspose2S<2,4,6,3,4>},
    {0x204060404ull,&rForceMultTranspose2S<2,4,6,4,4>},
    {0x204070504ull,&rForceMultTranspose2S<2,4,7,5,4>},
    {0x204070604ull,&rForceMultTranspose2S<2,4,7,6,4>},
    
    {0x205080405ull,&rForceMultTranspose2S<2,5,8,4,5>},
    {0x205080505ull,&rForceMultTranspose2S<2,5,8,5,5>},
    {0x205090605ull,&rForceMultTranspose2S<2,5,9,6,5>},
    
    {0x2060A0506ull,&rForceMultTranspose2S<2,6,10,5,6>},
    {0x2070C0607ull,&rForceMultTranspose2S<2,7,12,6,7>},
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
  else assert(false);
  pop();
#endif
}
