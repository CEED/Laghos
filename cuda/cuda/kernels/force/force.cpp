
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
#include "../cuda.hpp"

// *****************************************************************************
template<const int NUM_DOFS_1D,
         const int NUM_QUAD_1D,
         const int BZ,
         const int NBLOCK>
__launch_bounds__(NUM_QUAD_1D*NUM_QUAD_1D*BZ, NBLOCK)
kernel  
static void rForceMult2D_v2(const int numElements,
                            const double* restrict L2DofToQuad,
                            const double* restrict H1QuadToDof,
                            const double* restrict H1QuadToDofD,
                            const double* restrict stressJinvT,
                            const double* restrict e,
                            double* restrict v)
{
   int el = blockIdx.x*BZ + threadIdx.z;
   if (el >= numElements) return;   
   const int NUM_DIM = 2;
   const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
   const int L2_DOFS_1D = NUM_DOFS_1D-1;
   const int H1_DOFS_1D = NUM_DOFS_1D;
   __shared__ double buf1[BZ][NUM_QUAD_2D];
   __shared__ double buf2[BZ][NUM_QUAD_1D][H1_DOFS_1D];
   __shared__ double buf3[BZ][NUM_QUAD_1D][H1_DOFS_1D];
   double *e_xy = (double*)(buf1 + threadIdx.z);
   double (*Dxy)[H1_DOFS_1D] = (double (*)[H1_DOFS_1D])(buf2 + threadIdx.z);
   double (*xy)[H1_DOFS_1D] = (double (*)[H1_DOFS_1D])(buf3 + threadIdx.z);
   // e_x reuses Dxy
   double (*e_x)[NUM_QUAD_1D] = (double (*)[NUM_QUAD_1D])Dxy;
  
   for (int dy = threadIdx.y; dy < L2_DOFS_1D; dy += blockDim.y)
   {
      for (int qx = threadIdx.x; qx < NUM_QUAD_1D; qx += blockDim.x)
      {
         double t = 0;
         for (int dx = 0; dx < L2_DOFS_1D; ++dx)
         {
            t += L2DofToQuad[ijN(qx,dx,NUM_QUAD_1D)] * e[ijkN(dx,dy,el,L2_DOFS_1D)];
         }
         e_x[dy][qx] = t;
      }
   }
   __syncthreads();
   for (int qy = threadIdx.y; qy < NUM_QUAD_1D; qy += blockDim.y)
   {
      for (int qx = threadIdx.x; qx < NUM_QUAD_1D; qx += blockDim.x)
      {
         double t = 0;
         for (int dy = 0; dy < L2_DOFS_1D; ++dy)
         {          
            t += L2DofToQuad[ijN(qy,dy,NUM_QUAD_1D)] * e_x[dy][qx];
         }
         e_xy[ijN(qx,qy,NUM_QUAD_1D)] = t;
      }
   }
   __syncthreads();
   for (int c = 0; c < 2; ++c)
   {
      for (int qy = threadIdx.y; qy < NUM_QUAD_1D; qy += blockDim.y)
      {
         for (int dx = threadIdx.x; dx < H1_DOFS_1D; dx += blockDim.x)
         {
            double t1 = 0, t2 = 0;
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
            {            
               t1 += e_xy[ijN(qx,qy,NUM_QUAD_1D)] *
                  stressJinvT[ijklmNM(0,c,qx,qy,el,NUM_DIM,NUM_QUAD_1D)] *
                  H1QuadToDofD[ijN(dx,qx,H1_DOFS_1D)];
               t2  += e_xy[ijN(qx,qy,NUM_QUAD_1D)] *
                  stressJinvT[ijklmNM(1,c,qx,qy,el,NUM_DIM,NUM_QUAD_1D)] *
                  H1QuadToDof[ijN(dx,qx,H1_DOFS_1D)];
            }
            Dxy[qy][dx] = t1;
            xy[qy][dx] = t2;
         }
      }
      __syncthreads();
      for (int dy = threadIdx.y; dy < H1_DOFS_1D; dy += blockDim.y)
      {
         for (int dx = threadIdx.x; dx < H1_DOFS_1D; dx += blockDim.x)
         {
            double t = 0;
            for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
            {          
               t += H1QuadToDof[ijN(dy,qy,H1_DOFS_1D)] * Dxy[qy][dx] +
                  H1QuadToDofD[ijN(dy,qy,H1_DOFS_1D)]*xy[qy][dx];
            }
            v[_ijklNM(c,dx,dy,el,NUM_DOFS_1D,numElements)] = t;
         }
      }
      __syncthreads();
   }
}


template<const int NUM_DIM,
         const int NUM_DOFS_1D,
         const int NUM_QUAD_1D,
         const int L2_DOFS_1D,
         const int H1_DOFS_1D> kernel  
static void rForceMult2D(const int numElements,
                         const double* restrict L2DofToQuad,
                         const double* restrict H1QuadToDof,
                         const double* restrict H1QuadToDofD,
                         const double* restrict stressJinvT,
                         const double* restrict e,
                         double* restrict v)
{
   const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
   const int el = blockDim.x * blockIdx.x + threadIdx.x;
   if (el < numElements)
   {
      double e_xy[NUM_QUAD_2D];
      for (int i = 0; i < NUM_QUAD_2D; ++i)
      {
         e_xy[i] = 0;
      }
      for (int dy = 0; dy < L2_DOFS_1D; ++dy)
      {
         double e_x[NUM_QUAD_1D];
         for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
         {
            e_x[qy] = 0;
         }
         for (int dx = 0; dx < L2_DOFS_1D; ++dx)
         {
            const double r_e = e[ijkN(dx,dy,el,L2_DOFS_1D)];
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
            {
               e_x[qx] += L2DofToQuad[ijN(qx,dx,NUM_QUAD_1D)] * r_e;
            }
         }
         for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
         {
            const double wy = L2DofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
            {
               e_xy[ijN(qx,qy,NUM_QUAD_1D)] += wy * e_x[qx];
            }
         }
      }
      for (int c = 0; c < 2; ++c)
      {
         for (int dy = 0; dy < H1_DOFS_1D; ++dy)
         {
            for (int dx = 0; dx < H1_DOFS_1D; ++dx)
            {
               v[_ijklNM(c,dx,dy,el,NUM_DOFS_1D,numElements)] = 0.0;
            }
         }
         for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
         {
            double Dxy[H1_DOFS_1D];
            double xy[H1_DOFS_1D];
            for (int dx = 0; dx < H1_DOFS_1D; ++dx)
            {
               Dxy[dx] = 0.0;
               xy[dx]  = 0.0;
            }
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
            {
               const double esx = e_xy[ijN(qx,qy,NUM_QUAD_1D)] * stressJinvT[ijklmNM(0,c,qx,qy,
                                                                                     el,NUM_DIM,NUM_QUAD_1D)];
               const double esy = e_xy[ijN(qx,qy,NUM_QUAD_1D)] * stressJinvT[ijklmNM(1,c,qx,qy,
                                                                                     el,NUM_DIM,NUM_QUAD_1D)];
               for (int dx = 0; dx < H1_DOFS_1D; ++dx)
               {
                  Dxy[dx] += esx * H1QuadToDofD[ijN(dx,qx,H1_DOFS_1D)];
                  xy[dx]  += esy * H1QuadToDof[ijN(dx,qx,H1_DOFS_1D)];
               }
            }
            for (int dy = 0; dy < H1_DOFS_1D; ++dy)
            {
               const double wy  = H1QuadToDof[ijN(dy,qy,H1_DOFS_1D)];
               const double wDy = H1QuadToDofD[ijN(dy,qy,H1_DOFS_1D)];
               for (int dx = 0; dx < H1_DOFS_1D; ++dx)
               {
                  v[_ijklNM(c,dx,dy,el,NUM_DOFS_1D,numElements)] += wy* Dxy[dx] + wDy*xy[dx];
               }
            }
         }
      }
   }
}

// *****************************************************************************
template<const int NUM_DOFS_1D,
         const int NUM_QUAD_1D,
         const int BZ,
         const int NBLOCK>
__launch_bounds__(NUM_QUAD_1D*NUM_QUAD_1D*BZ, NBLOCK)
kernel
static void rForceMultTranspose2D_v2(const int numElements,
                                     const double* restrict L2QuadToDof,
                                     const double* restrict H1DofToQuad,
                                     const double* restrict H1DofToQuadD,
                                     const double* restrict stressJinvT,
                                     const double* restrict v,
                                     double* restrict e)
{
   int el = blockIdx.x*BZ + threadIdx.z;
   if (el >= numElements) return;   
   const int NUM_DIM = 2;
   const int L2_DOFS_1D = NUM_DOFS_1D - 1;
   const int H1_DOFS_1D = NUM_DOFS_1D;
   const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
   // const int el = blockDim.x * blockIdx.x + threadIdx.x;
   // if (el < numElements)
   int tid = threadIdx.x + threadIdx.y*blockDim.x;
   {
      __shared__ double buf1[BZ][NUM_QUAD_2D];
      __shared__ double buf2[BZ][H1_DOFS_1D][NUM_QUAD_1D];
      __shared__ double buf3[BZ][H1_DOFS_1D][NUM_QUAD_1D];      
      __shared__ double buf4[BZ][NUM_QUAD_2D];
      __shared__ double buf5[BZ][NUM_QUAD_2D];
      double *vStress = (double*)(buf1 + threadIdx.z);
      double (*v_x)[NUM_QUAD_1D] = (double (*)[NUM_QUAD_1D])(buf2 + threadIdx.z);
      double (*v_Dx)[NUM_QUAD_1D] = (double (*)[NUM_QUAD_1D])(buf3 + threadIdx.z);
      double *v_Dxy = (double *)(buf4 + threadIdx.z);
      double *v_xDy = (double *)(buf5 + threadIdx.z);
      // e_x reuses v_Dxy
      double (*e_x)[L2_DOFS_1D] = (double (*)[L2_DOFS_1D])v_Dxy;
      
      for (int i = tid; i < NUM_QUAD_2D; i += blockDim.x*blockDim.y)
      {
         vStress[i] = 0;
      }
      __syncthreads();
      for (int c = 0; c < NUM_DIM; ++c)
      {
         for (int dy = threadIdx.y; dy < H1_DOFS_1D; dy += blockDim.y)
         {
            for (int qx = threadIdx.x; qx < NUM_QUAD_1D; qx += blockDim.x)
            {
               double t1 = 0, t2 = 0;
               for (int dx = 0; dx < H1_DOFS_1D; ++dx)
               {             
                  t1  += v[_ijklNM(c,dx,dy,el,NUM_DOFS_1D,numElements)] * H1DofToQuad[ijN(qx,dx,NUM_QUAD_1D)];
                  t2 += v[_ijklNM(c,dx,dy,el,NUM_DOFS_1D,numElements)] * H1DofToQuadD[ijN(qx,dx,NUM_QUAD_1D)];
               }
               v_x[dy][qx] = t1;
               v_Dx[dy][qx] = t2;
            }
         }
         __syncthreads();

         for (int qy = threadIdx.y; qy < NUM_QUAD_1D; qy += blockDim.y)
         {
            for (int qx = threadIdx.x; qx < NUM_QUAD_1D; qx += blockDim.x)
            {
               double t1 = 0, t2 = 0;
               for (int dy = 0; dy < H1_DOFS_1D; ++dy)
               {           
                  t1 += v_Dx[dy][qx] * H1DofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
                  t2 += v_x[dy][qx]  * H1DofToQuadD[ijN(qy,dy,NUM_QUAD_1D)];
               }
               v_Dxy[ijN(qx,qy,NUM_QUAD_1D)] = t1;
               v_xDy[ijN(qx,qy,NUM_QUAD_1D)] = t2;
            }
         }
         __syncthreads();
       
         for (int qy = threadIdx.y; qy < NUM_QUAD_1D; qy += blockDim.y)
         {
            for (int qx = threadIdx.x; qx < NUM_QUAD_1D; qx += blockDim.x)
            {
               vStress[ijN(qx,qy,NUM_QUAD_1D)] +=
                  ((v_Dxy[ijN(qx,qy,NUM_QUAD_1D)] * stressJinvT[ijklmNM(0,c,qx,qy,el,NUM_DIM,
                                                                        NUM_QUAD_1D)]) +
                   (v_xDy[ijN(qx,qy,NUM_QUAD_1D)] * stressJinvT[ijklmNM(1,c,qx,qy,el,NUM_DIM,
                                                                        NUM_QUAD_1D)]));
            }
         }
         __syncthreads();
      }

      for (int qy = threadIdx.y; qy < NUM_QUAD_1D; qy += blockDim.y)
      {
         for (int dx = threadIdx.x; dx < L2_DOFS_1D; dx += blockDim.x)
         {
            double t = 0;
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
            {           
               t += vStress[ijN(qx,qy,NUM_QUAD_1D)] * L2QuadToDof[ijN(dx,qx,L2_DOFS_1D)];
            }
            e_x[qy][dx] = t;
         }
      }
      __syncthreads();

      for (int dy = threadIdx.y; dy < L2_DOFS_1D; dy += blockDim.y)
      {
         for (int dx = threadIdx.x; dx < L2_DOFS_1D; dx += blockDim.x)
         {
            double t = 0;
            for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
            {         
               t += e_x[qy][dx] * L2QuadToDof[ijN(dy,qy,L2_DOFS_1D)];
            }
            e[ijkN(dx,dy,el,L2_DOFS_1D)] = t;
         }
      }
   }
}

template<const int NUM_DIM,
         const int NUM_DOFS_1D,
         const int NUM_QUAD_1D,
         const int L2_DOFS_1D,
         const int H1_DOFS_1D> kernel
static void rForceMultTranspose2D(const int numElements,
                                  const double* restrict L2QuadToDof,
                                  const double* restrict H1DofToQuad,
                                  const double* restrict H1DofToQuadD,
                                  const double* restrict stressJinvT,
                                  const double* restrict v,
                                  double* restrict e)
{
   const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
   const int el = blockDim.x * blockIdx.x + threadIdx.x;
   if (el < numElements)
   {
      double vStress[NUM_QUAD_2D];
      for (int i = 0; i < NUM_QUAD_2D; ++i)
      {
         vStress[i] = 0;
      }
      for (int c = 0; c < NUM_DIM; ++c)
      {
         double v_Dxy[NUM_QUAD_2D];
         double v_xDy[NUM_QUAD_2D];
         for (int i = 0; i < NUM_QUAD_2D; ++i)
         {
            v_Dxy[i] = v_xDy[i] = 0;
         }
         for (int dy = 0; dy < H1_DOFS_1D; ++dy)
         {
            double v_x[NUM_QUAD_1D];
            double v_Dx[NUM_QUAD_1D];
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
            {
               v_x[qx] = v_Dx[qx] = 0;
            }

            for (int dx = 0; dx < H1_DOFS_1D; ++dx)
            {
               const double r_v = v[_ijklNM(c,dx,dy,el,NUM_DOFS_1D,numElements)];
               for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
               {
                  v_x[qx]  += r_v * H1DofToQuad[ijN(qx,dx,NUM_QUAD_1D)];
                  v_Dx[qx] += r_v * H1DofToQuadD[ijN(qx,dx,NUM_QUAD_1D)];
               }
            }
            for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
            {
               const double wy  = H1DofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
               const double wDy = H1DofToQuadD[ijN(qy,dy,NUM_QUAD_1D)];
               for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
               {
                  v_Dxy[ijN(qx,qy,NUM_QUAD_1D)] += v_Dx[qx] * wy;
                  v_xDy[ijN(qx,qy,NUM_QUAD_1D)] += v_x[qx]  * wDy;
               }
            }
         }
         for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
         {
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
            {
               vStress[ijN(qx,qy,NUM_QUAD_1D)] +=
                  ((v_Dxy[ijN(qx,qy,NUM_QUAD_1D)] * stressJinvT[ijklmNM(0,c,qx,qy,el,NUM_DIM,
                                                                        NUM_QUAD_1D)]) +
                   (v_xDy[ijN(qx,qy,NUM_QUAD_1D)] * stressJinvT[ijklmNM(1,c,qx,qy,el,NUM_DIM,
                                                                        NUM_QUAD_1D)]));
            }
         }
      }
      for (int dy = 0; dy < L2_DOFS_1D; ++dy)
      {
         for (int dx = 0; dx < L2_DOFS_1D; ++dx)
         {
            e[ijkN(dx,dy,el,L2_DOFS_1D)] = 0;
         }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
      {
         double e_x[L2_DOFS_1D];
         for (int dx = 0; dx < L2_DOFS_1D; ++dx)
         {
            e_x[dx] = 0;
         }
         for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
         {
            const double r_v = vStress[ijN(qx,qy,NUM_QUAD_1D)];
            for (int dx = 0; dx < L2_DOFS_1D; ++dx)
            {
               e_x[dx] += r_v * L2QuadToDof[ijN(dx,qx,L2_DOFS_1D)];
            }
         }
         for (int dy = 0; dy < L2_DOFS_1D; ++dy)
         {
            const double w = L2QuadToDof[ijN(dy,qy,L2_DOFS_1D)];
            for (int dx = 0; dx < L2_DOFS_1D; ++dx)
            {
               e[ijkN(dx,dy,el,L2_DOFS_1D)] += e_x[dx] * w;
            }
         }
      }
   }
}

// *****************************************************************************
template<const int NUM_DIM,
         const int NUM_DOFS_1D,
         const int NUM_QUAD_1D,
         const int L2_DOFS_1D,
         const int H1_DOFS_1D> kernel
void rForceMult3D(const int numElements,
                  const double* restrict L2DofToQuad,
                  const double* restrict H1QuadToDof,
                  const double* restrict H1QuadToDofD,
                  const double* restrict stressJinvT,
                  const double* restrict e,
                  double* restrict v)
{
   const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
   const int NUM_QUAD_3D = NUM_QUAD_1D*NUM_QUAD_1D*NUM_QUAD_1D;
   const int el = blockDim.x * blockIdx.x + threadIdx.x;
   if (el < numElements)
   {
      double e_xyz[NUM_QUAD_3D];
      for (int i = 0; i < NUM_QUAD_3D; ++i)
      {
         e_xyz[i] = 0;
      }
      for (int dz = 0; dz < L2_DOFS_1D; ++dz)
      {
         double e_xy[NUM_QUAD_2D];
         for (int i = 0; i < NUM_QUAD_2D; ++i)
         {
            e_xy[i] = 0;
         }
         for (int dy = 0; dy < L2_DOFS_1D; ++dy)
         {
            double e_x[NUM_QUAD_1D];
            for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
            {
               e_x[qy] = 0;
            }
            for (int dx = 0; dx < L2_DOFS_1D; ++dx)
            {
               const double r_e = e[ijklN(dx,dy,dz,el,L2_DOFS_1D)];
               for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
               {
                  e_x[qx] += L2DofToQuad[ijN(qx,dx,NUM_QUAD_1D)] * r_e;
               }
            }
            for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
            {
               const double wy = L2DofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
               for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
               {
                  e_xy[ijN(qx,qy,NUM_QUAD_1D)] += wy * e_x[qx];
               }
            }
         }
         for (int qz = 0; qz < NUM_QUAD_1D; ++qz)
         {
            const double wz = L2DofToQuad[ijN(qz,dz,NUM_QUAD_1D)];
            for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
            {
               for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
               {
                  e_xyz[ijkN(qx,qy,qz,NUM_QUAD_1D)] += wz * e_xy[ijN(qx,qy,NUM_QUAD_1D)];
               }
            }
         }
      }
      for (int c = 0; c < 3; ++c)
      {
         for (int dz = 0; dz < H1_DOFS_1D; ++dz)
         {
            for (int dy = 0; dy < H1_DOFS_1D; ++dy)
            {
               for (int dx = 0; dx < H1_DOFS_1D; ++dx)
               {
                  v[_ijklmNM(c,dx,dy,dz,el,NUM_DOFS_1D,numElements)] = 0;
               }
            }
         }
         for (int qz = 0; qz < NUM_QUAD_1D; ++qz)
         {
            double Dxy_x[H1_DOFS_1D * H1_DOFS_1D];
            double xDy_y[H1_DOFS_1D * H1_DOFS_1D];
            double xy_z[H1_DOFS_1D * H1_DOFS_1D] ;
            for (int d = 0; d < (H1_DOFS_1D * H1_DOFS_1D); ++d)
            {
               Dxy_x[d] = xDy_y[d] = xy_z[d] = 0;
            }
            for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
            {
               double Dx_x[H1_DOFS_1D];
               double x_y[H1_DOFS_1D];
               double x_z[H1_DOFS_1D];
               for (int dx = 0; dx < H1_DOFS_1D; ++dx)
               {
                  Dx_x[dx] = x_y[dx] = x_z[dx] = 0;
               }
               for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
               {
                  const double r_e = e_xyz[ijkN(qx,qy,qz,NUM_QUAD_1D)];
                  const double esx = r_e * stressJinvT[ijklmnNM(0,c,qx,qy,qz,el,NUM_DIM,
                                                                NUM_QUAD_1D)];
                  const double esy = r_e * stressJinvT[ijklmnNM(1,c,qx,qy,qz,el,NUM_DIM,
                                                                NUM_QUAD_1D)];
                  const double esz = r_e * stressJinvT[ijklmnNM(2,c,qx,qy,qz,el,NUM_DIM,
                                                                NUM_QUAD_1D)];
                  for (int dx = 0; dx < H1_DOFS_1D; ++dx)
                  {
                     Dx_x[dx] += esx * H1QuadToDofD[ijN(dx,qx,H1_DOFS_1D)];
                     x_y[dx]  += esy * H1QuadToDof[ijN(dx,qx,H1_DOFS_1D)];
                     x_z[dx]  += esz * H1QuadToDof[ijN(dx,qx,H1_DOFS_1D)];
                  }
               }
               for (int dy = 0; dy < H1_DOFS_1D; ++dy)
               {
                  const double wy  = H1QuadToDof[ijN(dy,qy,H1_DOFS_1D)];
                  const double wDy = H1QuadToDofD[ijN(dy,qy,H1_DOFS_1D)];
                  for (int dx = 0; dx < H1_DOFS_1D; ++dx)
                  {
                     Dxy_x[ijN(dx,dy,H1_DOFS_1D)] += Dx_x[dx] * wy;
                     xDy_y[ijN(dx,dy,H1_DOFS_1D)] += x_y[dx]  * wDy;
                     xy_z[ijN(dx,dy,H1_DOFS_1D)]  += x_z[dx]  * wy;
                  }
               }
            }
            for (int dz = 0; dz < H1_DOFS_1D; ++dz)
            {
               const double wz  = H1QuadToDof[ijN(dz,qz,H1_DOFS_1D)];
               const double wDz = H1QuadToDofD[ijN(dz,qz,H1_DOFS_1D)];
               for (int dy = 0; dy < H1_DOFS_1D; ++dy)
               {
                  for (int dx = 0; dx < H1_DOFS_1D; ++dx)
                  {
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
}

template<const int NUM_DOFS_1D,
         const int NUM_QUAD_1D,
         const int USE_SMEM,
         const int BLOCK,
         const int NBLOCK> kernel
__launch_bounds__(BLOCK, NBLOCK)
   void rForceMult3D_v2(const int numElements,
                        const double* restrict L2DofToQuad,
                        const double* restrict H1QuadToDof,
                        const double* restrict H1QuadToDofD,
                        const double* restrict stressJinvT,
                        const double* restrict e,
                        double* restrict v,
                        double* gbuf,
                        int bufSize)
{
   const int NUM_DIM = 3;
   const int L2_DOFS_1D = NUM_DOFS_1D - 1;
   const int H1_DOFS_1D = NUM_DOFS_1D;
   extern __shared__ double sbuf[];
   const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
   const int NUM_QUAD_3D = NUM_QUAD_1D*NUM_QUAD_1D*NUM_QUAD_1D;
   double *buf_ptr;
   if (USE_SMEM) 
      buf_ptr = sbuf;
   else
      buf_ptr = (double*)((char*)gbuf + blockIdx.x*bufSize);

   // __shared__ double s_L2DofToQuad[L2_DOFS_1D][NUM_QUAD_1D],
   //                   s_H1QuadToDof[NUM_QUAD_1D][H1_DOFS_1D],
   //                   s_H1QuadToDofD[NUM_QUAD_1D][H1_DOFS_1D],
   //                   e_xyz[NUM_QUAD_3D],
   //                   e_x[L2_DOFS_1D][L2_DOFS_1D][NUM_QUAD_1D],
   //                   e_xy[L2_DOFS_1D][NUM_QUAD_2D],
   //                   Dx_x[NUM_QUAD_1D][NUM_QUAD_1D][H1_DOFS_1D],
   //                   x_y[NUM_QUAD_1D][NUM_QUAD_1D][H1_DOFS_1D],
   //                   x_z[NUM_QUAD_1D][NUM_QUAD_1D][H1_DOFS_1D],
   //                   Dxy_x[NUM_QUAD_1D][H1_DOFS_1D*H1_DOFS_1D],
   //                   xDy_y[NUM_QUAD_1D][H1_DOFS_1D*H1_DOFS_1D],
   //                   xy_z[NUM_QUAD_1D][H1_DOFS_1D*H1_DOFS_1D];
   double *e_xyz,
      (*e_x)[L2_DOFS_1D][NUM_QUAD_1D],
      (*e_xy)[NUM_QUAD_2D],
      (*Dx_x)[NUM_QUAD_1D][H1_DOFS_1D], (*x_y)[NUM_QUAD_1D][H1_DOFS_1D], (*x_z)[NUM_QUAD_1D][H1_DOFS_1D],
      (*Dxy_x)[H1_DOFS_1D*H1_DOFS_1D], (*xDy_y)[H1_DOFS_1D*H1_DOFS_1D], (*xy_z)[H1_DOFS_1D*H1_DOFS_1D],
      (*s_L2DofToQuad)[NUM_QUAD_1D],
      (*s_H1QuadToDof)[H1_DOFS_1D], (*s_H1QuadToDofD)[H1_DOFS_1D];
   mallocBuf((void**)&e_xyz, (void**)&buf_ptr, NUM_QUAD_3D*sizeof(double));
   mallocBuf((void**)&Dx_x,  (void**)&buf_ptr, H1_DOFS_1D*NUM_QUAD_2D*sizeof(double));
   mallocBuf((void**)&x_y,   (void**)&buf_ptr, H1_DOFS_1D*NUM_QUAD_2D*sizeof(double));
   mallocBuf((void**)&x_z,   (void**)&buf_ptr, H1_DOFS_1D*NUM_QUAD_2D*sizeof(double));
   mallocBuf((void**)&Dxy_x, (void**)&buf_ptr, NUM_QUAD_1D*H1_DOFS_1D*H1_DOFS_1D*sizeof(double));
   mallocBuf((void**)&xDy_y, (void**)&buf_ptr, NUM_QUAD_1D*H1_DOFS_1D*H1_DOFS_1D*sizeof(double));
   mallocBuf((void**)&xy_z,  (void**)&buf_ptr, NUM_QUAD_1D*H1_DOFS_1D*H1_DOFS_1D*sizeof(double));
   mallocBuf((void**)&s_L2DofToQuad , (void**)&buf_ptr, L2_DOFS_1D*NUM_QUAD_1D*sizeof(double));
   mallocBuf((void**)&s_H1QuadToDof , (void**)&buf_ptr, NUM_QUAD_1D*H1_DOFS_1D*sizeof(double));
   mallocBuf((void**)&s_H1QuadToDofD, (void**)&buf_ptr, NUM_QUAD_1D*H1_DOFS_1D*sizeof(double));
   // e_x & e_xy reuses buffer space
   e_x = (double (*)[L2_DOFS_1D][NUM_QUAD_1D])e_xyz;
   e_xy = (double (*)[NUM_QUAD_2D])Dx_x;
  
   if (threadIdx.z == 0)
   {
      for (int dx = threadIdx.y; dx < L2_DOFS_1D; dx += blockDim.y)
      {
         for (int qx = threadIdx.x; qx < NUM_QUAD_1D; qx += blockDim.x)
         {
            s_L2DofToQuad[dx][qx] = L2DofToQuad[ijN(qx,dx,NUM_QUAD_1D)];
         }
      }
      for (int qx = threadIdx.y; qx < NUM_QUAD_1D; qx += blockDim.y)
      {
         for (int dx = threadIdx.x; dx < H1_DOFS_1D; dx += blockDim.x)
         {
            s_H1QuadToDof[qx][dx] = H1QuadToDof[ijN(dx,qx,H1_DOFS_1D)];
            s_H1QuadToDofD[qx][dx] = H1QuadToDofD[ijN(dx,qx,H1_DOFS_1D)];
         }
      }    
   }
  
   for (int el = blockIdx.x; el < numElements; el += gridDim.x)
   {
      __syncthreads();
      for (int dz = threadIdx.z; dz < L2_DOFS_1D; dz += blockDim.z)
      {
         for (int dy = threadIdx.y; dy < L2_DOFS_1D; dy += blockDim.y)
         {
            for (int qx = threadIdx.x; qx < NUM_QUAD_1D; qx += blockDim.x)
            {
               double t = 0;
               for (int dx = 0; dx < L2_DOFS_1D; ++dx)
               {
                  t += s_L2DofToQuad[dx][qx] * e[ijklN(dx,dy,dz,el,L2_DOFS_1D)];
               }
               e_x[dz][dy][qx] = t;
            }
         }
      }
      __syncthreads();
      for (int dz = threadIdx.z; dz < L2_DOFS_1D; dz += blockDim.z)
      {
         for (int qy = threadIdx.y; qy < NUM_QUAD_1D; qy += blockDim.y)
         {
            for (int qx = threadIdx.x; qx < NUM_QUAD_1D; qx += blockDim.x)
            {
               double t = 0;
               for (int dy = 0; dy < L2_DOFS_1D; ++dy)
               {
                  t += s_L2DofToQuad[dy][qy] * e_x[dz][dy][qx];
               }
               e_xy[dz][ijN(qx,qy,NUM_QUAD_1D)] = t;
            }
         }
      }
      __syncthreads();
      for (int qz = threadIdx.z; qz < NUM_QUAD_1D; qz += blockDim.z)
      {
         for (int qy = threadIdx.y; qy < NUM_QUAD_1D; qy += blockDim.y)
         {
            for (int qx = threadIdx.x; qx < NUM_QUAD_1D; qx += blockDim.x)
            {
               double t = 0;
               for (int dz = 0; dz < L2_DOFS_1D; ++dz)
               {
                  t += s_L2DofToQuad[dz][qz] * e_xy[dz][ijN(qx,qy,NUM_QUAD_1D)];
               }
               e_xyz[ijkN(qx,qy,qz,NUM_QUAD_1D)] = t;
            }
         }
      }
      for (int c = 0; c < 3; ++c)
      {
         __syncthreads();
         for (int qz = threadIdx.z; qz < NUM_QUAD_1D; qz += blockDim.z)
         {
            for (int qy = threadIdx.y; qy < NUM_QUAD_1D; qy += blockDim.y)
            {
               for (int dx = threadIdx.x; dx < H1_DOFS_1D; dx += blockDim.x)
               {
                  double t1 = 0, t2 = 0, t3 = 0;
                  for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
                  {
                     t1 += e_xyz[ijkN(qx,qy,qz,NUM_QUAD_1D)] *
                        stressJinvT[ijklmnNM(0,c,qx,qy,qz,el,NUM_DIM,NUM_QUAD_1D)] *
                        s_H1QuadToDofD[qx][dx];                  
                     t2  += e_xyz[ijkN(qx,qy,qz,NUM_QUAD_1D)] *
                        stressJinvT[ijklmnNM(1,c,qx,qy,qz,el,NUM_DIM,NUM_QUAD_1D)] *
                        s_H1QuadToDof[qx][dx];
                     t3  += e_xyz[ijkN(qx,qy,qz,NUM_QUAD_1D)] *
                        stressJinvT[ijklmnNM(2,c,qx,qy,qz,el,NUM_DIM,NUM_QUAD_1D)] *
                        s_H1QuadToDof[qx][dx];              
                  }
                  Dx_x[qz][qy][dx] = t1;
                  x_y[qz][qy][dx]  = t2;
                  x_z[qz][qy][dx]  = t3;
               }
            }
         }
         __syncthreads();

         for (int qz = threadIdx.z; qz < NUM_QUAD_1D; qz += blockDim.z)
         {
            for (int dy = threadIdx.y; dy < H1_DOFS_1D; dy += blockDim.y)
            {
               for (int dx = threadIdx.x; dx < H1_DOFS_1D; dx += blockDim.x)
               {
                  double t1 = 0, t2 = 0, t3 = 0;
                  for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
                  {
                     t1 += Dx_x[qz][qy][dx] * s_H1QuadToDof[qy][dy];
                     t2 += x_y[qz][qy][dx] * s_H1QuadToDofD[qy][dy];
                     t3 += x_z[qz][qy][dx] *  s_H1QuadToDof[qy][dy];
                  }
                  Dxy_x[qz][ijN(dx,dy,H1_DOFS_1D)] = t1;
                  xDy_y[qz][ijN(dx,dy,H1_DOFS_1D)] = t2;
                  xy_z[qz][ijN(dx,dy,H1_DOFS_1D)] = t3;
               }
            }
         }
         __syncthreads();

         for (int dz = threadIdx.z; dz < H1_DOFS_1D; dz += blockDim.z)
         {
            for (int dy = threadIdx.y; dy < H1_DOFS_1D; dy += blockDim.y)
            {
               for (int dx = threadIdx.x; dx < H1_DOFS_1D; dx += blockDim.x)
               {
                  double t = 0;
                  for (int qz = 0; qz < NUM_QUAD_1D; ++qz)
                  {
                     t += Dxy_x[qz][ijN(dx,dy,H1_DOFS_1D)] * s_H1QuadToDof[qz][dz] +
                        xDy_y[qz][ijN(dx,dy,H1_DOFS_1D)] * s_H1QuadToDof[qz][dz] +
                        xy_z[qz][ijN(dx,dy,H1_DOFS_1D)] * s_H1QuadToDofD[qz][dz];
                  }
                  v[_ijklmNM(c,dx,dy,dz,el,NUM_DOFS_1D,numElements)] = t;
               }
            }
         }
      }
   }
}


// *****************************************************************************
template<const int NUM_DIM,
         const int NUM_DOFS_1D,
         const int NUM_QUAD_1D,
         const int L2_DOFS_1D,
         const int H1_DOFS_1D> kernel
static void rForceMultTranspose3D(const int numElements,
                                  const double* restrict L2QuadToDof,
                                  const double* restrict H1DofToQuad,
                                  const double* restrict H1DofToQuadD,
                                  const double* restrict stressJinvT,
                                  const double* restrict v,
                                  double* restrict e)
{
   const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
   const int NUM_QUAD_3D = NUM_QUAD_1D*NUM_QUAD_1D*NUM_QUAD_1D;
   const int el = blockDim.x * blockIdx.x + threadIdx.x;
   if (el < numElements)
   {
      double vStress[NUM_QUAD_3D];
      for (int i = 0; i < NUM_QUAD_3D; ++i)
      {
         vStress[i] = 0;
      }
      for (int c = 0; c < NUM_DIM; ++c)
      {
         for (int dz = 0; dz < H1_DOFS_1D; ++dz)
         {
            double Dxy_x[NUM_QUAD_2D];
            double xDy_y[NUM_QUAD_2D];
            double xy_z[NUM_QUAD_2D] ;
            for (int i = 0; i < NUM_QUAD_2D; ++i)
            {
               Dxy_x[i] = xDy_y[i] = xy_z[i] = 0;
            }
            for (int dy = 0; dy < H1_DOFS_1D; ++dy)
            {
               double Dx_x[NUM_QUAD_1D];
               double x_y[NUM_QUAD_1D];
               for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
               {
                  Dx_x[qx] = x_y[qx] = 0;
               }
               for (int dx = 0; dx < H1_DOFS_1D; ++dx)
               {
                  const double r_v = v[_ijklmNM(c,dx,dy,dz,el,NUM_DOFS_1D,numElements)];
                  for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
                  {
                     Dx_x[qx] += r_v * H1DofToQuadD[ijN(qx,dx,NUM_QUAD_1D)];
                     x_y[qx]  += r_v * H1DofToQuad[ijN(qx,dx,NUM_QUAD_1D)];
                  }
               }
               for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
               {
                  const double wy  = H1DofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
                  const double wDy = H1DofToQuadD[ijN(qy,dy,NUM_QUAD_1D)];
                  for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
                  {
                     Dxy_x[ijN(qx,qy,NUM_QUAD_1D)] += Dx_x[qx] * wy;
                     xDy_y[ijN(qx,qy,NUM_QUAD_1D)] += x_y[qx]  * wDy;
                     xy_z[ijN(qx,qy,NUM_QUAD_1D)]  += x_y[qx]  * wy;
                  }
               }
            }
            for (int qz = 0; qz < NUM_QUAD_1D; ++qz)
            {
               const double wz  = H1DofToQuad[ijN(qz,dz,NUM_QUAD_1D)];
               const double wDz = H1DofToQuadD[ijN(qz,dz,NUM_QUAD_1D)];
               for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
               {
                  for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
                  {
                     vStress[ijkN(qx,qy,qz,NUM_QUAD_1D)] +=
                        ((Dxy_x[ijN(qx,qy,NUM_QUAD_1D)]*wz *stressJinvT[ijklmnNM(0,c,qx,qy,qz,el,
                                                                                 NUM_DIM,NUM_QUAD_1D)]) +
                         (xDy_y[ijN(qx,qy,NUM_QUAD_1D)]*wz *stressJinvT[ijklmnNM(1,c,qx,qy,qz,el,NUM_DIM,
                                                                                 NUM_QUAD_1D)]) +
                         (xy_z[ijN(qx,qy,NUM_QUAD_1D)] *wDz*stressJinvT[ijklmnNM(2,c,qx,qy,qz,el,NUM_DIM,
                                                                                 NUM_QUAD_1D)]));
                  }
               }
            }
         }
      }
      for (int dz = 0; dz < L2_DOFS_1D; ++dz)
      {
         for (int dy = 0; dy < L2_DOFS_1D; ++dy)
         {
            for (int dx = 0; dx < L2_DOFS_1D; ++dx)
            {
               e[ijklN(dx,dy,dz,el,L2_DOFS_1D)] = 0;
            }
         }
      }
      for (int qz = 0; qz < NUM_QUAD_1D; ++qz)
      {
         double e_xy[L2_DOFS_1D * L2_DOFS_1D];
         for (int d = 0; d < (L2_DOFS_1D * L2_DOFS_1D); ++d)
         {
            e_xy[d] = 0;
         }
         for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
         {
            double e_x[L2_DOFS_1D];
            for (int dx = 0; dx < L2_DOFS_1D; ++dx)
            {
               e_x[dx] = 0;
            }
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
            {
               const double r_v = vStress[ijkN(qx,qy,qz,NUM_QUAD_1D)];
               for (int dx = 0; dx < L2_DOFS_1D; ++dx)
               {
                  e_x[dx] += r_v * L2QuadToDof[ijN(dx,qx,L2_DOFS_1D)];
               }
            }
            for (int dy = 0; dy < L2_DOFS_1D; ++dy)
            {
               const double w = L2QuadToDof[ijN(dy,qy,L2_DOFS_1D)];
               for (int dx = 0; dx < L2_DOFS_1D; ++dx)
               {
                  e_xy[ijN(dx,dy,L2_DOFS_1D)] += e_x[dx] * w;
               }
            }
         }
         for (int dz = 0; dz < L2_DOFS_1D; ++dz)
         {
            const double w = L2QuadToDof[ijN(dz,qz,L2_DOFS_1D)];
            for (int dy = 0; dy < L2_DOFS_1D; ++dy)
            {
               for (int dx = 0; dx < L2_DOFS_1D; ++dx)
               {
                  e[ijklN(dx,dy,dz,el,L2_DOFS_1D)] += w * e_xy[ijN(dx,dy,L2_DOFS_1D)];
               }
            }
         }
      }
   }
}

template<const int NUM_DOFS_1D,
         const int NUM_QUAD_1D,
         const int USE_SMEM,
         const int BLOCK,
         const int NBLOCK> kernel
__launch_bounds__(BLOCK, NBLOCK)
   static void rForceMultTranspose3D_v2(const int numElements,
                                        const double* restrict L2QuadToDof,
                                        const double* restrict H1DofToQuad,
                                        const double* restrict H1DofToQuadD,
                                        const double* restrict stressJinvT,
                                        const double* restrict v,
                                        double* restrict e,
                                        double *gbuf,
                                        int bufSize)
{
   const int NUM_DIM = 3;
   const int L2_DOFS_1D = NUM_DOFS_1D - 1;
   const int H1_DOFS_1D = NUM_DOFS_1D;
   const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
   const int NUM_QUAD_3D = NUM_QUAD_1D*NUM_QUAD_1D*NUM_QUAD_1D;
   int tid = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
   
   extern __shared__ double sbuf[];
   double *buf_ptr;
   if (USE_SMEM)
      buf_ptr = sbuf;
   else
      buf_ptr = (double*)((char*)gbuf + blockIdx.x*bufSize);

   // __shared__ double vStress[NUM_QUAD_3D],
   //                   s_H1DofToQuad[H1_DOFS_1D][NUM_QUAD_1D],
   //                   s_H1DofToQuadD[H1_DOFS_1D][NUM_QUAD_1D],
   //                   s_L2QuadToDof[NUM_QUAD_1D][L2_DOFS_1D],
   //                   Dx_x[H1_DOFS_1D][H1_DOFS_1D][NUM_QUAD_1D],
   //                   x_y[H1_DOFS_1D][H1_DOFS_1D][NUM_QUAD_1D],   
   //                   Dxy_x[H1_DOFS_1D][NUM_QUAD_2D],
   //                   xDy_y[H1_DOFS_1D][NUM_QUAD_2D],
   //                   xy_z[H1_DOFS_1D][NUM_QUAD_2D] ,
   //                   e_x[NUM_QUAD_1D][NUM_QUAD_1D][L2_DOFS_1D],   
   //                   e_xy[NUM_QUAD_1D][L2_DOFS_1D * L2_DOFS_1D];   
   double *vStress,
      (*s_H1DofToQuad)[NUM_QUAD_1D],
      (*s_H1DofToQuadD)[NUM_QUAD_1D],
      (*s_L2QuadToDof)[L2_DOFS_1D],
      (*Dx_x)[H1_DOFS_1D][NUM_QUAD_1D],
      (*x_y)[H1_DOFS_1D][NUM_QUAD_1D],
      (*Dxy_x)[NUM_QUAD_2D],
      (*xDy_y)[NUM_QUAD_2D],
      (*xy_z)[NUM_QUAD_2D];
   mallocBuf((void**)&vStress, (void**)&buf_ptr, NUM_QUAD_3D*sizeof(double));
   mallocBuf((void**)&Dx_x   , (void**)&buf_ptr, H1_DOFS_1D*H1_DOFS_1D*NUM_QUAD_1D*sizeof(double));
   mallocBuf((void**)&x_y    , (void**)&buf_ptr, H1_DOFS_1D*H1_DOFS_1D*NUM_QUAD_1D*sizeof(double));
   mallocBuf((void**)&Dxy_x  , (void**)&buf_ptr, H1_DOFS_1D*NUM_QUAD_2D*sizeof(double));
   mallocBuf((void**)&xDy_y  , (void**)&buf_ptr, H1_DOFS_1D*NUM_QUAD_2D*sizeof(double));
   mallocBuf((void**)&xy_z   , (void**)&buf_ptr, H1_DOFS_1D*NUM_QUAD_2D*sizeof(double));
   mallocBuf((void**)&s_H1DofToQuad , (void**)&buf_ptr, H1_DOFS_1D*NUM_QUAD_1D*sizeof(double));
   mallocBuf((void**)&s_H1DofToQuadD, (void**)&buf_ptr, H1_DOFS_1D*NUM_QUAD_1D*sizeof(double));
   mallocBuf((void**)&s_L2QuadToDof , (void**)&buf_ptr, L2_DOFS_1D*NUM_QUAD_1D*sizeof(double));
   // (e_xy & e_x) reuses Dx_x   
   double (*e_x)[NUM_QUAD_1D][L2_DOFS_1D] = (double (*)[NUM_QUAD_1D][L2_DOFS_1D])Dx_x;   
   double (*e_xy)[L2_DOFS_1D*L2_DOFS_1D]  = (double (*)[L2_DOFS_1D*L2_DOFS_1D])(e_x + NUM_QUAD_1D);

   if (threadIdx.z == 0)
   {
      for (int dx = threadIdx.y; dx < H1_DOFS_1D; dx += blockDim.y)
      {
         for (int qx = threadIdx.x; qx < NUM_QUAD_1D; qx += blockDim.x)
         {
            s_H1DofToQuad[dx][qx] = H1DofToQuad[ijN(qx,dx,NUM_QUAD_1D)];
            s_H1DofToQuadD[dx][qx] = H1DofToQuadD[ijN(qx,dx,NUM_QUAD_1D)];
         }
      }
      for (int qx = threadIdx.y; qx < NUM_QUAD_1D; qx += blockDim.y)
      {
         for (int dx = threadIdx.x; dx < L2_DOFS_1D; dx += blockDim.x)
         {
            s_L2QuadToDof[qx][dx] = L2QuadToDof[ijN(dx,qx,L2_DOFS_1D)];
         }
      }
   }
   __syncthreads();
   
   for (int el = blockIdx.x; el < numElements; el += gridDim.x)
   {
      for (int i = tid; i < NUM_QUAD_3D; i += blockDim.x*blockDim.y*blockDim.z)
      {
         vStress[i] = 0;
      }
      __syncthreads();
      for (int c = 0; c < NUM_DIM; ++c)
      {
         for (int dz = threadIdx.z; dz < H1_DOFS_1D; dz += blockDim.z)
         {
            for (int dy = threadIdx.y; dy < H1_DOFS_1D; dy += blockDim.y)
            {
               for (int qx = threadIdx.x; qx < NUM_QUAD_1D; qx += blockDim.x)
               {
                  double t1 = 0, t2 = 0;
                  for (int dx = 0; dx < H1_DOFS_1D; ++dx)
                  {             
                     t1 += v[_ijklmNM(c,dx,dy,dz,el,NUM_DOFS_1D,numElements)] * s_H1DofToQuadD[dx][qx];
                     t2 += v[_ijklmNM(c,dx,dy,dz,el,NUM_DOFS_1D,numElements)] * s_H1DofToQuad[dx][qx];
                  }
                  Dx_x[dz][dy][qx] = t1;
                  x_y[dz][dy][qx] = t2;
               }
            }
         }
         __syncthreads();
       
         for (int dz = threadIdx.z; dz < H1_DOFS_1D; dz += blockDim.z)
         {
            for (int qy = threadIdx.y; qy < NUM_QUAD_1D; qy += blockDim.y)
            {
               for (int qx = threadIdx.x; qx < NUM_QUAD_1D; qx += blockDim.x)
               {
                  double t1 = 0, t2 = 0, t3 = 0;
                  for (int dy = 0; dy < H1_DOFS_1D; ++dy)
                  {               
                     t1 += Dx_x[dz][dy][qx] * s_H1DofToQuad[dy][qy];
                     t2 += x_y[dz][dy][qx]  * s_H1DofToQuadD[dy][qy];
                     t3 += x_y[dz][dy][qx]  * s_H1DofToQuad[dy][qy];
                  }
                  Dxy_x[dz][ijN(qx,qy,NUM_QUAD_1D)] = t1;
                  xDy_y[dz][ijN(qx,qy,NUM_QUAD_1D)] = t2;
                  xy_z[dz][ijN(qx,qy,NUM_QUAD_1D)] = t3;
               }
            }
         }
         __syncthreads();
       
         for (int qz = threadIdx.z; qz < NUM_QUAD_1D; qz += blockDim.z)
         {
            for (int qy = threadIdx.y; qy < NUM_QUAD_1D; qy += blockDim.y)
            {
               for (int qx = threadIdx.x; qx < NUM_QUAD_1D; qx += blockDim.x)
               {
                  double t = vStress[ijkN(qx,qy,qz,NUM_QUAD_1D)];
                  for (int dz = 0; dz < H1_DOFS_1D; ++dz)
                  {                    
                     t +=
                        ((Dxy_x[dz][ijN(qx,qy,NUM_QUAD_1D)] *
                          s_H1DofToQuad[dz][qz] *
                          stressJinvT[ijklmnNM(0,c,qx,qy,qz,el,NUM_DIM,NUM_QUAD_1D)]) +
                         (xDy_y[dz][ijN(qx,qy,NUM_QUAD_1D)] *
                          s_H1DofToQuad[dz][qz] *
                          stressJinvT[ijklmnNM(1,c,qx,qy,qz,el,NUM_DIM,NUM_QUAD_1D)]) +
                         (xy_z[dz][ijN(qx,qy,NUM_QUAD_1D)] *
                          s_H1DofToQuadD[dz][qz] *
                          stressJinvT[ijklmnNM(2,c,qx,qy,qz,el,NUM_DIM,NUM_QUAD_1D)]));
                  }
                  vStress[ijkN(qx,qy,qz,NUM_QUAD_1D)] = t;
               }
            }
         }
         __syncthreads();
      }
    
      for (int qz = threadIdx.z; qz < NUM_QUAD_1D; qz += blockDim.z)
      {
         for (int qy = threadIdx.y; qy < NUM_QUAD_1D; qy += blockDim.y)
         {
            for (int dx = threadIdx.x; dx < L2_DOFS_1D; dx += blockDim.x)
            {
               double t = 0;
               for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
               {           
                  t += vStress[ijkN(qx,qy,qz,NUM_QUAD_1D)] * s_L2QuadToDof[qx][dx];
               }
               e_x[qz][qy][dx] = t;
            }
         }
      }
      __syncthreads();

      for (int qz = threadIdx.z; qz < NUM_QUAD_1D; qz += blockDim.z)
      {
         for (int dy = threadIdx.y; dy < L2_DOFS_1D; dy += blockDim.y)
         {
            for (int dx = threadIdx.x; dx < L2_DOFS_1D; dx += blockDim.x)
            {
               double t = 0;
               for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
               {           
                  t += e_x[qz][qy][dx] * s_L2QuadToDof[qy][dy];
               }
               e_xy[qz][ijN(dx,dy,L2_DOFS_1D)] = t;
            }
         }
      }
      __syncthreads();

      for (int dz = threadIdx.z; dz < L2_DOFS_1D; dz += blockDim.z)
      {
         for (int dy = threadIdx.y; dy < L2_DOFS_1D; dy += blockDim.y)
         {
            for (int dx = threadIdx.x; dx < L2_DOFS_1D; dx += blockDim.x)
            {
               double t = 0;
               for (int qz = 0; qz < NUM_QUAD_1D; ++qz)
               {           
                  t += s_L2QuadToDof[qz][dz] * e_xy[qz][ijN(dx,dy,L2_DOFS_1D)];
               }
               e[ijklN(dx,dy,dz,el,L2_DOFS_1D)] = t;
            }
         }
      }
      __syncthreads();
   }
}


// *****************************************************************************
typedef void (*fForceMult)(const int numElements,
                           const double* restrict L2QuadToDof,
                           const double* restrict H1DofToQuad,
                           const double* restrict H1DofToQuadD,
                           const double* restrict stressJinvT,
                           const double* restrict e,
                           double* restrict v);

// *****************************************************************************
void rForceMult(const int NUM_DIM,
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
                double* restrict v)
{
   assert(NUM_QUAD_1D==2*(NUM_DOFS_1D-1));
   assert(NUM_DOFS_1D==H1_DOFS_1D);
   assert(L2_DOFS_1D==NUM_DOFS_1D-1);
   const unsigned int id = ((NUM_DIM)<<4)|(NUM_DOFS_1D-2);
   assert(LOG2(NUM_DIM)<=4);
   assert(LOG2(NUM_DOFS_1D-2)<=4);
   static std::unordered_map<unsigned long long, fForceMult> call =
      {
         // {0x20,&rForceMult2D<2,2,2,1,2>},
         // {0x21,&rForceMult2D<2,3,4,2,3>},
         // {0x22,&rForceMult2D<2,4,6,3,4>},
         // {0x23,&rForceMult2D<2,5,8,4,5>},
         // {0x24,&rForceMult2D<2,6,10,5,6>},
         // {0x25,&rForceMult2D<2,7,12,6,7>},
         // {0x26,&rForceMult2D<2,8,14,7,8>},
         // {0x27,&rForceMult2D<2,9,16,8,9>},
         // {0x28,&rForceMult2D<2,10,18,9,10>},
         // {0x29,&rForceMult2D<2,11,20,10,11>},
         // {0x2A,&rForceMult2D<2,12,22,11,12>},
         // {0x2B,&rForceMult2D<2,13,24,12,13>},
         // {0x2C,&rForceMult2D<2,14,26,13,14>},
         // {0x2D,&rForceMult2D<2,15,28,14,15>},
         // {0x2E,&rForceMult2D<2,16,30,15,16>},
         // {0x2F,&rForceMult2D<2,17,32,16,17>},
         // 3D
         // {0x30,&rForceMult3D<3,2,2,1,2>},
         // {0x31,&rForceMult3D<3,3,4,2,3>},
         // {0x32,&rForceMult3D<3,4,6,3,4>},
         // {0x33,&rForceMult3D<3,5,8,4,5>},
         // {0x34,&rForceMult3D<3,6,10,5,6>},
         // {0x35,&rForceMult3D<3,7,12,6,7>},
         // {0x36,&rForceMult3D<3,8,14,7,8>},
         // {0x37,&rForceMult3D<3,9,16,8,9>},
         // {0x38,&rForceMult3D<3,10,18,9,10>},
         // {0x39,&rForceMult3D<3,11,20,10,11>},
         // {0x3A,&rForceMult3D<3,12,22,11,12>},
         // {0x3B,&rForceMult3D<3,13,24,12,13>},
         // {0x3C,&rForceMult3D<3,14,26,13,14>},
         // {0x3D,&rForceMult3D<3,15,28,14,15>},
         // {0x3E,&rForceMult3D<3,16,30,15,16>},
         // {0x3F,&rForceMult3D<3,17,32,16,17>},
      };

#define call_2d(DOFS,QUAD,BZ,NBLOCK)                                    \
   call_2d_ker(rForceMult2D,nzones,DOFS,QUAD,BZ,NBLOCK,                 \
               nzones,L2QuadToDof,H1DofToQuad,H1DofToQuadD,stressJinvT,e,v)
#define call_3d(DOFS,QUAD,BZ,NBLOCK)                                    \
   call_3d_ker(rForceMult3D,nzones,DOFS,QUAD,BZ,NBLOCK,                 \
               nzones,L2QuadToDof,H1DofToQuad,H1DofToQuadD,stressJinvT,e,v,gbuf,rForceMult3D_BufSize)

   // 2D
   if      (id == 0x20) { call_2d(2 ,2,16,1); }
   else if (id == 0x21) { call_2d(3 ,4 ,8,1); }
   else if (id == 0x22) { call_2d(4 ,6 ,4,1); }   
   else if (id == 0x23) { call_2d(5 ,8 ,2,1); }
   else if (id == 0x24) { call_2d(6 ,10,2,1); }
   else if (id == 0x25) { call_2d(7 ,12,2,1); }
   else if (id == 0x26) { call_2d(8 ,14,2,1); }
   else if (id == 0x27) { call_2d(9 ,16,2,1); }
   else if (id == 0x28) { call_2d(10,18,2,1); }
   else if (id == 0x29) { call_2d(11,20,2,1); }
   else if (id == 0x2A) { call_2d(12,22,2,1); }
   else if (id == 0x2B) { call_2d(13,24,1,1); }
   else if (id == 0x2C) { call_2d(14,26,1,1); }
   else if (id == 0x2D) { call_2d(15,28,1,1); }
   else if (id == 0x2E) { call_2d(16,30,1,1); }
   else if (id == 0x2F) { call_2d(17,32,1,1); }   
   // 3D
   else if (id == 0x30) { call_3d(2 ,2 ,2,1); }
   else if (id == 0x31) { call_3d(3 ,4 ,4,1); }   
   else if (id == 0x32) { call_3d(4 ,6 ,6,1); }
   else if (id == 0x33) { call_3d(5 ,8 ,8,1); }
   else if (id == 0x34) { call_3d(6 ,10,2,1); }
   else if (id == 0x35) { call_3d(7 ,12,2,1); }
   else if (id == 0x36) { call_3d(8 ,14,2,1); }
   else if (id == 0x37) { call_3d(9 ,16,2,1); }
   else if (id == 0x38) { call_3d(10,18,2,1); }
   else if (id == 0x39) { call_3d(11,20,2,1); }
   else if (id == 0x3A) { call_3d(12,22,2,1); }
   else if (id == 0x3B) { call_3d(13,24,1,1); }
   else if (id == 0x3C) { call_3d(14,26,1,1); }
   else if (id == 0x3D) { call_3d(15,28,1,1); }
   else if (id == 0x3E) { call_3d(16,30,1,1); }
   else if (id == 0x3F) { call_3d(17,32,1,1); }                        
   else
   {
      const int blck = CUDA_BLOCK_SIZE;
      const int grid = (nzones+blck-1)/blck;   
      if (!call[id])
      {
         printf("\n[rForceMult] id \033[33m0x%X\033[m ",id);
         fflush(stdout);
      }
      assert(call[id]);
      call0(id,grid,blck,
            nzones,L2QuadToDof,H1DofToQuad,H1DofToQuadD,stressJinvT,e,v);
   }
   CUCHK(cudaGetLastError());
#undef call_2d   
#undef call_3d   
}

// *****************************************************************************
typedef void (*fForceMultTranspose)(const int nzones,
                                    const double* restrict L2QuadToDof,
                                    const double* restrict H1DofToQuad,
                                    const double* restrict H1DofToQuadD,
                                    const double* restrict stressJinvT,
                                    const double* restrict v,
                                    double* restrict e);

// *****************************************************************************
void rForceMultTranspose(const int NUM_DIM,
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
                         double* restrict e)
{
   assert(NUM_DOFS_1D==H1_DOFS_1D);
   assert(L2_DOFS_1D==NUM_DOFS_1D-1);
   assert(NUM_QUAD_1D==2*(NUM_DOFS_1D-1));
   assert(NUM_QUAD_1D==2*(NUM_DOFS_1D-1));
   const unsigned int id = ((NUM_DIM)<<4)|(NUM_DOFS_1D-2);
   static std::unordered_map<unsigned long long, fForceMultTranspose> call =
      {
         // 2D
         // {0x20,&rForceMultTranspose2D<2,2,2,1,2>},
         // {0x21,&rForceMultTranspose2D<2,3,4,2,3>},
         // {0x22,&rForceMultTranspose2D<2,4,6,3,4>},
         // {0x23,&rForceMultTranspose2D<2,5,8,4,5>},
         // {0x24,&rForceMultTranspose2D<2,6,10,5,6>},
         // {0x25,&rForceMultTranspose2D<2,7,12,6,7>},
         // {0x26,&rForceMultTranspose2D<2,8,14,7,8>},
         // {0x27,&rForceMultTranspose2D<2,9,16,8,9>},
         // {0x28,&rForceMultTranspose2D<2,10,18,9,10>},
         // {0x29,&rForceMultTranspose2D<2,11,20,10,11>},
         // {0x2A,&rForceMultTranspose2D<2,12,22,11,12>},
         // {0x2B,&rForceMultTranspose2D<2,13,24,12,13>},
         // {0x2C,&rForceMultTranspose2D<2,14,26,13,14>},
         // {0x2D,&rForceMultTranspose2D<2,15,28,14,15>},
         // {0x2E,&rForceMultTranspose2D<2,16,30,15,16>},
         // {0x2F,&rForceMultTranspose2D<2,17,32,16,17>},
         // 3D
         // {0x30,&rForceMultTranspose3D<3,2,2,1,2>},
         // {0x31,&rForceMultTranspose3D<3,3,4,2,3>},
         // {0x32,&rForceMultTranspose3D<3,4,6,3,4>},
         // {0x33,&rForceMultTranspose3D<3,5,8,4,5>},
         // {0x34,&rForceMultTranspose3D<3,6,10,5,6>},
         // {0x35,&rForceMultTranspose3D<3,7,12,6,7>},
         // {0x36,&rForceMultTranspose3D<3,8,14,7,8>},
         // {0x37,&rForceMultTranspose3D<3,9,16,8,9>},
         // {0x38,&rForceMultTranspose3D<3,10,18,9,10>},
         // {0x39,&rForceMultTranspose3D<3,11,20,10,11>},
         // {0x3A,&rForceMultTranspose3D<3,12,22,11,12>},
         // {0x3B,&rForceMultTranspose3D<3,13,24,12,13>},
         // {0x3C,&rForceMultTranspose3D<3,14,26,13,14>},
         // {0x3D,&rForceMultTranspose3D<3,15,28,14,15>},
         // {0x3E,&rForceMultTranspose3D<3,16,30,15,16>},
         // {0x3F,&rForceMultTranspose3D<3,17,32,16,17>},
      };

#define call_2d(DOFS,QUAD,BZ,NBLOCK)                                    \
   call_2d_ker(rForceMultTranspose2D,nzones,DOFS,QUAD,BZ,NBLOCK,                 \
               nzones,L2QuadToDof,H1DofToQuad,H1DofToQuadD,stressJinvT,v,e)
#define call_3d(DOFS,QUAD,BZ,NBLOCK)                                    \
   call_3d_ker(rForceMultTranspose3D,nzones,DOFS,QUAD,BZ,NBLOCK,        \
               nzones,L2QuadToDof,H1DofToQuad,H1DofToQuadD,stressJinvT,v,e,gbuf,rForceMultTranspose3D_BufSize)
   // 2D
   if      (id == 0x20) { call_2d(2 ,2,16,1); }
   else if (id == 0x21) { call_2d(3 ,4 ,8,1); }
   else if (id == 0x22) { call_2d(4 ,6 ,4,1); }
   else if (id == 0x23) { call_2d(5 ,8 ,2,1); }
   else if (id == 0x24) { call_2d(6 ,10,2,1); }
   else if (id == 0x25) { call_2d(7 ,12,2,1); }
   else if (id == 0x26) { call_2d(8 ,14,2,1); }
   else if (id == 0x27) { call_2d(9 ,16,2,1); }
   else if (id == 0x28) { call_2d(10,18,2,1); }
   else if (id == 0x29) { call_2d(11,20,2,1); }
   else if (id == 0x2A) { call_2d(12,22,2,1); }
   else if (id == 0x2B) { call_2d(13,24,1,1); }
   else if (id == 0x2C) { call_2d(14,26,1,1); }
   else if (id == 0x2D) { call_2d(15,28,1,1); }
   else if (id == 0x2E) { call_2d(16,30,1,1); }
   else if (id == 0x2F) { call_2d(17,32,1,1); }   
   // 3D
   else if (id == 0x30) { call_3d(2 ,2 ,2,1); }
   else if (id == 0x31) { call_3d(3 ,4 ,4,1); }   
   else if (id == 0x32) { call_3d(4 ,6 ,6,1); }
   else if (id == 0x33) { call_3d(5 ,8 ,8,1); }
   else if (id == 0x34) { call_3d(6 ,10,2,1); }
   else if (id == 0x35) { call_3d(7 ,12,2,1); }
   else if (id == 0x36) { call_3d(8 ,14,2,1); }
   else if (id == 0x37) { call_3d(9 ,16,2,1); }
   else if (id == 0x38) { call_3d(10,18,2,1); }
   else if (id == 0x39) { call_3d(11,20,2,1); }
   else if (id == 0x3A) { call_3d(12,22,2,1); }
   else if (id == 0x3B) { call_3d(13,24,1,1); }
   else if (id == 0x3C) { call_3d(14,26,1,1); }
   else if (id == 0x3D) { call_3d(15,28,1,1); }
   else if (id == 0x3E) { call_3d(16,30,1,1); }
   else if (id == 0x3F) { call_3d(17,32,1,1); }
   else
   {
      const int blck = CUDA_BLOCK_SIZE;
      const int grid = (nzones+blck-1)/blck;   
      if (!call[id])
      {
         printf("\n[rForceMultTranspose] id \033[33m0x%X\033[m ",id);
         fflush(stdout);
      }
      assert(call[id]);
      call0(id,grid,blck,
            nzones,L2QuadToDof,H1DofToQuad,H1DofToQuadD,stressJinvT,v,e);
   }
   CUCHK(cudaGetLastError());
}

