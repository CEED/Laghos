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
template<const int NUM_VDIM,
         const int NUM_DOFS_1D,
         const int NUM_QUAD_1D> kernel
void rGridFuncToQuad1D(const int numElements,
                       const double* restrict dofToQuad,
                       const int* restrict l2gMap,
                       const double* restrict gf,
                       double* restrict out)
{
   const int e = blockDim.x * blockIdx.x + threadIdx.x;
   if (e < numElements)
   {
      double r_out[NUM_VDIM][NUM_QUAD_1D];
      for (int v = 0; v < NUM_VDIM; ++v)
      {
         for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
         {
            r_out[v][qx] = 0;
         }
      }
      for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
      {
         const int gid = l2gMap[(dx) + (NUM_DOFS_1D) * (e)];
         for (int v = 0; v < NUM_VDIM; ++v)
         {
            const double r_gf = gf[v + gid * NUM_VDIM];
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
            {
               r_out[v][qx] += r_gf * dofToQuad[(qx) + (NUM_QUAD_1D) * (dx)];
            }
         }
      }
      for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
      {
         for (int v = 0; v < NUM_VDIM; ++v)
         {
            out[(qx) + (NUM_QUAD_1D) * ((e) + (numElements) * (v))] = r_out[v][qx];
         }
      }
   }
}

// *****************************************************************************
template<const int NUM_VDIM,
         const int NUM_DOFS_1D,
         const int NUM_QUAD_1D> kernel
void rGridFuncToQuad2D(const int numElements,
                       const double* restrict dofToQuad,
                       const int* restrict l2gMap,
                       const double* restrict gf,
                       double* restrict out)
{
   const int e = blockDim.x * blockIdx.x + threadIdx.x;
   if (e < numElements)
   {
      double out_xy[NUM_VDIM][NUM_QUAD_1D][NUM_QUAD_1D];
      for (int v = 0; v < NUM_VDIM; ++v)
      {
         for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
         {
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
            {
               out_xy[v][qy][qx] = 0;
            }
         }
      }
      for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
      {
         double out_x[NUM_VDIM][NUM_QUAD_1D];
         for (int v = 0; v < NUM_VDIM; ++v)
         {
            for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
            {
               out_x[v][qy] = 0;
            }
         }
         for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
         {
            const int gid = l2gMap[ijkN(dx, dy, e,NUM_DOFS_1D)];
            for (int v = 0; v < NUM_VDIM; ++v)
            {
               const double r_gf = gf[v + gid*NUM_VDIM];
               for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
               {
                  out_x[v][qy] += r_gf * dofToQuad[ijN(qy, dx,NUM_QUAD_1D)];
               }
            }
         }
         for (int v = 0; v < NUM_VDIM; ++v)
         {
            for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
            {
               const double d2q = dofToQuad[ijN(qy, dy,NUM_QUAD_1D)];
               for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
               {
                  out_xy[v][qy][qx] += d2q * out_x[v][qx];
               }
            }
         }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
      {
         for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
         {
            for (int v = 0; v < NUM_VDIM; ++v)
            {
               out[_ijklNM(v, qx, qy, e,NUM_QUAD_1D,numElements)] = out_xy[v][qy][qx];
            }
         }
      }
   }
}

// *****************************************************************************
template<const int NUM_VDIM,
         const int NUM_DOFS_1D,
         const int NUM_QUAD_1D> kernel
void rGridFuncToQuad3D(const int numElements,
                       const double* restrict dofToQuad,
                       const int* restrict l2gMap,
                       const double* restrict gf,
                       double* restrict out)
{
   const int e = blockDim.x * blockIdx.x + threadIdx.x;
   if (e < numElements)
   {
      double out_xyz[NUM_VDIM][NUM_QUAD_1D][NUM_QUAD_1D][NUM_QUAD_1D];
      for (int v = 0; v < NUM_VDIM; ++v)
      {
         for (int qz = 0; qz < NUM_QUAD_1D; ++qz)
         {
            for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
            {
               for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
               {
                  out_xyz[v][qz][qy][qx] = 0;
               }
            }
         }
      }
      for (int dz = 0; dz < NUM_DOFS_1D; ++dz)
      {
         double out_xy[NUM_VDIM][NUM_QUAD_1D][NUM_QUAD_1D];
         for (int v = 0; v < NUM_VDIM; ++v)
         {
            for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
            {
               for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
               {
                  out_xy[v][qy][qx] = 0;
               }
            }
         }
         for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
         {
            double out_x[NUM_VDIM][NUM_QUAD_1D];
            for (int v = 0; v < NUM_VDIM; ++v)
            {
               for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
               {
                  out_x[v][qx] = 0;
               }
            }
            for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
            {
               const int gid = l2gMap[ijklN(dx, dy, dz, e,NUM_DOFS_1D)];
               for (int v = 0; v < NUM_VDIM; ++v)
               {
                  const double r_gf = gf[v + gid*NUM_VDIM];
                  for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
                  {
                     out_x[v][qx] += r_gf * dofToQuad[ijN(qx, dx, NUM_QUAD_1D)];
                  }
               }
            }
            for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
            {
               const double wy = dofToQuad[ijN(qy, dy, NUM_QUAD_1D)];
               for (int v = 0; v < NUM_VDIM; ++v)
               {
                  for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
                  {
                     out_xy[v][qy][qx] += wy * out_x[v][qx];
                  }
               }
            }
         }
         for (int qz = 0; qz < NUM_QUAD_1D; ++qz)
         {
            const double wz = dofToQuad[ijN(qz, dz, NUM_QUAD_1D)];
            for (int v = 0; v < NUM_VDIM; ++v)
            {
               for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
               {
                  for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
                  {
                     out_xyz[v][qz][qy][qx] += wz * out_xy[v][qy][qx];
                  }
               }
            }
         }
      }

      for (int qz = 0; qz < NUM_QUAD_1D; ++qz)
      {
         for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
         {
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
            {
               for (int v = 0; v < NUM_VDIM; ++v)
               {
                  out[_ijklmNM(v, qx, qy, qz, e,NUM_QUAD_1D,
                               numElements)] = out_xyz[v][qz][qy][qx];
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
void rGridFuncToQuad3D_v2(const int numElements,
                          const double* restrict dofToQuad,
                          const int* restrict l2gMap,
                          const double* restrict gf,
                          double* restrict out,
                          double *gbuf,
                          int bufSize)
{
  const int NUM_VDIM = 1;
  extern __shared__ double sbuf[];
  double *buf_ptr;
  if (USE_SMEM)
    buf_ptr = sbuf;
  else
    buf_ptr = (double*)((char*)gbuf + blockIdx.x*bufSize);

  // __shared__ double out_xy[NUM_DOFS_1D][NUM_QUAD_1D][NUM_QUAD_1D],
  //   out_x[NUM_DOFS_1D][NUM_DOFS_1D][NUM_QUAD_1D],
  //   s_dofToQuad[NUM_DOFS_1D][NUM_QUAD_1D];  
  double (*out_xy)[NUM_QUAD_1D][NUM_QUAD_1D],
    (*out_x)[NUM_DOFS_1D][NUM_QUAD_1D],
    (*s_dofToQuad)[NUM_QUAD_1D];
  mallocBuf((void**)&out_xy, (void**)&buf_ptr, NUM_DOFS_1D*NUM_QUAD_1D*NUM_QUAD_1D*sizeof(double));
  mallocBuf((void**)&out_x , (void**)&buf_ptr, NUM_DOFS_1D*NUM_DOFS_1D*NUM_QUAD_1D*sizeof(double));
  mallocBuf((void**)&s_dofToQuad, (void**)&buf_ptr, NUM_DOFS_1D*NUM_QUAD_1D*sizeof(double));
    
  if (threadIdx.z == 0)
  {
    for (int dx = threadIdx.y; dx < NUM_DOFS_1D; dx += blockDim.y)
    {
      for (int qx = threadIdx.x; qx < NUM_QUAD_1D; qx += blockDim.x)
      {
        s_dofToQuad[dx][qx] = dofToQuad[ijN(qx, dx, NUM_QUAD_1D)];
      }
    }
  }
  __syncthreads();

  for (int e = blockIdx.x; e < numElements; e += gridDim.x)
  {
    for (int v = 0; v < NUM_VDIM; ++v)
    {    
      for (int dz = threadIdx.z; dz < NUM_DOFS_1D; dz += blockDim.z)
      {
        for (int dy = threadIdx.y; dy < NUM_DOFS_1D; dy += blockDim.y)
        {        
          for (int qx = threadIdx.x; qx < NUM_QUAD_1D; qx += blockDim.x)
          {
            double t = 0;
            for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
            {              
              int gid = l2gMap[ijklN(dx, dy, dz, e,NUM_DOFS_1D)];
              t += gf[v + gid*NUM_VDIM] * s_dofToQuad[dx][qx];
            }
            out_x[dz][dy][qx] = t;
          }
        }
      }
      __syncthreads();

      for (int dz = threadIdx.z; dz < NUM_DOFS_1D; dz += blockDim.z)
      {
        for (int qy = threadIdx.y; qy < NUM_QUAD_1D; qy += blockDim.y)
        {
          for (int qx = threadIdx.x; qx < NUM_QUAD_1D; qx += blockDim.x)
          {
            double t = 0;
            for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
            {            
              t += s_dofToQuad[dy][qy] * out_x[dz][dy][qx];
            }
            out_xy[dz][qy][qx] = t;
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
            for (int dz = 0; dz < NUM_DOFS_1D; ++dz)
            {            
              t += s_dofToQuad[dz][qz] * out_xy[dz][qy][qx];
            }
            out[_ijklmNM(v, qx, qy, qz, e,NUM_QUAD_1D,numElements)] = t;
          }
        }
      }
      __syncthreads();
    }
  }
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
                     double* __restrict out)
{
   const unsigned int id = (DIM<<8)|(NUM_VDIM<<4)|(NUM_DOFS_1D-1);
   assert(LOG2(DIM)<=4);
   assert(LOG2(NUM_VDIM)<=4);
   assert(LOG2(NUM_DOFS_1D-1)<=4);
   assert(NUM_QUAD_1D==2*NUM_DOFS_1D);
   if (NUM_QUAD_1D!=2*NUM_DOFS_1D)
   {
      return exit(
                printf("\033[31;1m[rGridFuncToQuad] order ERROR: -ok=p -ot=p-1, p in [1,16]\033[m\n"));
   }
   static std::unordered_map<unsigned int, fGridFuncToQuad> call =
   {
      // 2D
      {0x210,&rGridFuncToQuad2D<1,1,2>},
      {0x211,&rGridFuncToQuad2D<1,2,4>},
      {0x212,&rGridFuncToQuad2D<1,3,6>},
      {0x213,&rGridFuncToQuad2D<1,4,8>},
      {0x214,&rGridFuncToQuad2D<1,5,10>},
      {0x215,&rGridFuncToQuad2D<1,6,12>},
      {0x216,&rGridFuncToQuad2D<1,7,14>},
      {0x217,&rGridFuncToQuad2D<1,8,16>},
      {0x218,&rGridFuncToQuad2D<1,9,18>},
      {0x219,&rGridFuncToQuad2D<1,10,20>},
      {0x21A,&rGridFuncToQuad2D<1,11,22>},
      {0x21B,&rGridFuncToQuad2D<1,12,24>},
      {0x21C,&rGridFuncToQuad2D<1,13,26>},
      {0x21D,&rGridFuncToQuad2D<1,14,28>},
      {0x21E,&rGridFuncToQuad2D<1,15,30>},
      {0x21F,&rGridFuncToQuad2D<1,16,32>},

      // 3D
      // {0x310,&rGridFuncToQuad3D<1,1,2>},
      // {0x311,&rGridFuncToQuad3D<1,2,4>},
      // {0x312,&rGridFuncToQuad3D<1,3,6>},
      // {0x313,&rGridFuncToQuad3D<1,4,8>},
      // {0x314,&rGridFuncToQuad3D<1,5,10>},
      // {0x315,&rGridFuncToQuad3D<1,6,12>},
      // {0x316,&rGridFuncToQuad3D<1,7,14>},
      // {0x317,&rGridFuncToQuad3D<1,8,16>},
      // {0x318,&rGridFuncToQuad3D<1,9,18>},
      // {0x319,&rGridFuncToQuad3D<1,10,20>},
      // {0x31A,&rGridFuncToQuad3D<1,11,22>},
      // {0x31B,&rGridFuncToQuad3D<1,12,24>},
      // {0x31C,&rGridFuncToQuad3D<1,13,26>},
      // {0x31D,&rGridFuncToQuad3D<1,14,28>},
      // {0x31E,&rGridFuncToQuad3D<1,15,30>},
      // {0x31F,&rGridFuncToQuad3D<1,16,32>},
   };

#define call_3d(DOFS,QUAD,BZ,NBLOCK) \
   call_3d_ker(rGridFuncToQuad3D,numElements,DOFS,QUAD,BZ,NBLOCK,\
               numElements,dofToQuad,l2gMap,gf,out,gbuf,rGridFuncToQuad3D_BufSize)
 
   if      (id == 0x310) { call_3d(1 ,2 ,2,1); }
   else if (id == 0x311) { call_3d(2 ,4 ,4,1); }   
   else if (id == 0x312) { call_3d(3 ,6 ,6,1); }
   else if (id == 0x313) { call_3d(4 ,8 ,8,1); }
   else if (id == 0x314) { call_3d(5 ,10,4,1); }
   else if (id == 0x315) { call_3d(6 ,12,2,1); }
   else if (id == 0x316) { call_3d(7 ,14,2,1); }
   else if (id == 0x317) { call_3d(8 ,16,2,1); }
   else if (id == 0x318) { call_3d(9 ,18,2,1); }
   else if (id == 0x319) { call_3d(10,20,2,1); }
   else if (id == 0x31A) { call_3d(11,22,2,1); }
   else if (id == 0x31B) { call_3d(12,24,1,1); }
   else if (id == 0x31C) { call_3d(13,26,1,1); }
   else if (id == 0x31D) { call_3d(14,28,1,1); }
   else if (id == 0x31E) { call_3d(15,30,1,1); }
   else if (id == 0x31E) { call_3d(16,32,1,1); }   
   else
   {
     const int blck = CUDA_BLOCK_SIZE;
     const int grid = (numElements+blck-1)/blck;   
     if (!call[id])
     {
       printf("\n[rGridFuncToQuad] id \033[33m0x%X\033[m ",id);
       fflush(stdout);
     }
     assert(call[id]);
     call0(id,grid,blck,
           numElements,dofToQuad,l2gMap,gf,out);
   }
   CUCHK(cudaGetLastError());
}
