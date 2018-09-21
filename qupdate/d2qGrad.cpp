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

#include "qupdate.hpp"

using namespace std;

namespace mfem {

namespace hydrodynamics {
   
   // **************************************************************************
   template <const int NUM_DOFS_1D,
             const int NUM_QUAD_1D>
   __kernel__
   static void qGradVector2D(const int numElements,
                             const double* __restrict dofToQuad,
                             const double* __restrict dofToQuadD,
                             const double* __restrict in,
                             double* __restrict out){
      const int NUM_QUAD = NUM_QUAD_1D*NUM_QUAD_1D;
#ifdef __NVCC__
      const int e = blockDim.x * blockIdx.x + threadIdx.x;
      if (e < numElements)
#else
      for(int e=0; e<numElements; e+=1)
#endif
      {
         double s_gradv[4*NUM_QUAD];
         for (int i = 0; i < (4*NUM_QUAD); ++i) {
            s_gradv[i] = 0.0;
         }
         
         for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
            double vDx[2*NUM_QUAD_1D];
            double vx[2*NUM_QUAD_1D];
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
               for (int vi = 0; vi < 2; ++vi) {
                  vDx[ijN(vi,qx,2)] = 0.0;
                   vx[ijN(vi,qx,2)] = 0.0;
               }
            }
            for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
               for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
                  const double wDx = dofToQuadD[ijN(qx,dx,NUM_QUAD_1D)];
                  const double wx  = dofToQuad[ijN(qx,dx,NUM_QUAD_1D)];
                  for (int c = 0; c < 2; ++c) {
                     const double input = in[_ijklNM(c,dx,dy,e,NUM_DOFS_1D,numElements)];
                     vDx[ijN(c,qx,2)] += input * wDx;
                      vx[ijN(c,qx,2)] += input * wx;
                  }
               }
            }
            for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
               const double vy  = dofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
               const double vDy = dofToQuadD[ijN(qy,dy,NUM_QUAD_1D)];
               for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
                  const int q = qx+NUM_QUAD_1D*qy;
                  for (int c = 0; c < 2; ++c) {
                     s_gradv[ijkN(c,0,q,2)] += vy*vDx[ijN(c,qx,2)];
                     s_gradv[ijkN(c,1,q,2)] += vDy*vx[ijN(c,qx,2)];
                  }
               }
            }
         }         
         for (int q = 0; q < NUM_QUAD; ++q) {
            out[ijklNM(0,0,q,e,2,NUM_QUAD)] = s_gradv[ijkN(0,0,q,2)];
            out[ijklNM(1,0,q,e,2,NUM_QUAD)] = s_gradv[ijkN(1,0,q,2)];
            out[ijklNM(0,1,q,e,2,NUM_QUAD)] = s_gradv[ijkN(0,1,q,2)];
            out[ijklNM(1,1,q,e,2,NUM_QUAD)] = s_gradv[ijkN(1,1,q,2)];
         }
      }
   }
   
   // **************************************************************************
   // * Dof2DQuad
   // **************************************************************************
   void Dof2QuadGrad(ParFiniteElementSpace &fes,
                     const IntegrationRule& ir,
                     const double *d_in,
                     double **d_out){
      push();
      const kernels::kFiniteElementSpace &kfes =
         fes.Get_PFESpace()->As<kernels::kFiniteElementSpace>();
      const kernels::kDofQuadMaps* maps = kernels::kDofQuadMaps::Get(fes,ir);
      
      const int dim = fes.GetMesh()->Dimension();
      const int vdim = fes.GetVDim();
      const int vsize = fes.GetVSize();
      assert(dim==2);
      assert(vdim==2);
      const mfem::FiniteElement& fe = *fes.GetFE(0);
      const size_t numDofs  = fe.GetDof();
      const size_t nzones = fes.GetNE();
      const size_t nqp = ir.GetNPoints();

      const size_t local_size = vdim * numDofs * nzones;
      static double *d_local_in = NULL;
      if (!d_local_in){
         d_local_in=
            (double*) kernels::kmalloc<double>::operator new(local_size);
      }
      
      kfes.GlobalToLocal(d_in,d_local_in);
            
      const size_t out_size = vdim * vdim * nqp * nzones;
      if (!(*d_out)){
         *d_out =
            (double*) kernels::kmalloc<double>::operator new(out_size);
      }
    
      const int dofs1D = fes.GetFE(0)->GetOrder() + 1;
      const int quad1D = IntRules.Get(Geometry::SEGMENT,ir.GetOrder()).GetNPoints();

      assert(dofs1D==3);
      assert(quad1D==4);      
      qGradVector2D<3,4> __config(nzones)
         (nzones, maps->dofToQuad, maps->dofToQuadD, d_local_in, *d_out);      
      pop();
   }
   
} // namespace hydrodynamics
   
} // namespace mfem
