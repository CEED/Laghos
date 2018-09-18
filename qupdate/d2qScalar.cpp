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
   template<const int NUM_VDIM,
            const int NUM_DOFS_1D,
            const int NUM_QUAD_1D>
   __kernel__ void vecToQuad2D(const int numElements,
                               const double* __restrict dofToQuad,
                               const double* __restrict in,
                               double* __restrict out) {
#ifdef __NVCC__
      const int e = blockDim.x * blockIdx.x + threadIdx.x;
      if (e < numElements)
#else
      for(int e=0;e<numElements;e+=1)
#endif
      {
         double out_xy[NUM_VDIM][NUM_QUAD_1D][NUM_QUAD_1D];
         for (int v = 0; v < NUM_VDIM; ++v) {
            for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
               for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
                  out_xy[v][qy][qx] = 0;
               }
            }
         }
         for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
            double out_x[NUM_VDIM][NUM_QUAD_1D];
            for (int v = 0; v < NUM_VDIM; ++v) {
               for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
                  out_x[v][qy] = 0;
               }
            }
            for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
               for (int v = 0; v < NUM_VDIM; ++v) {
                  const double r_gf = in[_ijklNM(v,dx,dy,e,NUM_DOFS_1D,numElements)];
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
      }
   }
   
   // ***************************************************************************
   void Dof2QuadScalar(ParFiniteElementSpace &fes,
                       const IntegrationRule& ir,
                       const double *d_in,
                       double **d_out) {
      push();
      const kernels::kFiniteElementSpace &kfes =
         fes.Get_PFESpace()->As<kernels::kFiniteElementSpace>();
      const kernels::kDofQuadMaps* maps = kernels::kDofQuadMaps::Get(fes,ir);

      const int dim = fes.GetMesh()->Dimension();
      const int vdim = fes.GetVDim();
      const int vsize = fes.GetVSize();
      assert(dim==2);
      assert(vdim==1);
      const mfem::FiniteElement& fe = *fes.GetFE(0);
      const size_t numDofs  = fe.GetDof();
      const size_t nzones = fes.GetNE();
      const size_t nqp = ir.GetNPoints();

      const size_t local_size = numDofs * nzones;
      double *d_local_in =
         (double*)mfem::kernels::kmalloc<double>::operator new(local_size);

      kfes.GlobalToLocal(d_in,d_local_in);
      dbg("GlobalToLocal done");
      
      const size_t out_size =  nqp * nzones;
      *d_out = (double*) kernels::kmalloc<double>::operator new(out_size);
      
      const int dofs1D = fes.GetFE(0)->GetOrder() + 1;
      const int quad1D = IntRules.Get(Geometry::SEGMENT,ir.GetOrder()).GetNPoints();
      
      assert(dofs1D==2);
      assert(quad1D==4);
      vecToQuad2D<1,2,4> __config(nzones)
         (nzones, maps->dofToQuad, d_local_in, *d_out);
      pop();
   }

} // namespace hydrodynamics
   
} // namespace mfem
