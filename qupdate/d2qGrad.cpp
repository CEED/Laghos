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
          const int NUM_QUAD_1D> static
void qGradVector2D(const int numElements,
                   const double* __restrict dofToQuad,
                   const double* __restrict dofToQuadD,
                   const double* __restrict in,
                   double* __restrict out){
   const int NUM_QUAD = NUM_QUAD_1D*NUM_QUAD_1D;
   GET_CONST_ADRS(dofToQuad);
   GET_CONST_ADRS(dofToQuadD);
   GET_CONST_ADRS(in);
   GET_ADRS(out);
   MFEM_FORALL(e, numElements,
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
               const double wDx = d_dofToQuadD[ijN(qx,dx,NUM_QUAD_1D)];
               const double wx  = d_dofToQuad[ijN(qx,dx,NUM_QUAD_1D)];
               for (int c = 0; c < 2; ++c) {
                  const double input = d_in[_ijklNM(c,dx,dy,e,NUM_DOFS_1D,numElements)];
                  vDx[ijN(c,qx,2)] += input * wDx;
                  vx[ijN(c,qx,2)] += input * wx;
               }
            }
         }
         for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
            const double vy  = d_dofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
            const double vDy = d_dofToQuadD[ijN(qy,dy,NUM_QUAD_1D)];
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
         d_out[ijklNM(0,0,q,e,2,NUM_QUAD)] = s_gradv[ijkN(0,0,q,2)];
         d_out[ijklNM(1,0,q,e,2,NUM_QUAD)] = s_gradv[ijkN(1,0,q,2)];
         d_out[ijklNM(0,1,q,e,2,NUM_QUAD)] = s_gradv[ijkN(0,1,q,2)];
         d_out[ijklNM(1,1,q,e,2,NUM_QUAD)] = s_gradv[ijkN(1,1,q,2)];
      }
   });
}

// *****************************************************************************
typedef void (*fGradVector2D)(const int numElements,
                              const double* __restrict dofToQuad,
                              const double* __restrict dofToQuadD,
                              const double* __restrict in,
                              double* __restrict out);
              
// **************************************************************************
void QUpdate::Dof2QuadGrad(const kFiniteElementSpace *kfes,
                           const FiniteElementSpace &fes,
                           const kDofQuadMaps *maps,
                           const IntegrationRule& ir,
                           const double *d_in,
                           double **d_out){
   const int dim = fes.GetMesh()->Dimension();
   assert(dim==2);
   const int vdim = fes.GetVDim();
   assert(vdim==2);
   const int vsize = fes.GetVSize();
   const mfem::FiniteElement& fe = *fes.GetFE(0);
   const size_t numDofs  = fe.GetDof();
   const size_t nzones = fes.GetNE();
   const size_t nqp = ir.GetNPoints();
   const size_t local_size = vdim * numDofs * nzones;
   const size_t out_size = vdim * vdim * nqp * nzones;
   const int dofs1D = fes.GetFE(0)->GetOrder() + 1;
   const int quad1D = IntRules.Get(Geometry::SEGMENT,ir.GetOrder()).GetNPoints();
   
   static double *d_local_in = NULL;
   if (!d_local_in){
      d_local_in = (double*) mm::malloc<double>(local_size);
   }
      
   dbg("GlobalToLocal");
   Vector v_in = Vector((double*)d_in, vsize);
   Vector v_local_in = Vector(d_local_in, local_size);
   kfes->GlobalToLocal(v_in, v_local_in);
            
   if (!(*d_out)){
      *d_out = (double*) mm::malloc<double>(out_size);
   }

   // **************************************************************************
   assert(LOG2(dofs1D)<=4);
   assert(LOG2(quad1D)<=4);
   const size_t id = (dofs1D<<4)|(quad1D);
   static std::unordered_map<unsigned int, fGradVector2D> call = {
      {0x34,&qGradVector2D<3,4>},
      {0x58,&qGradVector2D<5,8>},
   };
   if(!call[id]){
      printf("\n[Dof2QuadGrad] id \033[33m0x%lX\033[m ",id);
      fflush(0);
   }
   assert(call[id]);
   call[id](nzones,
            maps->dofToQuad,
            maps->dofToQuadD,
            d_local_in,
            *d_out);
}
   
} // namespace hydrodynamics
   
} // namespace mfem
