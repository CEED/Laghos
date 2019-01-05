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
         const int NUM_QUAD_1D> static
void vecToQuad2D(const int numElements,
                 const double* __restrict dofToQuad,
                 const double* __restrict in,
                 double* __restrict out) {
   GET_CONST_ADRS(dofToQuad);
   GET_CONST_ADRS(in);
   GET_ADRS(out);
   MFEM_FORALL(e, numElements,
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
               const double r_gf = d_in[_ijklNM(v,dx,dy,e,NUM_DOFS_1D,numElements)];
               for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
                  out_x[v][qy] += r_gf * d_dofToQuad[ijN(qy, dx,NUM_QUAD_1D)];
               }
            }
         }
         for (int v = 0; v < NUM_VDIM; ++v) {
            for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
               const double d2q = d_dofToQuad[ijN(qy, dy,NUM_QUAD_1D)];
               for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
                  out_xy[v][qy][qx] += d2q * out_x[v][qx];
               }
            }
         }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
         for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            for (int v = 0; v < NUM_VDIM; ++v) {
               d_out[_ijklNM(v, qx, qy, e,NUM_QUAD_1D,numElements)] = out_xy[v][qy][qx];
            }
         }
      }
   });
}

// *****************************************************************************
typedef void (*fVecToQuad2D)(const int numElements,
                             const double* __restrict dofToQuad,
                             const double* __restrict in,
                             double* __restrict out);

// ***************************************************************************
void QUpdate::Dof2QuadScalar(const kFiniteElementSpace *kfes,
                             const FiniteElementSpace &fes,
                             const kDofQuadMaps *maps,
                             const IntegrationRule& ir,
                             const double *d_in,
                             double **d_out) {
   const int dim = fes.GetMesh()->Dimension();
   assert(dim==2);
   const int vdim = fes.GetVDim();
   const int vsize = fes.GetVSize();
   const mfem::FiniteElement& fe = *fes.GetFE(0);
   const size_t numDofs  = fe.GetDof();
   const size_t nzones = fes.GetNE();
   const size_t nqp = ir.GetNPoints();
   const size_t local_size = numDofs * nzones;
   const size_t out_size =  nqp * nzones;
   const int dofs1D = fes.GetFE(0)->GetOrder() + 1;
   const int quad1D = IntRules.Get(Geometry::SEGMENT,ir.GetOrder()).GetNPoints();
   
   static double *d_local_in = NULL;
   if (!d_local_in){
      d_local_in = (double*) mm::malloc<double>(local_size);
   }

   dbg("\033[7mGlobalToLocal");
   Vector v_in = Vector((double*)d_in, vsize);
   Vector v_local_in = Vector(d_local_in,local_size);
   kfes->GlobalToLocal(v_in,v_local_in);
   
   if (!(*d_out)){
      *d_out = (double*) mm::malloc<double>(out_size);
   }

   dbg("\033[7mvecToQuad2D");
   assert(vdim==1);   
   assert(LOG2(vdim)<=4);
   assert(LOG2(dofs1D)<=4);
   assert(LOG2(quad1D)<=4);
   const size_t id = (vdim<<8)|(dofs1D<<4)|(quad1D);
   static std::unordered_map<unsigned int, fVecToQuad2D> call = {
      {0x124,&vecToQuad2D<1,2,4>},
      {0x148,&vecToQuad2D<1,4,8>},
   };
   if(!call[id]){
      printf("\n[Dof2QuadScalar] id \033[33m0x%lX\033[m ",id);
      fflush(0);
   }
   assert(call[id]);
   call[id](nzones, maps->dofToQuad, d_local_in, *d_out);
}

} // namespace hydrodynamics
   
} // namespace mfem
