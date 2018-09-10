
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

#include "../laghos_solver.hpp"
#include "qupdate.hpp"

using namespace std;

namespace mfem {
   
namespace hydrodynamics {
   
   // **************************************************************************
   static void vecToQuad2D(const int NUM_VDIM,
                           const int NUM_DOFS_1D,
                           const int NUM_QUAD_1D,
                           const int numElements,
                           const double* dofToQuad,
                           const int* l2gMap,
                           const double* gf,
                           double* out) {
      for(int e=0;e<numElements;e+=1){
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
               const int gid = l2gMap[ijkN(dx, dy, e,NUM_DOFS_1D)];
               for (int v = 0; v < NUM_VDIM; ++v) {
                  const double r_gf = gf[v + gid*NUM_VDIM];
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
   void d2q(ParFiniteElementSpace &fes,
            const IntegrationRule& ir,
            const double *vec,
            double *quad) {
      const FiniteElement& fe = *fes.GetFE(0);
      const int dim  = fe.GetDim(); assert(dim==2);
      const int vdim = fes.GetVDim();
      const int elements = fes.GetNE();
      const qDofQuadMaps* maps = qDofQuadMaps::GetTensorMaps(fe,fe,ir);
      const double* dofToQuad = maps->dofToQuad;
      const int* l2gMap = global2LocalMap(fes);
      const int quad1D = IntRules.Get(Geometry::SEGMENT,ir.GetOrder()).GetNPoints();
      const int dofs1D = fes.GetFE(0)->GetOrder() + 1;     
      vecToQuad2D(vdim, dofs1D, quad1D, elements, dofToQuad, l2gMap, vec, quad);
   }

} // namespace hydrodynamics
   
} // namespace mfem
