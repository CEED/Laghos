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
                               const double* dofToQuad,
                               const int* l2gMap,
                               const double* gf,
                               double* out) {
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

   // **************************************************************************
   static void global2LocalMap(ParFiniteElementSpace &fes, qarray<int> &map){
      const int elements = fes.GetNE();
      const int localDofs = fes.GetFE(0)->GetDof();

      const FiniteElement *fe = fes.GetFE(0);
      const TensorBasisElement* el = dynamic_cast<const TensorBasisElement*>(fe);
      const Array<int> &dof_map = el->GetDofMap();
      const bool dof_map_is_identity = dof_map.Size()==0;
      const Table& e2dTable = fes.GetElementToDofTable();
      const int *elementMap = e2dTable.GetJ();
      mfem::Array<int> h_map(localDofs*elements);
      
      for (int e = 0; e < elements; ++e) {
         for (int d = 0; d < localDofs; ++d) {
            const int did = dof_map_is_identity?d:dof_map[d];
            const int gid = elementMap[localDofs*e + did];
            const int lid = localDofs*e + d;
            h_map[lid] = gid;
         }
      }
      map = h_map;
   }
   
   // ***************************************************************************
   void Dof2QuadScalar(ParFiniteElementSpace &fes,
                       const IntegrationRule& ir,
                       const double *vec,
                       double *quad) {
      const FiniteElement& fe = *fes.GetFE(0);
      const int dim  = fe.GetDim(); assert(dim==2);
      const int vdim = fes.GetVDim();
      const int elements = fes.GetNE();
      const qDofQuadMaps* maps = qDofQuadMaps::GetTensorMaps(fe,fe,ir);
      const double* dofToQuad = maps->dofToQuad;
      const int localDofs = fes.GetFE(0)->GetDof();
      qarray<int> l2gMap(localDofs, elements);
      global2LocalMap(fes,l2gMap);
      const int quad1D = IntRules.Get(Geometry::SEGMENT,ir.GetOrder()).GetNPoints();
      const int dofs1D = fes.GetFE(0)->GetOrder() + 1;
      assert(vdim==1);
      assert(dofs1D==2);
      assert(quad1D==4);
      vecToQuad2D<1,2,4> __config(elements)
         (elements, dofToQuad, l2gMap, vec, quad);
   }

} // namespace hydrodynamics
   
} // namespace mfem
