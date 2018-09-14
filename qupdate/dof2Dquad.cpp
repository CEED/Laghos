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
   template <const int NUM_DOFS,
             const int NUM_QUAD>
   __kernel__
   static void qGradVector2D(const int numElements,
                             const double* __restrict dofToQuadD,
                             const double* __restrict in,
                             double* __restrict out){
#ifdef __NVCC__
      const int e = blockDim.x * blockIdx.x + threadIdx.x;
      if (e < numElements)
#else
      for(int e=0; e<numElements; e+=1)
#endif
      {
         double s_in[2 * NUM_DOFS];
         for (int q = 0; q < NUM_QUAD; ++q) {
            for (int d = q; d < NUM_DOFS; d+=NUM_QUAD) {
               s_in[ijN(0,d,2)] = in[ijkNM(0,d,e,2,NUM_DOFS)];
               s_in[ijN(1,d,2)] = in[ijkNM(1,d,e,2,NUM_DOFS)];
            }
         }
         for (int q = 0; q < NUM_QUAD; ++q) {
            double J11 = 0.0; double J12 = 0.0;
            double J21 = 0.0; double J22 = 0.0;
            for (int d = 0; d < NUM_DOFS; ++d) {
               const double wx = dofToQuadD[ijkNM(0,q,d,2,NUM_QUAD)];
               const double wy = dofToQuadD[ijkNM(1,q,d,2,NUM_QUAD)];
               const double x = s_in[ijN(0,d,2)];
               const double y = s_in[ijN(1,d,2)];
               J11 += (wx * x); J12 += (wx * y);
               J21 += (wy * x); J22 += (wy * y);
            }
            out[ijklNM(0,0,q,e,2,NUM_QUAD)] = J11;
            out[ijklNM(1,0,q,e,2,NUM_QUAD)] = J12;
            out[ijklNM(0,1,q,e,2,NUM_QUAD)] = J21;
            out[ijklNM(1,1,q,e,2,NUM_QUAD)] = J22;
         }
      }
   }

   // **************************************************************************
   static void reorderByVDim(const FiniteElementSpace& fes,
                             const size_t size,
                             double *data){
      push();
      const size_t vdim = fes.GetVDim();
      const size_t ndofs = fes.GetNDofs();
      double *temp = new double[size];
      for (size_t k=0; k<size; k++) temp[k]=0.0;
      size_t k=0;
      for (size_t d = 0; d < ndofs; d++)
         for (size_t v = 0; v < vdim; v++)      
            temp[k++] = data[d+v*ndofs];
      for (size_t i=0; i<size; i++){
         data[i] = temp[i];
      }
      delete [] temp;
      pop();
   }

   // ***************************************************************************
   static void reorderByNodes(const FiniteElementSpace& fes,
                              const size_t size,
                              double *data){
      push();
      const size_t vdim = fes.GetVDim();
      const size_t ndofs = fes.GetNDofs(); 
      double *temp = new double[size];
      for (size_t k=0; k<size; k++) temp[k]=0.0;
      size_t k=0;
      for (size_t j=0; j < ndofs; j++)
         for (size_t i=0; i < vdim; i++)
            temp[j+i*ndofs] = data[k++];
      for (size_t i = 0; i < size; i++){
         data[i] = temp[i];
      }
      delete [] temp;
      pop();
   }

   // **************************************************************************
   static void qGradVector(const FiniteElementSpace& fes,
                           const IntegrationRule& ir,
                           const qDofQuadMaps* maps,
                           const size_t size,
                           double *in,
                           double *out){
      push();
      const mfem::FiniteElement &fe = *(fes.GetFE(0));
      const int dim = fe.GetDim(); assert(dim==2);
      const int ndf  = fe.GetDof();
      const int nqp  = ir.GetNPoints();
      const int nzones = fes.GetNE();
      assert(ndf==9);
      assert(nqp==16);
      qGradVector2D<9,16> __config(nzones) (nzones, maps->dofToQuadD, in, out);
      pop();
   }

   // **************************************************************************
   // * Dof2DQuad
   // **************************************************************************
   void Dof2QuadGrad(ParFiniteElementSpace &fes,
                     const IntegrationRule& ir,
                     double *velocity,
                     double **d_grad_v_data){
      push();
      const int dim = fes.GetMesh()->Dimension();
      assert(dim==2);
      const int size = fes.GetVSize();
      const int dims = fes.GetVDim();
      assert(dims==2);
      const mfem::FiniteElement& fe = *fes.GetFE(0);
      const int numDofs  = fe.GetDof();
      const int nzones = fes.GetNE();
      const int nqp = ir.GetNPoints();
      
      reorderByVDim(fes, size, velocity);
      
      const size_t v_local_size = dims * numDofs * nzones;
      mfem::Array<double> local_velocity(v_local_size);
      const Table& e2dTable = fes.GetElementToDofTable();
      const int* elementMap = e2dTable.GetJ();
      
      for (int e = 0; e < nzones; ++e) {
         for (int d = 0; d < numDofs; ++d) {
            const int lid = d+numDofs*e;
            const int gid = elementMap[lid];
            for (int v = 0; v < dims; ++v) {
               const int moffset = v+dims*lid;
               const int xoffset = v+dims*gid;
               local_velocity[moffset] = velocity[xoffset];
            }
         }
      }
      
      double *d_v_data = (double*)mfem::kernels::kmalloc<double>::operator new(v_local_size);
      mfem::kernels::kmemcpy::rHtoD(d_v_data,
                                    local_velocity.GetData(),
                                    v_local_size*sizeof(double));

      reorderByNodes(fes, size, velocity);
      const size_t grad_v_size = dim * dim * nqp * nzones;
      *d_grad_v_data = (double*) mfem::kernels::kmalloc<double>::operator new(grad_v_size);
      const qDofQuadMaps *simplex_maps = qDofQuadMaps::GetSimplexMaps(fe,ir);
      qGradVector(fes, ir,
                  simplex_maps,
                  v_local_size,
                  d_v_data,
                  *d_grad_v_data);
      pop();
   }
   
} // namespace hydrodynamics
   
} // namespace mfem
