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
   static void qGradVector2DSimplex(const int numElements,
                                    const double* __restrict simplx_dofToQuadD,
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
               const double wx = simplx_dofToQuadD[ijkNM(0,q,d,2,NUM_QUAD)];
               const double wy = simplx_dofToQuadD[ijkNM(1,q,d,2,NUM_QUAD)];
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
   template <const int NUM_DOFS,
             const int NUM_QUAD>
   __kernel__
   static void qGradVector2DTensor(const int numElements,
                                   const double* __restrict dofToQuad,
                                   const double* __restrict dofToQuadD,
                                   const double* __restrict in,
                                   double* __restrict out){
      const int NUM_DOFS_1D = 3;
      const int NUM_QUAD_1D = 4;
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
                  for (int vi = 0; vi < 2; ++vi) {
                     s_gradv[ijkN(vi,0,q,2)] += vy*vDx[ijN(vi,qx,2)];
                     s_gradv[ijkN(vi,1,q,2)] += vDy*vx[ijN(vi,qx,2)];
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
   static void qGradVector(const FiniteElementSpace& fes,
                           const IntegrationRule& ir,
                           const bool use_tensor,
                           const qDofQuadMaps* simplex_maps,
                           const qDofQuadMaps* tensor_maps,
                           const size_t size,
                           const double *in,
                           double *out,
                           double *bis){
      push();
      const mfem::FiniteElement &fe = *(fes.GetFE(0));
      const int dim = fe.GetDim(); assert(dim==2);
      const int ndf  = fe.GetDof();
      const int nqp  = ir.GetNPoints();
      const int nzones = fes.GetNE();
      assert(ndf==9);
      assert(nqp==16);
      if (!use_tensor){
         qGradVector2DSimplex<9,16> __config(nzones)
            (nzones,
             simplex_maps->dofToQuadD,
             in, out);
      }else{
         qGradVector2DTensor<9,16> __config(nzones)
            (nzones,
             tensor_maps->dofToQuad,
             tensor_maps->dofToQuadD,
             in, out);
      }
      pop();
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
   // * Dof2DQuad
   // **************************************************************************
   void Dof2QuadGrad(ParFiniteElementSpace &fes,
                     const IntegrationRule& ir,
                     double *velocity,
                     double **d_grad_v_data){
      push();
      const bool use_tensor = true;
      const int dim = fes.GetMesh()->Dimension();
      const int dims = fes.GetVDim();
      const int vsize = fes.GetVSize();
      assert(dim==2);
      assert(dims==2);
      const mfem::FiniteElement& fe = *fes.GetFE(0);
      const size_t numDofs  = fe.GetDof();
      const size_t nzones = fes.GetNE();
      const size_t nqp = ir.GetNPoints();
      const size_t v_local_size = dims * numDofs * nzones;
      mfem::Array<double> local_velocity(v_local_size);
      
      if (use_tensor){
         const kernels::kFiniteElementSpace &h1k =
            fes.Get_PFESpace()->As<kernels::kFiniteElementSpace>();
         h1k.GlobalToLocal(velocity,local_velocity.GetData());
      }else{
         const bool ordering = fes.GetOrdering();
         const bool vdim_ordering = ordering == Ordering::byVDIM;
         const Table& e2dTable = fes.GetElementToDofTable();
         const int* elementMap = e2dTable.GetJ();
         if (!vdim_ordering) reorderByVDim(fes, vsize, velocity);
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
         if (!vdim_ordering) reorderByNodes(fes, vsize, velocity);
      }
      double *d_v_data =
         (double*)mfem::kernels::kmalloc<double>::operator new(v_local_size);
      mfem::kernels::kmemcpy::rHtoD(d_v_data,
                                    local_velocity.GetData(),
                                    v_local_size*sizeof(double));
      const size_t grad_v_size = dim * dim * nqp * nzones;
      *d_grad_v_data =
         (double*) mfem::kernels::kmalloc<double>::operator new(grad_v_size);
      double *d_grad_v_data_bis =
         (double*) mfem::kernels::kmalloc<double>::operator new(grad_v_size);
      const qDofQuadMaps *simplex_maps = qDofQuadMaps::GetSimplexMaps(fe,ir);
      const qDofQuadMaps *tensor_maps = qDofQuadMaps::GetTensorMaps(fe,fe,ir);
      qGradVector(fes, ir,use_tensor,
                  simplex_maps,
                  tensor_maps,
                  v_local_size,
                  d_v_data,
                  *d_grad_v_data,
                  d_grad_v_data_bis);
      pop();
   }
   
} // namespace hydrodynamics
   
} // namespace mfem
