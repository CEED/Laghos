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

namespace mfem {

namespace hydrodynamics {

   // **************************************************************************
   __device__ static inline double det2D(const double *d){
      return d[0] * d[3] - d[1] * d[2];
   }

   // **************************************************************************
   __device__
   void calcInverse2D(const size_t n, const double *a, double *i){
      const double d = det2D(a);
      const double t = 1.0 / d;
      i[0*n+0] =  a[1*n+1] * t ;
      i[0*n+1] = -a[0*n+1] * t ;
      i[1*n+0] = -a[1*n+0] * t ;
      i[1*n+1] =  a[0*n+0] * t ;
   }

   // **************************************************************************
   /*__attribute__((unused))
   static void getL2Values(const int dim,
                           const int nL2dof1D,
                           const int nqp1D,
                           const double* __restrict__ vecL2,
                           double* __restrict__ vecQ){     
      assert(dim == 2);      
      double LQ[nL2dof1D*nqp1D];         
      // LQ_j2_k1 = vecL2_j1_j2 LQs_j1_k1  -- contract in x direction.
      multAtB(nL2dof1D, nL2dof1D, tensors1D->LQshape1D.Width(),
              vecL2, tensors1D->LQshape1D.Data(), LQ);      
      // QQ_k1_k2 = LQ_j2_k1 LQs_j2_k2 -- contract in y direction.
      multAtB(nL2dof1D, nqp1D, tensors1D->LQshape1D.Width(),
              LQ, tensors1D->LQshape1D.Data(), vecQ);
              }*/

   // **************************************************************************
   /*__attribute__((unused))
   static void getVectorGrad(const int dim,
                             const int nH1dof1D,
                             const int nqp1D,
                             const Array<int> &dof_map,
                             const DenseMatrix &vec,
                             DenseTensor &J) {
      assert(dim == 2);
      
      const int nH1dof = nH1dof1D * nH1dof1D;
      
      double X[nH1dof];
      double HQ[nH1dof1D * nqp1D];
      double QQ[nqp1D * nqp1D];
      
      for (int c = 0; c < 2; c++) {
         
         // Transfer from the mfem's H1 local numbering to the tensor structure
         for (int j = 0; j < nH1dof; j++) X[j] = vec(dof_map[j], c);         

         // HQ_i2_k1  = X_i1_i2 HQg_i1_k1  -- gradients in x direction.
         // QQ_k1_k2  = HQ_i2_k1 HQs_i2_k2 -- contract  in y direction.
         multAtB(nH1dof1D, nH1dof1D, tensors1D->HQgrad1D.Width(),
                 X, tensors1D->HQgrad1D.Data(), HQ);
         multAtB(nH1dof1D, nqp1D, tensors1D->HQshape1D.Width(),
                 HQ, tensors1D->HQshape1D.Data(), QQ);
         // Set the (c,0) component of the Jacobians at all quadrature points.
         for (int k1 = 0; k1 < nqp1D; k1++) {
            for (int k2 = 0; k2 < nqp1D; k2++) {
               const int idx = k2 * nqp1D + k1;
               J(idx)(c, 0) = QQ[idx];
            }
         }

         // HQ_i2_k1  = X_i1_i2 HQs_i1_k1  -- contract  in x direction.
         // QQ_k1_k2  = HQ_i2_k1 HQg_i2_k2 -- gradients in y direction.
         multAtB(nH1dof1D, nH1dof1D, tensors1D->HQshape1D.Width(),
                 X, tensors1D->HQshape1D.Data(), HQ);
         multAtB(nH1dof1D, nqp1D, tensors1D->HQgrad1D.Width(),
                 HQ, tensors1D->HQgrad1D.Data(), QQ);
         // Set the (c,1) component of the Jacobians at all quadrature points.
         for (int k1 = 0; k1 < nqp1D; k1++) {
            for (int k2 = 0; k2 < nqp1D; k2++) {
               const int idx = k2 * nqp1D + k1;
               J(idx)(c, 1) = QQ[idx];
            }
         }
      }
      }*/

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
         //dbg("elem #%d",e);
         double s_in[2 * NUM_DOFS];
         for (int q = 0; q < NUM_QUAD; ++q) {
            //dbg("\tq #%d",q);
            for (int d = q; d < NUM_DOFS; d+=NUM_QUAD) {
               //dbg("\t\td=%d",d);
               //const int x0 = ijN(0,d,2);
               //const int x1 = ijkNM(0,d,e,2,NUM_DOFS);
               //const int y0 = ijN(1,d,2);
               //const int y1 = ijkNM(1,d,e,2,NUM_DOFS);
               //const double x = in[ijkNM(0,d,e,2,NUM_DOFS)];
               //const double y = in[ijkNM(1,d,e,2,NUM_DOFS)];
               //dbg("\t\t%d <= %d: %f",x0,x1,x);
               //dbg("\t\t%d <= %d: %f",y0,y1,y);
               s_in[ijN(0,d,2)] = in[ijkNM(0,d,e,2,NUM_DOFS)];
               s_in[ijN(1,d,2)] = in[ijkNM(1,d,e,2,NUM_DOFS)];
            }
         }
         //dbg("eof share, returning to elem #%d",e);
         for (int q = 0; q < NUM_QUAD; ++q) {
            //dbg("\tq #%d",q);
            double J11 = 0.0; double J12 = 0.0;
            double J21 = 0.0; double J22 = 0.0;
            for (int d = 0; d < NUM_DOFS; ++d) {
               //dbg("\t\td=%d",d);
               const double wx = dofToQuadD[ijkNM(0,q,d,2,NUM_QUAD)];
               const double wy = dofToQuadD[ijkNM(1,q,d,2,NUM_QUAD)];
               const double x = s_in[ijN(0,d,2)];
               const double y = s_in[ijN(1,d,2)];
               //dbg("\t\twx=%f, wy=%f",wx,wy);
               //dbg("\t\t x=%f,  y=%f", x, y);
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
      dbg("size=%d",size);    
      dbg("vdim=%d ndofs=%d",vdim, ndofs);
      double *temp = new double[size];
      for (size_t k=0; k<size; k++) temp[k]=0.0;
      size_t k=0;
      for (size_t d = 0; d < ndofs; d++)
         for (size_t v = 0; v < vdim; v++)      
            temp[k++] = data[d+v*ndofs];
      for (size_t i=0; i<size; i++){
         data[i] = temp[i];
         //dbg("data[%d]=%f",i,data[i]);
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
      //dbg("size=%d",size);      
      //dbg("vdim=%d ndofs=%d",vdim, ndofs);      
      double *temp = new double[size];
      for (size_t k=0; k<size; k++) temp[k]=0.0;
      size_t k=0;
      for (size_t j=0; j < ndofs; j++)
         for (size_t i=0; i < vdim; i++)
            temp[j+i*ndofs] = data[k++];
      for (size_t i = 0; i < size; i++){
         data[i] = temp[i];
         //dbg("data[%d]=%f",i,data[i]);
      }
      delete [] temp;
      pop();
   }

   // **************************************************************************
   void qGradVector(const FiniteElementSpace& fes,
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
   __device__ double Det(const size_t dim, const double *J){
      assert(dim==2);
      return J[0] * J[3] - J[1] * J[2];
   }
   
   // **************************************************************************
   __device__ double norml2(const int size, const double *data) {
      if (0 == size) return 0.0;
      if (1 == size) return std::abs(data[0]);
      double scale = 0.0;
      double sum = 0.0;
      for (int i = 0; i < size; i++) {
         if (data[i] != 0.0)
         {
            const double absdata = fabs(data[i]);
            if (scale <= absdata)
            {
               const double sqr_arg = scale / absdata;
               sum = 1.0 + sum * (sqr_arg * sqr_arg);
               scale = absdata;
               continue;
            } // end if scale <= absdata
            const double sqr_arg = absdata / scale;
            sum += (sqr_arg * sqr_arg); // else scale > absdata
         } // end if data[i] != 0
      }
      return scale * sqrt(sum);
   }
   
   // **************************************************************************
   template<const int dim>
   __kernel__ void qkernel(const int nzones,
                           const int nqp,
                           const int nqp1D,
                           const double gamma,
                           const bool use_viscosity,
                           const double h0,
                           const double h1order,
                           const double cfl,
                           const double infinity,
                           
                           const double *weights,
                           const double *_J,
                           const double *rho0DetJ0w,
                           const double *e_quads,
                           const double *grad_v_ext,
                           const double *Jac0inv,
                           double *dt_est,
                           double *stressJinvT){
      double min_detJ = infinity;
#ifdef __NVCC__
      //const int z = blockDim.x * blockIdx.x + threadIdx.x;
      //if (z < nzones)
      const int _z = blockDim.x * blockIdx.x + threadIdx.x;
      if (_z >= 1) return;
      for (int z = 0; z < nzones; z++)
#else
      for (int z = 0; z < nzones; z++)
#endif
      {
         // ********************************************************************
         for (int q = 0; q < nqp; q++) {
            const int idx = z * nqp + q;
            const double weight =  weights[q];
            //printf("\nweight=%f",weight);
            const double inv_weight = 1. / weight;
            const double *J = &_J[(z*nqp+q)*nzones];
            const double detJ = Det(dim,J);
            //printf("\ndetJ=%f",detJ);
            min_detJ = fmin(min_detJ, detJ);
            //printf("\nmin_detJ=%f",min_detJ);
            double Jinv[dim*dim];
            calcInverse2D(dim, J, Jinv);    
            //for(int k=0;k<dim*dim;k+=1) printf("%f ",Jinv[k]);    
            // *****************************************************************
            const double rho = inv_weight * rho0DetJ0w[idx] / detJ;
            const double e   = fmax(0.0, e_quads[z*nqp1D*nqp1D+q]);
            const double p  = (gamma - 1.0) * rho * e;
            const double sound_speed = sqrt(gamma * (gamma-1.0) * e);
            //printf("\nrho=%f, e=%f, p=%f, sound_speed=%f",rho,e,p,sound_speed);
            // *****************************************************************
            double stress[dim*dim];
            for (int k=0;k<dim*dim;k+=1) stress[k] = 0.0;
            for (int d = 0; d < dim; d++)  stress[d*dim+d] = -p;
            // *****************************************************************
            double visc_coeff = 0.0;
            if (use_viscosity) {
               assert(false);
               // Compression-based length scale at the point. The first
               // eigenvector of the symmetric velocity gradient gives the
               // direction of maximal compression. This is used to define the
               // relative change of the initial length scale.               
               const double *dV = &grad_v_ext[(z*nqp+q)*nzones];
               double sgrad_v[dim*dim];
               mult(dim,dim,dim, dV, Jinv, sgrad_v);
               symmetrize(dim,sgrad_v);
               double eig_val_data[3], eig_vec_data[9];
               if (dim==1) {
                  eig_val_data[0] = sgrad_v[0*dim+0];
                  eig_vec_data[0] = 1.;
               }
               else {
                  calcEigenvalues(dim, &sgrad_v[0], eig_val_data, eig_vec_data);
               }
               double *compr_dir = eig_vec_data;
               // Computes the initial->physical transformation Jacobian.
               double Jpi[dim*dim];
               mult(dim,dim,dim, J, &Jac0inv[idx], Jpi);
               double ph_dir[dim];
               //Jpi.Mult(compr_dir, ph_dir);
               multV(dim, dim, Jpi, compr_dir, ph_dir);
               // Change of the initial mesh size in the compression direction.
               const double h = h0 * norml2(dim,ph_dir) / norml2(9,compr_dir);
               // Measure of maximal compression.
               const double mu = eig_val_data[0];
               visc_coeff = 2.0 * rho * h * h * fabs(mu);
               if (mu < 0.0) { visc_coeff += 0.5 * rho * h * sound_speed; }
               add(dim, dim, visc_coeff, sgrad_v, stress);
            }
            // Time step estimate at the point. Here the more relevant length
            // scale is related to the actual mesh deformation; we use the min
            // singular value of the ref->physical Jacobian. In addition, the
            // time step estimate should be aware of the presence of shocks.
            const double h_min = calcSingularvalue(dim, dim-1, J) / h1order;
            const double inv_h_min = 1. / h_min;
            const double inv_rho_inv_h_min_sq = inv_h_min * inv_h_min / rho ;
            const double inv_dt = sound_speed * inv_h_min + 2.5 * visc_coeff * inv_rho_inv_h_min_sq;
            //printf("\nh_min=%f, inv_h_min=%f, inv_rho_inv_h_min_sq=%f, inv_dt=%f",h_min,inv_h_min,inv_rho_inv_h_min_sq,inv_dt);
            //printf("\nmin_detJ=%f, dt_est=%f, cfl=%f, inv_dt=%f",min_detJ,*dt_est,cfl,inv_dt);
            if (min_detJ < 0.0) {
               // This will force repetition of the step with smaller dt.
               *dt_est = 0.0;
            } else {
               *dt_est = fmin(*dt_est, cfl * (1.0 / inv_dt) );
            }
            //printf("\ndt_est=%f",*dt_est);
            // Quadrature data for partial assembly of the force operator.
            double stressJiT[dim*dim];
            multABt(dim, dim, dim, stress, Jinv, stressJiT);
            for(int k=0;k<dim*dim;k+=1) stressJiT[k] *= weight * detJ;
            for (int vd = 0 ; vd < dim; vd++) {
               for (int gd = 0; gd < dim; gd++) {
                  double *base = stressJinvT;
                  double *offset = &stressJinvT[q + z*nqp + nqp*nzones*(gd+vd*dim)];
                  assert(offset>=base);
                  const size_t delta = offset-base;
                  //tdata+k*Mk.Height()*Mk.Width()
                  stressJinvT[q + z*nqp + nqp*nzones*(gd+vd*dim)] =
                     stressJiT[vd+gd*dim];
                  //printf("\nz=%d,q=%d,stressJiT=%f, delta=%ld",z,q,stressJiT[vd+gd*dim],delta);
               }
            }
         }
      }
   }
   
   // **************************************************************************
   // * Last kernel QUpdate
   // **************************************************************************
   void QUpdate(const int dim,
                const int nzones,
                const int l2dofs_cnt,
                const int h1dofs_cnt,
                const bool use_viscosity,
                const bool p_assembly,
                const double cfl,
                TimingData &timer,
                Coefficient *material_pcf,
                const IntegrationRule &integ_rule,
                ParFiniteElementSpace &H1FESpace,
                ParFiniteElementSpace &L2FESpace,
                const Vector &S,
                bool &quad_data_is_current,
                QuadratureData &quad_data) {
      push();
      assert(dim==2);
      assert(p_assembly);
      assert(material_pcf);

      // ***********************************************************************
      ElementTransformation *T = H1FESpace.GetElementTransformation(0);
      const IntegrationPoint &ip = integ_rule.IntPoint(0);
      const double gamma = material_pcf->Eval(*T,ip);

      // ***********************************************************************
      if (quad_data_is_current) return;

      // ***********************************************************************
      timer.sw_qdata.Start();
      const mfem::FiniteElement& fe = *H1FESpace.GetFE(0);
      const int dims     = H1FESpace.GetVDim();
      const int elements = H1FESpace.GetNE();
      const int numDofs  = fe.GetDof();
      const int nqp = integ_rule.GetNPoints();
      assert(elements==nzones);
      dbg("numDofs=%d, nqp=%d, nzones=%d",numDofs,nqp,nzones);

      Vector* sptr = (Vector*) &S;
      
      const Engine &engine = L2FESpace.GetMesh()->GetEngine();
      const size_t H1_size = H1FESpace.GetVSize();
      const size_t L2_size = L2FESpace.GetVSize();
                 
      const int nqp1D    = tensors1D->LQshape1D.Width();
      const int nL2dof1D = tensors1D->LQshape1D.Height();
      const int nH1dof1D = tensors1D->HQshape1D.Height();
      
      // Energy dof => quads ***************************************************
      const size_t e_quads_size = nzones * nqp;
      double *d_e_data = (double*)mfem::kernels::kmalloc<double>::operator new(L2_size);
      double *e_data = sptr->GetData()+2*H1_size;
      //for(int k=0;k<L2_size;k+=1) dbg("e[%d]=%f",k,e_data[k]); assert(false);
      mfem::kernels::kmemcpy::rHtoD(d_e_data, e_data, L2_size*sizeof(double));
      double *d_e_quads_data = (double*)mfem::kernels::kmalloc<double>::operator new(e_quads_size);
      d2q(L2FESpace, integ_rule, d_e_data, d_e_quads_data);
      // double h_e_quads_data[e_quads_size];mfem::kernels::kmemcpy::rDtoH(h_e_quads_data, d_e_quads_data, e_quads_size*sizeof(double));
      //for(int k=0;k<e_quads_size;k+=1) dbg("d_e_quads_data[%d]=%f",k,h_e_quads_data[k]); assert(false);

      // Refresh Geom J, invJ & detJ *******************************************
      const qGeometry *geom = qGeometry::Get(H1FESpace,integ_rule);

      // Integration Points Weights (tensor) ***********************************
      const qDofQuadMaps* maps = qDofQuadMaps::Get(H1FESpace,integ_rule);
      /*{
         const size_t quadWeights1D_size = nqp;
         double h_maps_quadWeights[quadWeights1D_size];
         mfem::kernels::kmemcpy::rDtoH(h_maps_quadWeights, maps->quadWeights.GetData(), quadWeights1D_size*sizeof(double));
         for(int k=0;k<quadWeights1D_size;k+=1) dbg("h_maps_quadWeights[%d]=%f",k,h_maps_quadWeights[k]); assert(false);
         }*/
      
      // Velocity **************************************************************     
      ParGridFunction velocity;
      velocity.MakeRef(&H1FESpace, *sptr, H1FESpace.GetVSize());
      reorderByVDim(H1FESpace, velocity.Size(), velocity.GetData());
      const size_t v_local_size = dims * numDofs * elements;
      mfem::Array<double> local_velocity(v_local_size);
      const Table& e2dTable = H1FESpace.GetElementToDofTable();
      const int* elementMap = e2dTable.GetJ();
      for (int e = 0; e < elements; ++e) {
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
      /*{
         double h_v_data[v_local_size];
         mfem::kernels::kmemcpy::rDtoH(h_v_data, d_v_data, v_local_size*sizeof(double));
         for(int k=0;k<v_local_size;k+=1) dbg("h_v_data[%d]=%f",k,h_v_data[k]); assert(false);
         }*/

      reorderByNodes(H1FESpace, velocity.Size(), velocity.GetData());
      const size_t grad_v_size = dim * dim * nqp * nzones;
      double *d_grad_v_data = (double*) mfem::kernels::kmalloc<double>::operator new(grad_v_size);
      const qDofQuadMaps *simplex_maps = qDofQuadMaps::GetSimplexMaps(fe,integ_rule);
      qGradVector(H1FESpace,
                  integ_rule,
                  simplex_maps,
                  v_local_size,
                  d_v_data,
                  d_grad_v_data);
      /*{
         double h_grad_v_data[grad_v_size];
         mfem::kernels::kmemcpy::rDtoH(h_grad_v_data, d_grad_v_data, grad_v_size*sizeof(double));
         for(int k=0;k<grad_v_size;k+=1) dbg("h_grad_v_data[%d]=%f",k,h_grad_v_data[k]); assert(false);
         }*/

      // ***********************************************************************      
      const double h1order = (double) H1FESpace.GetOrder(0);
      const double infinity = std::numeric_limits<double>::infinity();
      dbg("infinity=%f",infinity);
      
      const size_t rho0DetJ0w_sz = nzones * nqp;
      double *d_rho0DetJ0w =
         (double*)mfem::kernels::kmalloc<double>::operator new(rho0DetJ0w_sz);
      mfem::kernels::kmemcpy::rHtoD(d_rho0DetJ0w,
                                    quad_data.rho0DetJ0w.GetData(),
                                    rho0DetJ0w_sz*sizeof(double));
      /*{
         double h_rho0DetJ0w[rho0DetJ0w_sz];
         mfem::kernels::kmemcpy::rDtoH(h_rho0DetJ0w, d_rho0DetJ0w, rho0DetJ0w_sz*sizeof(double));
         for(int k=0;k<rho0DetJ0w_sz;k+=1) dbg("h_rho0DetJ0w[%d]=%f",k,h_rho0DetJ0w[k]); assert(false);
         }*/

      const size_t Jac0inv_sz = dim * dim * nzones * nqp;
      double *d_Jac0inv =
         (double*)mfem::kernels::kmalloc<double>::operator new(Jac0inv_sz);
      mfem::kernels::kmemcpy::rHtoD(d_Jac0inv,
                                    quad_data.Jac0inv.Data(),
                                    Jac0inv_sz*sizeof(double));
      //for(int k=0;k<Jac0inv_sz;k+=1) dbg("d_Jac0inv[%d]=%f",k,quad_data.Jac0inv.Data()[k]); assert(false);
      
      double *d_dt_est = (double*)mfem::kernels::kmalloc<double>::operator new(1);
      mfem::kernels::kmemcpy::rHtoD(d_dt_est, &quad_data.dt_est, sizeof(double));

      const size_t stressJinvT_sz = nzones * nqp * dim * dim;
      double *d_stressJinvT =
         (double*)mfem::kernels::kmalloc<double>::operator new(stressJinvT_sz);
      
      qkernel<2> __config(nzones) (nzones,
                                   nqp,
                                   nqp1D,
                                   gamma,
                                   use_viscosity,
                                   quad_data.h0,
                                   h1order,
                                   cfl,
                                   infinity,
                                    
                                   maps->quadWeights,                                   
                                   geom->J,
                                   d_rho0DetJ0w,
                                   d_e_quads_data,
                                   d_grad_v_data,
                                   d_Jac0inv,
                                   d_dt_est,
                                   d_stressJinvT);
      
      mfem::kernels::kmemcpy::rDtoH(quad_data.stressJinvT.Data(),
                                    d_stressJinvT, stressJinvT_sz*sizeof(double));
      
      mfem::kernels::kmemcpy::rDtoH(&quad_data.dt_est, d_dt_est, sizeof(double));
      
      dbg("dt_est=%.21e",quad_data.dt_est);
      //assert(false);
      quad_data_is_current = true;
      timer.sw_qdata.Stop();
      timer.quad_tstep += nzones;
   }

} // namespace hydrodynamics

} // namespace mfem
