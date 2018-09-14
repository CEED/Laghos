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
      const int z = blockDim.x * blockIdx.x + threadIdx.x;
      if (z < nzones)
#else
      for (int z = 0; z < nzones; z++)
#endif
      {
         // ********************************************************************
         for (int q = 0; q < nqp; q++) {
            const int idx = z * nqp + q;
            const double weight =  weights[q];
            const double inv_weight = 1. / weight;
            const double *J = &_J[(z*nqp+q)*nzones];
            const double detJ = Det(dim,J);
            min_detJ = fmin(min_detJ, detJ);
            double Jinv[dim*dim];
            calcInverse2D(dim, J, Jinv);
            // *****************************************************************
            const double rho = inv_weight * rho0DetJ0w[idx] / detJ;
            const double e   = fmax(0.0, e_quads[z*nqp1D*nqp1D+q]);
            const double p  = (gamma - 1.0) * rho * e;
            const double sound_speed = sqrt(gamma * (gamma-1.0) * e);
            // *****************************************************************
            double stress[dim*dim];
            for (int k=0;k<dim*dim;k+=1) stress[k] = 0.0;
            for (int d = 0; d < dim; d++)  stress[d*dim+d] = -p;
            // *****************************************************************
            double visc_coeff = 0.0;
            if (use_viscosity) {
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
                  calcEigenvalues(dim, sgrad_v, eig_val_data, eig_vec_data);
               }
               double compr_dir[dim];
               for(int k=0;k<dim;k+=1) compr_dir[k]=eig_vec_data[k];
               // Computes the initial->physical transformation Jacobian.
               double Jpi[dim*dim];
               mult(dim,dim,dim, J, Jac0inv+idx*dim*dim, Jpi);
               double ph_dir[dim];
               multV(dim, dim, Jpi, compr_dir, ph_dir);
               // Change of the initial mesh size in the compression direction.
               const double h = h0 * norml2(dim,ph_dir) / norml2(dim,compr_dir);
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
            const double sv = calcSingularvalue(dim, dim-1, J);
            const double h_min = sv / h1order;
            const double inv_h_min = 1. / h_min;
            const double inv_rho_inv_h_min_sq = inv_h_min * inv_h_min / rho ;
            const double inv_dt = sound_speed * inv_h_min + 2.5 * visc_coeff * inv_rho_inv_h_min_sq;
            if (min_detJ < 0.0) {
               // This will force repetition of the step with smaller dt.
               *dt_est = 0.0;
            } else {
               *dt_est = fmin(*dt_est, cfl * (1.0 / inv_dt) );
            }
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
                  stressJinvT[q + z*nqp + nqp*nzones*(gd+vd*dim)] =
                     stressJiT[vd+gd*dim];
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
      mfem::kernels::kmemcpy::rHtoD(d_e_data, e_data, L2_size*sizeof(double));
      double *d_e_quads_data = (double*)mfem::kernels::kmalloc<double>::operator new(e_quads_size);
      Dof2Quad(L2FESpace, integ_rule, d_e_data, d_e_quads_data);

      // Refresh Geom J, invJ & detJ *******************************************
      const qGeometry *geom = qGeometry::Get(H1FESpace,integ_rule);

      // Integration Points Weights (tensor) ***********************************
      const qDofQuadMaps* maps = qDofQuadMaps::Get(H1FESpace,integ_rule);
      
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

      const size_t Jac0inv_sz = dim * dim * nzones * nqp;
      double *d_Jac0inv =
         (double*)mfem::kernels::kmalloc<double>::operator new(Jac0inv_sz);
      mfem::kernels::kmemcpy::rHtoD(d_Jac0inv,
                                    quad_data.Jac0inv.Data(),
                                    Jac0inv_sz*sizeof(double));
      
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
