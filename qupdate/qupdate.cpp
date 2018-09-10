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
   //__attribute__((unused))
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
   }

   // **************************************************************************
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
   }

   // **************************************************************************
   static void qGradVector2D( const int NUM_DOFS,
                              const int NUM_QUAD,
                              const int numElements,
                              const double* __restrict dofToQuadD,
                              const double* __restrict in,
                              double* __restrict out){
      push();
      for(int e=0; e<numElements; e+=1){
         dbg("elem #%d",e);
         double s_in[2 * NUM_DOFS];
         for (int q = 0; q < NUM_QUAD; ++q) {
            dbg("\tq #%d",q);
            for (int d = q; d < NUM_DOFS; d+=NUM_QUAD) {
               dbg("\t\td=%d",d);
               const int x0 = ijN(0,d,2);
               const int x1 = ijkNM(0,d,e,2,NUM_DOFS);
               const int y0 = ijN(1,d,2);
               const int y1 = ijkNM(1,d,e,2,NUM_DOFS);
               const double x = in[ijkNM(0,d,e,2,NUM_DOFS)];
               const double y = in[ijkNM(1,d,e,2,NUM_DOFS)];
               dbg("\t\t%d <= %d: %f",x0,x1,x);
               dbg("\t\t%d <= %d: %f",y0,y1,y);
               s_in[ijN(0,d,2)] = in[ijkNM(0,d,e,2,NUM_DOFS)];
               s_in[ijN(1,d,2)] = in[ijkNM(1,d,e,2,NUM_DOFS)];
            }
         }
         dbg("eof share, returning to elem #%d",e);
         for (int q = 0; q < NUM_QUAD; ++q) {
            dbg("\tq #%d",q);
            double J11 = 0.0; double J12 = 0.0;
            double J21 = 0.0; double J22 = 0.0;
            for (int d = 0; d < NUM_DOFS; ++d) {
               dbg("\t\td=%d",d);
               const double wx = dofToQuadD[ijkNM(0,q,d,2,NUM_QUAD)];
               const double wy = dofToQuadD[ijkNM(1,q,d,2,NUM_QUAD)];
               const double x = s_in[ijN(0,d,2)];
               const double y = s_in[ijN(1,d,2)];
               dbg("\t\twx=%f, wy=%f",wx,wy);
               dbg("\t\t x=%f,  y=%f", x, y);
               J11 += (wx * x); J12 += (wx * y);
               J21 += (wy * x); J22 += (wy * y);
            }
            out[ijklNM(0,0,q,e,2,NUM_QUAD)] = J11;
            out[ijklNM(1,0,q,e,2,NUM_QUAD)] = J12;
            out[ijklNM(0,1,q,e,2,NUM_QUAD)] = J21;
            out[ijklNM(1,1,q,e,2,NUM_QUAD)] = J22;
         }
      }
      pop();
   }

   // **************************************************************************
   static void reorderByVDim(const FiniteElementSpace& fes,
                             const size_t size,
                             double *data){
      push();
      const int vdim = fes.GetVDim();
      const int ndofs = fes.GetNDofs();
      dbg("size=%d",size);    
      dbg("vdim=%d ndofs=%d",vdim, ndofs);
      double *temp = new double[size];
      for (int k=0; k<size; k++) temp[k]=0.0;
      int k=0;
      for (int d = 0; d < ndofs; d++)
         for (int v = 0; v < vdim; v++)      
            temp[k++] = data[d+v*ndofs];
      for (int i=0; i<size; i++){
         data[i] = temp[i];
         dbg("data[%d]=%f",i,data[i]);
      }
      delete [] temp;
      pop();
   }

   // ***************************************************************************
   static void reorderByNodes(const FiniteElementSpace& fes,
                              const size_t size,
                              double *data){
      push();
      const int vdim = fes.GetVDim();
      const int ndofs = fes.GetNDofs();
      dbg("size=%d",size);      
      dbg("vdim=%d ndofs=%d",vdim, ndofs);      
      double *temp = new double[size];
      for (int k=0; k<size; k++) temp[k]=0.0;
      int k=0;
      for (int j=0; j < ndofs; j++)
         for (int i=0; i < vdim; i++)
            temp[j+i*ndofs] = data[k++];
      for (int i = 0; i < size; i++){
         data[i] = temp[i];
         dbg("data[%d]=%f",i,data[i]);
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
      //const bool orderedByNODES = (fes.GetOrdering() == Ordering::byNODES);
      
      /*if (orderedByNODES) {
         dbg("\033[7morderedByNODES, ReorderByVDim");
         reorderByVDim(fes, size, in);
         }*/
      
      qGradVector2D(ndf, nqp, nzones,
                    maps->dofToQuadD,
                    in, out);
      
      /*if (orderedByNODES) {
         dbg("Reorder the original gf back");
         reorderByNodes(fes, size, in);
         }*/
      pop();
   }

   
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
      assert(p_assembly);
      assert(material_pcf);
      
      ElementTransformation *T = H1FESpace.GetElementTransformation(0);
      const IntegrationPoint &ip = integ_rule.IntPoint(0);
      const double gamma = material_pcf->Eval(*T,ip);

      // ***********************************************************************
      if (quad_data_is_current) return;
      timer.sw_qdata.Start();

      const int nqp = integ_rule.GetNPoints();
      dbg("nqp=%d, nzones=%d",nqp,nzones);

      ParGridFunction x, velocity, energy;
      Vector* sptr = (Vector*) &S;
      
      x.MakeRef(&H1FESpace, *sptr, 0);
//#warning x for(int i=0;i<x.Size();i+=1) x[i] = 1.123456789*drand48();
         //dbg("x (size=%d)",x.Size());//x.Print();
      
      velocity.MakeRef(&H1FESpace, *sptr, H1FESpace.GetVSize());
//#warning v
//      srand48(0xDEADBEEFul);
//      for(int i=0;i<velocity.Size();i+=1) velocity[i] = 0.123456789*drand48();
//      dbg("velocity (size=%d)",velocity.Size());velocity.Print();
      
      energy.MakeRef(&L2FESpace, *sptr, 2*H1FESpace.GetVSize());
      //dbg("energy (size=%d)",energy.Size());//energy.Print();
      
      Vector e_loc(l2dofs_cnt), vector_loc(h1dofs_cnt * dim);
      DenseMatrix Jpi(dim), sgrad_v(dim), Jinv(dim), stress(dim), stressJiT(dim);
      DenseMatrix vector_loc_mtx(vector_loc.GetData(), h1dofs_cnt, dim);
      DenseTensor grad_v_ref(dim, dim, nqp);
      Array<int> L2dofs, H1dofs;

      const H1_QuadrilateralElement *fe =
         dynamic_cast<const H1_QuadrilateralElement *>(H1FESpace.GetFE(0));
      const Array<int> &h1_dof_map = fe->GetDofMap();
      
      const int nqp1D    = tensors1D->LQshape1D.Width();
      const int nL2dof1D = tensors1D->LQshape1D.Height();
      const int nH1dof1D = tensors1D->HQshape1D.Height();
      
      // Energy values at quadrature point *************************************
      const bool use_external_e = false;
      Vector e_vals(nqp);
      Vector e_quads(nzones * nqp);
      if (use_external_e)
         d2q(L2FESpace, integ_rule, energy.GetData(), e_quads.GetData());

      // Jacobian **************************************************************
      DenseTensor Jpr(dim, dim, nqp);

      // ***********************************************************************
      const bool use_external_J = true;
      qGeometry *geom;
      if (use_external_J)
         geom = qGeometry::Get(H1FESpace,integ_rule);

      // IP Weights ************************************************************
      Vector weights;
      const bool use_external_w = true;
      const qDofQuadMaps* maps;
      if (use_external_w){
         maps = qDofQuadMaps::Get(H1FESpace,integ_rule);
         weights.SetDataAndSize((double*)maps->quadWeights.ptr(),nqp);
      }

      // Velocity **************************************************************
      const bool use_external_grad_v = true;
      Vector grad_v_ext;
      if (use_external_grad_v){
         
         const mfem::FiniteElement& fe = *H1FESpace.GetFE(0);
         const int dims     = H1FESpace.GetVDim();
         const int elements = H1FESpace.GetNE();
         const int numDofs  = fe.GetDof();
         
         //dbg("velocity:\n");velocity.Print();
         reorderByVDim(H1FESpace, velocity.Size(), velocity.GetData());
         //dbg("reorderByVDim:");velocity.Print();
        
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
         //dbg("local_velocity:");local_velocity.Print();

         //dbg("dims=%d, elements=%d, numDofs=%d, v_local_size=%d",dims, elements, numDofs, v_local_size);
         Vector v_local(v_local_size);
         v_local = local_velocity;
         //dbg("v_local:");v_local.Print();
        
         reorderByNodes(H1FESpace, velocity.Size(), velocity.GetData());
         
         //globalToLocal(H1FESpace,velocity.GetData(),v_local.GetData(),false);
         //dbg("v_local:");v_local.Print();

         const size_t grad_v_size = dim * dim * nqp * nzones;
         grad_v_ext.SetSize(grad_v_size);
         //dbg("grad_v_size=%d",grad_v_size);
         
         maps = qDofQuadMaps::GetSimplexMaps(fe,integ_rule);
         
         qGradVector(H1FESpace,
                     integ_rule,
                     maps,
                     v_local_size,
                     v_local.GetData(),
                     grad_v_ext.GetData());
         
         //dbg("grad_v_ext:\n");grad_v_ext.Print();
                  
         //assert(false);
/*
         for (int z = 0; z < nzones; z++) {
            H1FESpace.GetElementVDofs(z, H1dofs);
            velocity.GetSubVector(H1dofs, vector_loc);
            //dbg("(z=%d) vector_loc:",z);vector_loc.Print();
            getVectorGrad(dim, nH1dof1D, nqp1D, h1_dof_map, vector_loc_mtx, grad_v_ref);
            for (int q = 0; q < nqp; q++) {
               DenseMatrix qgrad_v_ext(&grad_v_ext.GetData()[(z*nqp+q)*nzones],dim,dim);
               dbg("(%d,%d) grad_v_ref:",z,q);
               grad_v_ref(q).Print();
               dbg("\033[7mVERSUS:");               
               qgrad_v_ext.Print();
            }
            }
*/
         
         //assert(false);
      }
   
      // ***********************************************************************
      const double h1order = (double) H1FESpace.GetOrder(0);
      const double infinity = std::numeric_limits<double>::infinity();
      double min_detJ = infinity;
      
      for (int z = 0; z < nzones; z++) {
         ElementTransformation *T = H1FESpace.GetElementTransformation(z);
         
         // Energy values at quadrature point **********************************
         if (!use_external_e){
            L2FESpace.GetElementDofs(z, L2dofs);
            energy.GetSubVector(L2dofs, e_loc);
            getL2Values(dim, nL2dof1D, nqp1D, e_loc.GetData(), e_vals.GetData());
         }
 
         // Jacobians at quadrature points *************************************
         if (!use_external_J)
         {
            H1FESpace.GetElementVDofs(z, H1dofs);
            x.GetSubVector(H1dofs, vector_loc);
            getVectorGrad(dim, nH1dof1D, nqp1D, h1_dof_map, vector_loc_mtx, Jpr);
         }

         // Velocity gradient at quadrature points *****************************
         if (use_viscosity) {
            if (!use_external_grad_v){
               H1FESpace.GetElementVDofs(z, H1dofs);
               velocity.GetSubVector(H1dofs, vector_loc);
               getVectorGrad(dim, nH1dof1D, nqp1D, h1_dof_map, vector_loc_mtx, grad_v_ref);
            }
         }
         
         // ********************************************************************
         for (int q = 0; q < nqp; q++) {
            const int idx = z * nqp + q;
            double ip_w = 0.0;
            if (!use_external_w){
               const IntegrationPoint &ip = integ_rule.IntPoint(q);
               T->SetIntPoint(&ip);
               ip_w = ip.weight;
            }
            const double weight = (!use_external_w) ? ip_w : weights[q];
            const double inv_weight = 1. / weight;

            const DenseMatrix &J = !use_external_J? Jpr(q):
               DenseMatrix(&geom->J[(z*nqp+q)*nzones],dim,dim);

            const double detJ = J.Det();
            min_detJ = fmin(min_detJ, detJ);   
            calcInverse2D(J.Height(), J.Data(), Jinv.Data());        
            
            // *****************************************************************
            const double rho = inv_weight * quad_data.rho0DetJ0w(idx) / detJ;
            const double e   = !use_external_e?
               fmax(0.0, e_vals(q)):
               fmax(0.0, e_quads.GetData()[z*nqp1D*nqp1D+q]);
            const double p  = (gamma - 1.0) * rho * e;
            const double sound_speed = sqrt(gamma * (gamma-1.0) * e);
            // *****************************************************************
            stress = 0.0;
            for (int d = 0; d < dim; d++)  stress(d,d) = -p;
            // *****************************************************************
            double visc_coeff = 0.0;
            if (use_viscosity) {
               // Compression-based length scale at the point. The first
               // eigenvector of the symmetric velocity gradient gives the
               // direction of maximal compression. This is used to define the
               // relative change of the initial length scale.               
               const DenseMatrix &dV = !use_external_grad_v? grad_v_ref(q):
                  DenseMatrix(&grad_v_ext.GetData()[(z*nqp+q)*nzones],dim,dim);                  
               mult(sgrad_v.Height(),sgrad_v.Width(),grad_v_ref(q).Width(),
                    dV.Data(), Jinv.Data(), sgrad_v.Data());
               symmetrize(sgrad_v.Height(),sgrad_v.Data());
               double eig_val_data[3], eig_vec_data[9];
               if (dim==1) {
                  eig_val_data[0] = sgrad_v(0, 0);
                  eig_vec_data[0] = 1.;
               }
               else {
                  calcEigenvalues(sgrad_v.Height(),sgrad_v.Data(),
                                  eig_val_data, eig_vec_data);
               }
               Vector compr_dir(eig_vec_data, dim);
               // Computes the initial->physical transformation Jacobian.
               mult(Jpi.Height(),Jpi.Width(),J.Width(),
                 J.Data(), quad_data.Jac0inv(idx).Data(), Jpi.Data());
               Vector ph_dir(dim);
               //Jpi.Mult(compr_dir, ph_dir);
               multV(Jpi.Height(), Jpi.Width(), Jpi.Data(),
                     compr_dir.GetData(), ph_dir.GetData());
               // Change of the initial mesh size in the compression direction.
               const double h = quad_data.h0 * ph_dir.Norml2() / compr_dir.Norml2();
               // Measure of maximal compression.
               const double mu = eig_val_data[0];
               visc_coeff = 2.0 * rho * h * h * fabs(mu);
               if (mu < 0.0) { visc_coeff += 0.5 * rho * h * sound_speed; }
               add(stress.Height(), stress.Width(),visc_coeff, sgrad_v.Data(), stress.Data());
            }
            // Time step estimate at the point. Here the more relevant length
            // scale is related to the actual mesh deformation; we use the min
            // singular value of the ref->physical Jacobian. In addition, the
            // time step estimate should be aware of the presence of shocks.
            const double h_min = calcSingularvalue(J.Height(), dim-1, J.Data()) / h1order;
            const double inv_h_min = 1. / h_min;
            const double inv_rho_inv_h_min_sq = inv_h_min * inv_h_min / rho ;
            const double inv_dt = sound_speed * inv_h_min + 2.5 * visc_coeff * inv_rho_inv_h_min_sq;
            if (min_detJ < 0.0) {
               // This will force repetition of the step with smaller dt.
               quad_data.dt_est = 0.0;
            } else {
               quad_data.dt_est = fmin(quad_data.dt_est, cfl * (1.0 / inv_dt) );
            }
            // Quadrature data for partial assembly of the force operator.
            multABt(stress.Height(), stress.Width(), Jinv.Height(),
                    stress.Data(), Jinv.Data(), stressJiT.Data());
            stressJiT *= weight * detJ;
            for (int vd = 0 ; vd < dim; vd++) {
               for (int gd = 0; gd < dim; gd++) {
                  quad_data.stressJinvT(vd)(z*nqp + q, gd) =
                     stressJiT(vd, gd);
               }
            }
         }
      }
      //assert(false);
      quad_data_is_current = true;
      timer.sw_qdata.Stop();
      timer.quad_tstep += nzones;
   }

} // namespace hydrodynamics

} // namespace mfem
