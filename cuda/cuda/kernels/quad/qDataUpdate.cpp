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
#include "../cuda.hpp"
#include <limits>

using namespace std;

// *****************************************************************************
__device__ inline double getScalingFactor(const double d_max)
{
   int d_exp;
   if (d_max > 0.0)
   {
      double mult = frexp(d_max, &d_exp);
      if (d_exp == numeric_limits<double>::max_exponent)
      {
         mult *= numeric_limits<double>::radix;
      }
      return d_max/mult;
   }
   return 1.0;
   // mult = 2^d_exp is such that d_max/mult is in [0.5,1)
   // or in other words d_max is in the interval [0.5,1)*mult
}

// *****************************************************************************
__device__ double calcSingularvalue(const int n, const int i, const double *d)
{
   const double J0 = d[0];
   const double J1 = d[1];
   const double J2 = d[2];
   const double J3 = d[3];
   double J_max = fabs(J0);
   if (J_max < fabs(J1)) { J_max = fabs(J1); }
   if (J_max < fabs(J2)) { J_max = fabs(J2); }
   if (J_max < fabs(J3)) { J_max = fabs(J3); }
   const double mult = getScalingFactor(J_max);
   const double d0 = J0/mult;
   const double d1 = J1/mult;
   const double d2 = J2/mult;
   const double d3 = J3/ mult;
   const double t = 0.5*((d0+d2)*(d0-d2)+(d1-d3)*(d1+d3));
   const double s = d0*d2 + d1*d3;
   const double s1 = sqrt(0.5*(d0*d0 + d1*d1 + d2*d2 + d3*d3) + sqrt(t*t + s*s));
   if (s1 == 0.0) { return 0.0; }
   const double t1 = fabs(d0*d3 - d1*d2) / s1;
   if (t1 > s1) { return s1*mult; }
   return t1*mult;
}

// *****************************************************************************
__device__ inline double cpysign(const double x, const double y)
{
   if ((x < 0 && y > 0) || (x > 0 && y < 0))
   {
      return -x;
   }
   return x;
}

// *****************************************************************************
__device__ inline void eigensystem2S(const double &d12, double &d1, double &d2,
                                     double &c, double &s)
{
   const double epsilon = 1.e-16;
   const double sqrt_1_eps = sqrt(1./epsilon);
   if (d12 == 0.)
   {
      c = 1.;
      s = 0.;
   }
   else
   {
      // "The Symmetric Eigenvalue Problem", B. N. Parlett, pp.189-190
      double t, zeta = (d2 - d1)/(2*d12);
      if (fabs(zeta) < sqrt_1_eps)
      {
         t = cpysign(1./(fabs(zeta) + sqrt(1. + zeta*zeta)), zeta);
      }
      else
      {
         t = cpysign(0.5/fabs(zeta), zeta);
      }
      c = sqrt(1./(1. + t*t));
      s = c*t;
      t *= d12;
      d1 -= t;
      d2 += t;
   }
}

// *****************************************************************************
__device__ inline void calcEigenvalues2D(const double *d,
                                         double *val, double *vec)
{
   double d0 = d[0];
   double d2 = d[2]; // use the upper triangular entry
   double d3 = d[3];
   double c, s;
   eigensystem2S(d2, d0, d3, c, s);
   if (d0 <= d3)
   {
      val[0] = d0; val[1] = d3;
      vec[0] =  c;
      vec[1] = -s;
      vec[2] =  s;
      vec[3] =  c;
   }
   else
   {
      val[0] = d3; val[1] = d0;
      vec[0] =  s;
      vec[1] =  c;
      vec[2] =  c;
      vec[3] = -s;
   }
}

// *****************************************************************************
__device__ inline void mult(const int ah,
                            const int aw,
                            const int bw,
                            const double* __restrict__ B,
                            const double* __restrict__ C,
                            double* __restrict__ A)
{
   const int ah_x_aw = ah*aw;
   for (int i = 0; i < ah_x_aw; i++) { A[i] = 0.0; }
   for (int j = 0; j < aw; j++)
   {
      for (int k = 0; k < bw; k++)
      {
         for (int i = 0; i < ah; i++)
         {
            A[i+j*ah] += B[i+k*ah] * C[k+j*bw];
         }
      }
   }
}

// *****************************************************************************
__device__ inline void multV(const int height,
                             const int width,
                             double *data,
                             const double* __restrict__ x,
                             double* __restrict__ y)
{
   if (width == 0)
   {
      for (int row = 0; row < height; row++)
      {
         y[row] = 0.0;
      }
      return;
   }
   double *d_col = data;
   double x_col = x[0];
   for (int row = 0; row < height; row++)
   {
      y[row] = x_col*d_col[row];
   }
   d_col += height;
   for (int col = 1; col < width; col++)
   {
      x_col = x[col];
      for (int row = 0; row < height; row++)
      {
         y[row] += x_col*d_col[row];
      }
      d_col += height;
   }
}

// *****************************************************************************
__device__  static
double norml2(const int size, const double *data)
{
   if (0 == size) { return 0.0; }
   if (1 == size) { return std::abs(data[0]); }
   double scale = 0.0;
   double sum = 0.0;
   for (int i = 0; i < size; i++)
   {
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

// *****************************************************************************
__device__ inline double smooth_step_01(const double x, const double eps)
{
   const double y = (x + eps) / (2.0 * eps);
   if (y < 0.0) { return 0.0; }
   if (y > 1.0) { return 1.0; }
   return (3.0 - 2.0 * y) * y * y;
}

// *****************************************************************************
__device__ inline void symmetrize(const int n, double* __restrict__ d)
{
   for (int i = 0; i<n; i++)
   {
      for (int j = 0; j<i; j++)
      {
         const double a = 0.5 * (d[i*n+j] + d[j*n+i]);
         d[j*n+i] = d[i*n+j] = a;
      }
   }
}

// *****************************************************************************
template<const int NUM_DOFS_1D,
         const int NUM_QUAD_1D,
         const int BZ,
         const int NBLOCK>
__launch_bounds__(NUM_QUAD_1D*NUM_QUAD_1D*BZ,NBLOCK)
kernel
void rUpdateQuadratureData2D_v2(const double GAMMA,
                                const double H0,
                                const int h1order,
                                const double CFL,
                                const double infinity,
                                const bool USE_VISCOSITY,
                                const int numElements,
                                const double* restrict dofToQuad,
                                const double* restrict dofToQuadD,
                                const double* restrict quadWeights,
                                const double* restrict v,
                                const double* restrict e,
                                const double* restrict rho0DetJ0w,
                                const double* restrict invJ0,
                                const double* restrict J,
                                const double* restrict invJ,
                                const double* restrict detJ,
                                double* restrict stressJinvT,
                                double* restrict dtEst)
{
   constexpr int DIM = 2;
   const int NUM_QUAD = NUM_QUAD_1D*NUM_QUAD_1D;
   const int VDIMQ = DIM*DIM*NUM_QUAD;
   double min_detJ = infinity;

   const int el = blockIdx.x*BZ+threadIdx.z;
   if (el >= numElements) { return; }
   int tid = threadIdx.x + threadIdx.y*blockDim.x;
   __shared__ double buf1[BZ][VDIMQ];
   __shared__ double buf2[BZ][NUM_DOFS_1D][DIM*NUM_QUAD_1D];
   // __shared__ double buf3[BZ][NUM_DOFS_1D][DIM*NUM_QUAD_1D];

   double *s_gradv = (double*)(buf1 + threadIdx.z);
   double (*vDx)[DIM*NUM_QUAD_1D] =
      (double (*)[DIM*NUM_QUAD_1D])(buf2 + threadIdx.z);
   // double (*vx)[DIM*NUM_QUAD_1D] =
   //    (double (*)[DIM*NUM_QUAD_1D])(buf3 + threadIdx.z);
   double (*vx)[DIM*NUM_QUAD_1D] = vDx;

   for (int dy = threadIdx.y; dy < NUM_DOFS_1D; dy += blockDim.y)
   {
      for (int qx = threadIdx.x; qx < NUM_QUAD_1D; qx += blockDim.x)
      {
         for (int c = 0; c < DIM; ++c)
         {
            double t = 0;
            for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
            {
               t += dofToQuadD[ijN(qx,dx,NUM_QUAD_1D)]*v[_ijklNM(c,dx,dy,el,NUM_DOFS_1D,
                                                                 numElements)];
            }
            vDx[dy][ijN(c,qx,DIM)] = t;
         }
      }
   }
   __syncthreads();

   for (int qy = threadIdx.y; qy < NUM_QUAD_1D; qy += blockDim.y)
   {
      for (int qx = threadIdx.x; qx < NUM_QUAD_1D; qx += blockDim.x)
      {
         for (int c = 0; c < DIM; ++c)
         {
            double t = 0;
            for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
            {
               t += dofToQuad[ijN(qy,dy,NUM_QUAD_1D)]*vDx[dy][ijN(c,qx,DIM)];
            }
            s_gradv[ijkN(c,0,qx+qy*NUM_QUAD_1D,DIM)] = t;
         }
      }
   }
   __syncthreads();
   for (int dy = threadIdx.y; dy < NUM_DOFS_1D; dy += blockDim.y)
   {
      for (int qx = threadIdx.x; qx < NUM_QUAD_1D; qx += blockDim.x)
      {
         for (int c = 0; c < DIM; ++c)
         {
            double t = 0;
            for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
            {
               t += dofToQuad[ijN(qx,dx,NUM_QUAD_1D)]*v[_ijklNM(c,dx,dy,el,NUM_DOFS_1D,
                                                                numElements)];
            }
            vx[dy][ijN(c,qx,DIM)] = t;
         }
      }
   }
   __syncthreads();

   for (int qy = threadIdx.y; qy < NUM_QUAD_1D; qy += blockDim.y)
   {
      for (int qx = threadIdx.x; qx < NUM_QUAD_1D; qx += blockDim.x)
      {
         for (int c = 0; c < DIM; ++c)
         {
            double t = 0;
            for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
            {
               t += dofToQuadD[ijN(qy,dy,NUM_QUAD_1D)]*vx[dy][ijN(c,qx,DIM)];
            }
            s_gradv[ijkN(c,1,qx+qy*NUM_QUAD_1D,DIM)] = t;
         }
      }
   }
   __syncthreads();

   for (int q = tid; q < NUM_QUAD; q += blockDim.x*blockDim.y)
   {
      const double det = detJ[ijN(q,el,NUM_QUAD)];
      min_detJ = fmin(min_detJ, det);
      const double q_Jw = det * quadWeights[q];
      const double q_rho = rho0DetJ0w[ijN(q,el,NUM_QUAD)] / q_Jw;
      const double q_e   = fmax(0.0,e[ijN(q,el,NUM_QUAD)]);
      const double p = -(GAMMA-1.0)*q_rho*q_e;
      const double soundSpeed = sqrt(GAMMA*(GAMMA-1.0)*q_e);
      // *****************************************************************
      double q_stress[DIM*DIM];
      q_stress[ijN(0,0,2)] = p; q_stress[ijN(1,0,2)] = 0.0;
      q_stress[ijN(0,1,2)] = 0.0; q_stress[ijN(1,1,2)] = p;
      // *****************************************************************
      const double J_00 = J[ijklNM(0,0,q,el,DIM,NUM_QUAD)];
      const double J_10 = J[ijklNM(1,0,q,el,DIM,NUM_QUAD)];
      const double J_01 = J[ijklNM(0,1,q,el,DIM,NUM_QUAD)];
      const double J_11 = J[ijklNM(1,1,q,el,DIM,NUM_QUAD)];
      const double xJ[4] = {J_00, J_10, J_01, J_11};
      // *****************************************************************
      const double invJ_00 = invJ[ijklNM(0,0,q,el,DIM,NUM_QUAD)];
      const double invJ_10 = invJ[ijklNM(1,0,q,el,DIM,NUM_QUAD)];
      const double invJ_01 = invJ[ijklNM(0,1,q,el,DIM,NUM_QUAD)];
      const double invJ_11 = invJ[ijklNM(1,1,q,el,DIM,NUM_QUAD)];
      const double xiJ[4] = {invJ_00, invJ_10, invJ_01, invJ_11};
      // *****************************************************************
      const double invJ0_00 = invJ0[ijklNM(0,0,q,el,DIM,NUM_QUAD)];
      const double invJ0_10 = invJ0[ijklNM(1,0,q,el,DIM,NUM_QUAD)];
      const double invJ0_01 = invJ0[ijklNM(0,1,q,el,DIM,NUM_QUAD)];
      const double invJ0_11 = invJ0[ijklNM(1,1,q,el,DIM,NUM_QUAD)];
      const double xiJ0[4] = {invJ0_00, invJ0_10, invJ0_01, invJ0_11};
      // *****************************************************************
      double visc_coeff = 0.0;
      if (USE_VISCOSITY)
      {
         // Compression-based length scale at the point. The first
         // eigenvector of the symmetric velocity gradient gives the
         // direction of maximal compression. This is used to define the
         // relative change of the initial length scale.const double *_dV = d_grad_v_ext + (q+nqp*dim*dim*z);
         double q_gradv[DIM*DIM];
         const double dV[4] = {s_gradv[ijkN(0,0,q,2)],
                               s_gradv[ijkN(1,0,q,2)],
                               s_gradv[ijkN(0,1,q,2)],
                               s_gradv[ijkN(1,1,q,2)]
                              };
         mult(DIM,DIM,DIM, dV, xiJ, q_gradv);
         symmetrize(DIM, q_gradv);
         double eig_val_data[3];
         double eig_vec_data[9];
         double compr_dir[DIM];
         double ph_dir[DIM];
         double Jpi[DIM*DIM];
         calcEigenvalues2D(q_gradv, eig_val_data, eig_vec_data);
         for (int k=0; k<DIM; k+=1) { compr_dir[k] = eig_vec_data[k]; }
         // Computes the initial->physical transformation Jacobian.
         mult(DIM,DIM,DIM, xJ, xiJ0, Jpi);
         multV(DIM, DIM, Jpi, compr_dir, ph_dir);
         // Change of the initial mesh size in the compression direction.
         const double h = H0 * norml2(DIM, ph_dir) / norml2(DIM, compr_dir);
         // Measure of maximal compression.
         const double mu = eig_val_data[0];
         visc_coeff = 2.0 * q_rho * h * h * fabs(mu);
         // The following represents a "smooth" version of the statement
         // "if (mu < 0) visc_coeff += 0.5 rho h sound_speed".  Note that
         // eps must be scaled appropriately if a different unit system is
         // being used.
         const double eps = 1e-12;
         visc_coeff += 0.5 * q_rho * h * soundSpeed *
                       (1.0 - smooth_step_01(mu - 2.0 * eps, eps));
         //if (mu < 0.0) { visc_coeff += 0.5 * rho * h * sound_speed; }
         for (int y = 0; y < DIM; ++y)
         {
            for (int x = 0; x < DIM; ++x)
            {
               q_stress[ijN(x,y,2)] += visc_coeff * q_gradv[ijN(x,y,2)];
            }
         }
      }
      // Time step estimate at the point. Here the more relevant length
      // scale is related to the actual mesh deformation; we use the min
      // singular value of the ref->physical Jacobian. In addition, the
      // time step estimate should be aware of the presence of shocks.
      const double sv = calcSingularvalue(DIM, DIM-1, xJ);
      const double h_min = sv / h1order;
      const double inv_h_min = 1. / h_min;
      const double inv_rho_inv_h_min_sq = inv_h_min * inv_h_min / q_rho ;
      const double inv_dt = soundSpeed * inv_h_min
                            + 2.5 * visc_coeff * inv_rho_inv_h_min_sq;
      if (min_detJ < 0.0)
      {
         // This will force repetition of the step with smaller dt.
         dtEst[ijN(q,el,NUM_QUAD)] = 0.0;
      }
      else
      {
         if (inv_dt>0.0)
         {
            const double cfl_inv_dt = CFL / inv_dt;
            dtEst[ijN(q,el,NUM_QUAD)] =
               fmin(dtEst[ijN(q,el,NUM_QUAD)], cfl_inv_dt);
         }
      }
      const double S00 = q_stress[ijN(0,0,2)];
      const double S10 = q_stress[ijN(1,0,2)];
      const double S01 = q_stress[ijN(0,1,2)];
      const double S11 = q_stress[ijN(1,1,2)];
      stressJinvT[ijklNM(0,0,q,el,DIM,
                         NUM_QUAD)] = q_Jw*((S00*invJ_00)+(S10*invJ_01));
      stressJinvT[ijklNM(1,0,q,el,DIM,
                         NUM_QUAD)] = q_Jw*((S00*invJ_10)+(S10*invJ_11));
      stressJinvT[ijklNM(0,1,q,el,DIM,
                         NUM_QUAD)] = q_Jw*((S01*invJ_00)+(S11*invJ_01));
      stressJinvT[ijklNM(1,1,q,el,DIM,
                         NUM_QUAD)] = q_Jw*((S01*invJ_10)+(S11*invJ_11));
   }
}

template<const int NUM_DIM,
         const int NUM_QUAD,
         const int NUM_QUAD_1D,
         const int NUM_DOFS_1D> kernel
void rUpdateQuadratureData2D(const double GAMMA,
                             const double H0,
                             const double CFL,
                             const bool USE_VISCOSITY,
                             const int numElements,
                             const double* restrict dofToQuad,
                             const double* restrict dofToQuadD,
                             const double* restrict quadWeights,
                             const double* restrict v,
                             const double* restrict e,
                             const double* restrict rho0DetJ0w,
                             const double* restrict invJ0,
                             const double* restrict J,
                             const double* restrict invJ,
                             const double* restrict detJ,
                             double* restrict stressJinvT,
                             double* restrict dtEst)
{
   const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
   const int el = blockDim.x * blockIdx.x + threadIdx.x;
   if (el < numElements)
   {
      const int DIM = 2;
      const int VDIMQ = DIM*DIM * NUM_QUAD_2D;
      double s_gradv[VDIMQ];

      for (int i = 0; i < VDIMQ; ++i) { s_gradv[i] = 0.0; }

      for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
      {
         double vDx[DIM*NUM_QUAD_1D];
         double  vx[DIM*NUM_QUAD_1D];

         for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
         {
            for (int c = 0; c < DIM; ++c)
            {
               vDx[ijN(c,qx,DIM)] = 0.0;
               vx[ijN(c,qx,DIM)] = 0.0;
            }
         }
         for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
         {
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
            {
               const double wx  =  dofToQuad[ijN(qx,dx,NUM_QUAD_1D)];
               const double wDx = dofToQuadD[ijN(qx,dx,NUM_QUAD_1D)];
               for (int c = 0; c < DIM; ++c)
               {
                  vDx[ijN(c,qx,DIM)] += wDx * v[_ijklNM(c,dx,dy,el,NUM_DOFS_1D,numElements)];
                  vx[ijN(c,qx,DIM)] +=  wx * v[_ijklNM(c,dx,dy,el,NUM_DOFS_1D,numElements)];
               }
            }
         }
         for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
         {
            const double  wy =  dofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
            const double wDy = dofToQuadD[ijN(qy,dy,NUM_QUAD_1D)];
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
            {
               for (int c = 0; c < DIM; ++c)
               {
                  s_gradv[ijkN(c,0,qx+qy*NUM_QUAD_1D,DIM)] += wy *vDx[ijN(c,qx,DIM)];
                  s_gradv[ijkN(c,1,qx+qy*NUM_QUAD_1D,DIM)] += wDy*vx[ijN(c,qx,DIM)];
               }
            }
         }
      }

      for (int q = 0; q < NUM_QUAD; ++q)
      {
         double q_gradv[NUM_DIM*NUM_DIM];
         double q_stress[NUM_DIM*NUM_DIM];

         const double invJ_00 = invJ[ijklNM(0,0,q,el,NUM_DIM,NUM_QUAD)];
         const double invJ_10 = invJ[ijklNM(1,0,q,el,NUM_DIM,NUM_QUAD)];
         const double invJ_01 = invJ[ijklNM(0,1,q,el,NUM_DIM,NUM_QUAD)];
         const double invJ_11 = invJ[ijklNM(1,1,q,el,NUM_DIM,NUM_QUAD)];

         q_gradv[ijN(0,0,2)] = ((s_gradv[ijkN(0,0,q,2)]*invJ_00)+(s_gradv[ijkN(1,0,q,
                                                                               2)]*invJ_01));
         q_gradv[ijN(1,0,2)] = ((s_gradv[ijkN(0,0,q,2)]*invJ_10)+(s_gradv[ijkN(1,0,q,
                                                                               2)]*invJ_11));
         q_gradv[ijN(0,1,2)] = ((s_gradv[ijkN(0,1,q,2)]*invJ_00)+(s_gradv[ijkN(1,1,q,
                                                                               2)]*invJ_01));
         q_gradv[ijN(1,1,2)] = ((s_gradv[ijkN(0,1,q,2)]*invJ_10)+(s_gradv[ijkN(1,1,q,
                                                                               2)]*invJ_11));

         const double q_Jw = detJ[ijN(q,el,NUM_QUAD)]*quadWeights[q];

         const double q_rho = rho0DetJ0w[ijN(q,el,NUM_QUAD)] / q_Jw;
         const double q_e   = fmax(0.0,e[ijN(q,el,NUM_QUAD)]);

         // TODO: Input OccaVector eos(q,e) -> (stress,soundSpeed)
         const double s = -(GAMMA-1.0)*q_rho*q_e;
         q_stress[ijN(0,0,2)] = s; q_stress[ijN(1,0,2)] = 0;
         q_stress[ijN(0,1,2)] = 0; q_stress[ijN(1,1,2)] = s;

         const double gradv00 = q_gradv[ijN(0,0,2)];
         const double gradv11 = q_gradv[ijN(1,1,2)];
         const double gradv10 = 0.5*(q_gradv[ijN(1,0,2)]+q_gradv[ijN(0,1,2)]);
         q_gradv[ijN(1,0,2)] = gradv10;
         q_gradv[ijN(0,1,2)] = gradv10;

         double comprDirX = 1;
         double comprDirY = 0;
         double minEig = 0;
         // linalg/densemat.cpp: Eigensystem2S()
         if (gradv10 == 0)
         {
            minEig = (gradv00 < gradv11) ? gradv00 : gradv11;
         }
         else
         {
            const double zeta  = (gradv11-gradv00) / (2.0*gradv10);
            const double azeta = fabs(zeta);
            double t = 1.0 / (azeta+sqrt(1.0+zeta*zeta));
            if ((t < 0) != (zeta < 0))
            {
               t = -t;
            }
            const double c = sqrt(1.0 / (1.0+t*t));
            const double s = c*t;
            t *= gradv10;
            if ((gradv00-t) <= (gradv11+t))
            {
               minEig = gradv00-t;
               comprDirX = c;
               comprDirY = -s;
            }
            else
            {
               minEig = gradv11+t;
               comprDirX = s;
               comprDirY = c;
            }
         }

         // Computes the initial->physical transformation Jacobian.
         const double J_00 = J[ijklNM(0,0,q,el,NUM_DIM,NUM_QUAD)];
         const double J_10 = J[ijklNM(1,0,q,el,NUM_DIM,NUM_QUAD)];
         const double J_01 = J[ijklNM(0,1,q,el,NUM_DIM,NUM_QUAD)];
         const double J_11 = J[ijklNM(1,1,q,el,NUM_DIM,NUM_QUAD)];
         const double invJ0_00 = invJ0[ijklNM(0,0,q,el,NUM_DIM,NUM_QUAD)];
         const double invJ0_10 = invJ0[ijklNM(1,0,q,el,NUM_DIM,NUM_QUAD)];
         const double invJ0_01 = invJ0[ijklNM(0,1,q,el,NUM_DIM,NUM_QUAD)];
         const double invJ0_11 = invJ0[ijklNM(1,1,q,el,NUM_DIM,NUM_QUAD)];
         const double Jpi_00 = ((J_00*invJ0_00)+(J_10*invJ0_01));
         const double Jpi_10 = ((J_00*invJ0_10)+(J_10*invJ0_11));
         const double Jpi_01 = ((J_01*invJ0_00)+(J_11*invJ0_01));
         const double Jpi_11 = ((J_01*invJ0_10)+(J_11*invJ0_11));
         const double physDirX = (Jpi_00*comprDirX)+(Jpi_10*comprDirY);
         const double physDirY = (Jpi_01*comprDirX)+(Jpi_11*comprDirY);
         const double q_h = H0*sqrt((physDirX*physDirX)+(physDirY*physDirY));
         // TODO: soundSpeed will be an input as well (function call or values per q)
         const double soundSpeed = sqrt(GAMMA*(GAMMA-1.0)*q_e);
         dtEst[ijN(q,el,NUM_QUAD)] = CFL*q_h / soundSpeed;
         //const double cfl_inv_dt = CFL*q_h / soundSpeed;
         //dtEst[el] = fmin(dtEst[el], cfl_inv_dt);
         if (USE_VISCOSITY)
         {
            // TODO: Check how we can extract outside of kernel
            const double mu = minEig;
            double coeff = 2.0*q_rho*q_h*q_h*fabs(mu);
            if (mu < 0)
            {
               coeff += 0.5*q_rho*q_h*soundSpeed;
            }
            for (int y = 0; y < NUM_DIM; ++y)
            {
               for (int x = 0; x < NUM_DIM; ++x)
               {
                  q_stress[ijN(x,y,2)] += coeff*q_gradv[ijN(x,y,2)];
               }
            }
         }
         const double S00 = q_stress[ijN(0,0,2)];
         const double S10 = q_stress[ijN(1,0,2)];
         const double S01 = q_stress[ijN(0,1,2)];
         const double S11 = q_stress[ijN(1,1,2)];
         stressJinvT[ijklNM(0,0,q,el,NUM_DIM,
                            NUM_QUAD)] = q_Jw*((S00*invJ_00)+(S10*invJ_01));
         stressJinvT[ijklNM(1,0,q,el,NUM_DIM,
                            NUM_QUAD)] = q_Jw*((S00*invJ_10)+(S10*invJ_11));
         stressJinvT[ijklNM(0,1,q,el,NUM_DIM,
                            NUM_QUAD)] = q_Jw*((S01*invJ_00)+(S11*invJ_01));
         stressJinvT[ijklNM(1,1,q,el,NUM_DIM,
                            NUM_QUAD)] = q_Jw*((S01*invJ_10)+(S11*invJ_11));
      }
   }
}

// *****************************************************************************
template<const int NUM_DIM,
         const int NUM_QUAD,
         const int NUM_QUAD_1D,
         const int NUM_DOFS_1D> kernel
void rUpdateQuadratureData3D(const double GAMMA,
                             const double H0,
                             const double CFL,
                             const bool USE_VISCOSITY,
                             const int numElements,
                             const double* restrict dofToQuad,
                             const double* restrict dofToQuadD,
                             const double* restrict quadWeights,
                             const double* restrict v,
                             const double* restrict e,
                             const double* restrict rho0DetJ0w,
                             const double* restrict invJ0,
                             const double* restrict J,
                             const double* restrict invJ,
                             const double* restrict detJ,
                             double* restrict stressJinvT,
                             double* restrict dtEst)
{
   const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
   const int NUM_QUAD_3D = NUM_QUAD_1D*NUM_QUAD_1D*NUM_QUAD_1D;
   const int el = blockDim.x * blockIdx.x + threadIdx.x;
   if (el < numElements)
   {
      double s_gradv[9*NUM_QUAD_3D];

      for (int i = 0; i < (9*NUM_QUAD_3D); ++i)
      {
         s_gradv[i] = 0;
      }

      for (int dz = 0; dz < NUM_DOFS_1D; ++dz)
      {
         double vDxy[3*NUM_QUAD_2D] ;
         double vxDy[3*NUM_QUAD_2D] ;
         double vxy[3*NUM_QUAD_2D]  ;
         for (int i = 0; i < (3*NUM_QUAD_2D); ++i)
         {
            vDxy[i] = 0;
            vxDy[i] = 0;
            vxy[i]  = 0;
         }

         for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
         {
            double vDx[3*NUM_QUAD_1D] ;
            double vx[3*NUM_QUAD_1D]  ;
            for (int i = 0; i < (3*NUM_QUAD_1D); ++i)
            {
               vDx[i] = 0;
               vx[i]  = 0;
            }

            for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
            {
               for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
               {
                  for (int vi = 0; vi < 3; ++vi)
                  {
                     vDx[ijN(vi,qx,3)] += v[_ijklmNM(vi,dx,dy,dz,el,NUM_DOFS_1D,
                                                     numElements)]*dofToQuadD[ijN(qx,dx,NUM_QUAD_1D)];
                     vx[ijN(vi,qx,3)]  += v[_ijklmNM(vi,dx,dy,dz,el,NUM_DOFS_1D,
                                                     numElements)]*dofToQuad[ijN(qx,dx,NUM_QUAD_1D)];
                  }
               }
            }

            for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
            {
               const double wy  = dofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
               const double wDy = dofToQuadD[ijN(qy,dy,NUM_QUAD_1D)];
               for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
               {
                  for (int vi = 0; vi < 3; ++vi)
                  {
                     vDxy[ijkNM(vi,qx,qy,3,NUM_QUAD_1D)] += wy *vDx[ijN(vi,qx,3)];
                     vxDy[ijkNM(vi,qx,qy,3,NUM_QUAD_1D)] += wDy*vx[ijN(vi,qx,3)];
                     vxy[ijkNM(vi,qx,qy,3,NUM_QUAD_1D)]  += wy *vx[ijN(vi,qx,3)];
                  }
               }
            }
         }
         // for (int qz = 0; qz < NUM_DOFS_1D; ++qz)
         for (int qz = 0; qz < NUM_QUAD_1D; ++qz)
         {
            const double wz  = dofToQuad[ijN(qz,dz,NUM_QUAD_1D)];
            const double wDz = dofToQuadD[ijN(qz,dz,NUM_QUAD_1D)];
            for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
            {
               for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
               {
                  const int q = qx+qy*NUM_QUAD_1D+qz*NUM_QUAD_2D;
                  for (int vi = 0; vi < 3; ++vi)
                  {
                     s_gradv[ijkN(vi,0,q,3)] += wz *vDxy[ijkNM(vi,qx,qy,3,NUM_QUAD_1D)];
                     s_gradv[ijkN(vi,1,q,3)] += wz *vxDy[ijkNM(vi,qx,qy,3,NUM_QUAD_1D)];
                     s_gradv[ijkN(vi,2,q,3)] += wDz*vxy[ijkNM(vi,qx,qy,3,NUM_QUAD_1D)];
                  }
               }
            }
         }
      }

      for (int q = 0; q < NUM_QUAD; ++q)
      {
         double q_gradv[9]  ;
         double q_stress[9] ;

         const double invJ_00 = invJ[ijklNM(0,0,q,el,NUM_DIM,NUM_QUAD)];
         const double invJ_10 = invJ[ijklNM(1,0,q,el,NUM_DIM,NUM_QUAD)];
         const double invJ_20 = invJ[ijklNM(2,0,q,el,NUM_DIM,NUM_QUAD)];
         const double invJ_01 = invJ[ijklNM(0,1,q,el,NUM_DIM,NUM_QUAD)];
         const double invJ_11 = invJ[ijklNM(1,1,q,el,NUM_DIM,NUM_QUAD)];
         const double invJ_21 = invJ[ijklNM(2,1,q,el,NUM_DIM,NUM_QUAD)];
         const double invJ_02 = invJ[ijklNM(0,2,q,el,NUM_DIM,NUM_QUAD)];
         const double invJ_12 = invJ[ijklNM(1,2,q,el,NUM_DIM,NUM_QUAD)];
         const double invJ_22 = invJ[ijklNM(2,2,q,el,NUM_DIM,NUM_QUAD)];

         q_gradv[ijN(0,0,3)] = ((s_gradv[ijkN(0,0,q,3)]*invJ_00) +
                                (s_gradv[ijkN(1,0,q,3)]*invJ_01) +
                                (s_gradv[ijkN(2,0,q,3)]*invJ_02));
         q_gradv[ijN(1,0,3)] = ((s_gradv[ijkN(0,0,q,3)]*invJ_10) +
                                (s_gradv[ijkN(1,0,q,3)]*invJ_11) +
                                (s_gradv[ijkN(2,0,q,3)]*invJ_12));
         q_gradv[ijN(2,0,3)] = ((s_gradv[ijkN(0,0,q,3)]*invJ_20) +
                                (s_gradv[ijkN(1,0,q,3)]*invJ_21) +
                                (s_gradv[ijkN(2,0,q,3)]*invJ_22));

         q_gradv[ijN(0,1,3)] = ((s_gradv[ijkN(0,1,q,3)]*invJ_00) +
                                (s_gradv[ijkN(1,1,q,3)]*invJ_01) +
                                (s_gradv[ijkN(2,1,q,3)]*invJ_02));
         q_gradv[ijN(1,1,3)] = ((s_gradv[ijkN(0,1,q,3)]*invJ_10) +
                                (s_gradv[ijkN(1,1,q,3)]*invJ_11) +
                                (s_gradv[ijkN(2,1,q,3)]*invJ_12));
         q_gradv[ijN(2,1,3)] = ((s_gradv[ijkN(0,1,q,3)]*invJ_20) +
                                (s_gradv[ijkN(1,1,q,3)]*invJ_21) +
                                (s_gradv[ijkN(2,1,q,3)]*invJ_22));

         q_gradv[ijN(0,2,3)] = ((s_gradv[ijkN(0,2,q,3)]*invJ_00) +
                                (s_gradv[ijkN(1,2,q,3)]*invJ_01) +
                                (s_gradv[ijkN(2,2,q,3)]*invJ_02));
         q_gradv[ijN(1,2,3)] = ((s_gradv[ijkN(0,2,q,3)]*invJ_10) +
                                (s_gradv[ijkN(1,2,q,3)]*invJ_11) +
                                (s_gradv[ijkN(2,2,q,3)]*invJ_12));
         q_gradv[ijN(2,2,3)] = ((s_gradv[ijkN(0,2,q,3)]*invJ_20) +
                                (s_gradv[ijkN(1,2,q,3)]*invJ_21) +
                                (s_gradv[ijkN(2,2,q,3)]*invJ_22));

         const double q_Jw = detJ[ijN(q,el,NUM_QUAD)]*quadWeights[q];

         const double q_rho = rho0DetJ0w[ijN(q,el,NUM_QUAD)] / q_Jw;
         const double q_e   = fmax(0.0,e[ijN(q,el,NUM_QUAD)]);

         const double s = -(GAMMA-1.0)*q_rho*q_e;
         q_stress[ijN(0,0,3)] = s; q_stress[ijN(1,0,3)] = 0; q_stress[ijN(2,0,3)] = 0;
         q_stress[ijN(0,1,3)] = 0; q_stress[ijN(1,1,3)] = s; q_stress[ijN(2,1,3)] = 0;
         q_stress[ijN(0,2,3)] = 0; q_stress[ijN(1,2,3)] = 0; q_stress[ijN(2,2,3)] = s;

         const double gradv00 = q_gradv[ijN(0,0,3)];
         const double gradv11 = q_gradv[ijN(1,1,3)];
         const double gradv22 = q_gradv[ijN(2,2,3)];
         const double gradv10 = 0.5*(q_gradv[ijN(1,0,3)]+q_gradv[ijN(0,1,3)]);
         const double gradv20 = 0.5*(q_gradv[ijN(2,0,3)]+q_gradv[ijN(0,2,3)]);
         const double gradv21 = 0.5*(q_gradv[ijN(2,1,3)]+q_gradv[ijN(1,2,3)]);
         q_gradv[ijN(1,0,3)] = gradv10; q_gradv[ijN(2,0,3)] = gradv20;
         q_gradv[ijN(0,1,3)] = gradv10; q_gradv[ijN(2,1,3)] = gradv21;
         q_gradv[ijN(0,2,3)] = gradv20; q_gradv[ijN(1,2,3)] = gradv21;

         double minEig = 0;
         double comprDirX = 1;
         double comprDirY = 0;
         double comprDirZ = 0;

         {
            // Compute eigenvalues using quadrature formula
            const double q_ = (gradv00+gradv11+gradv22) / 3.0;
            const double gradv_q00 = (gradv00-q_);
            const double gradv_q11 = (gradv11-q_);
            const double gradv_q22 = (gradv22-q_);

            const double p1 = ((gradv10*gradv10) +
                               (gradv20*gradv20) +
                               (gradv21*gradv21));
            const double p2 = ((gradv_q00*gradv_q00) +
                               (gradv_q11*gradv_q11) +
                               (gradv_q22*gradv_q22) +
                               (2.0*p1));
            const double p    = sqrt(p2 / 6.0);
            const double pinv = 1.0 / p;
            // det(pinv*(gradv-q*I))
            const double r = (0.5*pinv*pinv*pinv *
                              ((gradv_q00*gradv_q11*gradv_q22) +
                               (2.0*gradv10*gradv21*gradv20) -
                               (gradv_q11*gradv20*gradv20) -
                               (gradv_q22*gradv10*gradv10) -
                               (gradv_q00*gradv21*gradv21)));

            double phi = 0;
            if (r <= -1.0)
            {
               phi = M_PI / 3.0;
            }
            else if (r < 1.0)
            {
               phi = acos(r) / 3.0;
            }

            minEig = q_+(2.0*p*cos(phi+(2.0*M_PI / 3.0)));
            const double eig3 = q_+(2.0*p*cos(phi));
            const double eig2 = 3.0*q_-minEig-eig3;
            double maxNorm = 0;

            for (int i = 0; i < 3; ++i)
            {
               const double x = q_gradv[i+3*0]-(i == 0)*eig3;
               const double y = q_gradv[i+3*1]-(i == 1)*eig3;
               const double z = q_gradv[i+3*2]-(i == 2)*eig3;
               const double cx = ((x*(gradv00-eig2)) +
                                  (y*gradv10) +
                                  (z*gradv20));
               const double cy = ((x*gradv10) +
                                  (y*(gradv11-eig2)) +
                                  (z*gradv21));
               const double cz = ((x*gradv20) +
                                  (y*gradv21) +
                                  (z*(gradv22-eig2)));
               const double cNorm = (cx*cx+cy*cy+cz*cz);
               //#warning 1e-16 to 1
               if ((cNorm > 1.e-16) && (maxNorm < cNorm))
               {
                  comprDirX = cx;
                  comprDirY = cy;
                  comprDirZ = cz;
                  maxNorm = cNorm;
               }
            }
            //#warning 1e-16 to 1
            if (maxNorm > 1.e-16)
            {
               const double maxNormInv = 1.0 / sqrt(maxNorm);
               comprDirX *= maxNormInv;
               comprDirY *= maxNormInv;
               comprDirZ *= maxNormInv;
            }
         }

         // Computes the initial->physical transformation Jacobian.
         const double J_00 = J[ijklNM(0,0,q,el,NUM_DIM,NUM_QUAD)];
         const double J_10 = J[ijklNM(1,0,q,el,NUM_DIM,NUM_QUAD)];
         const double J_20 = J[ijklNM(2,0,q,el,NUM_DIM,NUM_QUAD)];
         const double J_01 = J[ijklNM(0,1,q,el,NUM_DIM,NUM_QUAD)];
         const double J_11 = J[ijklNM(1,1,q,el,NUM_DIM,NUM_QUAD)];
         const double J_21 = J[ijklNM(2,1,q,el,NUM_DIM,NUM_QUAD)];
         const double J_02 = J[ijklNM(0,2,q,el,NUM_DIM,NUM_QUAD)];
         const double J_12 = J[ijklNM(1,2,q,el,NUM_DIM,NUM_QUAD)];
         const double J_22 = J[ijklNM(2,2,q,el,NUM_DIM,NUM_QUAD)];

         const double invJ0_00 = invJ0[ijklNM(0,0,q,el,NUM_DIM,NUM_QUAD)];
         const double invJ0_10 = invJ0[ijklNM(1,0,q,el,NUM_DIM,NUM_QUAD)];
         const double invJ0_20 = invJ0[ijklNM(2,0,q,el,NUM_DIM,NUM_QUAD)];
         const double invJ0_01 = invJ0[ijklNM(0,1,q,el,NUM_DIM,NUM_QUAD)];
         const double invJ0_11 = invJ0[ijklNM(1,1,q,el,NUM_DIM,NUM_QUAD)];
         const double invJ0_21 = invJ0[ijklNM(2,1,q,el,NUM_DIM,NUM_QUAD)];
         const double invJ0_02 = invJ0[ijklNM(0,2,q,el,NUM_DIM,NUM_QUAD)];
         const double invJ0_12 = invJ0[ijklNM(1,2,q,el,NUM_DIM,NUM_QUAD)];
         const double invJ0_22 = invJ0[ijklNM(2,2,q,el,NUM_DIM,NUM_QUAD)];

         const double Jpi_00 = ((J_00*invJ0_00)+(J_10*invJ0_01)+(J_20*invJ0_02));
         const double Jpi_10 = ((J_00*invJ0_10)+(J_10*invJ0_11)+(J_20*invJ0_12));
         const double Jpi_20 = ((J_00*invJ0_20)+(J_10*invJ0_21)+(J_20*invJ0_22));

         const double Jpi_01 = ((J_01*invJ0_00)+(J_11*invJ0_01)+(J_21*invJ0_02));
         const double Jpi_11 = ((J_01*invJ0_10)+(J_11*invJ0_11)+(J_21*invJ0_12));
         const double Jpi_21 = ((J_01*invJ0_20)+(J_11*invJ0_21)+(J_21*invJ0_22));

         const double Jpi_02 = ((J_02*invJ0_00)+(J_12*invJ0_01)+(J_22*invJ0_02));
         const double Jpi_12 = ((J_02*invJ0_10)+(J_12*invJ0_11)+(J_22*invJ0_12));
         const double Jpi_22 = ((J_02*invJ0_20)+(J_12*invJ0_21)+(J_22*invJ0_22));

         const double physDirX = ((Jpi_00*comprDirX)+(Jpi_10*comprDirY)+
                                  (Jpi_20*comprDirZ));
         const double physDirY = ((Jpi_01*comprDirX)+(Jpi_11*comprDirY)+
                                  (Jpi_21*comprDirZ));
         const double physDirZ = ((Jpi_02*comprDirX)+(Jpi_12*comprDirY)+
                                  (Jpi_22*comprDirZ));

         const double q_h = H0*sqrt((physDirX*physDirX)+
                                    (physDirY*physDirY)+
                                    (physDirZ*physDirZ));

         const double soundSpeed = sqrt(GAMMA*(GAMMA-1.0)*q_e);
         dtEst[ijN(q,el,NUM_QUAD)] = CFL*q_h / soundSpeed;

         if (USE_VISCOSITY)
         {
            // TODO: Check how we can extract outside of kernel
            const double mu = minEig;
            double coeff = 2.0*q_rho*q_h*q_h*fabs(mu);
            if (mu < 0)
            {
               coeff += 0.5*q_rho*q_h*soundSpeed;
            }
            for (int y = 0; y < 3; ++y)
            {
               for (int x = 0; x < 3; ++x)
               {
                  q_stress[ijN(x,y,3)] += coeff*q_gradv[ijN(x,y,3)];
               }
            }
         }

         const double S00 = q_stress[ijN(0,0,3)];
         const double S10 = q_stress[ijN(1,0,3)];
         const double S20 = q_stress[ijN(2,0,3)];
         const double S01 = q_stress[ijN(0,1,3)];
         const double S11 = q_stress[ijN(1,1,3)];
         const double S21 = q_stress[ijN(2,1,3)];
         const double S02 = q_stress[ijN(0,2,3)];
         const double S12 = q_stress[ijN(1,2,3)];
         const double S22 = q_stress[ijN(2,2,3)];

         stressJinvT[ijklNM(0,0,q,el,NUM_DIM,
                            NUM_QUAD)] = q_Jw*((S00*invJ_00)+(S10*invJ_01)+(S20*invJ_02));
         stressJinvT[ijklNM(1,0,q,el,NUM_DIM,
                            NUM_QUAD)] = q_Jw*((S00*invJ_10)+(S10*invJ_11)+(S20*invJ_12));
         stressJinvT[ijklNM(2,0,q,el,NUM_DIM,
                            NUM_QUAD)] = q_Jw*((S00*invJ_20)+(S10*invJ_21)+(S20*invJ_22));

         stressJinvT[ijklNM(0,1,q,el,NUM_DIM,
                            NUM_QUAD)] = q_Jw*((S01*invJ_00)+(S11*invJ_01)+(S21*invJ_02));
         stressJinvT[ijklNM(1,1,q,el,NUM_DIM,
                            NUM_QUAD)] = q_Jw*((S01*invJ_10)+(S11*invJ_11)+(S21*invJ_12));
         stressJinvT[ijklNM(2,1,q,el,NUM_DIM,
                            NUM_QUAD)] = q_Jw*((S01*invJ_20)+(S11*invJ_21)+(S21*invJ_22));

         stressJinvT[ijklNM(0,2,q,el,NUM_DIM,
                            NUM_QUAD)] = q_Jw*((S02*invJ_00)+(S12*invJ_01)+(S22*invJ_02));
         stressJinvT[ijklNM(1,2,q,el,NUM_DIM,
                            NUM_QUAD)] = q_Jw*((S02*invJ_10)+(S12*invJ_11)+(S22*invJ_12));
         stressJinvT[ijklNM(2,2,q,el,NUM_DIM,
                            NUM_QUAD)] = q_Jw*((S02*invJ_20)+(S12*invJ_21)+(S22*invJ_22));
      }
   }
}

template<const int NUM_DOFS_1D,
         const int NUM_QUAD_1D,
         const int USE_SMEM,
         const int BLOCK,
         const int NBLOCK>
__launch_bounds__(BLOCK, NBLOCK)
kernel
void rUpdateQuadratureData3D_v2(const double GAMMA,
                                const double H0,
                                const double CFL,
                                const bool USE_VISCOSITY,
                                const int numElements,
                                const double* restrict dofToQuad,
                                const double* restrict dofToQuadD,
                                const double* restrict quadWeights,
                                const double* restrict v,
                                const double* restrict e,
                                const double* restrict rho0DetJ0w,
                                const double* restrict invJ0,
                                const double* restrict J,
                                const double* restrict invJ,
                                const double* restrict detJ,
                                double* restrict stressJinvT,
                                double* restrict dtEst,
                                double *gbuf,
                                int bufSize)
{
   const int NUM_DIM = 3;
   const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
   const int NUM_QUAD_3D = NUM_QUAD_1D*NUM_QUAD_1D*NUM_QUAD_1D;
   int tid = threadIdx.x + threadIdx.y*blockDim.x +
             threadIdx.z*blockDim.x*blockDim.y;
   extern __shared__ double sbuf[];
   double *buf_ptr;

   if (USE_SMEM)
   {
      buf_ptr = sbuf;
   }
   else
   {
      buf_ptr = (double*)((char*)gbuf + blockIdx.x*bufSize);
   }

   // __shared__ double s_gradv[9*NUM_QUAD_3D];
   //                   vDx[NUM_DOFS_1D][NUM_DOFS_1D][3*NUM_QUAD_1D],
   //                   vx[NUM_DOFS_1D][NUM_DOFS_1D][3*NUM_QUAD_1D],
   //                   vDxy[NUM_DOFS_1D][3*NUM_QUAD_2D],
   //                   vxDy[NUM_DOFS_1D][3*NUM_QUAD_2D],
   //                   vxy[NUM_DOFS_1D][3*NUM_QUAD_2D]  ;
   double *s_gradv,
          (*vDx)[NUM_DOFS_1D][3*NUM_QUAD_1D],
          (*vx)[NUM_DOFS_1D][3*NUM_QUAD_1D],
          (*vDxy)[3*NUM_QUAD_2D],
          (*vxDy)[3*NUM_QUAD_2D],
          (*vxy)[3*NUM_QUAD_2D];
   mallocBuf((void**)&s_gradv, (void**)&buf_ptr, 9*NUM_QUAD_3D*sizeof(double));
   mallocBuf((void**)&vDx    , (void**)&buf_ptr,
             3*NUM_DOFS_1D*NUM_DOFS_1D*NUM_QUAD_1D*sizeof(double));
   mallocBuf((void**)&vx     , (void**)&buf_ptr,
             3*NUM_DOFS_1D*NUM_DOFS_1D*NUM_QUAD_1D*sizeof(double));
   mallocBuf((void**)&vDxy   , (void**)&buf_ptr,
             3*NUM_DOFS_1D*NUM_QUAD_2D*sizeof(double));
   mallocBuf((void**)&vxDy   , (void**)&buf_ptr,
             3*NUM_DOFS_1D*NUM_QUAD_2D*sizeof(double));
   mallocBuf((void**)&vxy    , (void**)&buf_ptr,
             3*NUM_DOFS_1D*NUM_QUAD_2D*sizeof(double));

   for (int el = blockIdx.x; el < numElements; el += gridDim.x)
   {
      for (int dz = threadIdx.z; dz < NUM_DOFS_1D; dz += blockDim.z)
      {
         for (int dy = threadIdx.y; dy < NUM_DOFS_1D; dy += blockDim.y)
         {
            for (int qx = threadIdx.x; qx < NUM_QUAD_1D; qx += blockDim.x)
            {
               for (int vi = 0; vi < 3; ++vi)
               {
                  double t1 = 0, t2 = 0;
                  for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
                  {
                     t1 += v[_ijklmNM(vi,dx,dy,dz,el,NUM_DOFS_1D,
                                      numElements)]*dofToQuadD[ijN(qx,dx,NUM_QUAD_1D)];
                     t2 += v[_ijklmNM(vi,dx,dy,dz,el,NUM_DOFS_1D,
                                      numElements)]*dofToQuad[ijN(qx,dx,NUM_QUAD_1D)];
                  }
                  vDx[dz][dy][ijN(vi,qx,3)] = t1;
                  vx[dz][dy][ijN(vi,qx,3)] = t2;
               }
            }
         }
      }
      __syncthreads();

      for (int dz = threadIdx.z; dz < NUM_DOFS_1D; dz += blockDim.z)
      {
         for (int qy = threadIdx.y; qy < NUM_QUAD_1D; qy += blockDim.y)
         {
            for (int qx = threadIdx.x; qx < NUM_QUAD_1D; qx += blockDim.x)
            {
               for (int vi = 0; vi < 3; ++vi)
               {
                  double t1 = 0, t2 = 0, t3 = 0;
                  for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
                  {
                     t1 += dofToQuad[ijN(qy,dy,NUM_QUAD_1D)] *vDx[dz][dy][ijN(vi,qx,3)];
                     t2 += dofToQuadD[ijN(qy,dy,NUM_QUAD_1D)]*vx[dz][dy][ijN(vi,qx,3)];
                     t3 += dofToQuad[ijN(qy,dy,NUM_QUAD_1D)]*vx[dz][dy][ijN(vi,qx,3)];
                  }
                  vDxy[dz][ijkNM(vi,qx,qy,3,NUM_QUAD_1D)] = t1;
                  vxDy[dz][ijkNM(vi,qx,qy,3,NUM_QUAD_1D)] = t2;
                  vxy[dz][ijkNM(vi,qx,qy,3,NUM_QUAD_1D)] = t3;
               }
            }
         }
      }
      __syncthreads();

      for (int qz = threadIdx.z; qz < NUM_QUAD_1D; qz += blockDim.z)
      {
         for (int qy = threadIdx.y; qy < NUM_QUAD_1D; qy += blockDim.y)
         {
            for (int qx = threadIdx.x; qx < NUM_QUAD_1D; qx += blockDim.x)
            {
               const int q = qx+qy*NUM_QUAD_1D+qz*NUM_QUAD_2D;
               for (int vi = 0; vi < 3; ++vi)
               {
                  double t1 = 0, t2 = 0, t3 = 0;
                  for (int dz = 0; dz < NUM_DOFS_1D; ++dz)
                  {
                     t1 += dofToQuad[ijN(qz,dz,NUM_QUAD_1D)]*vDxy[dz][ijkNM(vi,qx,qy,3,NUM_QUAD_1D)];
                     t2 += dofToQuad[ijN(qz,dz,NUM_QUAD_1D)]*vxDy[dz][ijkNM(vi,qx,qy,3,NUM_QUAD_1D)];
                     t3 += dofToQuadD[ijN(qz,dz,NUM_QUAD_1D)]*vxy[dz][ijkNM(vi,qx,qy,3,NUM_QUAD_1D)];
                  }
                  s_gradv[ijkN(vi,0,q,3)] = t1;
                  s_gradv[ijkN(vi,1,q,3)] = t2;
                  s_gradv[ijkN(vi,2,q,3)] = t3;
               }
            }
         }
      }
      __syncthreads();

      for (int q = tid; q < NUM_QUAD_3D; q += blockDim.x*blockDim.y*blockDim.z)
      {
         double q_gradv[9]  ;
         double q_stress[9] ;

         const double invJ_00 = invJ[ijklNM(0,0,q,el,NUM_DIM,NUM_QUAD_3D)];
         const double invJ_10 = invJ[ijklNM(1,0,q,el,NUM_DIM,NUM_QUAD_3D)];
         const double invJ_20 = invJ[ijklNM(2,0,q,el,NUM_DIM,NUM_QUAD_3D)];
         const double invJ_01 = invJ[ijklNM(0,1,q,el,NUM_DIM,NUM_QUAD_3D)];
         const double invJ_11 = invJ[ijklNM(1,1,q,el,NUM_DIM,NUM_QUAD_3D)];
         const double invJ_21 = invJ[ijklNM(2,1,q,el,NUM_DIM,NUM_QUAD_3D)];
         const double invJ_02 = invJ[ijklNM(0,2,q,el,NUM_DIM,NUM_QUAD_3D)];
         const double invJ_12 = invJ[ijklNM(1,2,q,el,NUM_DIM,NUM_QUAD_3D)];
         const double invJ_22 = invJ[ijklNM(2,2,q,el,NUM_DIM,NUM_QUAD_3D)];

         q_gradv[ijN(0,0,3)] = ((s_gradv[ijkN(0,0,q,3)]*invJ_00) +
                                (s_gradv[ijkN(1,0,q,3)]*invJ_01) +
                                (s_gradv[ijkN(2,0,q,3)]*invJ_02));
         q_gradv[ijN(1,0,3)] = ((s_gradv[ijkN(0,0,q,3)]*invJ_10) +
                                (s_gradv[ijkN(1,0,q,3)]*invJ_11) +
                                (s_gradv[ijkN(2,0,q,3)]*invJ_12));
         q_gradv[ijN(2,0,3)] = ((s_gradv[ijkN(0,0,q,3)]*invJ_20) +
                                (s_gradv[ijkN(1,0,q,3)]*invJ_21) +
                                (s_gradv[ijkN(2,0,q,3)]*invJ_22));

         q_gradv[ijN(0,1,3)] = ((s_gradv[ijkN(0,1,q,3)]*invJ_00) +
                                (s_gradv[ijkN(1,1,q,3)]*invJ_01) +
                                (s_gradv[ijkN(2,1,q,3)]*invJ_02));
         q_gradv[ijN(1,1,3)] = ((s_gradv[ijkN(0,1,q,3)]*invJ_10) +
                                (s_gradv[ijkN(1,1,q,3)]*invJ_11) +
                                (s_gradv[ijkN(2,1,q,3)]*invJ_12));
         q_gradv[ijN(2,1,3)] = ((s_gradv[ijkN(0,1,q,3)]*invJ_20) +
                                (s_gradv[ijkN(1,1,q,3)]*invJ_21) +
                                (s_gradv[ijkN(2,1,q,3)]*invJ_22));

         q_gradv[ijN(0,2,3)] = ((s_gradv[ijkN(0,2,q,3)]*invJ_00) +
                                (s_gradv[ijkN(1,2,q,3)]*invJ_01) +
                                (s_gradv[ijkN(2,2,q,3)]*invJ_02));
         q_gradv[ijN(1,2,3)] = ((s_gradv[ijkN(0,2,q,3)]*invJ_10) +
                                (s_gradv[ijkN(1,2,q,3)]*invJ_11) +
                                (s_gradv[ijkN(2,2,q,3)]*invJ_12));
         q_gradv[ijN(2,2,3)] = ((s_gradv[ijkN(0,2,q,3)]*invJ_20) +
                                (s_gradv[ijkN(1,2,q,3)]*invJ_21) +
                                (s_gradv[ijkN(2,2,q,3)]*invJ_22));

         const double q_Jw = detJ[ijN(q,el,NUM_QUAD_3D)]*quadWeights[q];

         const double q_rho = rho0DetJ0w[ijN(q,el,NUM_QUAD_3D)] / q_Jw;
         const double q_e   = fmax(0.0,e[ijN(q,el,NUM_QUAD_3D)]);

         const double s = -(GAMMA-1.0)*q_rho*q_e;
         q_stress[ijN(0,0,3)] = s; q_stress[ijN(1,0,3)] = 0; q_stress[ijN(2,0,3)] = 0;
         q_stress[ijN(0,1,3)] = 0; q_stress[ijN(1,1,3)] = s; q_stress[ijN(2,1,3)] = 0;
         q_stress[ijN(0,2,3)] = 0; q_stress[ijN(1,2,3)] = 0; q_stress[ijN(2,2,3)] = s;

         const double gradv00 = q_gradv[ijN(0,0,3)];
         const double gradv11 = q_gradv[ijN(1,1,3)];
         const double gradv22 = q_gradv[ijN(2,2,3)];
         const double gradv10 = 0.5*(q_gradv[ijN(1,0,3)]+q_gradv[ijN(0,1,3)]);
         const double gradv20 = 0.5*(q_gradv[ijN(2,0,3)]+q_gradv[ijN(0,2,3)]);
         const double gradv21 = 0.5*(q_gradv[ijN(2,1,3)]+q_gradv[ijN(1,2,3)]);
         q_gradv[ijN(1,0,3)] = gradv10; q_gradv[ijN(2,0,3)] = gradv20;
         q_gradv[ijN(0,1,3)] = gradv10; q_gradv[ijN(2,1,3)] = gradv21;
         q_gradv[ijN(0,2,3)] = gradv20; q_gradv[ijN(1,2,3)] = gradv21;

         double minEig = 0;
         double comprDirX = 1;
         double comprDirY = 0;
         double comprDirZ = 0;

         {
            // Compute eigenvalues using quadrature formula
            const double q_ = (gradv00+gradv11+gradv22) / 3.0;
            const double gradv_q00 = (gradv00-q_);
            const double gradv_q11 = (gradv11-q_);
            const double gradv_q22 = (gradv22-q_);

            const double p1 = ((gradv10*gradv10) +
                               (gradv20*gradv20) +
                               (gradv21*gradv21));
            const double p2 = ((gradv_q00*gradv_q00) +
                               (gradv_q11*gradv_q11) +
                               (gradv_q22*gradv_q22) +
                               (2.0*p1));
            const double p    = sqrt(p2 / 6.0);
            const double pinv = 1.0 / p;
            // det(pinv*(gradv-q*I))
            const double r = (0.5*pinv*pinv*pinv *
                              ((gradv_q00*gradv_q11*gradv_q22) +
                               (2.0*gradv10*gradv21*gradv20) -
                               (gradv_q11*gradv20*gradv20) -
                               (gradv_q22*gradv10*gradv10) -
                               (gradv_q00*gradv21*gradv21)));

            double phi = 0;
            if (r <= -1.0)
            {
               phi = M_PI / 3.0;
            }
            else if (r < 1.0)
            {
               phi = acos(r) / 3.0;
            }

            minEig = q_+(2.0*p*cos(phi+(2.0*M_PI / 3.0)));
            const double eig3 = q_+(2.0*p*cos(phi));
            const double eig2 = 3.0*q_-minEig-eig3;
            double maxNorm = 0;

            for (int i = 0; i < 3; ++i)
            {
               const double x = q_gradv[i+3*0]-(i == 0)*eig3;
               const double y = q_gradv[i+3*1]-(i == 1)*eig3;
               const double z = q_gradv[i+3*2]-(i == 2)*eig3;
               const double cx = ((x*(gradv00-eig2)) +
                                  (y*gradv10) +
                                  (z*gradv20));
               const double cy = ((x*gradv10) +
                                  (y*(gradv11-eig2)) +
                                  (z*gradv21));
               const double cz = ((x*gradv20) +
                                  (y*gradv21) +
                                  (z*(gradv22-eig2)));
               const double cNorm = (cx*cx+cy*cy+cz*cz);
               //#warning 1e-16 to 1
               if ((cNorm > 1.e-16) && (maxNorm < cNorm))
               {
                  comprDirX = cx;
                  comprDirY = cy;
                  comprDirZ = cz;
                  maxNorm = cNorm;
               }
            }
            //#warning 1e-16 to 1
            if (maxNorm > 1.e-16)
            {
               const double maxNormInv = 1.0 / sqrt(maxNorm);
               comprDirX *= maxNormInv;
               comprDirY *= maxNormInv;
               comprDirZ *= maxNormInv;
            }
         }

         // Computes the initial->physical transformation Jacobian.
         const double J_00 = J[ijklNM(0,0,q,el,NUM_DIM,NUM_QUAD_3D)];
         const double J_10 = J[ijklNM(1,0,q,el,NUM_DIM,NUM_QUAD_3D)];
         const double J_20 = J[ijklNM(2,0,q,el,NUM_DIM,NUM_QUAD_3D)];
         const double J_01 = J[ijklNM(0,1,q,el,NUM_DIM,NUM_QUAD_3D)];
         const double J_11 = J[ijklNM(1,1,q,el,NUM_DIM,NUM_QUAD_3D)];
         const double J_21 = J[ijklNM(2,1,q,el,NUM_DIM,NUM_QUAD_3D)];
         const double J_02 = J[ijklNM(0,2,q,el,NUM_DIM,NUM_QUAD_3D)];
         const double J_12 = J[ijklNM(1,2,q,el,NUM_DIM,NUM_QUAD_3D)];
         const double J_22 = J[ijklNM(2,2,q,el,NUM_DIM,NUM_QUAD_3D)];

         const double invJ0_00 = invJ0[ijklNM(0,0,q,el,NUM_DIM,NUM_QUAD_3D)];
         const double invJ0_10 = invJ0[ijklNM(1,0,q,el,NUM_DIM,NUM_QUAD_3D)];
         const double invJ0_20 = invJ0[ijklNM(2,0,q,el,NUM_DIM,NUM_QUAD_3D)];
         const double invJ0_01 = invJ0[ijklNM(0,1,q,el,NUM_DIM,NUM_QUAD_3D)];
         const double invJ0_11 = invJ0[ijklNM(1,1,q,el,NUM_DIM,NUM_QUAD_3D)];
         const double invJ0_21 = invJ0[ijklNM(2,1,q,el,NUM_DIM,NUM_QUAD_3D)];
         const double invJ0_02 = invJ0[ijklNM(0,2,q,el,NUM_DIM,NUM_QUAD_3D)];
         const double invJ0_12 = invJ0[ijklNM(1,2,q,el,NUM_DIM,NUM_QUAD_3D)];
         const double invJ0_22 = invJ0[ijklNM(2,2,q,el,NUM_DIM,NUM_QUAD_3D)];

         const double Jpi_00 = ((J_00*invJ0_00)+(J_10*invJ0_01)+(J_20*invJ0_02));
         const double Jpi_10 = ((J_00*invJ0_10)+(J_10*invJ0_11)+(J_20*invJ0_12));
         const double Jpi_20 = ((J_00*invJ0_20)+(J_10*invJ0_21)+(J_20*invJ0_22));

         const double Jpi_01 = ((J_01*invJ0_00)+(J_11*invJ0_01)+(J_21*invJ0_02));
         const double Jpi_11 = ((J_01*invJ0_10)+(J_11*invJ0_11)+(J_21*invJ0_12));
         const double Jpi_21 = ((J_01*invJ0_20)+(J_11*invJ0_21)+(J_21*invJ0_22));

         const double Jpi_02 = ((J_02*invJ0_00)+(J_12*invJ0_01)+(J_22*invJ0_02));
         const double Jpi_12 = ((J_02*invJ0_10)+(J_12*invJ0_11)+(J_22*invJ0_12));
         const double Jpi_22 = ((J_02*invJ0_20)+(J_12*invJ0_21)+(J_22*invJ0_22));

         const double physDirX = ((Jpi_00*comprDirX)+(Jpi_10*comprDirY)+
                                  (Jpi_20*comprDirZ));
         const double physDirY = ((Jpi_01*comprDirX)+(Jpi_11*comprDirY)+
                                  (Jpi_21*comprDirZ));
         const double physDirZ = ((Jpi_02*comprDirX)+(Jpi_12*comprDirY)+
                                  (Jpi_22*comprDirZ));

         const double q_h = H0*sqrt((physDirX*physDirX)+
                                    (physDirY*physDirY)+
                                    (physDirZ*physDirZ));

         const double soundSpeed = sqrt(GAMMA*(GAMMA-1.0)*q_e);
         dtEst[ijN(q,el,NUM_QUAD_3D)] = CFL*q_h / soundSpeed;

         if (USE_VISCOSITY)
         {
            // TODO: Check how we can extract outside of kernel
            const double mu = minEig;
            double coeff = 2.0*q_rho*q_h*q_h*fabs(mu);
            if (mu < 0)
            {
               coeff += 0.5*q_rho*q_h*soundSpeed;
            }
            for (int y = 0; y < 3; ++y)
            {
               for (int x = 0; x < 3; ++x)
               {
                  q_stress[ijN(x,y,3)] += coeff*q_gradv[ijN(x,y,3)];
               }
            }
         }

         const double S00 = q_stress[ijN(0,0,3)];
         const double S10 = q_stress[ijN(1,0,3)];
         const double S20 = q_stress[ijN(2,0,3)];
         const double S01 = q_stress[ijN(0,1,3)];
         const double S11 = q_stress[ijN(1,1,3)];
         const double S21 = q_stress[ijN(2,1,3)];
         const double S02 = q_stress[ijN(0,2,3)];
         const double S12 = q_stress[ijN(1,2,3)];
         const double S22 = q_stress[ijN(2,2,3)];

         stressJinvT[ijklNM(0,0,q,el,NUM_DIM,
                            NUM_QUAD_3D)] = q_Jw*((S00*invJ_00)+(S10*invJ_01)+(S20*invJ_02));
         stressJinvT[ijklNM(1,0,q,el,NUM_DIM,
                            NUM_QUAD_3D)] = q_Jw*((S00*invJ_10)+(S10*invJ_11)+(S20*invJ_12));
         stressJinvT[ijklNM(2,0,q,el,NUM_DIM,
                            NUM_QUAD_3D)] = q_Jw*((S00*invJ_20)+(S10*invJ_21)+(S20*invJ_22));

         stressJinvT[ijklNM(0,1,q,el,NUM_DIM,
                            NUM_QUAD_3D)] = q_Jw*((S01*invJ_00)+(S11*invJ_01)+(S21*invJ_02));
         stressJinvT[ijklNM(1,1,q,el,NUM_DIM,
                            NUM_QUAD_3D)] = q_Jw*((S01*invJ_10)+(S11*invJ_11)+(S21*invJ_12));
         stressJinvT[ijklNM(2,1,q,el,NUM_DIM,
                            NUM_QUAD_3D)] = q_Jw*((S01*invJ_20)+(S11*invJ_21)+(S21*invJ_22));

         stressJinvT[ijklNM(0,2,q,el,NUM_DIM,
                            NUM_QUAD_3D)] = q_Jw*((S02*invJ_00)+(S12*invJ_01)+(S22*invJ_02));
         stressJinvT[ijklNM(1,2,q,el,NUM_DIM,
                            NUM_QUAD_3D)] = q_Jw*((S02*invJ_10)+(S12*invJ_11)+(S22*invJ_12));
         stressJinvT[ijklNM(2,2,q,el,NUM_DIM,
                            NUM_QUAD_3D)] = q_Jw*((S02*invJ_20)+(S12*invJ_21)+(S22*invJ_22));
      }
      __syncthreads();
   }
}


// *****************************************************************************
typedef void (*fUpdateQuadratureData)(const double GAMMA,
                                      const double H0,
                                      const int h1order,
                                      const double CFL,
                                      const double infinity,
                                      const bool USE_VISCOSITY,
                                      const int numElements,
                                      const double* restrict dofToQuad,
                                      const double* restrict dofToQuadD,
                                      const double* restrict quadWeights,
                                      const double* restrict v,
                                      const double* restrict e,
                                      const double* restrict rho0DetJ0w,
                                      const double* restrict invJ0,
                                      const double* restrict J,
                                      const double* restrict invJ,
                                      const double* restrict detJ,
                                      double* restrict stressJinvT,
                                      double* restrict dtEst);

// *****************************************************************************
void rUpdateQuadratureData(const double GAMMA,
                           const double H0,
                           const int h1order,
                           const double CFL,
                           const double infinity,
                           const bool USE_VISCOSITY,
                           const int NUM_DIM,
                           const int NUM_QUAD,
                           const int NUM_QUAD_1D,
                           const int NUM_DOFS_1D,
                           const int nzones,
                           const double* restrict dofToQuad,
                           const double* restrict dofToQuadD,
                           const double* restrict quadWeights,
                           const double* restrict v,
                           const double* restrict e,
                           const double* restrict rho0DetJ0w,
                           const double* restrict invJ0,
                           const double* restrict J,
                           const double* restrict invJ,
                           const double* restrict detJ,
                           double* restrict stressJinvT,
                           double* restrict dtEst)
{
   assert(LOG2(NUM_DIM)<=4);
   assert(LOG2(NUM_DOFS_1D-2)<=4);
   assert(NUM_QUAD_1D==2*(NUM_DOFS_1D-1));
   assert(IROOT(NUM_DIM,NUM_QUAD)==NUM_QUAD_1D);
   const unsigned int id = (NUM_DIM<<4)|(NUM_DOFS_1D-2);
   static std::unordered_map<unsigned int, fUpdateQuadratureData> call =
   {
      // 2D
      // {0x20,&rUpdateQuadratureData2D<2,2*2,2,2>},
      // {0x21,&rUpdateQuadratureData2D<2,4*4,4,3>},
      // {0x22,&rUpdateQuadratureData2D<2,6*6,6,4>},
      // {0x23,&rUpdateQuadratureData2D<2,8*8,8,5>},
      // {0x24,&rUpdateQuadratureData2D<2,10*10,10,6>},
      // {0x25,&rUpdateQuadratureData2D<2,12*12,12,7>},
      // {0x26,&rUpdateQuadratureData2D<2,14*14,14,8>},
      // {0x27,&rUpdateQuadratureData2D<2,16*16,16,9>},
      // {0x28,&rUpdateQuadratureData2D<2,18*18,18,10>},
      // {0x29,&rUpdateQuadratureData2D<2,20*20,20,11>},
      // {0x2A,&rUpdateQuadratureData2D<2,22*22,22,12>},
      // {0x2B,&rUpdateQuadratureData2D<2,24*24,24,13>},
      // {0x2C,&rUpdateQuadratureData2D<2,26*26,26,14>},
      // {0x2D,&rUpdateQuadratureData2D<2,28*28,28,15>},
      // {0x2E,&rUpdateQuadratureData2D<2,30*30,30,16>},
      // {0x2F,&rUpdateQuadratureData2D<2,32*32,32,17>},
      // 3D
      // {0x30,&rUpdateQuadratureData3D<3,2*2*2,2,2>},
      // {0x31,&rUpdateQuadratureData3D<3,4*4*4,4,3>},
      // {0x32,&rUpdateQuadratureData3D<3,6*6*6,6,4>},
      // {0x33,&rUpdateQuadratureData3D<3,8*8*8,8,5>},
      // {0x34,&rUpdateQuadratureData3D<3,10*10*10,10,6>},
      // {0x35,&rUpdateQuadratureData3D<3,12*12*12,12,7>},
      // {0x36,&rUpdateQuadratureData3D<3,14*14*14,14,8>},
      // {0x37,&rUpdateQuadratureData3D<3,16*16*16,16,9>},
      // {0x38,&rUpdateQuadratureData3D<3,18*18*18,18,10>},
      // {0x39,&rUpdateQuadratureData3D<3,20*20*20,20,11>},
      // {0x3A,&rUpdateQuadratureData3D<3,22*22*22,22,12>},
      // {0x3B,&rUpdateQuadratureData3D<3,24*24*24,24,13>},
      // {0x3C,&rUpdateQuadratureData3D<3,26*26*26,26,14>},
      // {0x3D,&rUpdateQuadratureData3D<3,28*28*28,28,15>},
      // {0x3E,&rUpdateQuadratureData3D<3,30*30*30,30,16>},
      // {0x3F,&rUpdateQuadratureData3D<3,32*32*32,32,17>},
   };

#define call_2d(DOFS,QUAD,BZ,NBLOCK)      \
   call_2d_ker(rUpdateQuadratureData2D,nzones,DOFS,QUAD,BZ,NBLOCK,\
           GAMMA,H0,h1order,CFL,infinity,USE_VISCOSITY, \
           nzones,dofToQuad,dofToQuadD,quadWeights, \
           v,e,rho0DetJ0w,invJ0,J,invJ,detJ, \
           stressJinvT,dtEst)
#define call_3d(DOFS,QUAD,BZ,NBLOCK) \
   call_3d_ker(rUpdateQuadratureData3D,nzones,DOFS,QUAD,BZ,NBLOCK,\
               GAMMA,H0,CFL,USE_VISCOSITY,                        \
               nzones,dofToQuad,dofToQuadD,quadWeights,           \
               v,e,rho0DetJ0w,invJ0,J,invJ,detJ,                  \
               stressJinvT,dtEst,gbuf,rUpdateQuadratureData3D_BufSize)

   // 2D
   if      (id == 0x20) { call_2d(2 ,2,16,1); }
   else if (id == 0x21) { call_2d(3 ,4 ,8,1); }
   else if (id == 0x22) { call_2d(4 ,6 ,4,1); }
   else if (id == 0x23) { call_2d(5 ,8 ,2,1); }
   else if (id == 0x24) { call_2d(6 ,10,1,1); }
   else if (id == 0x25) { call_2d(7 ,12,1,1); }
   else if (id == 0x26) { call_2d(8 ,14,1,1); }
   else if (id == 0x27) { call_2d(9 ,16,1,1); }
   else if (id == 0x28) { call_2d(10,18,1,1); }
   else if (id == 0x29) { call_2d(11,20,1,1); }
   else if (id == 0x2A) { call_2d(12,22,1,1); }
   else if (id == 0x2B) { call_2d(13,24,1,1); }
   else if (id == 0x2C) { call_2d(14,26,1,1); }
   else if (id == 0x2D) { call_2d(15,28,1,1); }
   else if (id == 0x2E) { call_2d(16,30,1,1); }
   else if (id == 0x2F) { call_2d(17,32,1,1); }
   // 3D
   /*
   else if (id == 0x30) { call_3d(2 ,2 ,2,1); }
   else if (id == 0x31) { call_3d(3 ,4 ,4,1); }
   else if (id == 0x32) { call_3d(4 ,6 ,6,1); }
   else if (id == 0x33) { call_3d(5 ,8 ,8,1); }
   else if (id == 0x34) { call_3d(6 ,10,2,1); }
   else if (id == 0x35) { call_3d(7 ,12,2,1); }
   else if (id == 0x36) { call_3d(8 ,14,2,1); }
   else if (id == 0x37) { call_3d(9 ,16,2,1); }
   else if (id == 0x38) { call_3d(10,18,2,1); }
   else if (id == 0x39) { call_3d(11,20,2,1); }
   else if (id == 0x3A) { call_3d(12,22,2,1); }
   else if (id == 0x3B) { call_3d(13,24,1,1); }
   else if (id == 0x3C) { call_3d(14,26,1,1); }
   else if (id == 0x3D) { call_3d(15,28,1,1); }
   else if (id == 0x3E) { call_3d(16,30,1,1); }
   else if (id == 0x3F) { call_3d(17,32,1,1); }
   */
   else
   {
      const int blck = CUDA_BLOCK_SIZE;
      const int grid = (nzones+blck-1)/blck;
      if (!call[id])
      {
         printf("\n[rUpdateQuadratureData] id \033[33m0x%X\033[m ",id);
         fflush(stdout);
      }
      assert(call[id]);
      call0(id,grid,blck,
            GAMMA,H0,h1order,CFL,infinity,USE_VISCOSITY,
            nzones,dofToQuad,dofToQuadD,quadWeights,
            v,e,rho0DetJ0w,invJ0,J,invJ,detJ,
            stressJinvT,dtEst);
   }
   CUCHK(cudaGetLastError());
}
