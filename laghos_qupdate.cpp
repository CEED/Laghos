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

#include "laghos_qupdate.hpp"
#include "laghos_solver.hpp"
#include "linalg/dtensor.hpp"
#include "general/forall.hpp"
#include <unordered_map>

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{

// *****************************************************************************
namespace kernels
{
namespace vector
{
double Min(const int, const double*);
}
}

namespace hydrodynamics
{

// *****************************************************************************
// * Dense matrix
// *****************************************************************************
MFEM_ATTR_HOST_DEVICE static
void multABt(const int ah,
             const int aw,
             const int bh,
             const double* __restrict__ A,
             const double* __restrict__ B,
             double* __restrict__ C)
{
   const int ah_x_bh = ah*bh;
   for (int i=0; i<ah_x_bh; i+=1)
   {
      C[i] = 0.0;
   }
   for (int k=0; k<aw; k+=1)
   {
      double *c = C;
      for (int j=0; j<bh; j+=1)
      {
         const double bjk = B[j];
         for (int i=0; i<ah; i+=1)
         {
            c[i] += A[i] * bjk;
         }
         c += ah;
      }
      A += ah;
      B += bh;
   }
}

// *****************************************************************************
MFEM_ATTR_HOST_DEVICE static
void mult(const int ah,
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
MFEM_ATTR_HOST_DEVICE static
void multV(const int height,
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
MFEM_ATTR_HOST_DEVICE static
void add(const int height, const int width,
         const double c, const double *A,
         double *D)
{
   for (int j = 0; j < width; j++)
   {
      for (int i = 0; i < height; i++)
      {
         D[i*width+j] += c * A[i*width+j];
      }
   }
}

// *****************************************************************************
// * Eigen
// *****************************************************************************
MFEM_ATTR_HOST_DEVICE  static
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
MFEM_ATTR_HOST_DEVICE static
inline double det2D(const double *d)
{
   return d[0] * d[3] - d[1] * d[2];
}

// *****************************************************************************
MFEM_ATTR_HOST_DEVICE static
inline double det3D(const double *d)
{
   return
      d[0] * (d[4] * d[8] - d[5] * d[7]) +
      d[3] * (d[2] * d[7] - d[1] * d[8]) +
      d[6] * (d[1] * d[5] - d[2] * d[4]);
}

// *****************************************************************************
MFEM_ATTR_HOST_DEVICE static
double det(const int dim, const double *J)
{
   if (dim==2) { return det2D(J); }
   if (dim==3) { return det3D(J); }
#ifndef MFEM_USE_CUDA
   MFEM_ASSERT(false, "dim!=2 && dim!=3");
#endif
   return 0.0;
}

// *****************************************************************************
MFEM_ATTR_HOST_DEVICE static
void calcInverse2D(const int n, const double *a, double *i)
{
   const double d = det2D(a);
   const double t = 1.0 / d;
   i[0*n+0] =  a[1*n+1] * t ;
   i[0*n+1] = -a[0*n+1] * t ;
   i[1*n+0] = -a[1*n+0] * t ;
   i[1*n+1] =  a[0*n+0] * t ;
}

// *****************************************************************************
MFEM_ATTR_HOST_DEVICE static
void symmetrize(const int n, double* __restrict__ d)
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
MFEM_ATTR_HOST_DEVICE static
inline double cpysign(const double x, const double y)
{
   if ((x < 0 && y > 0) || (x > 0 && y < 0))
   {
      return -x;
   }
   return x;
}

// *****************************************************************************
MFEM_ATTR_HOST_DEVICE static
inline void eigensystem2S(const double &d12, double &d1, double &d2,
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
MFEM_ATTR_HOST_DEVICE static
void calcEigenvalues(const int n, const double *d,
                     double *lambda,
                     double *vec)
{
#ifndef MFEM_USE_CUDA
   MFEM_ASSERT(n==2, "n!=2");
#endif
   double d0 = d[0];
   double d2 = d[2]; // use the upper triangular entry
   double d3 = d[3];
   double c, s;
   eigensystem2S(d2, d0, d3, c, s);
   if (d0 <= d3)
   {
      lambda[0] = d0;
      lambda[1] = d3;
      vec[0] =  c;
      vec[1] = -s;
      vec[2] =  s;
      vec[3] =  c;
   }
   else
   {
      lambda[0] = d3;
      lambda[1] = d0;
      vec[0] =  s;
      vec[1] =  c;
      vec[2] =  c;
      vec[3] = -s;
   }
}

// *****************************************************************************
MFEM_ATTR_HOST_DEVICE static
inline void getScalingFactor(const double &d_max, double &mult)
{
   int d_exp;
   if (d_max > 0.)
   {
      mult = frexp(d_max, &d_exp);
      if (d_exp == numeric_limits<double>::max_exponent)
      {
         mult *= numeric_limits<double>::radix;
      }
      mult = d_max/mult;
   }
   else
   {
      mult = 1.;
   }
   // mult = 2^d_exp is such that d_max/mult is in [0.5,1)
   // or in other words d_max is in the interval [0.5,1)*mult
}

// *****************************************************************************
MFEM_ATTR_HOST_DEVICE static
double calcSingularvalue(const int n, const int i, const double *d)
{
#ifndef MFEM_USE_CUDA
   MFEM_ASSERT(n==2, "n!=2");
#endif

   double d0, d1, d2, d3;
   d0 = d[0];
   d1 = d[1];
   d2 = d[2];
   d3 = d[3];
   double mult;

   {
      double d_max = fabs(d0);
      if (d_max < fabs(d1)) { d_max = fabs(d1); }
      if (d_max < fabs(d2)) { d_max = fabs(d2); }
      if (d_max < fabs(d3)) { d_max = fabs(d3); }

      getScalingFactor(d_max, mult);
   }

   d0 /= mult;
   d1 /= mult;
   d2 /= mult;
   d3 /= mult;

   double t = 0.5*((d0+d2)*(d0-d2)+(d1-d3)*(d1+d3));
   double s = d0*d2 + d1*d3;
   s = sqrt(0.5*(d0*d0 + d1*d1 + d2*d2 + d3*d3) + sqrt(t*t + s*s));

   if (s == 0.0)
   {
      return 0.0;
   }
   t = fabs(d0*d3 - d1*d2) / s;
   if (t > s)
   {
      if (i == 0)
      {
         return t*mult;
      }
      return s*mult;
   }
   if (i == 0)
   {
      return s*mult;
   }
   return t*mult;
}

// *****************************************************************************
// * Smooth transition between 0 and 1 for x in [-eps, eps].
// *****************************************************************************
MFEM_ATTR_HOST_DEVICE static
inline double smooth_step_01(const double x, const double eps)
{
   const double y = (x + eps) / (2.0 * eps);
   if (y < 0.0) { return 0.0; }
   if (y > 1.0) { return 1.0; }
   return (3.0 - 2.0 * y) * y * y;
}

// *****************************************************************************
// * qupdate
// *****************************************************************************
template<const int dim> static
void qupdate(const int nzones,
             const int nqp,
             const int nqp1D,
             const double gamma,
             const bool use_viscosity,
             const double h0,
             const double h1order,
             const double cfl,
             const double infinity,
             const Array<double> &weights,
             const Vector &Jacobians,
             const Vector &rho0DetJ0w,
             const Vector &e_quads,
             const Vector &grad_v_ext,
             const DenseTensor &Jac0inv,
             Vector &dt_est,
             DenseTensor &stressJinvT)
{
   auto d_weights = weights.ReadAccess();
   auto d_Jacobians = Jacobians.ReadAccess();
   auto d_rho0DetJ0w = rho0DetJ0w.ReadAccess();
   auto d_e_quads = e_quads.ReadAccess();
   auto d_grad_v_ext = grad_v_ext.ReadAccess();
   auto d_Jac0inv = ReadAccess(Jac0inv.GetMemory(), Jac0inv.TotalSize());
   auto d_dt_est = dt_est.ReadWriteAccess();
   auto d_stressJinvT = WriteAccess(stressJinvT.GetMemory(),
                                    stressJinvT.TotalSize());
   MFEM_FORALL(z, nzones,
   {
      double min_detJ = infinity;
      double Jinv[dim*dim];
      double stress[dim*dim];
      double sgrad_v[dim*dim];
      double eig_val_data[3];
      double eig_vec_data[9];
      double compr_dir[dim];
      double Jpi[dim*dim];
      double ph_dir[dim];
      double stressJiT[dim*dim];
      // ********************************************************************
      for (int q = 0; q < nqp; q++)   // this for-loop should be kernel'd too
      {
         const int zdx = z * nqp + q;
         //const int zdx_ = z*nqp*dim*dim + q;
         const int zdx1 = z*nqp*dim + q;
         const double weight =  d_weights[q];
         const double inv_weight = 1. / weight;
         const double *_J = d_Jacobians + (q+nqp*dim*dim*z);
         const double J[4] = {_J[0], _J[nqp], _J[2*nqp], _J[3*nqp]};
         const double detJ = det(dim,J);
         min_detJ = fmin(min_detJ,detJ);
         calcInverse2D(dim,J,Jinv);
         // *****************************************************************
         const double rho = inv_weight * d_rho0DetJ0w[zdx] / detJ;
         const double e   = fmax(0.0, d_e_quads[zdx]);
         const double p  = (gamma - 1.0) * rho * e;
         const double sound_speed = sqrt(gamma * (gamma-1.0) * e);
         // *****************************************************************
         for (int k = 0; k < dim*dim; k+=1) { stress[k] = 0.0; }
         for (int d = 0; d < dim; d++) { stress[d*dim+d] = -p; }
         // *****************************************************************
         double visc_coeff = 0.0;
         if (use_viscosity)
         {
            // Compression-based length scale at the point. The first
            // eigenvector of the symmetric velocity gradient gives the
            // direction of maximal compression. This is used to define the
            // relative change of the initial length scale.
            const double *_dV = d_grad_v_ext + (q+nqp*dim*dim*z);//+ zdx*dim*dim;
            const double dV[4] = {_dV[0], _dV[nqp], _dV[2*nqp], _dV[3*nqp]};
            mult(dim,dim,dim, dV, Jinv, sgrad_v);
            symmetrize(dim,sgrad_v);
            if (dim==1)
            {
               eig_val_data[0] = sgrad_v[0];
               eig_vec_data[0] = 1.;
            }
            else
            {
               calcEigenvalues(dim, sgrad_v, eig_val_data, eig_vec_data);
            }
            for (int k=0; k<dim; k+=1) { compr_dir[k]=eig_vec_data[k]; }
            // Computes the initial->physical transformation Jacobian.
            mult(dim,dim,dim, J, d_Jac0inv+zdx*dim*dim, Jpi);
            multV(dim, dim, Jpi, compr_dir, ph_dir);
            // Change of the initial mesh size in the compression direction.
            const double h = h0 * norml2(dim,ph_dir) / norml2(dim,compr_dir);
            // Measure of maximal compression.
            const double mu = eig_val_data[0];
            visc_coeff = 2.0 * rho * h * h * fabs(mu);
            // The following represents a "smooth" version of the statement
            // "if (mu < 0) visc_coeff += 0.5 rho h sound_speed".  Note that
            // eps must be scaled appropriately if a different unit system is
            // being used.
            const double eps = 1e-12;
            visc_coeff += 0.5 * rho * h * sound_speed *
                          (1.0 - smooth_step_01(mu - 2.0 * eps, eps));
            //if (mu < 0.0) { visc_coeff += 0.5 * rho * h * sound_speed; }
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
         const double inv_dt = sound_speed * inv_h_min
                               + 2.5 * visc_coeff * inv_rho_inv_h_min_sq;
         if (min_detJ < 0.0)
         {
            // This will force repetition of the step with smaller dt.
            d_dt_est[z] = 0.0;
         }
         else
         {
            if (inv_dt>0.0)
            {
               const double cfl_inv_dt = cfl / inv_dt;
               d_dt_est[z] = fmin(d_dt_est[z], cfl_inv_dt);
            }
         }
         // Quadrature data for partial assembly of the force operator.
         multABt(dim, dim, dim, stress, Jinv, stressJiT);
         for (int k=0; k<dim*dim; k+=1) { stressJiT[k] *= weight * detJ; }
         for (int vd = 0 ; vd < dim; vd++)
         {
            for (int gd = 0; gd < dim; gd++)
            {
               const int offset = zdx + nqp*nzones*(gd+vd*dim);
               d_stressJinvT[offset] = stressJiT[vd+gd*dim];
            }
         }
      }
   });
}

// *****************************************************************************
QUpdate::~QUpdate()
{
   delete l2_ElemRestrict;
   delete h1_ElemRestrict;
   // delete l2_maps; // FIXME: can lead to double delete
   // delete h1_maps; // FIXME: can lead to double delete
}

// *****************************************************************************
QUpdate::QUpdate(const int _dim,
                 const int _nzones,
                 const int _l2dofs_cnt,
                 const int _h1dofs_cnt,
                 const bool _use_viscosity,
                 const bool _p_assembly,
                 const double _cfl,
                 const double _gamma,
                 TimingData *_timer,
                 Coefficient *_material_pcf,
                 const IntegrationRule &_ir,
                 ParFiniteElementSpace &_H1FESpace,
                 ParFiniteElementSpace &_L2FESpace):
   dim(_dim),
   nzones(_nzones),
   l2dofs_cnt(_l2dofs_cnt),
   h1dofs_cnt(_h1dofs_cnt),
   use_viscosity(_use_viscosity),
   p_assembly(_p_assembly),
   cfl(_cfl),
   gamma(_gamma),
   timer(_timer),
   material_pcf(_material_pcf),
   ir(_ir),
   H1FESpace(_H1FESpace),
   L2FESpace(_L2FESpace),
   h1_maps(&H1FESpace.GetFE(0)->GetDofToQuad(ir, DofToQuad::TENSOR)),
   l2_maps(&L2FESpace.GetFE(0)->GetDofToQuad(ir, DofToQuad::TENSOR)),
   h1_ElemRestrict(H1FESpace.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC)),
   l2_ElemRestrict(L2FESpace.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC)),
   //d_e_quads_data(NULL),
   //d_grad_x_data(NULL),
   //d_grad_v_data(NULL),
   nqp(ir.GetNPoints())
{
   MFEM_ASSERT(material_pcf, "!material_pcf");
}

// **************************************************************************
template<const int VDIM,
         const int D1D,
         const int Q1D> static
void vecToQuad2D(const int NE,
                 const Array<double> &_B,
                 const Vector &_x,
                 Vector &_y)
{
   auto B = Reshape(_B.ReadAccess(), Q1D,D1D);
   auto x = Reshape(_x.ReadAccess(), D1D,D1D,VDIM,NE);
   auto y = Reshape(_y.WriteAccess(), Q1D,Q1D,VDIM,NE);
   MFEM_FORALL(e, NE,
   {
      double out_xy[VDIM][Q1D][Q1D];
      for (int v = 0; v < VDIM; ++v)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               out_xy[v][qy][qx] = 0.0;
            }
         }
      }
      for (int dy = 0; dy < D1D; ++dy)
      {
         double out_x[VDIM][Q1D];
         for (int v = 0; v < VDIM; ++v)
         {
            for (int qy = 0; qy < Q1D; ++qy)
            {
               out_x[v][qy] = 0.0;
            }
         }
         for (int dx = 0; dx < D1D; ++dx)
         {
            for (int v = 0; v < VDIM; ++v)
            {
               const double r_gf = x(dx,dy,v,e);
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  out_x[v][qy] += r_gf * B(qy,dx);
               }
            }
         }
         for (int v = 0; v < VDIM; ++v)
         {
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const double d2q = B(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  out_xy[v][qy][qx] += d2q * out_x[v][qx];
               }
            }
         }
      }
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int v = 0; v < VDIM; ++v)
            {
               y(qx,qy,v,e) = out_xy[v][qy][qx];
            }
         }
      }
   });
}

// *****************************************************************************
typedef void (*fVecToQuad2D)(const int E,
                             const Array<double> &dofToQuad,
                             const Vector &in,
                             Vector &out);

// ***************************************************************************
static void Dof2QuadScalar(const Operator *erestrict,
                           const FiniteElementSpace &fes,
                           const DofToQuad *maps,
                           const IntegrationRule& ir,
                           const Vector &d_in,
                           Vector &d_out)
{
   const int dim = fes.GetMesh()->Dimension();
   MFEM_ASSERT(dim==2, "dim!=2");
   const int vdim = fes.GetVDim();
   // const int vsize = fes.GetVSize(); // not used
   const mfem::FiniteElement& fe = *fes.GetFE(0);
   const int numDofs  = fe.GetDof();
   const int nzones = fes.GetNE();
   const int nqp = ir.GetNPoints();
//   const int local_size = numDofs * nzones;
   const int out_size =  nqp * nzones;
   const int dofs1D = fes.GetFE(0)->GetOrder() + 1;
   const int quad1D = IntRules.Get(Geometry::SEGMENT,ir.GetOrder()).GetNPoints();
   //Vector v_local_in(local_size);
   //erestrict->Mult(d_in,v_local_in);
   d_out.SetSize(out_size);
   /*
   if (!(*d_out))
   {
      *d_out = (double*) mfem::New<double>(out_size);
      }*/
   MFEM_ASSERT(vdim==1, "vdim!=1");
   const int id = (vdim<<8)|(dofs1D<<4)|(quad1D);
   static std::unordered_map<unsigned int, fVecToQuad2D> call =
   {
      {0x124,&vecToQuad2D<1,2,4>},
      {0x134,&vecToQuad2D<1,3,4>},
      {0x135,&vecToQuad2D<1,3,5>},
      {0x136,&vecToQuad2D<1,3,6>},
      {0x148,&vecToQuad2D<1,4,8>},
   };
   if (!call[id])
   {
      printf("\n[Dof2QuadScalar] id \033[33m0x%X\033[m ",id);
      fflush(0);
   }
   call[id](nzones, maps->B, d_in, *d_out);
}

// **************************************************************************
template <const int D1D,
          const int Q1D> static
void qGradVector2D(const int NE,
                   const Array<double> &_B,
                   const Array<double> &_G,
                   const Vector &_x,
                   Vector &_y)
{
   auto B = Reshape(_B.ReadAccess(), Q1D,D1D);
   auto G = Reshape(_G.ReadAccess(), Q1D,D1D);
   auto x = Reshape(_x.ReadAccess(), D1D,D1D,2,NE);
   auto y = Reshape(_y.WriteAccess(), Q1D,Q1D,2,2,NE);
   MFEM_FORALL(e, NE,
   {
      double s_gradv[2][2][Q1D][Q1D];
      for (int qx = 0; qx < Q1D; ++qx)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            s_gradv[0][0][qx][qy] = 0.0;
            s_gradv[0][1][qx][qy] = 0.0;
            s_gradv[1][0][qx][qy] = 0.0;
            s_gradv[1][1][qx][qy] = 0.0;
         }
      }
      for (int dy = 0; dy < D1D; ++dy)
      {
         double vDx[2][Q1D];
         double vx[2][Q1D];
         for (int qx = 0; qx < Q1D; ++qx)
         {
            for (int c = 0; c < 2; ++c)
            {
               vDx[c][qx] = 0.0;
               vx[c][qx] = 0.0;
            }
         }
         for (int dx = 0; dx < D1D; ++dx)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double wDx = G(qx,dx);
               const double wx  = B(qx,dx);
               for (int c = 0; c < 2; ++c)
               {
                  const double input = x(dx,dy,c,e);
                  vDx[c][qx] += input * wDx;
                  vx[c][qx] += input * wx;
               }
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            const double vy  = B(qy,dy);
            const double vDy = G(qy,dy);
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int c = 0; c < 2; ++c)
               {
                  s_gradv[c][0][qx][qy] += vy*vDx[c][qx];
                  s_gradv[c][1][qx][qy] += vDy*vx[c][qx];
               }
            }
         }
      }
      for (int qx = 0; qx < Q1D; ++qx)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            y(qx,qy,0,0,e) = s_gradv[0][0][qx][qy];
            y(qx,qy,1,0,e) = s_gradv[1][0][qx][qy];
            y(qx,qy,0,1,e) = s_gradv[0][1][qx][qy];
            y(qx,qy,1,1,e) = s_gradv[1][1][qx][qy];
         }
      }
   });
}

// *****************************************************************************
typedef void (*fGradVector2D)(const int E,
                              const Array<double> &dofToQuad,
                              const Array<double> &dofToQuadD,
                              const Vector &in,
                              Vector &out);

// **************************************************************************
static void Dof2QuadGrad(const Operator *erestrict,
                         const FiniteElementSpace &fes,
                         const DofToQuad *maps,
                         const IntegrationRule& ir,
                         const Vector &d_in,
                         Vector &d_out)
{
   const int dim = fes.GetMesh()->Dimension();
   MFEM_ASSERT(dim==2, "dim!=2");
   const int vdim = fes.GetVDim();
   MFEM_ASSERT(vdim==2, "vdim!=2");
   // const int vsize = fes.GetVSize(); // not used
   const mfem::FiniteElement& fe = *fes.GetFE(0);
   const int numDofs  = fe.GetDof();
   const int nzones = fes.GetNE();
   const int nqp = ir.GetNPoints();
   const int local_size = vdim * numDofs * nzones;
   const int out_size = vdim * vdim * nqp * nzones;
   const int dofs1D = fes.GetFE(0)->GetOrder() + 1;
   const int quad1D = IntRules.Get(Geometry::SEGMENT,ir.GetOrder()).GetNPoints();
   Vector v_local_in(local_size);
   erestrict->Mult(d_in, v_local_in);
   d_out.SetSize(out_size);
   const int id = (dofs1D<<4)|(quad1D);
   static std::unordered_map<unsigned int, fGradVector2D> call =
   {
      {0x34,&qGradVector2D<3,4>},
      {0x44,&qGradVector2D<4,4>},
      {0x45,&qGradVector2D<4,5>},
      {0x46,&qGradVector2D<4,6>},
      {0x58,&qGradVector2D<5,8>},
   };
   if (!call[id])
   {
      printf("\n[Dof2QuadGrad] id \033[33m0x%X\033[m ",id);
      fflush(0);
   }
   call[id](nzones,
            maps->B,
            maps->G,
            v_local_in,
            d_out);
}

// *****************************************************************************
// * QUpdate UpdateQuadratureData kernel
// *****************************************************************************
void QUpdate::UpdateQuadratureData(const Vector &S,
                                   bool &quad_data_is_current,
                                   QuadratureData &quad_data,
                                   const Tensors1D *tensors1D)
{
   // **************************************************************************
   if (quad_data_is_current) { return; }

   // **************************************************************************
   timer->sw_qdata.Start();
   Vector* S_p = (Vector*) &S;

   // **************************************************************************
   const int H1_size = H1FESpace.GetVSize();
   const int nqp1D = tensors1D->LQshape1D.Width();

   // Energy dof => quads ******************************************************
   ParGridFunction d_e;
   d_e.MakeRef(&L2FESpace, *S_p, 2*H1_size);
   Dof2QuadScalar(l2_ElemRestrict, L2FESpace, l2_maps, ir, d_e, d_e_quads_data);

   // Coords to Jacobians ******************************************************
   ParGridFunction d_x;
   d_x.MakeRef(&H1FESpace,*S_p, 0);
   Dof2QuadGrad(h1_ElemRestrict, H1FESpace, h1_maps, ir, d_x, d_grad_x_data);

   // Velocity *****************************************************************
   ParGridFunction d_v;
   d_v.MakeRef(&H1FESpace,*S_p, H1_size);
   Dof2QuadGrad(h1_ElemRestrict, H1FESpace, h1_maps, ir, d_v, d_grad_v_data);

   // **************************************************************************
   const double h1order = (double) H1FESpace.GetOrder(0);
   const double infinity = std::numeric_limits<double>::infinity();

   // **************************************************************************
   const int dt_est_sz = nzones;
   Vector d_dt_est(dt_est_sz);
   d_dt_est = quad_data.dt_est;

   // **************************************************************************
   MFEM_VERIFY(dim==2, "Only UpdateQuadratureData with dim==2 is supported");
   qupdate<2>(nzones,
              nqp,
              nqp1D,
              gamma,
              use_viscosity,
              quad_data.h0,
              h1order,
              cfl,
              infinity,
              ir.GetWeights(),//h1_maps->W,
              d_grad_x_data,
              quad_data.rho0DetJ0w,
              d_e_quads_data,
              d_grad_v_data,
              quad_data.Jac0inv,
              d_dt_est,
              quad_data.stressJinvT);

   // **************************************************************************
   quad_data.dt_est = d_dt_est.Min();
   quad_data_is_current = true;
   timer->sw_qdata.Stop();
   timer->quad_tstep += nzones;
}

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI
