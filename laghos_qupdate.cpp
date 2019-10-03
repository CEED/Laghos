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
MFEM_HOST_DEVICE static inline
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
MFEM_HOST_DEVICE static inline
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
MFEM_HOST_DEVICE static inline
void multV(const int height,
           const int width,
           double* __restrict__ data,
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
MFEM_HOST_DEVICE static inline
void add(const int height, const int width, const double c,
         const double* __restrict__ A, double* __restrict__ D)
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
MFEM_HOST_DEVICE static inline
double norml2(const int size, const double* __restrict__ data)
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
template<int dim> static double det(const double *d);

// *****************************************************************************
template<> MFEM_HOST_DEVICE inline
double det<2>(const double* __restrict__ d)
{
   return d[0] * d[3] - d[1] * d[2];
}

// *****************************************************************************
template<> MFEM_HOST_DEVICE inline
double det<3>(const double* __restrict__ d)
{
   return d[0] * (d[4] * d[8] - d[5] * d[7]) +
          d[3] * (d[2] * d[7] - d[1] * d[8]) +
          d[6] * (d[1] * d[5] - d[2] * d[4]);
}

// *****************************************************************************
template<int n> static void calcInverse(const double *a, double *i);

// *****************************************************************************
template<> MFEM_HOST_DEVICE inline
void calcInverse<2>(const double* __restrict__ a,
                    double* __restrict__ i)
{
   constexpr int n = 2;
   const double d = det<2>(a);
   const double t = 1.0 / d;
   i[0*n+0] =  a[1*n+1] * t ;
   i[0*n+1] = -a[0*n+1] * t ;
   i[1*n+0] = -a[1*n+0] * t ;
   i[1*n+1] =  a[0*n+0] * t ;
}

// *****************************************************************************
template<> MFEM_HOST_DEVICE inline
void calcInverse<3>(const double* __restrict__ a,
                    double* __restrict__ inva)
{
   constexpr int n = 3;
   const double d = det<3>(a);
   const double t = 1.0 / d;
   inva[0*n+0] = (a[1*n+1]*a[2*n+2]-a[1*n+2]*a[2*n+1])*t;
   inva[0*n+1] = (a[0*n+2]*a[2*n+1]-a[0*n+1]*a[2*n+2])*t;
   inva[0*n+2] = (a[0*n+1]*a[1*n+2]-a[0*n+2]*a[1*n+1])*t;

   inva[1*n+0] = (a[1*n+2]*a[2*n+0]-a[1*n+0]*a[2*n+2])*t;
   inva[1*n+1] = (a[0*n+0]*a[2*n+2]-a[0*n+2]*a[2*n+0])*t;
   inva[1*n+2] = (a[0*n+2]*a[1*n+0]-a[0*n+0]*a[1*n+2])*t;

   inva[2*n+0] = (a[1*n+0]*a[2*n+1]-a[1*n+1]*a[2*n+0])*t;
   inva[2*n+1] = (a[0*n+1]*a[2*n+0]-a[0*n+0]*a[2*n+1])*t;
   inva[2*n+2] = (a[0*n+0]*a[1*n+1]-a[0*n+1]*a[1*n+0])*t;
}

// *****************************************************************************
MFEM_HOST_DEVICE static inline
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
MFEM_HOST_DEVICE static inline
double cpysign(const double x, const double y)
{
   if ((x < 0 && y > 0) || (x > 0 && y < 0))
   {
      return -x;
   }
   return x;
}

// *****************************************************************************
const double Epsilon = numeric_limits<double>::epsilon();

// *****************************************************************************
MFEM_HOST_DEVICE static inline
void eigensystem2S(const double &d12, double &d1, double &d2,
                   double &c, double &s)
{
   const double sqrt_1_eps = sqrt(1./Epsilon);
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
// * Eigen values 2D
// *****************************************************************************
template<int dim> static
void calcEigenvalues(const double *d, double *lambda, double *vec);

// *****************************************************************************
template<> MFEM_HOST_DEVICE inline
void calcEigenvalues<2>(const double* __restrict__ d,
                        double* __restrict__ lambda,
                        double* __restrict__ vec)
{
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
MFEM_HOST_DEVICE static inline
void getScalingFactor(const double &d_max, double &mult)
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
template<int dim> static double calcSingularvalue(const double *d);

// *****************************************************************************
template<> MFEM_HOST_DEVICE inline
double calcSingularvalue<2>(const double* __restrict__ d)
{
   constexpr int i = 2-1;
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
// * Eigen values 3D
// *****************************************************************************
MFEM_HOST_DEVICE static inline void Swap(double &a, double &b)
{
   double tmp = a;
   a = b;
   b = tmp;
}

// *****************************************************************************
MFEM_HOST_DEVICE static
inline bool KernelVector2G(const int &mode,
                           double &d1, double &d12, double &d21, double &d2)
{
   // Find a vector (z1,z2) in the "near"-kernel of the matrix
   // |  d1  d12 |
   // | d21   d2 |
   // using QR factorization.
   // The vector (z1,z2) is returned in (d1,d2). Return 'true' if the matrix
   // is zero without setting (d1,d2).
   // Note: in the current implementation |z1| + |z2| = 1.

   // l1-norms of the columns
   double n1 = fabs(d1) + fabs(d21);
   double n2 = fabs(d2) + fabs(d12);

   bool swap_columns = (n2 > n1);
   double mu;

   if (!swap_columns)
   {
      if (n1 == 0.)
      {
         return true;
      }

      if (mode == 0) // eliminate the larger entry in the column
      {
         if (fabs(d1) > fabs(d21))
         {
            Swap(d1, d21);
            Swap(d12, d2);
         }
      }
      else // eliminate the smaller entry in the column
      {
         if (fabs(d1) < fabs(d21))
         {
            Swap(d1, d21);
            Swap(d12, d2);
         }
      }
   }
   else
   {
      // n2 > n1, swap columns 1 and 2
      if (mode == 0) // eliminate the larger entry in the column
      {
         if (fabs(d12) > fabs(d2))
         {
            Swap(d1, d2);
            Swap(d12, d21);
         }
         else
         {
            Swap(d1, d12);
            Swap(d21, d2);
         }
      }
      else // eliminate the smaller entry in the column
      {
         if (fabs(d12) < fabs(d2))
         {
            Swap(d1, d2);
            Swap(d12, d21);
         }
         else
         {
            Swap(d1, d12);
            Swap(d21, d2);
         }
      }
   }

   n1 = hypot(d1, d21);

   if (d21 != 0.)
   {
      // v = (n1, n2)^t,  |v| = 1
      // Q = I - 2 v v^t,  Q (d1, d21)^t = (mu, 0)^t
      mu = copysign(n1, d1);
      n1 = -d21*(d21/(d1 + mu)); // = d1 - mu
      d1 = mu;
      // normalize (n1,d21) to avoid overflow/underflow
      // normalize (n1,d21) by the max-norm to avoid the sqrt call
      if (fabs(n1) <= fabs(d21))
      {
         // (n1,n2) <-- (n1/d21,1)
         n1 = n1/d21;
         mu = (2./(1. + n1*n1))*(n1*d12 + d2);
         d2  = d2  - mu;
         d12 = d12 - mu*n1;
      }
      else
      {
         // (n1,n2) <-- (1,d21/n1)
         n2 = d21/n1;
         mu = (2./(1. + n2*n2))*(d12 + n2*d2);
         d2  = d2  - mu*n2;
         d12 = d12 - mu;
      }
   }

   // Solve:
   // | d1 d12 | | z1 | = | 0 |
   // |  0  d2 | | z2 |   | 0 |

   // choose (z1,z2) to minimize |d1*z1 + d12*z2| + |d2*z2|
   // under the condition |z1| + |z2| = 1, z2 >= 0 (for uniqueness)
   // set t = z1, z2 = 1 - |t|, -1 <= t <= 1
   // objective function is:
   // |d1*t + d12*(1 - |t|)| + |d2|*(1 - |t|) -- piecewise linear with
   // possible minima are -1,0,1,t1 where t1: d1*t1 + d12*(1 - |t1|) = 0
   // values: @t=+/-1 -> |d1|, @t=0 -> |n1| + |d2|, @t=t1 -> |d2|*(1 - |t1|)

   // evaluate z2 @t=t1
   mu = -d12/d1;
   // note: |mu| <= 1,       if using l2-norm for column pivoting
   //       |mu| <= sqrt(2), if using l1-norm
   n2 = 1./(1. + fabs(mu));
   // check if |d1|<=|d2|*z2
   if (fabs(d1) <= n2*fabs(d2))
   {
      d2 = 0.;
      d1 = 1.;
   }
   else
   {
      d2 = n2;
      // d1 = (n2 < 0.5) ? copysign(1. - n2, mu) : mu*n2;
      d1 = mu*n2;
   }

   if (swap_columns)
   {
      Swap(d1, d2);
   }

   return false;
}

// *****************************************************************************
MFEM_HOST_DEVICE static
inline void vec_normalize3_aux(const double &x1, const double &x2,
                               const double &x3,
                               double &n1, double &n2, double &n3)
{
   double m, t, r;

   m = fabs(x1);
   r = x2/m;
   t = 1. + r*r;
   r = x3/m;
   t = sqrt(1./(t + r*r));
   n1 = copysign(t, x1);
   t /= m;
   n2 = x2*t;
   n3 = x3*t;
}

// *****************************************************************************
MFEM_HOST_DEVICE static
inline void vec_normalize3(const double &x1, const double &x2, const double &x3,
                           double &n1, double &n2, double &n3)
{
   // should work ok when xk is the same as nk for some or all k

   if (fabs(x1) >= fabs(x2))
   {
      if (fabs(x1) >= fabs(x3))
      {
         if (x1 != 0.)
         {
            vec_normalize3_aux(x1, x2, x3, n1, n2, n3);
         }
         else
         {
            n1 = n2 = n3 = 0.;
         }
         return;
      }
   }
   else if (fabs(x2) >= fabs(x3))
   {
      vec_normalize3_aux(x2, x1, x3, n2, n1, n3);
      return;
   }
   vec_normalize3_aux(x3, x1, x2, n3, n1, n2);
}

// *****************************************************************************
MFEM_HOST_DEVICE static
inline int KernelVector3G_aux(const int &mode,
                              double &d1, double &d2, double &d3,
                              double &c12, double &c13, double &c23,
                              double &c21, double &c31, double &c32)
{
   int kdim;
   double mu, n1, n2, n3, s1, s2, s3;

   s1 = hypot(c21, c31);
   n1 = hypot(d1, s1);

   if (s1 != 0.)
   {
      // v = (s1, s2, s3)^t,  |v| = 1
      // Q = I - 2 v v^t,  Q (d1, c12, c13)^t = (mu, 0, 0)^t
      mu = copysign(n1, d1);
      n1 = -s1*(s1/(d1 + mu)); // = d1 - mu
      d1 = mu;

      // normalize (n1,c21,c31) to avoid overflow/underflow
      // normalize (n1,c21,c31) by the max-norm to avoid the sqrt call
      if (fabs(n1) >= fabs(c21))
      {
         if (fabs(n1) >= fabs(c31))
         {
            // n1 is max, (s1,s2,s3) <-- (1,c21/n1,c31/n1)
            s2 = c21/n1;
            s3 = c31/n1;
            mu = 2./(1. + s2*s2 + s3*s3);
            n2  = mu*(c12 + s2*d2  + s3*c32);
            n3  = mu*(c13 + s2*c23 + s3*d3);
            c12 = c12 -    n2;
            d2  = d2  - s2*n2;
            c32 = c32 - s3*n2;
            c13 = c13 -    n3;
            c23 = c23 - s2*n3;
            d3  = d3  - s3*n3;
            goto done_column_1;
         }
      }
      else if (fabs(c21) >= fabs(c31))
      {
         // c21 is max, (s1,s2,s3) <-- (n1/c21,1,c31/c21)
         s1 = n1/c21;
         s3 = c31/c21;
         mu = 2./(1. + s1*s1 + s3*s3);
         n2  = mu*(s1*c12 + d2  + s3*c32);
         n3  = mu*(s1*c13 + c23 + s3*d3);
         c12 = c12 - s1*n2;
         d2  = d2  -    n2;
         c32 = c32 - s3*n2;
         c13 = c13 - s1*n3;
         c23 = c23 -    n3;
         d3  = d3  - s3*n3;
         goto done_column_1;
      }
      // c31 is max, (s1,s2,s3) <-- (n1/c31,c21/c31,1)
      s1 = n1/c31;
      s2 = c21/c31;
      mu = 2./(1. + s1*s1 + s2*s2);
      n2  = mu*(s1*c12 + s2*d2  + c32);
      n3  = mu*(s1*c13 + s2*c23 + d3);
      c12 = c12 - s1*n2;
      d2  = d2  - s2*n2;
      c32 = c32 -    n2;
      c13 = c13 - s1*n3;
      c23 = c23 - s2*n3;
      d3  = d3  -    n3;
   }

done_column_1:

   // Solve:
   // |  d2 c23 | | z2 | = | 0 |
   // | c32  d3 | | z3 |   | 0 |
   if (KernelVector2G(mode, d2, c23, c32, d3))
   {
      // Have two solutions:
      // two vectors in the kernel are P (-c12/d1, 1, 0)^t and
      // P (-c13/d1, 0, 1)^t where P is the permutation matrix swapping
      // entries 1 and col.

      // A vector orthogonal to both these vectors is P (1, c12/d1, c13/d1)^t
      d2 = c12/d1;
      d3 = c13/d1;
      d1 = 1.;
      kdim = 2;
   }
   else
   {
      // solve for z1:
      // note: |z1| <= a since |z2| + |z3| = 1, and
      // max{|c12|,|c13|} <= max{norm(col. 2),norm(col. 3)}
      //                  <= norm(col. 1) <= a |d1|
      // a = 1,       if using l2-norm for column pivoting
      // a = sqrt(3), if using l1-norm
      d1 = -(c12*d2 + c13*d3)/d1;
      kdim = 1;
   }

   vec_normalize3(d1, d2, d3, d1, d2, d3);

   return kdim;
}

// *****************************************************************************
MFEM_HOST_DEVICE static
inline int KernelVector3S(const int &mode, const double &d12,
                          const double &d13, const double &d23,
                          double &d1, double &d2, double &d3)
{
   // Find a unit vector (z1,z2,z3) in the "near"-kernel of the matrix
   // |  d1  d12  d13 |
   // | d12   d2  d23 |
   // | d13  d23   d3 |
   // using QR factorization.
   // The vector (z1,z2,z3) is returned in (d1,d2,d3).
   // Returns the dimension of the kernel, kdim, but never zero.
   // - if kdim == 3, then (d1,d2,d3) is not defined,
   // - if kdim == 2, then (d1,d2,d3) is a vector orthogonal to the kernel,
   // - otherwise kdim == 1 and (d1,d2,d3) is a vector in the "near"-kernel.

   double c12 = d12, c13 = d13, c23 = d23;
   double c21, c31, c32;
   int col, row;

   // l1-norms of the columns:
   c32 = fabs(d1) + fabs(c12) + fabs(c13);
   c31 = fabs(d2) + fabs(c12) + fabs(c23);
   c21 = fabs(d3) + fabs(c13) + fabs(c23);

   // column pivoting: choose the column with the largest norm
   if (c32 >= c21)
   {
      col = (c32 >= c31) ? 1 : 2;
   }
   else
   {
      col = (c31 >= c21) ? 2 : 3;
   }
   switch (col)
   {
      case 1:
         if (c32 == 0.) // zero matrix
         {
            return 3;
         }
         break;

      case 2:
         if (c31 == 0.) // zero matrix
         {
            return 3;
         }
         Swap(c13, c23);
         Swap(d1, d2);
         break;

      case 3:
         if (c21 == 0.) // zero matrix
         {
            return 3;
         }
         Swap(c12, c23);
         Swap(d1, d3);
   }

   // row pivoting depending on 'mode'
   if (mode == 0)
   {
      if (fabs(d1) <= fabs(c13))
      {
         row = (fabs(d1) <= fabs(c12)) ? 1 : 2;
      }
      else
      {
         row = (fabs(c12) <= fabs(c13)) ? 2 : 3;
      }
   }
   else
   {
      if (fabs(d1) >= fabs(c13))
      {
         row = (fabs(d1) >= fabs(c12)) ? 1 : 2;
      }
      else
      {
         row = (fabs(c12) >= fabs(c13)) ? 2 : 3;
      }
   }
   switch (row)
   {
      case 1:
         c21 = c12;
         c31 = c13;
         c32 = c23;
         break;

      case 2:
         c21 = d1;
         c31 = c13;
         c32 = c23;
         d1 = c12;
         c12 = d2;
         d2 = d1;
         c13 = c23;
         c23 = c31;
         break;

      case 3:
         c21 = c12;
         c31 = d1;
         c32 = c12;
         d1 = c13;
         c12 = c23;
         c13 = d3;
         d3 = d1;
   }
   row = KernelVector3G_aux(mode, d1, d2, d3, c12, c13, c23, c21, c31, c32);
   // row is kdim

   switch (col)
   {
      case 2:
         Swap(d1, d2);
         break;

      case 3:
         Swap(d1, d3);
   }
   return row;
}

// *****************************************************************************
MFEM_HOST_DEVICE static
inline int Reduce3S(const int &mode,
                    double &d1, double &d2, double &d3,
                    double &d12, double &d13, double &d23,
                    double &z1, double &z2, double &z3,
                    double &v1, double &v2, double &v3,
                    double &g)
{
   // Given the matrix
   //     |  d1  d12  d13 |
   // A = | d12   d2  d23 |
   //     | d13  d23   d3 |
   // and a unit eigenvector z=(z1,z2,z3), transform the matrix A into the
   // matrix B = Q P A P Q that has the form
   //                 | b1   0   0 |
   // B = Q P A P Q = | 0   b2 b23 |
   //                 | 0  b23  b3 |
   // where P is the permutation matrix switching entries 1 and k, and
   // Q is the reflection matrix Q = I - g v v^t, defined by: set y = P z and
   // v = c(y - e_1); if y = e_1, then v = 0 and Q = I.
   // Note: Q y = e_1, Q e_1 = y ==> Q P A P Q e_1 = ... = lambda e_1.
   // The entries (b1,b2,b3,b23) are returned in (d1,d2,d3,d23), and the
   // return value of the function is k. The variable g = 2/(v1^2+v2^2+v3^3).

   int k;
   double s, w1, w2, w3;

   if (mode == 0)
   {
      // choose k such that z^t e_k = zk has the smallest absolute value, i.e.
      // the angle between z and e_k is closest to pi/2
      if (fabs(z1) <= fabs(z3))
      {
         k = (fabs(z1) <= fabs(z2)) ? 1 : 2;
      }
      else
      {
         k = (fabs(z2) <= fabs(z3)) ? 2 : 3;
      }
   }
   else
   {
      // choose k such that zk is the largest by absolute value
      if (fabs(z1) >= fabs(z3))
      {
         k = (fabs(z1) >= fabs(z2)) ? 1 : 2;
      }
      else
      {
         k = (fabs(z2) >= fabs(z3)) ? 2 : 3;
      }
   }
   switch (k)
   {
      case 2:
         Swap(d13, d23);
         Swap(d1, d2);
         Swap(z1, z2);
         break;

      case 3:
         Swap(d12, d23);
         Swap(d1, d3);
         Swap(z1, z3);
   }

   s = hypot(z2, z3);

   if (s == 0.)
   {
      // s can not be zero, if zk is the smallest (mode == 0)
      v1 = v2 = v3 = 0.;
      g = 1.;
   }
   else
   {
      g = copysign(1., z1);
      v1 = -s*(s/(z1 + g)); // = z1 - g
      // normalize (v1,z2,z3) by its max-norm, avoiding the sqrt call
      g = fabs(v1);
      if (fabs(z2) > g) { g = fabs(z2); }
      if (fabs(z3) > g) { g = fabs(z3); }
      v1 = v1/g;
      v2 = z2/g;
      v3 = z3/g;
      g = 2./(v1*v1 + v2*v2 + v3*v3);

      // Compute Q A Q = A - v w^t - w v^t, where
      // w = u - (g/2)(v^t u) v, and u = g A v
      // set w = g A v
      w1 = g*( d1*v1 + d12*v2 + d13*v3);
      w2 = g*(d12*v1 +  d2*v2 + d23*v3);
      w3 = g*(d13*v1 + d23*v2 +  d3*v3);
      // w := w - (g/2)(v^t w) v
      s = (g/2)*(v1*w1 + v2*w2 + v3*w3);
      w1 -= s*v1;
      w2 -= s*v2;
      w3 -= s*v3;
      // dij -= vi*wj + wi*vj
      d1  -= 2*v1*w1;
      d2  -= 2*v2*w2;
      d23 -= v2*w3 + v3*w2;
      d3  -= 2*v3*w3;
      // compute the offdiagonal entries on the first row/column of B which
      // should be zero (for debugging):
#if 0
      s = d12 - v1*w2 - v2*w1;  // b12 = 0
      s = d13 - v1*w3 - v3*w1;  // b13 = 0
#endif
   }

   switch (k)
   {
      case 2:
         Swap(z1, z2);
         break;

      case 3:
         Swap(z1, z3);
   }

   return k;
}

// *****************************************************************************
template<> MFEM_HOST_DEVICE inline
void calcEigenvalues<3>(const double* __restrict__ d,
                        double* __restrict__ lambda,
                        double* __restrict__ vec)
{
   double d11 = d[0];
   double d12 = d[3]; // use the upper triangular entries
   double d22 = d[4];
   double d13 = d[6];
   double d23 = d[7];
   double d33 = d[8];

   double mult;
   {
      double d_max = fabs(d11);
      if (d_max < fabs(d22)) { d_max = fabs(d22); }
      if (d_max < fabs(d33)) { d_max = fabs(d33); }
      if (d_max < fabs(d12)) { d_max = fabs(d12); }
      if (d_max < fabs(d13)) { d_max = fabs(d13); }
      if (d_max < fabs(d23)) { d_max = fabs(d23); }

      getScalingFactor(d_max, mult);
   }

   d11 /= mult;  d22 /= mult;  d33 /= mult;
   d12 /= mult;  d13 /= mult;  d23 /= mult;

   double aa = (d11 + d22 + d33)/3;  // aa = tr(A)/3
   double c1 = d11 - aa;
   double c2 = d22 - aa;
   double c3 = d33 - aa;

   double Q, R;

   Q = (2*(d12*d12 + d13*d13 + d23*d23) + c1*c1 + c2*c2 + c3*c3)/6;
   R = (c1*(d23*d23 - c2*c3)+ d12*(d12*c3 - 2*d13*d23) + d13*d13*c2)/2;

   if (Q <= 0.)
   {
      lambda[0] = lambda[1] = lambda[2] = aa;
      vec[0] = 1.; vec[3] = 0.; vec[6] = 0.;
      vec[1] = 0.; vec[4] = 1.; vec[7] = 0.;
      vec[2] = 0.; vec[5] = 0.; vec[8] = 1.;
   }
   else
   {
      double sqrtQ = sqrt(Q);
      double sqrtQ3 = Q*sqrtQ;
      // double sqrtQ3 = sqrtQ*sqrtQ*sqrtQ;
      // double sqrtQ3 = pow(Q, 1.5);
      double r;
      if (fabs(R) >= sqrtQ3)
      {
         if (R < 0.)
         {
            // R = -1.;
            r = 2*sqrtQ;
         }
         else
         {
            // R = 1.;
            r = -2*sqrtQ;
         }
      }
      else
      {
         R = R/sqrtQ3;

         if (R < 0.)
         {
            r = -2*sqrtQ*cos((acos(R) + 2.0*M_PI)/3); // max
         }
         else
         {
            r = -2*sqrtQ*cos(acos(R)/3); // min
         }
      }

      aa += r;
      c1 = d11 - aa;
      c2 = d22 - aa;
      c3 = d33 - aa;

      // Type of Householder reflections: z --> mu ek, where k is the index
      // of the entry in z with:
      // mode == 0: smallest absolute value --> angle closest to pi/2
      // mode == 1: largest absolute value --> angle farthest from pi/2
      // Observations:
      // mode == 0 produces better eigenvectors, less accurate eigenvalues?
      // mode == 1 produces better eigenvalues, less accurate eigenvectors?
      const int mode = 0;

      // Find a unit vector z = (z1,z2,z3) in the "near"-kernel of
      //  |  c1  d12  d13 |
      //  | d12   c2  d23 | = A - aa*I
      //  | d13  d23   c3 |
      // This vector is also an eigenvector for A corresponding to aa.
      // The vector z overwrites (c1,c2,c3).
      switch (KernelVector3S(mode, d12, d13, d23, c1, c2, c3))
      {
         case 3:
            // 'aa' is a triple eigenvalue
            lambda[0] = lambda[1] = lambda[2] = aa;
            vec[0] = 1.; vec[3] = 0.; vec[6] = 0.;
            vec[1] = 0.; vec[4] = 1.; vec[7] = 0.;
            vec[2] = 0.; vec[5] = 0.; vec[8] = 1.;
            goto done_3d;

         case 2:
         // ok, continue with the returned vector orthogonal to the kernel
         case 1:
            // ok, continue with the returned vector in the "near"-kernel
            ;
      }

      // Using the eigenvector c=(c1,c2,c3) transform A into
      //                   | d11   0   0 |
      // A <-- Q P A P Q = |  0  d22 d23 |
      //                   |  0  d23 d33 |
      double v1, v2, v3, g;
      int k = Reduce3S(mode, d11, d22, d33, d12, d13, d23,
                       c1, c2, c3, v1, v2, v3, g);
      // Q = I - 2 v v^t
      // P - permutation matrix switching entries 1 and k

      // find the eigenvalues and eigenvectors for
      // | d22 d23 |
      // | d23 d33 |
      double c, s;
      eigensystem2S(d23, d22, d33, c, s);
      // d22 <-> P Q (0, c, -s), d33 <-> P Q (0, s, c)

      double *vec_1, *vec_2, *vec_3;
      if (d11 <= d22)
      {
         if (d22 <= d33)
         {
            lambda[0] = d11;  vec_1 = vec;
            lambda[1] = d22;  vec_2 = vec + 3;
            lambda[2] = d33;  vec_3 = vec + 6;
         }
         else if (d11 <= d33)
         {
            lambda[0] = d11;  vec_1 = vec;
            lambda[1] = d33;  vec_3 = vec + 3;
            lambda[2] = d22;  vec_2 = vec + 6;
         }
         else
         {
            lambda[0] = d33;  vec_3 = vec;
            lambda[1] = d11;  vec_1 = vec + 3;
            lambda[2] = d22;  vec_2 = vec + 6;
         }
      }
      else
      {
         if (d11 <= d33)
         {
            lambda[0] = d22;  vec_2 = vec;
            lambda[1] = d11;  vec_1 = vec + 3;
            lambda[2] = d33;  vec_3 = vec + 6;
         }
         else if (d22 <= d33)
         {
            lambda[0] = d22;  vec_2 = vec;
            lambda[1] = d33;  vec_3 = vec + 3;
            lambda[2] = d11;  vec_1 = vec + 6;
         }
         else
         {
            lambda[0] = d33;  vec_3 = vec;
            lambda[1] = d22;  vec_2 = vec + 3;
            lambda[2] = d11;  vec_1 = vec + 6;
         }
      }

      vec_1[0] = c1;
      vec_1[1] = c2;
      vec_1[2] = c3;
      d22 = g*(v2*c - v3*s);
      d33 = g*(v2*s + v3*c);
      vec_2[0] =    - v1*d22;  vec_3[0] =   - v1*d33;
      vec_2[1] =  c - v2*d22;  vec_3[1] = s - v2*d33;
      vec_2[2] = -s - v3*d22;  vec_3[2] = c - v3*d33;
      switch (k)
      {
         case 2:
            Swap(vec_2[0], vec_2[1]);
            Swap(vec_3[0], vec_3[1]);
            break;

         case 3:
            Swap(vec_2[0], vec_2[2]);
            Swap(vec_3[0], vec_3[2]);
      }
   }

done_3d:
   lambda[0] *= mult;
   lambda[1] *= mult;
   lambda[2] *= mult;
}

// *****************************************************************************
MFEM_HOST_DEVICE static
inline void Eigenvalues2S(const double &d12, double &d1, double &d2)
{
   const double sqrt_1_eps = sqrt(1./Epsilon);
   if (d12 != 0.)
   {
      // "The Symmetric Eigenvalue Problem", B. N. Parlett, pp.189-190
      double t, zeta = (d2 - d1)/(2*d12); // inf/inf from overflows?
      if (fabs(zeta) < sqrt_1_eps)
      {
         t = d12*copysign(1./(fabs(zeta) + sqrt(1. + zeta*zeta)), zeta);
      }
      else
      {
         t = d12*copysign(0.5/fabs(zeta), zeta);
      }
      d1 -= t;
      d2 += t;
   }
}


// *****************************************************************************
template<> MFEM_HOST_DEVICE inline
double calcSingularvalue<3>(const double* __restrict__ d)
{
   constexpr int i = 3-1;
   double d0, d1, d2, d3, d4, d5, d6, d7, d8;
   d0 = d[0];  d3 = d[3];  d6 = d[6];
   d1 = d[1];  d4 = d[4];  d7 = d[7];
   d2 = d[2];  d5 = d[5];  d8 = d[8];
   double mult;
   {
      double d_max = fabs(d0);
      if (d_max < fabs(d1)) { d_max = fabs(d1); }
      if (d_max < fabs(d2)) { d_max = fabs(d2); }
      if (d_max < fabs(d3)) { d_max = fabs(d3); }
      if (d_max < fabs(d4)) { d_max = fabs(d4); }
      if (d_max < fabs(d5)) { d_max = fabs(d5); }
      if (d_max < fabs(d6)) { d_max = fabs(d6); }
      if (d_max < fabs(d7)) { d_max = fabs(d7); }
      if (d_max < fabs(d8)) { d_max = fabs(d8); }

      getScalingFactor(d_max, mult);
   }

   d0 /= mult;  d1 /= mult;  d2 /= mult;
   d3 /= mult;  d4 /= mult;  d5 /= mult;
   d6 /= mult;  d7 /= mult;  d8 /= mult;

   double b11 = d0*d0 + d1*d1 + d2*d2;
   double b12 = d0*d3 + d1*d4 + d2*d5;
   double b13 = d0*d6 + d1*d7 + d2*d8;
   double b22 = d3*d3 + d4*d4 + d5*d5;
   double b23 = d3*d6 + d4*d7 + d5*d8;
   double b33 = d6*d6 + d7*d7 + d8*d8;

   // double a, b, c;
   // a = -(b11 + b22 + b33);
   // b = b11*(b22 + b33) + b22*b33 - b12*b12 - b13*b13 - b23*b23;
   // c = b11*(b23*b23 - b22*b33) + b12*(b12*b33 - 2*b13*b23) + b13*b13*b22;

   // double Q = (a * a - 3 * b) / 9;
   // double Q = (b12*b12 + b13*b13 + b23*b23 +
   //             ((b11 - b22)*(b11 - b22) +
   //              (b11 - b33)*(b11 - b33) +
   //              (b22 - b33)*(b22 - b33))/6)/3;
   // Q = (3*(b12^2 + b13^2 + b23^2) +
   //      ((b11 - b22)^2 + (b11 - b33)^2 + (b22 - b33)^2)/2)/9
   //   or
   // Q = (1/6)*|B-tr(B)/3|_F^2
   // Q >= 0 and
   // Q = 0  <==> B = scalar * I
   // double R = (2 * a * a * a - 9 * a * b + 27 * c) / 54;
   double aa = (b11 + b22 + b33)/3;  // aa = tr(B)/3
   double c1, c2, c3;
   // c1 = b11 - aa; // ((b11 - b22) + (b11 - b33))/3
   // c2 = b22 - aa; // ((b22 - b11) + (b22 - b33))/3
   // c3 = b33 - aa; // ((b33 - b11) + (b33 - b22))/3
   {
      double b11_b22 = ((d0-d3)*(d0+d3)+(d1-d4)*(d1+d4)+(d2-d5)*(d2+d5));
      double b22_b33 = ((d3-d6)*(d3+d6)+(d4-d7)*(d4+d7)+(d5-d8)*(d5+d8));
      double b33_b11 = ((d6-d0)*(d6+d0)+(d7-d1)*(d7+d1)+(d8-d2)*(d8+d2));
      c1 = (b11_b22 - b33_b11)/3;
      c2 = (b22_b33 - b11_b22)/3;
      c3 = (b33_b11 - b22_b33)/3;
   }
   double Q, R;
   Q = (2*(b12*b12 + b13*b13 + b23*b23) + c1*c1 + c2*c2 + c3*c3)/6;
   R = (c1*(b23*b23 - c2*c3)+ b12*(b12*c3 - 2*b13*b23) +b13*b13*c2)/2;
   // R = (-1/2)*det(B-(tr(B)/3)*I)
   // Note: 54*(det(S))^2 <= |S|_F^6, when S^t=S and tr(S)=0, S is 3x3
   // Therefore: R^2 <= Q^3

   if (Q <= 0.) { ; }

   // else if (fabs(R) >= sqrtQ3)
   // {
   //    double det = (d[0] * (d[4] * d[8] - d[5] * d[7]) +
   //                  d[3] * (d[2] * d[7] - d[1] * d[8]) +
   //                  d[6] * (d[1] * d[5] - d[2] * d[4]));
   //
   //    if (R > 0.)
   //    {
   //       if (i == 2)
   //          // aa -= 2*sqrtQ;
   //          return fabs(det)/(aa + sqrtQ);
   //       else
   //          aa += sqrtQ;
   //    }
   //    else
   //    {
   //       if (i != 0)
   //          aa -= sqrtQ;
   //          // aa = fabs(det)/sqrt(aa + 2*sqrtQ);
   //       else
   //          aa += 2*sqrtQ;
   //    }
   // }

   else
   {
      double sqrtQ = sqrt(Q);
      double sqrtQ3 = Q*sqrtQ;
      // double sqrtQ3 = sqrtQ*sqrtQ*sqrtQ;
      // double sqrtQ3 = pow(Q, 1.5);
      double r;

      if (fabs(R) >= sqrtQ3)
      {
         if (R < 0.)
         {
            // R = -1.;
            r = 2*sqrtQ;
         }
         else
         {
            // R = 1.;
            r = -2*sqrtQ;
         }
      }
      else
      {
         R = R/sqrtQ3;

         // if (fabs(R) <= 0.95)
         if (fabs(R) <= 0.9)
         {
            if (i == 2)
            {
               aa -= 2*sqrtQ*cos(acos(R)/3);   // min
            }
            else if (i == 0)
            {
               aa -= 2*sqrtQ*cos((acos(R) + 2.0*M_PI)/3);   // max
            }
            else
            {
               aa -= 2*sqrtQ*cos((acos(R) - 2.0*M_PI)/3);   // mid
            }
            goto have_aa;
         }

         if (R < 0.)
         {
            r = -2*sqrtQ*cos((acos(R) + 2.0*M_PI)/3); // max
            if (i == 0)
            {
               aa += r;
               goto have_aa;
            }
         }
         else
         {
            r = -2*sqrtQ*cos(acos(R)/3); // min
            if (i == 2)
            {
               aa += r;
               goto have_aa;
            }
         }
      }

      // (tr(B)/3 + r) is the root which is separated from the other
      // two roots which are close to each other when |R| is close to 1

      c1 -= r;
      c2 -= r;
      c3 -= r;
      // aa += r;

      // Type of Householder reflections: z --> mu ek, where k is the index
      // of the entry in z with:
      // mode == 0: smallest absolute value --> angle closest to pi/2
      //            (eliminate large entries)
      // mode == 1: largest absolute value --> angle farthest from pi/2
      //            (eliminate small entries)
      const int mode = 1;

      // Find a unit vector z = (z1,z2,z3) in the "near"-kernel of
      //  |  c1  b12  b13 |
      //  | b12   c2  b23 | = B - aa*I
      //  | b13  b23   c3 |
      // This vector is also an eigenvector for B corresponding to aa
      // The vector z overwrites (c1,c2,c3).
      switch (KernelVector3S(mode, b12, b13, b23, c1, c2, c3))
      {
         case 3:
            aa += r;
            goto have_aa;
         case 2:
         // ok, continue with the returned vector orthogonal to the kernel
         case 1:
            // ok, continue with the returned vector in the "near"-kernel
            ;
      }

      // Using the eigenvector c = (c1,c2,c3) to transform B into
      //                   | b11   0   0 |
      // B <-- Q P B P Q = |  0  b22 b23 |
      //                   |  0  b23 b33 |
      double v1, v2, v3, g;
      Reduce3S(mode, b11, b22, b33, b12, b13, b23,
               c1, c2, c3, v1, v2, v3, g);
      // Q = I - g v v^t
      // P - permutation matrix switching rows and columns 1 and k

      // find the eigenvalues of
      //  | b22 b23 |
      //  | b23 b33 |
      Eigenvalues2S(b23, b22, b33);

      if (i == 2)
      {
         aa = fmin(fmin(b11, b22), b33);
      }
      else if (i == 1)
      {
         if (b11 <= b22)
         {
            aa = (b22 <= b33) ? b22 : fmax(b11, b33);
         }
         else
         {
            aa = (b11 <= b33) ? b11 : fmax(b33, b22);
         }
      }
      else
      {
         aa = fmax(fmax(b11, b22), b33);
      }
   }

have_aa:

   return sqrt(fabs(aa))*mult; // take abs before we sort?
}

// *****************************************************************************
// * Smooth transition between 0 and 1 for x in [-eps, eps].
// *****************************************************************************
MFEM_HOST_DEVICE static
inline double smooth_step_01(const double x, const double eps)
{
   const double y = (x + eps) / (2.0 * eps);
   if (y < 0.0) { return 0.0; }
   if (y > 1.0) { return 1.0; }
   return (3.0 - 2.0 * y) * y * y;
}

// *****************************************************************************
template<int dim> MFEM_HOST_DEVICE static inline
void QBody(const int nzones, const int z,
           const int nqp, const int q,
           const double gamma,
           const bool use_viscosity,
           const double h0,
           const double h1order,
           const double cfl,
           const double infinity,
           double* __restrict__ Jinv,
           double* __restrict__ stress,
           double* __restrict__ sgrad_v,
           double* __restrict__ eig_val_data,
           double* __restrict__ eig_vec_data,
           double* __restrict__ compr_dir,
           double* __restrict__ Jpi,
           double* __restrict__ ph_dir,
           double* __restrict__ stressJiT,
           const double* __restrict__ d_weights,
           const double* __restrict__ d_Jacobians,
           const double* __restrict__ d_rho0DetJ0w,
           const double* __restrict__ d_e_quads,
           const double* __restrict__ d_grad_v_ext,
           const double* __restrict__ d_Jac0inv,
           double *d_dt_est,
           double *d_stressJinvT)
{
   constexpr int dim2 = dim*dim;
   double min_detJ = infinity;

   const int zq = z * nqp + q;
   const double weight =  d_weights[q];
   const double inv_weight = 1. / weight;
   const double *J = d_Jacobians + dim2*(nqp*z + q);
   const double detJ = det<dim>(J);
   min_detJ = fmin(min_detJ,detJ);
   calcInverse<dim>(J,Jinv);
   // *****************************************************************
   const double rho = inv_weight * d_rho0DetJ0w[zq] / detJ;
   const double e   = fmax(0.0, d_e_quads[zq]);
   const double p   = (gamma - 1.0) * rho * e;
   const double sound_speed = sqrt(gamma * (gamma-1.0) * e);
   // *****************************************************************
   for (int k = 0; k < dim2; k+=1) { stress[k] = 0.0; }
   for (int d = 0; d < dim; d++) { stress[d*dim+d] = -p; }
   // *****************************************************************
   double visc_coeff = 0.0;
   if (use_viscosity)
   {
      // Compression-based length scale at the point. The first
      // eigenvector of the symmetric velocity gradient gives the
      // direction of maximal compression. This is used to define the
      // relative change of the initial length scale.
      const double *dV = d_grad_v_ext + dim2*(nqp*z + q);
      mult(dim,dim,dim, dV, Jinv, sgrad_v);
      symmetrize(dim,sgrad_v);
      if (dim==1)
      {
         eig_val_data[0] = sgrad_v[0];
         eig_vec_data[0] = 1.;
      }
      else
      {
         calcEigenvalues<dim>(sgrad_v, eig_val_data, eig_vec_data);
      }
      for (int k=0; k<dim; k+=1) { compr_dir[k]=eig_vec_data[k]; }
      // Computes the initial->physical transformation Jacobian.
      mult(dim,dim,dim, J, d_Jac0inv+zq*dim*dim, Jpi);
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
      add(dim, dim, visc_coeff, sgrad_v, stress);
   }
   // Time step estimate at the point. Here the more relevant length
   // scale is related to the actual mesh deformation; we use the min
   // singular value of the ref->physical Jacobian. In addition, the
   // time step estimate should be aware of the presence of shocks.
   const double sv = calcSingularvalue<dim>(J);
   const double h_min = sv / h1order;
   const double inv_h_min = 1. / h_min;
   const double inv_rho_inv_h_min_sq = inv_h_min * inv_h_min / rho ;
   const double inv_dt = sound_speed * inv_h_min
                         + 2.5 * visc_coeff * inv_rho_inv_h_min_sq;
   if (min_detJ < 0.0)
   {
      // This will force repetition of the step with smaller dt.
      d_dt_est[zq] = 0.0;
   }
   else
   {
      if (inv_dt>0.0)
      {
         const double cfl_inv_dt = cfl / inv_dt;
         d_dt_est[zq] = fmin(d_dt_est[zq], cfl_inv_dt);
      }
   }
   // Quadrature data for partial assembly of the force operator.
   multABt(dim, dim, dim, stress, Jinv, stressJiT);
   for (int k=0; k<dim2; k+=1) { stressJiT[k] *= weight * detJ; }
   for (int vd = 0 ; vd < dim; vd++)
   {
      for (int gd = 0; gd < dim; gd++)
      {
         const int offset = zq + nqp*nzones*(gd+vd*dim);
         d_stressJinvT[offset] = stressJiT[vd+gd*dim];
      }
   }
}

// *****************************************************************************
// * qupdate
// *****************************************************************************
typedef void (*fQUpdate)(const int nzones,
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
                         DenseTensor &stressJinvT);

// *****************************************************************************
template<int Q1D, int dim=2> static inline
void QUpdate2D(const int nzones,
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
   constexpr int dim2 = dim*dim;
   auto d_weights = weights.Read();
   auto d_Jacobians = Jacobians.Read();
   auto d_rho0DetJ0w = rho0DetJ0w.Read();
   auto d_e_quads = e_quads.Read();
   auto d_grad_v_ext = grad_v_ext.Read();
   auto d_Jac0inv = Read(Jac0inv.GetMemory(), Jac0inv.TotalSize());
   auto d_dt_est = dt_est.ReadWrite();
   auto d_stressJinvT = Write(stressJinvT.GetMemory(),
                              stressJinvT.TotalSize());
   MFEM_FORALL_2D(z, nzones, Q1D, Q1D, 1,
   {
      double Jinv[dim2];
      double stress[dim2];
      double sgrad_v[dim2];
      double eig_val_data[3];
      double eig_vec_data[9];
      double compr_dir[dim];
      double Jpi[dim2];
      double ph_dir[dim];
      double stressJiT[dim2];
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            QBody<dim>(nzones, z, nqp, qx + qy * Q1D,
            gamma, use_viscosity, h0, h1order, cfl, infinity,
            Jinv,stress,sgrad_v,eig_val_data,eig_vec_data,
            compr_dir,Jpi,ph_dir,stressJiT,
            d_weights, d_Jacobians, d_rho0DetJ0w,
            d_e_quads, d_grad_v_ext, d_Jac0inv,
            d_dt_est, d_stressJinvT);
         }
      }
      MFEM_SYNC_THREAD;
   });
}

// *****************************************************************************
template<int Q1D, int dim=3> static inline
void QUpdate3D(const int nzones,
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
   constexpr int dim2 = dim*dim;
   auto d_weights = weights.Read();
   auto d_Jacobians = Jacobians.Read();
   auto d_rho0DetJ0w = rho0DetJ0w.Read();
   auto d_e_quads = e_quads.Read();
   auto d_grad_v_ext = grad_v_ext.Read();
   auto d_Jac0inv = Read(Jac0inv.GetMemory(), Jac0inv.TotalSize());
   auto d_dt_est = dt_est.ReadWrite();
   auto d_stressJinvT = Write(stressJinvT.GetMemory(),
                              stressJinvT.TotalSize());
   MFEM_FORALL_3D(z, nzones, Q1D, Q1D, Q1D,
   {
      double Jinv[dim2];
      double stress[dim2];
      double sgrad_v[dim2];
      double eig_val_data[3];
      double eig_vec_data[9];
      double compr_dir[dim];
      double Jpi[dim2];
      double ph_dir[dim];
      double stressJiT[dim2];
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qz,z,Q1D)
            {
               QBody<dim>(nzones, z, nqp, qx + Q1D * (qy + qz * Q1D),
               gamma, use_viscosity, h0, h1order, cfl, infinity,
               Jinv,stress,sgrad_v,eig_val_data,eig_vec_data,
               compr_dir,Jpi,ph_dir,stressJiT,
               d_weights, d_Jacobians, d_rho0DetJ0w,
               d_e_quads, d_grad_v_ext, d_Jac0inv,
               d_dt_est, d_stressJinvT);
            }
         }
      }
      MFEM_SYNC_THREAD;
   });
}

// *****************************************************************************
template<int D1D, int Q1D, int NBZ> static inline
void D2QInterp2D(const int NE, const Array<double> &b_,
                 const Vector &x_, Vector &y_)
{
   auto b = Reshape(b_.Read(), Q1D, D1D);
   auto x = Reshape(x_.Read(), D1D, D1D, NE);
   auto y = Reshape(y_.Write(), Q1D, Q1D, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int zid = MFEM_THREAD_ID(z);
      MFEM_SHARED double B[Q1D][D1D];

      MFEM_SHARED double DDz[NBZ][D1D*D1D];
      double (*DD)[D1D] = (double (*)[D1D])(DDz + zid);

      MFEM_SHARED double DQz[NBZ][D1D*Q1D];
      double (*DQ)[Q1D] = (double (*)[Q1D])(DQz + zid);

      if (zid == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B[q][d] = b(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            DD[dy][dx] = x(dx,dy,e);
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dy,y,D1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double dq = 0.0;
            for (int dx = 0; dx < D1D; ++dx)
            {
               dq += B[qx][dx] * DD[dy][dx];
            }
            DQ[dy][qx] = dq;
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double qq = 0.0;
            for (int dy = 0; dy < D1D; ++dy)
            {
               qq += DQ[dy][qx] * B[qy][dy];
            }
            y(qx,qy,e) = qq;
         }
      }
      MFEM_SYNC_THREAD;
   });
}

// *****************************************************************************
template<int D1D, int Q1D> static inline
void D2QInterp3D(const int NE, const Array<double> &b_,
                 const Vector &x_, Vector &y_)
{
   auto b = Reshape(b_.Read(), Q1D, D1D);
   auto x = Reshape(x_.Read(), D1D, D1D, D1D, NE);
   auto y = Reshape(y_.Write(), Q1D, Q1D, Q1D, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int tidz = MFEM_THREAD_ID(z);
      MFEM_SHARED double B[Q1D][D1D];
      MFEM_SHARED double sm0[Q1D*Q1D*Q1D];
      MFEM_SHARED double sm1[Q1D*Q1D*Q1D];
      double (*X)[D1D][D1D]   = (double (*)[D1D][D1D]) sm0;
      double (*DDQ)[D1D][Q1D] = (double (*)[D1D][Q1D]) sm1;
      double (*DQQ)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) sm0;

      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B[q][d] = b(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;

      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(dx,x,D1D)
            {
               X[dz][dy][dx] = x(dx,dy,dz,e);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0;
               for (int dx = 0; dx < D1D; ++dx)
               {
                  u += B[qx][dx] * X[dz][dy][dx];
               }
               DDQ[dz][dy][qx] = u;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(dz,z,D1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0;
               for (int dy = 0; dy < D1D; ++dy)
               {
                  u += DDQ[dz][dy][qx] * B[qy][dy];
               }
               DQQ[dz][qy][qx] = u;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(qz,z,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0;
               for (int dz = 0; dz < D1D; ++dz)
               {
                  u += DQQ[dz][qy][qx] * B[qz][dz];
               }
               y(qx,qy,qz,e) = u;
            }
         }
      }
      MFEM_SYNC_THREAD;
   });
}

// *****************************************************************************
typedef void (*fD2QInterp)(const int NE,
                           const Array<double> &B,
                           const Vector &x,
                           Vector &y);

// ***************************************************************************
static void D2QInterp(const Operator *erestrict,
                      const FiniteElementSpace &fes,
                      const DofToQuad *maps,
                      const IntegrationRule& ir,
                      const Vector &d_in,
                      Vector &d_out)
{
   const int dim = fes.GetMesh()->Dimension();
   const int nzones = fes.GetNE();
   const int dofs1D = fes.GetFE(0)->GetOrder() + 1;
   const int quad1D = IntRules.Get(Geometry::SEGMENT,ir.GetOrder()).GetNPoints();
   const int id = (dim<<8)|(dofs1D<<4)|(quad1D);
   static std::unordered_map<int, fD2QInterp> call =
   {
      // 2D
      {0x224,&D2QInterp2D<2,4,8>},
      {0x236,&D2QInterp2D<3,6,4>},
      {0x248,&D2QInterp2D<4,8,2>},
      // 3D
      {0x324,&D2QInterp3D<2,4>},
      {0x336,&D2QInterp3D<3,6>},
      {0x348,&D2QInterp3D<4,8>},
   };
   if (!call[id])
   {
      mfem::out << "Unknown kernel 0x" << std::hex << id << std::endl;
      MFEM_ABORT("Unknown kernel");
   }
   call[id](nzones, maps->B, d_in, d_out);
}

// **************************************************************************
template<int D1D, int Q1D, int NBZ> static inline
void D2QGrad2D(const int NE,
               const Array<double> &b_,
               const Array<double> &g_,
               const Vector &x_,
               Vector &y_)
{
   constexpr int VDIM = 2;

   auto b = Reshape(b_.Read(), Q1D, D1D);
   auto g = Reshape(g_.Read(), Q1D, D1D);
   auto x = Reshape(x_.Read(), D1D, D1D, VDIM, NE);
   auto y = Reshape(y_.Write(), VDIM, VDIM, Q1D, Q1D, NE);

   MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
   {
      const int tidz = MFEM_THREAD_ID(z);
      MFEM_SHARED double B[Q1D][D1D];
      MFEM_SHARED double G[Q1D][D1D];

      MFEM_SHARED double Xz[NBZ][D1D][D1D];
      double (*X)[D1D] = (double (*)[D1D])(Xz + tidz);

      MFEM_SHARED double GD[2][NBZ][D1D][Q1D];
      double (*DQ0)[Q1D] = (double (*)[Q1D])(GD[0] + tidz);
      double (*DQ1)[Q1D] = (double (*)[Q1D])(GD[1] + tidz);

      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B[q][d] = b(q,d);
               G[q][d] = g(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;

      for (int c = 0; c < 2; ++c)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               X[dx][dy] = x(dx,dy,c,e);
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dy,y,D1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0;
               double v = 0.0;
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const double input = X[dx][dy];
                  u += B[qx][dx] * input;
                  v += G[qx][dx] * input;
               }
               DQ0[dy][qx] = u;
               DQ1[dy][qx] = v;
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = 0.0;
               double v = 0.0;
               for (int dy = 0; dy < D1D; ++dy)
               {
                  u += DQ1[dy][qx] * B[qy][dy];
                  v += DQ0[dy][qx] * G[qy][dy];
               }
               y(c,0,qx,qy,e) = u;
               y(c,1,qx,qy,e) = v;
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

// **************************************************************************
template<int D1D, int Q1D> static inline
void D2QGrad3D(const int NE,
               const Array<double> &b_,
               const Array<double> &g_,
               const Vector &x_,
               Vector &y_)
{
   constexpr int VDIM = 3;
   auto b = Reshape(b_.Read(), Q1D, D1D);
   auto g = Reshape(g_.Read(), Q1D, D1D);
   auto x = Reshape(x_.Read(), D1D, D1D, D1D, VDIM, NE);
   auto y = Reshape(y_.Write(), VDIM, VDIM, Q1D, Q1D, Q1D, NE);

   MFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,
   {
      const int tidz = MFEM_THREAD_ID(z);
      MFEM_SHARED double B[Q1D][D1D];
      MFEM_SHARED double G[Q1D][D1D];

      MFEM_SHARED double sm0[3][Q1D*Q1D*Q1D];
      MFEM_SHARED double sm1[3][Q1D*Q1D*Q1D];
      double (*X)[D1D][D1D]    = (double (*)[D1D][D1D]) (sm0+2);
      double (*DDQ0)[D1D][Q1D] = (double (*)[D1D][Q1D]) (sm0+0);
      double (*DDQ1)[D1D][Q1D] = (double (*)[D1D][Q1D]) (sm0+1);
      double (*DQQ0)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm1+0);
      double (*DQQ1)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm1+1);
      double (*DQQ2)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm1+2);

      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D)
            {
               B[q][d] = b(q,d);
               G[q][d] = g(q,d);
            }
         }
      }
      MFEM_SYNC_THREAD;

      for (int c = 0; c < VDIM; ++c)
      {
         MFEM_FOREACH_THREAD(dx,x,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(dz,z,D1D)
               {

                  X[dx][dy][dz] = x(dx,dy,dz,c,e);
               }
            }
         }
         MFEM_SYNC_THREAD;

         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(dy,y,D1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  double u = 0.0;
                  double v = 0.0;
                  for (int dx = 0; dx < D1D; ++dx)
                  {
                     const double coords = X[dx][dy][dz];
                     u += coords * B[qx][dx];
                     v += coords * G[qx][dx];
                  }
                  DDQ0[dz][dy][qx] = u;
                  DDQ1[dz][dy][qx] = v;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dz,z,D1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  double u = 0.0;
                  double v = 0.0;
                  double w = 0.0;
                  for (int dy = 0; dy < D1D; ++dy)
                  {
                     u += DDQ1[dz][dy][qx] * B[qy][dy];
                     v += DDQ0[dz][dy][qx] * G[qy][dy];
                     w += DDQ0[dz][dy][qx] * B[qy][dy];
                  }
                  DQQ0[dz][qy][qx] = u;
                  DQQ1[dz][qy][qx] = v;
                  DQQ2[dz][qy][qx] = w;
               }
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  double u = 0.0;
                  double v = 0.0;
                  double w = 0.0;
                  for (int dz = 0; dz < D1D; ++dz)
                  {
                     u += DQQ0[dz][qy][qx] * B[qz][dz];
                     v += DQQ1[dz][qy][qx] * B[qz][dz];
                     w += DQQ2[dz][qy][qx] * G[qz][dz];
                  }
                  y(c,0,qx,qy,qz,e) = u;
                  y(c,1,qx,qy,qz,e) = v;
                  y(c,2,qx,qy,qz,e) = w;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

// *****************************************************************************
typedef void (*fD2QGrad)(const int NE,
                         const Array<double> &B,
                         const Array<double> &G,
                         const Vector &x,
                         Vector &y);

// **************************************************************************
static void D2QGrad(const Operator *erestrict,
                    const FiniteElementSpace &fes,
                    const DofToQuad *maps,
                    const IntegrationRule& ir,
                    const Vector &d_in,
                    Vector &d_h1_v_local_in,
                    Vector &d_out)
{
   const int dim = fes.GetMesh()->Dimension();
   const int nzones = fes.GetNE();
   const int dofs1D = fes.GetFE(0)->GetOrder() + 1;
   const int quad1D = IntRules.Get(Geometry::SEGMENT,ir.GetOrder()).GetNPoints();
   erestrict->Mult(d_in, d_h1_v_local_in);
   const int id = (dim<<8)|(dofs1D<<4)|(quad1D);
   static std::unordered_map<int, fD2QGrad> call =
   {
      // 2D
      {0x234,&D2QGrad2D<3,4,8>},
      {0x246,&D2QGrad2D<4,6,4>},
      {0x258,&D2QGrad2D<5,8,2>},
      // 3D
      {0x334,&D2QGrad3D<3,4>},
      {0x346,&D2QGrad3D<4,6>},
      {0x358,&D2QGrad3D<5,8>},
   };
   if (!call[id])
   {
      mfem::out << "Unknown kernel 0x" << std::hex << id << std::endl;
      MFEM_ABORT("Unknown kernel");
   }
   call[id](nzones,
            maps->B,
            maps->G,
            d_h1_v_local_in,
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
   Vector* S_p = const_cast<Vector*>(&S);

   // **************************************************************************
   const int H1_size = H1FESpace.GetVSize();
   const int nqp1D = tensors1D->LQshape1D.Width();

   // Energy dof => quads ******************************************************
   ParGridFunction d_e;
   d_e.MakeRef(&L2FESpace, *S_p, 2*H1_size);
   D2QInterp(l2_ElemRestrict, L2FESpace, l2_maps, ir, d_e, d_l2_e_quads_data);

   // Coords to Jacobians ******************************************************
   ParGridFunction d_x;
   d_x.MakeRef(&H1FESpace,*S_p, 0);
   D2QGrad(h1_ElemRestrict, H1FESpace, h1_maps, ir, d_x,
           d_h1_v_local_in, d_h1_grad_x_data);

   // Velocity *****************************************************************
   ParGridFunction d_v;
   d_v.MakeRef(&H1FESpace,*S_p, H1_size);
   D2QGrad(h1_ElemRestrict, H1FESpace, h1_maps, ir, d_v,
           d_h1_v_local_in, d_h1_grad_v_data);

   // **************************************************************************
   const double h1order = (double) H1FESpace.GetOrder(0);
   const double infinity = std::numeric_limits<double>::infinity();

   // **************************************************************************
   d_dt_est = quad_data.dt_est;

   // **************************************************************************
   const int id = (dim<<4)|nqp1D;
   static std::unordered_map<int, fQUpdate> qupdate =
   {
      // 2D
      {0x24,&QUpdate2D<4>},
      {0x26,&QUpdate2D<6>},
      {0x28,&QUpdate2D<8>},
      // 3D
      {0x34,&QUpdate3D<4>},
      {0x36,&QUpdate3D<6>},
      {0x38,&QUpdate3D<8>}
   };
   if (!qupdate[id])
   {
      mfem::out << "Unknown kernel 0x" << std::hex << id << std::endl;
      MFEM_ABORT("Unknown kernel");
   }
   qupdate[id](nzones, nqp, nqp1D, gamma, use_viscosity, quad_data.h0,
               h1order, cfl, infinity, ir.GetWeights(), d_h1_grad_x_data,
               quad_data.rho0DetJ0w, d_l2_e_quads_data, d_h1_grad_v_data,
               quad_data.Jac0inv, d_dt_est, quad_data.stressJinvT);

   // **************************************************************************
   quad_data.dt_est = d_dt_est.Min();
   quad_data_is_current = true;
   timer->sw_qdata.Stop();
   timer->quad_tstep += nzones;
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
   nqp(_ir.GetNPoints()),
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
   h1_ElemRestrict(H1FESpace.GetElementRestriction(
                      ElementDofOrdering::LEXICOGRAPHIC)),
   l2_ElemRestrict(L2FESpace.GetElementRestriction(
                      ElementDofOrdering::LEXICOGRAPHIC)),
   d_l2_e_quads_data(nzones * nqp),
   h1_vdim(H1FESpace.GetVDim()),
   d_h1_v_local_in(           h1_vdim * nqp * nzones),
   d_h1_grad_x_data(h1_vdim * h1_vdim * nqp * nzones),
   d_h1_grad_v_data(h1_vdim * h1_vdim * nqp * nzones),
   d_dt_est(nzones * nqp)
{
   MFEM_ASSERT(material_pcf, "!material_pcf");
}

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI
