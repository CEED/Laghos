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

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{

namespace hydrodynamics
{
   // **************************************************************************
   static inline double det2D(const double *d){
      return d[0] * d[3] - d[1] * d[2];
   }
   
   // **************************************************************************
   /*static inline double det3D(const double *d){
      return
         d[0] * (d[4] * d[8] - d[5] * d[7]) +
         d[3] * (d[2] * d[7] - d[1] * d[8]) +
         d[6] * (d[1] * d[5] - d[2] * d[4]);
         }*/

   // **************************************************************************
   void calcInverse2D(const size_t n, const double *a, double *i){
      const double d = det2D(a);
      const double t = 1.0 / d;
      i[0*n+0] =  a[1*n+1] * t ;
      i[0*n+1] = -a[0*n+1] * t ;
      i[1*n+0] = -a[1*n+0] * t ;
      i[1*n+1] =  a[0*n+0] * t ;
   }

   // **************************************************************************
   void symmetrize(const size_t n, double* __restrict__ d){
      for (size_t i = 0; i<n; i++){
         for (size_t j = 0; j<i; j++) {
            const double a = 0.5 * (d[i*n+j] + d[j*n+i]);
            d[j*n+i] = d[i*n+j] = a;
         }
      }
   }
   
   // **************************************************************************
   static inline double cpysign(const double x, const double y) {
      if ((x < 0 && y > 0) || (x > 0 && y < 0))
         return -x;
      return x;
   }

   // **************************************************************************
   static inline void eigensystem2S(const double &d12, double &d1, double &d2,
                                    double &c, double &s) {
      static const double sqrt_1_eps = sqrt(1./numeric_limits<double>::epsilon());
      if (d12 == 0.) {
         c = 1.;
         s = 0.;
      } else {
         // "The Symmetric Eigenvalue Problem", B. N. Parlett, pp.189-190
         double t, zeta = (d2 - d1)/(2*d12);
         if (fabs(zeta) < sqrt_1_eps) {
            t = cpysign(1./(fabs(zeta) + sqrt(1. + zeta*zeta)), zeta);
         } else {
            t = cpysign(0.5/fabs(zeta), zeta);
         }
         c = sqrt(1./(1. + t*t));
         s = c*t;
         t *= d12;
         d1 -= t;
         d2 += t;
      }
   }
   
   // **************************************************************************
   void calcEigenvalues(const size_t n, const double *d,
                        double *lambda,
                        double *vec) {
      assert(n == 2);   
      double d0 = d[0];
      double d2 = d[2]; // use the upper triangular entry
      double d3 = d[3];
      double c, s;
      eigensystem2S(d2, d0, d3, c, s);
      if (d0 <= d3) {
         lambda[0] = d0;
         lambda[1] = d3;
         vec[0] =  c;
         vec[1] = -s;
         vec[2] =  s;
         vec[3] =  c;
      } else {
         lambda[0] = d3;
         lambda[1] = d0;
         vec[0] =  s;
         vec[1] =  c;
         vec[2] =  c;
         vec[3] = -s;
      }
   }

   // **************************************************************************
   static inline void getScalingFactor(const double &d_max, double &mult){
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

      // **************************************************************************
   double calcSingularvalue(const int n, const int i, const double *d) {
      assert (n == 2);
      
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
   

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI
