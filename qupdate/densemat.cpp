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

namespace mfem
{

namespace hydrodynamics
{
   // **************************************************************************
   void multABt(const size_t ah,
                const size_t aw,
                const size_t bh,
                const double* __restrict__ A,
                const double* __restrict__ B,
                double* __restrict__ C){
      const size_t ah_x_bh = ah*bh;
      for(size_t i=0; i<ah_x_bh; i+=1)
         C[i] = 0.0;  
      for(size_t k=0; k<aw; k+=1) {
         double *c = C;
         for(size_t j=0; j<bh; j+=1){
            const double bjk = B[j];
            for(size_t i=0; i<ah; i+=1)
               c[i] += A[i] * bjk;            
            c += ah;
         }
         A += ah;
         B += bh;
      }
   }

   // **************************************************************************
   void multAtB(const size_t ah,
                const size_t aw,
                const size_t bw,
                const double* __restrict__ A,
                const double* __restrict__ B,
                double* __restrict__ C) {
      for(size_t j = 0; j < bw; j+=1) {
         const double *a = A;
         for(size_t i = 0; i < aw; i+=1) {
            double d = 0.0;
            for(size_t k = 0; k < ah; k+=1) 
               d += a[k] * B[k];
            *(C++) = d;
            a += ah;
         }
         B += ah;
      }
   }

   // **************************************************************************
   void mult(const size_t ah,
             const size_t aw,
             const size_t bw,
             const double* __restrict__ B,
             const double* __restrict__ C,
             double* __restrict__ A){
      const size_t ah_x_aw = ah*aw;
      for (int i = 0; i < ah_x_aw; i++) A[i] = 0.0;
      for (int j = 0; j < aw; j++) {
         for (int k = 0; k < bw; k++) {
            for (int i = 0; i < ah; i++) {
               A[i+j*ah] += B[i+k*ah] * C[k+j*bw];
            }
         }
      }
   }

   // **************************************************************************
   void multV(const size_t height,
              const size_t width,
              double *data,
              const double* __restrict__ x,
              double* __restrict__ y) {
      if (width == 0) {
         for (int row = 0; row < height; row++) 
            y[row] = 0.0;         
         return;
      }
      double *d_col = data;
      double x_col = x[0];
      for (int row = 0; row < height; row++) {
         y[row] = x_col*d_col[row];
      }
      d_col += height;
      for (int col = 1; col < width; col++) {
         x_col = x[col];
         for (int row = 0; row < height; row++) {
            y[row] += x_col*d_col[row];
         }
         d_col += height;
      }
   }
   
   // **************************************************************************
   void add(const size_t height, const size_t width,
            const double c, const double *A,
            double *D){
      for (int j = 0; j < width; j++){
         for (int i = 0; i < height; i++) {
            D[i*width+j] += c * A[i*width+j];
         }
      }
   }


} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI
