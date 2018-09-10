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

#ifndef MFEM_LAGHOS_QUPDATE
#define MFEM_LAGHOS_QUPDATE

// *****************************************************************************
#include "memcpy.hpp"
#include "alloc.hpp"
#include "array.hpp"
#include "dof2quad.hpp"
#include "geom.hpp"

// Offsets *********************************************************************
#define   ijN(i,j,N) (i)+(N)*(j)
#define  ijkN(i,j,k,N) (i)+(N)*((j)+(N)*(k))
#define ijklN(i,j,k,l,N) (i)+(N)*((j)+(N)*((k)+(N)*(l)))

#define    ijNMt(i,j,N,M,t) (t)?((i)+(N)*(j)):((j)+(M)*(i))
#define    ijkNM(i,j,k,N,M) (i)+(N)*((j)+(M)*(k))
#define   _ijkNM(i,j,k,N,M) (j)+(N)*((k)+(M)*(i))
#define   ijklNM(i,j,k,l,N,M) (i)+(N)*((j)+(N)*((k)+(M)*(l)))
#define  _ijklNM(i,j,k,l,N,M) (j)+(N)*((k)+(N)*((l)+(M)*(i)))
#define   ijklmNM(i,j,k,l,m,N,M) (i)+(N)*((j)+(N)*((k)+(M)*((l)+(M)*(m))))
#define __ijklmNM(i,j,k,l,m,N,M) (k)+(M)*((l)+(M)*((m)+(N*N)*((i)+(N)*j)))

#define _ijklmNM(i,j,k,l,m,N,M) (j)+(N)*((k)+(N)*((l)+(N)*((m)+(M)*(i))))
#define ijklmnNM(i,j,k,l,m,n,N,M) (i)+(N)*((j)+(N)*((k)+(M)*((l)+(M)*((m)+(M)*(n)))))

namespace mfem {

namespace hydrodynamics {
   
   // **************************************************************************
   void multABt(const size_t, const size_t, const size_t,
                const double*, const double*, double*);
   
   // **************************************************************************
   void multAtB(const size_t, const size_t, const size_t,
                const double*, const double*, double*);
   
   // **************************************************************************
   void mult(const size_t, const size_t, const size_t,
             const double*, const double*, double*);

   // **************************************************************************
   void multV(const size_t, const size_t,
              double*, const double*, double*);

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
                QuadratureData &quad_data);

   // **************************************************************************
   void add(const size_t, const size_t,
            const double, const double*, double*);

   // **************************************************************************
   void calcInverse2D(const size_t, const double*, double*);
   void symmetrize(const size_t, double*);
   void calcEigenvalues(const size_t, const double*, double*, double*);
   double calcSingularvalue(const int, const int, const double*);
   
   // **************************************************************************
   int *global2LocalMap(ParFiniteElementSpace&);
   void globalToLocal(ParFiniteElementSpace&, const double*, double*,
                      const bool =false);

   // **************************************************************************
   void d2q(ParFiniteElementSpace&, const IntegrationRule&,
            const double*, double*);

} // namespace hydrodynamics

} // namespace mfem

#endif // LAGHOS_KERNELS_QUPDATE
