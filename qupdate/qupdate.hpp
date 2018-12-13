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
#include "mfem.hpp"
#include "../laghos_assembly.hpp"
#include "d2q.hpp"
#include "eigen.hpp"
#include "densemat.hpp"

// Offsets *********************************************************************
#define      ijN(i,j,N) (i)+(N)*(j)
#define     ijkN(i,j,k,N) (i)+(N)*((j)+(N)*(k))
#define    _ijkN(i,j,k,N) (j)+(N)*((k)+(N)*(i))
#define    ijNMt(i,j,N,M,t) (t)?((i)+(N)*(j)):((j)+(M)*(i))
#define    ijkNM(i,j,k,N,M) (i)+(N)*((j)+(M)*(k))
#define   ijklNM(i,j,k,l,N,M) (i)+(N)*((j)+(N)*((k)+(M)*(l)))
#define  _ijklNM(i,j,k,l,N,M) (j)+(N)*((k)+(N)*((l)+(M)*(i)))

namespace mfem {

double kVectorMin(const size_t, const double*);

namespace hydrodynamics {
struct TimingData;

// *****************************************************************************
class QUpdate{
private:
   const int dim;
   const int nzones;
   const int l2dofs_cnt;
   const int h1dofs_cnt;
   const bool use_viscosity;
   const bool p_assembly;
   const double cfl;
   const double gamma;
   TimingData *timer;
   mfem::Coefficient *material_pcf;
   const mfem::IntegrationRule &ir;
   mfem::ParFiniteElementSpace &H1FESpace;
   mfem::ParFiniteElementSpace &L2FESpace;
   mfem::kDofQuadMaps *h1_maps;
   mfem::kDofQuadMaps *l2_maps;
   const mfem::kFiniteElementSpace *h1_kfes;
   const mfem::kFiniteElementSpace *l2_kfes;
   double *d_e_quads_data;
   double *d_grad_x_data;
   double *d_grad_v_data;
   const int nqp;
public:
   // **************************************************************************
   QUpdate(const int dim,
           const int nzones,
           const int l2dofs_cnt,
           const int h1dofs_cnt,
           const bool use_viscosity,
           const bool p_assembly,
           const double cfl,
           const double gamma,
           TimingData *timer,
           Coefficient *material_pcf,
           const IntegrationRule &integ_rule,
           ParFiniteElementSpace &H1FESpace,
           ParFiniteElementSpace &L2FESpace);
   
   // **************************************************************************
   ~QUpdate();
   
   // **************************************************************************
   void UpdateQuadratureData(const Vector &S,
                             bool &quad_data_is_current,
                             QuadratureData &quad_data);

   // **************************************************************************
   // * Dof2QuadScalar
   // **************************************************************************
   void Dof2QuadScalar(const kFiniteElementSpace *kfes,
                       const FiniteElementSpace &fes,
                       const kDofQuadMaps *maps,
                       const IntegrationRule&,
                       const double*,
                       double**);
   
   // **************************************************************************
   // * Dof2QuadGrad
   // **************************************************************************
   void Dof2QuadGrad(const kFiniteElementSpace *kfes,
                     const FiniteElementSpace &fes,
                     const kDofQuadMaps *maps,
                     const IntegrationRule&,
                     const double*,
                     double**);   
};

} // namespace hydrodynamics

} // namespace mfem

#endif // LAGHOS_KERNELS_QUPDATE
