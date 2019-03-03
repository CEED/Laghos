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

#include "mfem.hpp"
#include "laghos_assembly.hpp"

namespace mfem {

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
   mfem::DofToQuad *h1_maps;
   mfem::DofToQuad *l2_maps;
   const mfem::ElemRestriction *h1_ElemRestrict;
   const mfem::ElemRestriction *l2_ElemRestrict;
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
};

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_LAGHOS_QUPDATE
