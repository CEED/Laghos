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

namespace mfem
{

namespace hydrodynamics
{

struct TimingData;

class QUpdate
{
private:
   const int dim;
   const int NQ;
   const int NE;
   const bool use_viscosity;
   const bool p_assembly;
   const double cfl;
   const double gamma;
   TimingData *timer;
   const IntegrationRule &ir;
   ParFiniteElementSpace &H1;
   ParFiniteElementSpace &L2;
   const Operator *H1ER;
   Vector d_l2_e_quads_data; // scalar L2
   const int vdim;
   Vector d_h1_v_local_in;   // vector H1
   Vector d_h1_grad_x_data;  // grad vector H1
   Vector d_h1_grad_v_data;  // grad vector H1
   Vector d_dt_est;
   const QuadratureInterpolator *q1,*q2;

public:
   QUpdate(const int dim,
           const int NE,
           const bool use_viscosity,
           const bool p_assembly,
           const double cfl,
           const double gamma,
           TimingData *timer,
           const IntegrationRule &ir,
           ParFiniteElementSpace &H1,
           ParFiniteElementSpace &L2);

   void UpdateQuadratureData(const Vector &S,
                             bool &quad_data_is_current,
                             QuadratureData &quad_data,
                             const Tensors1D *tensors1D);
};

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_LAGHOS_QUPDATE
