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

#ifndef MFEM_LAGHOS_ASSEMBLY
#define MFEM_LAGHOS_ASSEMBLY

#include "mfem.hpp"


#ifdef MFEM_USE_MPI

#include <memory>
#include <iostream>
#include "mpi_utils.hpp"

namespace mfem
{
namespace hydrodynamics
{
// Container for all data needed at quadrature points.
struct QuadratureData
{
   // TODO: use QuadratureFunctions?
   occa::device device;

   // Reference to physical Jacobian for the initial mesh. These are computed
   // only at time zero and stored here.
   OccaVector Jac0inv;

   // Quadrature data used for full/partial assembly of the force operator. At
   // each quadrature point, it combines the stress, inverse Jacobian,
   // determinant of the Jacobian and the integration weight. It must be
   // recomputed in every time step.
   OccaVector stressJinvT;

   // Quadrature data used for full/partial assembly of the mass matrices. At
   // time zero, we compute and store (rho0 * det(J0) * qp_weight) at each
   // quadrature point. Note the at any other time, we can compute
   // rho = rho0 * det(J0) / det(J), representing the notion of pointwise mass
   // conservation.
   OccaVector rho0DetJ0w;

   // Initial length scale. This represents a notion of local mesh size. We
   // assume that all initial zones have similar size.
   double h0;

   // Estimate of the minimum time step over all quadrature points. This is
   // recomputed at every time step to achieve adaptive time stepping.
   double dt_est;

   // Occa stuff
   occa::properties props;

   OccaDofQuadMaps dqMaps;
   OccaGeometry geom;
   OccaVector dtEst;

   QuadratureData(int dim,
                  int elements,
                  int nqp);

   QuadratureData(occa::device device_,
                  int dim,
                  int elements,
                  int nqp);

   void Setup(occa::device device_,
              int dim,
              int elements,
              int nqp);
};

// This class is used only for visualization. It assembles (rho, phi) in each
// zone, which is used by LagrangianHydroOperator::ComputeDensity to do an L2
// projection of the density.
class DensityIntegrator
{
private:
   const QuadratureData &quad_data;

public:
   DensityIntegrator(QuadratureData &quad_data_) : quad_data(quad_data_) { }

   void AssembleRHSElementVect(const FiniteElement &fe,
                               ElementTransformation &Tr,
                               const IntegrationRule &integ_rule,
                               Vector &rho0DetJ0w,
                               Vector &elvect);
};

class OccaMassOperator : public Operator
{
private:
   occa::device device;

   int dim, elements;
   OccaFiniteElementSpace &fes;

   const IntegrationRule &integ_rule;

   int ess_tdofs_count;
   occa::memory ess_tdofs;

   OccaBilinearForm bilinearForm;
   Operator *massOperator;

   QuadratureData *quad_data;

   // For distributing X
   mutable OccaVector distX;
   mutable OccaGridFunction x_gf, y_gf;

public:
   OccaMassOperator(OccaFiniteElementSpace &fes_,
                    const IntegrationRule &integ_rule_,
                    QuadratureData *quad_data_);

   OccaMassOperator(occa::device device_,
                    OccaFiniteElementSpace &fes_,
                    const IntegrationRule &integ_rule_,
                    QuadratureData *quad_data_);

   void Setup();

   void SetEssentialTrueDofs(Array<int> &dofs);

   // Can be used for both velocity and specific internal energy. For the case
   // of velocity, we only work with one component at a time.
   virtual void Mult(const OccaVector &x, OccaVector &y) const;

   void EliminateRHS(OccaVector &b);
};

// Performs partial assembly, which corresponds to (and replaces) the use of the
// LagrangianHydroOperator::Force global matrix.
class OccaForceOperator : public Operator
{
private:
   occa::device device;
   int dim, elements;

   OccaFiniteElementSpace &h1fes, &l2fes;
   const IntegrationRule &integ_rule;

   QuadratureData *quad_data;

   occa::kernel multKernel, multTransposeKernel;

   OccaDofQuadMaps l2D2Q, h1D2Q;
   mutable OccaVector gVecL2, gVecH1;

   void MultHex(const Vector &vecL2, Vector &vecH1) const;
   void MultTransposeHex(const Vector &vecH1, Vector &vecL2) const;

public:
   OccaForceOperator(OccaFiniteElementSpace &h1fes_,
                     OccaFiniteElementSpace &l2fes_,
                     const IntegrationRule &integ_rule,
                     QuadratureData *quad_data_);

   OccaForceOperator(occa::device device_,
                     OccaFiniteElementSpace &h1fes_,
                     OccaFiniteElementSpace &l2fes_,
                     const IntegrationRule &integ_rule,
                     QuadratureData *quad_data_);

   void Setup();

   virtual void Mult(const OccaVector &vecL2, OccaVector &vecH1) const;
   virtual void MultTranspose(const OccaVector &vecH1, OccaVector &vecL2) const;

   ~OccaForceOperator() { }
};

} // namespace hydrodynamics
} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_LAGHOS_ASSEMBLY
