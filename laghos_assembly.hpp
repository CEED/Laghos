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
#include "general/forall.hpp"
#include "linalg/dtensor.hpp"

namespace mfem
{

namespace hydrodynamics
{

// Container for all data needed at quadrature points.
struct QuadratureData
{
   // Reference to physical Jacobian for the initial mesh.
   // These are computed only at time zero and stored here.
   DenseTensor Jac0inv;

   // Quadrature data used for full/partial assembly of the force operator.
   // At each quadrature point, it combines the stress, inverse Jacobian,
   // determinant of the Jacobian and the integration weight.
   // It must be recomputed in every time step.
   DenseTensor stressJinvT_1, stressJinvT_2, stressJinvT_tot;

   // Quadrature data used for full/partial assembly of the mass matrices.
   // At time zero, we compute and store (rho0 * det(J0) * qp_weight) at each
   // quadrature point. Note the at any other time, we can compute
   // rho = rho0 * det(J0) / det(J), representing the notion of pointwise mass
   // conservation.
   Vector rho0DetJ0w_1, rho0DetJ0w_2;

   // Initial length scale. This represents a notion of local mesh size.
   // We assume that all initial zones have similar size.
   double h0;

   // Estimate of the minimum time step over all quadrature points. This is
   // recomputed at every time step to achieve adaptive time stepping.
   double dt_est;

   QuadratureData(int dim, int NE, int quads_per_el)
      : Jac0inv(dim, dim, NE * quads_per_el),
        stressJinvT_1(NE * quads_per_el, dim, dim),
        stressJinvT_2(NE * quads_per_el, dim, dim),
        stressJinvT_tot(NE * quads_per_el, dim, dim),
        rho0DetJ0w_1(NE * quads_per_el), rho0DetJ0w_2(NE * quads_per_el) { }
};

// Container for all data needed at quadrature points.
struct CutFaceQuadratureData
{
   // Quadrature data used for full/partial assembly of the mass matrices.
   // At time zero, we compute and store (rho0 * det(J0) * qp_weight) at each
   // quadrature point. Note the at any other time, we can compute
   // rho = rho0 * det(J0) / det(J), representing the notion of pointwise mass
   // conservation.
   Vector rho0DetJ0, rhocs;

   CutFaceQuadratureData() : rho0DetJ0(), rhocs() { }
};

// This class is used only for visualization. It assembles (rho, phi) in each
// zone, which is used by LagrangianHydroOperator::ComputeDensity to do an L2
// projection of the density.
class DensityIntegrator : public LinearFormIntegrator
{
   using LinearFormIntegrator::AssembleRHSElementVect;
private:
   const Vector &rho0DetJ0w;

public:
   DensityIntegrator(const Vector &rDJw) : rho0DetJ0w(rDJw) { }
   virtual void AssembleRHSElementVect(const FiniteElement &fe,
                                       ElementTransformation &Tr,
                                       Vector &elvect);
};

// Performs full assembly for the force operator:
// F_ij = int_zone [ (sigma:h1_shape_j) l2_shape_i ].
class ForceIntegrator : public BilinearFormIntegrator
{
private:
   const DenseTensor &stressJinvT;

public:
   ForceIntegrator(const DenseTensor &stressJiT) : stressJinvT(stressJiT) { }

   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Tr,
                                       DenseMatrix &elmat);
};

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_LAGHOS_ASSEMBLY
