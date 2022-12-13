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

#ifndef MFEM_AUXILIARY_FUNCTIONS
#define MFEM_AUXILIARY_FUNCTIONS

#include "mfem.hpp"

#ifdef MFEM_USE_MPI

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

      ParGridFunction &rho_gf;

      ParGridFunction &gamma_gf;

      // Quadrature data used for full/partial assembly of the force operator.
      // At each quadrature point, it combines the stress, inverse Jacobian,
      // determinant of the Jacobian and the integration weight.
      // It must be recomputed in every time step.
      DenseTensor stressJinvT;

      // Quadrature data used for full/partial assembly of the mass matrices.
      // At time zero, we compute and store (rho0 * det(J0) * qp_weight) at each
      // quadrature point. Note the at any other time, we can compute
      // rho = rho0 * det(J0) / det(J), representing the notion of pointwise mass
      // conservation.
      Vector rho0DetJ0w;

      // Initial length scale. This represents a notion of local mesh size.
      // We assume that all initial zones have similar size.
      double h0;

      // Estimate of the minimum time step over all quadrature points. This is
      // recomputed at every time step to achieve adaptive time stepping.
      double dt_est;
  
      QuadratureData(int dim, int NE, int quads_per_el, ParGridFunction &rho_gf, ParGridFunction &gamma_gf)
	: Jac0inv(dim, dim, NE * quads_per_el),
	  stressJinvT(NE * quads_per_el, dim, dim),
	  rho0DetJ0w(NE * quads_per_el),
	  rho_gf(rho_gf),
	  gamma_gf(gamma_gf){ }
    };

    // Container for all data needed at quadrature points.
    struct FaceQuadratureData
    {
      // evaluation of the norma stress at the face quadrature points
      DenseMatrix weightedNormalStress;

      // Scaling of the penalty term evaluated at the face quadrature points:
      // tau * (c_s * (rho + mu / h) + rho * vorticity * h)
      // tau: user-defined non-dimensional constant 
      // c_s: max. sound speed over all boundary faces/edges
      // rho: max. density over all boundary faces/edges
      // mu: max. artificial viscosity over all boundary faces/edges
      // vorticity: max. vorticity over all boundary faces/edges
      double normalVelocityPenaltyScaling;

      // Reference to physical Jacobian for the initial mesh.
      // These are computed only at time zero and stored here.
      DenseTensor Jac0inv;

      // Quadrature data used for full/partial assembly of the mass matrices.
      // At time zero, we compute and store (rho0 * det(J0) * qp_weight) at each
      // quadrature point. Note the at any other time, we can compute
      // rho = rho0 * det(J0) / det(J), representing the notion of pointwise mass
      // conservation.
      Vector rho0DetJ0w;

      FaceQuadratureData(int dim, int NE, int quads_per_faceel) : weightedNormalStress(NE * quads_per_faceel, dim),normalVelocityPenaltyScaling(0.0), rho0DetJ0w(NE * quads_per_faceel),Jac0inv(dim, dim, NE * quads_per_faceel) { }
    };

    void ComputeMaterialProperty(const double gamma,
				   const double rho, const double e,
				   double &p, double &cs);
    void ComputeMaterialProperties(int nvalues, const double gamma[],
				   const double rho[], const double e[],
				   double p[], double cs[]);
      
    void ComputeStress(const double p, const int dim, DenseMatrix &stress);
  
    void ComputeViscousStress(ElementTransformation &T, const ParGridFunction &v, const QuadratureData &qdata, const int qdata_quad_index, const bool use_viscosity, const bool use_vorticity, const double rho, const double sound_speed, const int dim, DenseMatrix &stress);
    double smooth_step_01(double x, double eps);

  } // namespace hydrodynamics
  
} // namespace mfem
#endif // MFEM_USE_MPI

#endif // MFEM_LAGHOS
