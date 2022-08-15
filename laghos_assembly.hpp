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
#include "AnalyticalGeometricShape.hpp"
#include "AnalyticalSurface.hpp"

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
  
  QuadratureData(int dim, int NE, int quads_per_el)
      : Jac0inv(dim, dim, NE * quads_per_el),
        stressJinvT(NE * quads_per_el, dim, dim),
        rho0DetJ0w(NE * quads_per_el) { }
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
   Vector normalVelocityPenaltyScaling;

  // Reference to physical Jacobian for the initial mesh.
   // These are computed only at time zero and stored here.
   DenseTensor Jac0inv;

    // Quadrature data used for full/partial assembly of the mass matrices.
   // At time zero, we compute and store (rho0 * det(J0) * qp_weight) at each
   // quadrature point. Note the at any other time, we can compute
   // rho = rho0 * det(J0) / det(J), representing the notion of pointwise mass
   // conservation.
   Vector rho0DetJ0w;

  FaceQuadratureData(int dim, int NE, int quads_per_faceel) : weightedNormalStress(NE * quads_per_faceel, dim),normalVelocityPenaltyScaling(NE * quads_per_faceel), rho0DetJ0w(NE * quads_per_faceel), Jac0inv(dim, dim, NE * quads_per_faceel) { }
};

// This class is used only for visualization. It assembles (rho, phi) in each
// zone, which is used by LagrangianHydroOperator::ComputeDensity to do an L2
// projection of the density.
class DensityIntegrator : public LinearFormIntegrator
{
   using LinearFormIntegrator::AssembleRHSElementVect;
private:
   const QuadratureData &qdata;

public:
   DensityIntegrator(QuadratureData &qdata) : qdata(qdata) { }
   virtual void AssembleRHSElementVect(const FiniteElement &fe,
                                       ElementTransformation &Tr,
                                       Vector &elvect);
};

// Performs full assembly for the force operator.
class ForceIntegrator : public BilinearFormIntegrator
{
private:
   const QuadratureData &qdata;
public:
   ForceIntegrator(QuadratureData &qdata) : qdata(qdata) { }
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Tr,
                                       DenseMatrix &elmat);
};

  // Performs full assembly for the boundary force operator on the momentum equation.
   // < sigma_{ij} n_{j} , \psi_{i} > 
class VelocityBoundaryForceIntegrator : public BilinearFormIntegrator
{
private:
  const FaceQuadratureData &qdata;
  Array<int> elemStatus;

public:
  VelocityBoundaryForceIntegrator(FaceQuadratureData &qdata, Array<int> elementStatus) : qdata(qdata), elemStatus(elementStatus) { }
   virtual void AssembleFaceMatrix(const FiniteElement &trial_fe,
				   const FiniteElement &test_fe,
				   FaceElementTransformations &Tr,
				   DenseMatrix &elmat);
};

  // Performs full assembly for the boundary force operator on the energy equation.
  // < sigma_{ij} n_{j} n_{i}, \phi * v.n > 
class EnergyBoundaryForceIntegrator : public BilinearFormIntegrator
{
private:
  const ParMesh *pmesh;
  const FaceQuadratureData &qdata;
  Array<int> elemStatus;
  AnalyticalSurface *analyticalSurface;

public:
  EnergyBoundaryForceIntegrator(const ParMesh *pmesh, FaceQuadratureData &qdata, AnalyticalSurface *analyticalSurface, Array<int> elementStatus) : pmesh(pmesh), qdata(qdata), analyticalSurface(analyticalSurface), elemStatus(elementStatus) { }
   virtual void AssembleFaceMatrix(const FiniteElement &trial_fe,
				   const FiniteElement &test_fe,
				   FaceElementTransformations &Tr,
				   DenseMatrix &elmat);
};

// Performs full assembly for the normal velocity mass matrix operator.
class NormalVelocityMassIntegrator : public BilinearFormIntegrator
{
private:
  const ParMesh *pmesh;
   const FaceQuadratureData &qdata;
   Array<int> elemStatus;
   AnalyticalSurface *analyticalSurface;

public:
  NormalVelocityMassIntegrator(const ParMesh *pmesh, FaceQuadratureData &qdata, AnalyticalSurface *analyticalSurface, Array<int> elementStatus) : pmesh(pmesh), qdata(qdata), analyticalSurface(analyticalSurface), elemStatus(elementStatus) { }
   virtual void AssembleFaceMatrix(const FiniteElement &el1,
				   const FiniteElement &el2,
                                       FaceElementTransformations &Tr,
                                       DenseMatrix &elmat);

};

  // Performs full assembly for the boundary force operator on the momentum equation.
   // < sigma_{ij} n_{j} , \psi_{i} > 
class ShiftedVelocityBoundaryForceIntegrator : public BilinearFormIntegrator
{
private:
  const ParMesh *pmesh;
  AnalyticalSurface *analyticalSurface;
  const FaceQuadratureData &qdata;
  Array<int> elemStatus;
  Array<int> faceTags;
  ParFiniteElementSpace &trial_L2;
  ParFiniteElementSpace &test_H1;
  
public:
  ShiftedVelocityBoundaryForceIntegrator(const ParMesh *pmesh, AnalyticalSurface *analyticalSurface, FaceQuadratureData &qdata, Array<int> elementStatus, Array<int> faceTag, ParFiniteElementSpace &l2, ParFiniteElementSpace &h1) : pmesh(pmesh), analyticalSurface(analyticalSurface), qdata(qdata), elemStatus(elementStatus), faceTags(faceTag), trial_L2(l2), test_H1(h1) { }
  virtual void AssembleFaceMatrix(const FiniteElement &trial_fe1,
				  const FiniteElement &trial_fe2,
				  const FiniteElement &test_fe1,
				  const FiniteElement &test_fe2,
				  FaceElementTransformations &Trans,
				  DenseMatrix &elmat,
				  Array<int> &trial_vdofs,
				  Array<int> &test_vdofs);
};

  // Performs full assembly for the boundary force operator on the energy equation.
  // < sigma_{ij} n_{j} n_{i}, \phi * v.n > 
class ShiftedEnergyBoundaryForceIntegrator : public BilinearFormIntegrator
{
private:
   const ParMesh *pmesh;
   AnalyticalSurface *analyticalSurface;
   const FaceQuadratureData &qdata;
   Array<int> elemStatus;
   Array<int> faceTags;
   ParFiniteElementSpace &trial_L2;
   ParFiniteElementSpace &test_H1;
  
public:
  ShiftedEnergyBoundaryForceIntegrator(const ParMesh *pmesh, AnalyticalSurface *analyticalSurface, FaceQuadratureData &qdata, Array<int> elementStatus, Array<int> faceTag, ParFiniteElementSpace &l2, ParFiniteElementSpace &h1) : pmesh(pmesh), analyticalSurface(analyticalSurface), qdata(qdata), elemStatus(elementStatus), faceTags(faceTag), trial_L2(l2), test_H1(h1) { }
   virtual void AssembleFaceMatrix(const FiniteElement &trial_fe1,
				   const FiniteElement &trial_fe2,
				   const FiniteElement &test_fe1,
				   const FiniteElement &test_fe2,
				   FaceElementTransformations &Trans,
				   DenseMatrix &elmat,
				   Array<int> &trial_vdofs,
				   Array<int> &test_vdofs);
};

// Performs full assembly for the normal velocity mass matrix operator.
class ShiftedNormalVelocityMassIntegrator : public BilinearFormIntegrator
{
private:
   const ParMesh *pmesh;
   AnalyticalSurface *analyticalSurface;
   const FaceQuadratureData &qdata;
   Array<int> elemStatus;
   Array<int> faceTags;
   ParFiniteElementSpace &H1;
  
public:
  ShiftedNormalVelocityMassIntegrator(const ParMesh *pmesh, AnalyticalSurface *analyticalSurface, FaceQuadratureData &qdata, Array<int> elementStatus, Array<int> faceTag, ParFiniteElementSpace &h1) : pmesh(pmesh), analyticalSurface(analyticalSurface), qdata(qdata), elemStatus(elementStatus), faceTags(faceTag), H1(h1) { }
  virtual void AssembleFaceMatrix(const FiniteElement &el1,
				  const FiniteElement &el2,
				  FaceElementTransformations &Trans,
				  DenseMatrix &elmat,
				  Array<int> &vdofs);
  
};
  
}// namespace hydrodynamics

} // namespace mfem

#endif // MFEM_LAGHOS_ASSEMBLY
