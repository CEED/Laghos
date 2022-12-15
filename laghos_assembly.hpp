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
#include "auxiliary_functions.hpp"

namespace mfem
{

  namespace hydrodynamics
  {

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
      const ParGridFunction &v_gf;
      const ParGridFunction &e_gf;
      const ParGridFunction &p_gf;
      const ParGridFunction &cs_gf;
      const bool use_viscosity;
      const bool use_vorticity;
      
    public:
      ForceIntegrator(QuadratureData &qdata, const ParGridFunction &v_gf, const ParGridFunction &e_gf, const ParGridFunction &p_gf, const ParGridFunction &cs_gf, const bool use_viscosity, const bool use_vorticity) : qdata(qdata), v_gf(v_gf), e_gf(e_gf), p_gf(p_gf), cs_gf(cs_gf), use_viscosity(use_viscosity), use_vorticity(use_vorticity)  { }
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

    public:
      VelocityBoundaryForceIntegrator(FaceQuadratureData &qdata) : qdata(qdata) { }
      virtual void AssembleFaceMatrix(const FiniteElement &trial_fe,
				      const FiniteElement &test_fe1,
				      FaceElementTransformations &Tr,
				      DenseMatrix &elmat);
    };

    // Performs full assembly for the boundary force operator on the energy equation.
    // < sigma_{ij} n_{j} n_{i}, \phi * v.n > 
    class EnergyBoundaryForceIntegrator : public BilinearFormIntegrator
    {
    private:
      const FaceQuadratureData &qdata;
    public:
      EnergyBoundaryForceIntegrator(FaceQuadratureData &qdata) : qdata(qdata) { }
      virtual void AssembleFaceMatrix(const FiniteElement &trial_fe,
				      const FiniteElement &test_fe1,
				      FaceElementTransformations &Tr,
				      DenseMatrix &elmat);
    };

    // Performs full assembly for the normal velocity mass matrix operator.
    class NormalVelocityMassIntegrator : public BilinearFormIntegrator
    {
    private:
      const FaceQuadratureData &qdata;
    public:
      NormalVelocityMassIntegrator(FaceQuadratureData &qdata) : qdata(qdata) { }
      virtual void AssembleFaceMatrix(const FiniteElement &fe,
				      const FiniteElement &fe2,
				      FaceElementTransformations &Tr,
				      DenseMatrix &elmat);
    };

  } // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_LAGHOS_ASSEMBLY
