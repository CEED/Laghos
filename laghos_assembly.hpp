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
#include "marking.hpp"

using namespace std;
using namespace mfem;

namespace mfem
{

  namespace hydrodynamics
  {
    double factorial(int nTerms);
  
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
    class ForceIntegrator : public LinearFormIntegrator
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
      virtual void AssembleRHSElementVect(const FiniteElement &el,
					  ElementTransformation &Tr,
					  Vector &elvect);
    };

    // Performs full assembly for the force operator.
    class EnergyForceIntegrator : public LinearFormIntegrator
    {
    private:
      const QuadratureData &qdata;
      const ParGridFunction &v_gf;
      const ParGridFunction &e_gf;
      const ParGridFunction &p_gf;
      const ParGridFunction &cs_gf;
      const bool use_viscosity;
      const bool use_vorticity;
      const ParGridFunction *Vnpt_gf;
      
    public:
      EnergyForceIntegrator(QuadratureData &qdata, const ParGridFunction &v_gf, const ParGridFunction &e_gf, const ParGridFunction &p_gf, const ParGridFunction &cs_gf, const bool use_viscosity, const bool use_vorticity) : qdata(qdata), v_gf(v_gf), e_gf(e_gf), p_gf(p_gf), cs_gf(cs_gf), use_viscosity(use_viscosity), use_vorticity(use_vorticity), Vnpt_gf(NULL)  { }
      virtual void AssembleRHSElementVect(const FiniteElement &el,
					  ElementTransformation &Tr,
					  Vector &elvect);
      virtual void SetVelocityGridFunctionAtNewState(const ParGridFunction * velocity_NPT){
	Vnpt_gf = velocity_NPT;
      }
    };

    
    // Performs full assembly for the boundary force operator on the momentum equation.
    // < sigma_{ij} n_{j} , \psi_{i} > 
    class VelocityBoundaryForceIntegrator : public LinearFormIntegrator
    {
    private:
      const QuadratureDataGL &qdata;
      const ParGridFunction &pface_gf;
    public:
      VelocityBoundaryForceIntegrator(QuadratureDataGL &qdata, const ParGridFunction &pface_gf) : qdata(qdata), pface_gf(pface_gf) { }
      virtual void AssembleRHSElementVect(const FiniteElement &el,
					  FaceElementTransformations &Tr,
					  Vector &elvect);
      virtual void AssembleRHSElementVect(const FiniteElement &el,
					  ElementTransformation &Tr,
					  Vector &elvect);
    };

    // Performs full assembly for the boundary force operator on the energy equation.
    // < sigma_{ij} n_{j} n_{i}, \phi * v.n > 
    class EnergyBoundaryForceIntegrator : public LinearFormIntegrator
    {
    private:
      const QuadratureDataGL &qdata;
      const ParGridFunction &pface_gf;
      const ParGridFunction &v_gf;
      const ParGridFunction *Vnpt_gf;
      
    public:
      EnergyBoundaryForceIntegrator(QuadratureDataGL &qdata, const ParGridFunction &pface_gf, const ParGridFunction &v_gf) :  qdata(qdata), pface_gf(pface_gf), v_gf(v_gf), Vnpt_gf(NULL) { }
      virtual void AssembleRHSElementVect(const FiniteElement &el,
					  FaceElementTransformations &Tr,
					  Vector &elvect);
      virtual void AssembleRHSElementVect(const FiniteElement &el,
					  ElementTransformation &Tr,
					  Vector &elvect);
      virtual void SetVelocityGridFunctionAtNewState(const ParGridFunction * velocity_NPT){
	Vnpt_gf = velocity_NPT;
      }
 
    };

    // Performs full assembly for the normal velocity mass matrix operator.
    class NormalVelocityMassIntegrator : public BilinearFormIntegrator
    {
    private:
      const QuadratureDataGL &qdata;
    public:
      NormalVelocityMassIntegrator(const QuadratureDataGL &qdata) : qdata(qdata) { }
      virtual void AssembleFaceMatrix(const FiniteElement &fe,
				      const FiniteElement &fe2,
				      FaceElementTransformations &Tr,
				      DenseMatrix &elmat);
    };

    
    // Performs full assembly for the boundary force operator on the momentum equation.
    // < sigma_{ij} n_{j} , \psi_{i} > 
    class ShiftedVelocityBoundaryForceIntegrator : public LinearFormIntegrator
    {
    private:
      const ParMesh *pmesh;
      const QuadratureDataGL &qdata;
      const ParGridFunction &pface_gf;
      ShiftedFaceMarker *analyticalSurface;
      int par_shared_face_count;
      
    public:
      ShiftedVelocityBoundaryForceIntegrator(const ParMesh *pmesh, QuadratureDataGL &qdata, const ParGridFunction &pface_gf, ShiftedFaceMarker *analyticalSurface) : pmesh(pmesh), qdata(qdata), pface_gf(pface_gf), analyticalSurface(analyticalSurface), par_shared_face_count(0) { }
      virtual void AssembleRHSElementVect(const FiniteElement &el,
					  const FiniteElement &el2,
					  FaceElementTransformations &Tr,
					  Vector &elvect);
      virtual void AssembleRHSElementVect(const FiniteElement &el,
					  ElementTransformation &Tr,
					  Vector &elvect) {}
    };

    // Performs full assembly for the boundary force operator on the energy equation.
    // < sigma_{ij} n_{j} n_{i}, \phi * v.n > 
    class ShiftedEnergyBoundaryForceIntegrator : public LinearFormIntegrator
    {
    private:
      const ParMesh *pmesh;
      const QuadratureDataGL &qdata;
      const ParGridFunction &pface_gf;
      const ParGridFunction &v_gf;
      const ParGridFunction *Vnpt_gf;
      ShiftedFaceMarker *analyticalSurface;
      int par_shared_face_count;
      VectorCoefficient *vD;
      VectorCoefficient *vN;
      int nTerms;

    public:
      ShiftedEnergyBoundaryForceIntegrator(const ParMesh *pmesh, QuadratureDataGL &qdata, const ParGridFunction &pface_gf, const ParGridFunction &v_gf, ShiftedFaceMarker *analyticalSurface, VectorCoefficient *dist_vec, VectorCoefficient *normal_vec, int nTerms) :  pmesh(pmesh), qdata(qdata), pface_gf(pface_gf), v_gf(v_gf), Vnpt_gf(NULL), analyticalSurface(analyticalSurface), par_shared_face_count(0), vD(dist_vec), vN(normal_vec), nTerms(nTerms)  { }
      virtual void AssembleRHSElementVect(const FiniteElement &el,
					  const FiniteElement &el2,
					  FaceElementTransformations &Tr,
					  Vector &elvect);
      virtual void AssembleRHSElementVect(const FiniteElement &el,
					  ElementTransformation &Tr,
					  Vector &elvect) {}
      virtual void SetVelocityGridFunctionAtNewState(const ParGridFunction * velocity_NPT){
	Vnpt_gf = velocity_NPT;
      }
 
    };

    // Performs full assembly for the normal velocity mass matrix operator.
    class ShiftedNormalVelocityMassIntegrator : public BilinearFormIntegrator
    {
    private:
      const ParMesh *pmesh;
      const QuadratureDataGL &qdata;
      ShiftedFaceMarker *analyticalSurface;
      int par_shared_face_count;
      VectorCoefficient *vD;
      VectorCoefficient *vN;
      int nTerms;
      bool fullPenalty;
      
    public:
      ShiftedNormalVelocityMassIntegrator(const ParMesh *pmesh, QuadratureDataGL &qdata, ShiftedFaceMarker *analyticalSurface, VectorCoefficient *dist_vec, VectorCoefficient *normal_vec, int nTerms, bool fP = 0) : pmesh(pmesh), qdata(qdata), analyticalSurface(analyticalSurface), vD(dist_vec), vN(normal_vec), par_shared_face_count(0), nTerms(nTerms), fullPenalty(fP) { }
      virtual void AssembleFaceMatrix(const FiniteElement &fe,
				      const FiniteElement &fe2,
				      FaceElementTransformations &Tr,
				      DenseMatrix &elmat);
      
    };

    
  } // namespace hydrodynamics
  
} // namespace mfem

#endif // MFEM_LAGHOS_ASSEMBLY
