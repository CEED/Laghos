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
    void shift_shape(const ParFiniteElementSpace &pfes_e_const,
		     const ParFiniteElementSpace &pfes_p,
		     int e_id,
		     const IntegrationPoint &ip, const Vector &dist,
		     int nterms, Vector &shape_shift);
    void get_shifted_value(const ParGridFunction &g, int e_id,
			   const IntegrationPoint &ip, const Vector &dist,
			   int nterms, Vector &shifted_vec);
      
  
    // This class is used only for visualization. It assembles (rho, phi) in each
    // zone, which is used by LagrangianHydroOperator::ComputeDensity to do an L2
    // projection of the density.
    class DensityIntegrator : public LinearFormIntegrator
    {
      using LinearFormIntegrator::AssembleRHSElementVect;
    private:
      const ParGridFunction &rho0DetJ0_gf;

    public:
      DensityIntegrator(const ParGridFunction &rho0DetJ0_gf) : rho0DetJ0_gf(rho0DetJ0_gf) { }
      virtual void AssembleRHSElementVect(const FiniteElement &fe,
					  ElementTransformation &Tr,
					  Vector &elvect);
    };

    class SourceForceIntegrator : public LinearFormIntegrator
    {
      using LinearFormIntegrator::AssembleRHSElementVect;
    private:
      const ParGridFunction &rho_gf;
      const ParGridFunction *accel_src_gf;
      
    public:
      SourceForceIntegrator(const ParGridFunction &rho_gf) : rho_gf(rho_gf), accel_src_gf(NULL) { }
      virtual void AssembleRHSElementVect(const FiniteElement &fe,
					  ElementTransformation &Tr,
					  Vector &elvect);
      virtual void SetAccelerationGridFunction(const ParGridFunction * accel_gf){
	accel_src_gf = accel_gf;
      }
  
    };

    class WeightedVectorMassIntegrator : public BilinearFormIntegrator
    {
    private:
      const ParGridFunction &alpha;
      const ParGridFunction &rho_gf;
      
public:
      WeightedVectorMassIntegrator(const ParGridFunction &alphaF, const ParGridFunction &rho_gf, const IntegrationRule *ir) : BilinearFormIntegrator(ir), alpha(alphaF), rho_gf(rho_gf) {}
      virtual void AssembleElementMatrix(const FiniteElement &el,
					 ElementTransformation &Trans,
					 DenseMatrix &elmat);
      
      const IntegrationRule &GetRule(const FiniteElement &trial_fe,
				     const FiniteElement &test_fe,
				     ElementTransformation &Trans);
      
    };

    class WeightedMassIntegrator : public BilinearFormIntegrator
    {
    private:
      const ParGridFunction &alpha;
      const ParGridFunction &rho_gf;
      
public:
      WeightedMassIntegrator(const ParGridFunction &alphaF, const ParGridFunction &rho_gf, const IntegrationRule *ir) : BilinearFormIntegrator(ir), alpha(alphaF), rho_gf(rho_gf) {}
      virtual void AssembleElementMatrix(const FiniteElement &el,
					 ElementTransformation &Trans,
					 DenseMatrix &elmat);
      
      const IntegrationRule &GetRule(const FiniteElement &trial_fe,
				     const FiniteElement &test_fe,
				     ElementTransformation &Trans);
      
    };
    
    // Performs full assembly for the force operator.
    class ForceIntegrator : public LinearFormIntegrator
    {
    private:
      const ParGridFunction &alpha;
      const ParGridFunction &v_gf;
      const ParGridFunction &e_gf;
      const ParGridFunction &p_gf;
      const ParGridFunction &cs_gf;
      const ParGridFunction &Jac0inv_gf;
      const ParGridFunction &rho_gf;
      const double h0;
      const bool use_viscosity;
      const bool use_vorticity;
      
    public:
      ForceIntegrator(const double h0, const ParGridFunction &alphaF, const ParGridFunction &v_gf, const ParGridFunction &e_gf, const ParGridFunction &p_gf, const ParGridFunction &cs_gf, const ParGridFunction &rho_gf, const ParGridFunction &Jac0inv_gf, const bool use_viscosity, const bool use_vorticity) : h0(h0), alpha(alphaF), v_gf(v_gf), e_gf(e_gf), p_gf(p_gf), cs_gf(cs_gf), rho_gf(rho_gf), Jac0inv_gf(Jac0inv_gf), use_viscosity(use_viscosity), use_vorticity(use_vorticity)  { }
      virtual void AssembleRHSElementVect(const FiniteElement &el,
					  ElementTransformation &Tr,
					  Vector &elvect);
    };

    // Performs full assembly for the force operator.
    class EnergyForceIntegrator : public LinearFormIntegrator
    {
    private:
      const double h0;
      const ParGridFunction &alpha;
      const ParGridFunction &v_gf;
      const ParGridFunction &e_gf;
      const ParGridFunction &p_gf;
      const ParGridFunction &cs_gf;
      const ParGridFunction &Jac0inv_gf;
      const ParGridFunction &rho_gf;
      const bool use_viscosity;
      const bool use_vorticity;
      const ParGridFunction *Vnpt_gf;
      
    public:
      EnergyForceIntegrator(const double h0, const ParGridFunction &alphaF, const ParGridFunction &v_gf, const ParGridFunction &e_gf, const ParGridFunction &p_gf, const ParGridFunction &cs_gf, const ParGridFunction &rho_gf, const ParGridFunction &Jac0inv_gf, const bool use_viscosity, const bool use_vorticity) : h0(h0), alpha(alphaF), v_gf(v_gf), e_gf(e_gf), p_gf(p_gf), cs_gf(cs_gf), rho_gf(rho_gf), Jac0inv_gf(Jac0inv_gf), use_viscosity(use_viscosity), use_vorticity(use_vorticity), Vnpt_gf(NULL)  { }
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
      const double h0;
      const ParGridFunction &alpha;
      const ParGridFunction &pface_gf;
      const ParGridFunction &v_gf;
      const ParGridFunction &csface_gf;
      const ParGridFunction &rho0DetJ0face_gf;
      const ParGridFunction &Jac0invface_gf;
      const bool use_viscosity;
      const bool use_vorticity;
      
    public:
      VelocityBoundaryForceIntegrator(const double h0, const ParGridFunction &alphaF, const ParGridFunction &pface_gf, const ParGridFunction &v_gf, const ParGridFunction &csface_gf, const ParGridFunction &rho0DetJ0face_gf, const ParGridFunction &Jac0invface_gf, const bool use_viscosity, const bool use_vorticity) : h0(h0), alpha(alphaF), pface_gf(pface_gf), v_gf(v_gf), csface_gf(csface_gf), rho0DetJ0face_gf(rho0DetJ0face_gf), Jac0invface_gf(Jac0invface_gf), use_viscosity(use_viscosity), use_vorticity(use_vorticity) { }
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
      const double h0;
      const ParGridFunction &alpha;
      const ParGridFunction &pface_gf;
      const ParGridFunction &v_gf;
      const ParGridFunction *Vnpt_gf;
      const ParGridFunction &csface_gf;
      const ParGridFunction &rho0DetJ0face_gf;
      const ParGridFunction &Jac0invface_gf;
     	
      const bool use_viscosity;
      const bool use_vorticity;
      double c_N; 
      double c_NP1; 
      const ParGridFunction *Vn_gf;
      const ParGridFunction *Vnp1_gf;
        
    public:
      EnergyBoundaryForceIntegrator(const double h0, const ParGridFunction &alphaF, const ParGridFunction &pface_gf, const ParGridFunction &v_gf, const ParGridFunction &csface_gf, const ParGridFunction &rho0DetJ0face_gf,  const ParGridFunction &Jac0invface_gf , const bool use_viscosity, const bool use_vorticity) : h0(h0), alpha(alphaF), pface_gf(pface_gf), v_gf(v_gf), csface_gf(csface_gf), rho0DetJ0face_gf(rho0DetJ0face_gf), Jac0invface_gf(Jac0invface_gf), use_viscosity(use_viscosity), use_vorticity(use_vorticity), Vnpt_gf(NULL), Vn_gf(NULL), Vnp1_gf(NULL), c_N(0), c_NP1(0) { }
      virtual void AssembleRHSElementVect(const FiniteElement &el,
					  FaceElementTransformations &Tr,
					  Vector &elvect);
      virtual void AssembleRHSElementVect(const FiniteElement &el,
					  ElementTransformation &Tr,
					  Vector &elvect);
      virtual void SetVelocityGridFunctionAtNewState(const ParGridFunction * velocity_NPT, const ParGridFunction * velocity_N, const ParGridFunction * velocity_NP1){
	Vnpt_gf = velocity_NPT;
	Vn_gf = velocity_N;
    	Vnp1_gf = velocity_NP1;
      }
      virtual void SetCoefficients(double c_N_, double c_NP1_){
	c_N = c_N_;
	c_NP1 = c_NP1_;
      }
 
    };

    // Performs full assembly for the normal velocity mass matrix operator.
    class NormalVelocityMassIntegrator : public BilinearFormIntegrator
    {
    private:
      const double h0 ;
      const ParGridFunction &alpha;
      double penaltyParameter;
      const int order_v;
      const double &globalmax_rho;
      const double &globalmax_cs;
      const double &globalmax_viscous_coef;
      const ParGridFunction &rhoface_gf;
      const ParGridFunction &Jac0invface_gf;
      const ParGridFunction &rho0DetJ0face_gf;
      const ParGridFunction &v_gf;
      const ParGridFunction &csface_gf;
            
    public:
      NormalVelocityMassIntegrator(const double h0, const ParGridFunction &alphaF, double penaltyParameter, const int order_v, const ParGridFunction &rhoface_gf, const ParGridFunction &v_gf, const ParGridFunction &csface_gf, const ParGridFunction &Jac0invface_gf, const ParGridFunction & rho0DetJ0face_gf, const double &globalmax_rho, const double &globalmax_cs, const double &globalmax_viscous_coef) : h0(h0), alpha(alphaF), penaltyParameter(penaltyParameter), order_v(order_v), rhoface_gf(rhoface_gf), v_gf(v_gf), csface_gf(csface_gf), Jac0invface_gf(Jac0invface_gf), rho0DetJ0face_gf(rho0DetJ0face_gf), globalmax_rho(globalmax_rho), globalmax_cs(globalmax_cs), globalmax_viscous_coef(globalmax_viscous_coef) {  }
      virtual void AssembleFaceMatrix(const FiniteElement &fe,
				      const FiniteElement &fe2,
				      FaceElementTransformations &Tr,
				      DenseMatrix &elmat);
    };

    // Performs full assembly for the normal velocity mass matrix operator.
    class DiffusionNormalVelocityIntegrator : public LinearFormIntegrator
    {
    private:
      const double h0 ;
      const ParGridFunction &alpha;
      double penaltyParameter;
      const int order_v;
      const double &globalmax_rho;
      const double &globalmax_cs;
      const double &globalmax_viscous_coef;
      const ParGridFunction &rhoface_gf;
      const ParGridFunction &csface_gf;
      const ParGridFunction &Jac0invface_gf;
      const ParGridFunction &rho0DetJ0face_gf;
      const ParGridFunction &v_gf;
            
    public:
      DiffusionNormalVelocityIntegrator(const double h0, const ParGridFunction &alphaF, double penaltyParameter, const int order_v, const ParGridFunction &rhoface_gf, const ParGridFunction &v_gf, const ParGridFunction &Jac0invface_gf, const ParGridFunction & rho0DetJ0face_gf,  const ParGridFunction &csface_gf, const double &globalmax_rho, const double &globalmax_cs, const double &globalmax_viscous_coef) : h0(h0), alpha(alphaF), penaltyParameter(penaltyParameter), order_v(order_v), rhoface_gf(rhoface_gf), v_gf(v_gf), Jac0invface_gf(Jac0invface_gf), rho0DetJ0face_gf(rho0DetJ0face_gf), csface_gf(csface_gf), globalmax_rho(globalmax_rho), globalmax_cs(globalmax_cs), globalmax_viscous_coef(globalmax_viscous_coef) {  }
      virtual void AssembleRHSElementVect(const FiniteElement &el,
					  FaceElementTransformations &Tr,
					  Vector &elvect);
      virtual void AssembleRHSElementVect(const FiniteElement &el,
					  ElementTransformation &Tr,
					  Vector &elvect);
    };

      // Performs full assembly for the normal velocity mass matrix operator.
    class DiffusionEnergyNormalVelocityIntegrator : public LinearFormIntegrator
    {
    private:
      const double h0 ;
      const ParGridFunction &alpha;
      double penaltyParameter;
      const int order_v;
      const double &globalmax_rho;
      const double &globalmax_cs;
      const double &globalmax_viscous_coef;
      const ParGridFunction &rhoface_gf;
      const ParGridFunction &csface_gf;
      const ParGridFunction &Jac0invface_gf;
      const ParGridFunction &rho0DetJ0face_gf;
      const ParGridFunction &v_gf;
      const ParGridFunction *Vnpt_gf;
      double c_N; 
      double c_NP1; 
      const ParGridFunction *Vn_gf;
      const ParGridFunction *Vnp1_gf;
            
    public:
      DiffusionEnergyNormalVelocityIntegrator(const double h0, const ParGridFunction &alphaF, double penaltyParameter, const int order_v, const ParGridFunction &rhoface_gf, const ParGridFunction &v_gf, const ParGridFunction &Jac0invface_gf, const ParGridFunction & rho0DetJ0face_gf, const ParGridFunction &csface_gf, const double &globalmax_rho, const double &globalmax_cs, const double &globalmax_viscous_coef) : h0(h0), alpha(alphaF), penaltyParameter(penaltyParameter), order_v(order_v), rhoface_gf(rhoface_gf), v_gf(v_gf), Jac0invface_gf(Jac0invface_gf), rho0DetJ0face_gf(rho0DetJ0face_gf), csface_gf(csface_gf), globalmax_rho(globalmax_rho), globalmax_cs(globalmax_cs), globalmax_viscous_coef(globalmax_viscous_coef), Vnpt_gf(NULL), Vn_gf(NULL), Vnp1_gf(NULL), c_N(0), c_NP1(0) {  }
      virtual void AssembleRHSElementVect(const FiniteElement &el,
					  FaceElementTransformations &Tr,
					  Vector &elvect);
      virtual void AssembleRHSElementVect(const FiniteElement &el,
					  ElementTransformation &Tr,
					  Vector &elvect);
      virtual void SetVelocityGridFunctionAtNewState(const ParGridFunction * velocity_NPT, const ParGridFunction * velocity_N, const ParGridFunction * velocity_NP1){
	Vnpt_gf = velocity_NPT;
	Vn_gf = velocity_N;
    	Vnp1_gf = velocity_NP1;
      }
      virtual void SetCoefficients(double c_N_, double c_NP1_){
	c_N = c_N_;
	c_NP1 = c_NP1_;
      }
 
    };
    
    
    // Performs full assembly for the boundary force operator on the momentum equation.
    // < sigma_{ij} n_{j} , \psi_{i} > 
    class ShiftedVelocityBoundaryForceIntegrator : public LinearFormIntegrator
    {
    private:
      const ParMesh *pmesh;
      const ParGridFunction &alpha_gf;
      const ParGridFunction &pface_gf;
     
    public:
      ShiftedVelocityBoundaryForceIntegrator(const ParMesh *pmesh, const ParGridFunction &alpha_gf, const ParGridFunction &pface_gf) : pmesh(pmesh), alpha_gf(alpha_gf), pface_gf(pface_gf) { }
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
      const ParGridFunction &alpha_gf;
      const ParGridFunction &pface_gf;
      const ParGridFunction &v_gf;
      const ParGridFunction *Vnpt_gf;

      const ParGridFunction *distance_gf;
      const ParGridFunction *normal_gf;

      const ParGridFunction *distance_n_gf;
      const ParGridFunction *normal_n_gf;
      const ParGridFunction *distance_np1_gf;
      const ParGridFunction *normal_np1_gf;
     
      VectorCoefficient *vD;
      VectorCoefficient *vN;
      int nTerms;

      double c_N; 
      double c_NP1; 
      
      const ParGridFunction *Vn_gf;
      const ParGridFunction *Vnp1_gf;
        
    public:
      ShiftedEnergyBoundaryForceIntegrator(const ParMesh *pmesh, const ParGridFunction &alpha_gf, const ParGridFunction &pface_gf, const ParGridFunction &v_gf, VectorCoefficient *dist_vec, VectorCoefficient *normal_vec,  int nTerms) :  pmesh(pmesh), alpha_gf(alpha_gf), pface_gf(pface_gf), v_gf(v_gf), Vnpt_gf(NULL), Vn_gf(NULL), Vnp1_gf(NULL), c_N(0), c_NP1(0), vD(dist_vec), vN(normal_vec), distance_gf(NULL), normal_gf(NULL), distance_n_gf(NULL), normal_n_gf(NULL), distance_np1_gf(NULL), normal_np1_gf(NULL), nTerms(nTerms)  { }
      virtual void AssembleRHSElementVect(const FiniteElement &el,
					  const FiniteElement &el2,
					  FaceElementTransformations &Tr,
					  Vector &elvect);
      virtual void AssembleRHSElementVect(const FiniteElement &el,
					  ElementTransformation &Tr,
					  Vector &elvect) {}
      virtual void SetVelocityGridFunctionAtNewState(const ParGridFunction * velocity_NPT, const ParGridFunction * velocity_N, const ParGridFunction * velocity_NP1){
	Vnpt_gf = velocity_NPT;
	Vn_gf = velocity_N;
    	Vnp1_gf = velocity_NP1;
     }
      virtual void SetDistanceGridFunctionAtNewState(const ParGridFunction * distance_NPT, const ParGridFunction * distance_N, const ParGridFunction * distance_NP1){
	distance_gf = distance_NPT;
	distance_n_gf = distance_N;
    	distance_np1_gf = distance_NP1;
      }
      virtual void SetNormalGridFunctionAtNewState(const ParGridFunction * normal_NPT, const ParGridFunction * normal_N, const ParGridFunction * normal_NP1){
	normal_gf = normal_NPT;
	normal_n_gf = normal_N;
    	normal_np1_gf = normal_NP1;
     }
    
       virtual void SetCoefficients(double c_N_, double c_NP1_){
	c_N = c_N_;
	c_NP1 = c_NP1_;
      }

    };

    // Performs full assembly for the normal velocity mass matrix operator.
    class ShiftedNormalVelocityMassIntegrator : public BilinearFormIntegrator
    {
    private:
      const ParMesh *pmesh;
      const double hinit;
      const int order_v;
      const ParGridFunction &alpha_gf;
      VectorCoefficient *vD;
      VectorCoefficient *vN;
      int nTerms;
      bool fullPenalty;
      double penaltyParameter;
      const double &globalmax_rho;
      const double &globalmax_cs;
      const double &globalmax_viscous_coef;
      const ParGridFunction &csface_gf;
      const ParGridFunction &rhoface_gf;
      const ParGridFunction &viscousface_gf;
      const ParFiniteElementSpace &h1;
      const ParGridFunction &v_gf;
      const ParGridFunction &Jac0invface_gf;
      
    public:
      ShiftedNormalVelocityMassIntegrator(const double hinit, const ParMesh *pmesh, const ParFiniteElementSpace &h1, const ParGridFunction &alpha_gf, double penaltyParameter, const int order_v, const double &globalmax_rho, const double &globalmax_cs, const double &globalmax_viscous_coef, const ParGridFunction &rhoface_gf, const ParGridFunction &viscousface_gf, const ParGridFunction &csface_gf, const ParGridFunction &v_gf, const ParGridFunction &Jac0invface_gf, VectorCoefficient *dist_vec, VectorCoefficient *normal_vec, int nTerms, bool fP = 0) : hinit(hinit), pmesh(pmesh), h1(h1), alpha_gf(alpha_gf), penaltyParameter(penaltyParameter), order_v(order_v), globalmax_rho(globalmax_rho), globalmax_cs(globalmax_cs), globalmax_viscous_coef(globalmax_viscous_coef), rhoface_gf(rhoface_gf), viscousface_gf(viscousface_gf), csface_gf(csface_gf), v_gf(v_gf), Jac0invface_gf(Jac0invface_gf), vD(dist_vec), vN(normal_vec), nTerms(nTerms), fullPenalty(fP) { }
      virtual void AssembleFaceMatrix(const FiniteElement &fe,
				      const FiniteElement &fe2,
				      FaceElementTransformations &Tr,
				      DenseMatrix &elmat);
      
    };

    // Performs full assembly for the normal velocity mass matrix operator.
    class ShiftedDiffusionNormalVelocityIntegrator : public LinearFormIntegrator
    {
    private:
      const ParMesh *pmesh;
      const double hinit;
      const int order_v;
      const ParGridFunction &alpha_gf;
      VectorCoefficient *vD;
      VectorCoefficient *vN;
      int nTerms;
      bool fullPenalty;
      double penaltyParameter;
      const double &globalmax_rho;
      const double &globalmax_cs;
      const double &globalmax_viscous_coef;
      const ParGridFunction &csface_gf;
      const ParGridFunction &rhoface_gf;
      const ParGridFunction &viscousface_gf;
      const ParFiniteElementSpace &h1;
      const ParGridFunction &v_gf;
      const ParGridFunction &Jac0invface_gf;
      
    public:
      ShiftedDiffusionNormalVelocityIntegrator(const double hinit, const ParMesh *pmesh, const ParFiniteElementSpace &h1, const ParGridFunction &alpha_gf, double penaltyParameter, const int order_v, const double &globalmax_rho, const double &globalmax_cs, const double &globalmax_viscous_coef, const ParGridFunction &rhoface_gf, const ParGridFunction &viscousface_gf, const ParGridFunction &csface_gf, const ParGridFunction &v_gf, const ParGridFunction &Jac0invface_gf, VectorCoefficient *dist_vec, VectorCoefficient *normal_vec, int nTerms, bool fP = 0) : hinit(hinit), pmesh(pmesh), h1(h1), alpha_gf(alpha_gf), penaltyParameter(penaltyParameter), order_v(order_v), globalmax_rho(globalmax_rho), globalmax_cs(globalmax_cs), globalmax_viscous_coef(globalmax_viscous_coef), rhoface_gf(rhoface_gf), viscousface_gf(viscousface_gf), csface_gf(csface_gf), v_gf(v_gf), Jac0invface_gf(Jac0invface_gf), vD(dist_vec), vN(normal_vec), nTerms(nTerms), fullPenalty(fP) { }
      virtual void AssembleRHSElementVect(const FiniteElement &el,
					  const FiniteElement &el2,
					  FaceElementTransformations &Tr,
					  Vector &elvect);
      virtual void AssembleRHSElementVect(const FiniteElement &el,
					  ElementTransformation &Tr,
					  Vector &elvect) {}
    };

    // Performs full assembly for the normal velocity mass matrix operator.
    class ShiftedDiffusionEnergyNormalVelocityIntegrator : public LinearFormIntegrator
    {
    private:
      const ParMesh *pmesh;
      const double hinit;
      const int order_v;
      const ParGridFunction &alpha_gf;
      VectorCoefficient *vD;
      VectorCoefficient *vN;
      int nTerms;
      bool fullPenalty;
      double penaltyParameter;
      const double &globalmax_rho;
      const double &globalmax_cs;
      const double &globalmax_viscous_coef;
      const ParGridFunction &csface_gf;
      const ParGridFunction &rhoface_gf;
      const ParGridFunction &viscousface_gf;
      const ParFiniteElementSpace &h1;
      const ParGridFunction &v_gf;
      const ParGridFunction &Jac0invface_gf;
      const ParGridFunction *Vnpt_gf;

      double c_N; 
      double c_NP1; 
      
      const ParGridFunction *Vn_gf;
      const ParGridFunction *Vnp1_gf;

      const ParGridFunction *distance_gf;
      const ParGridFunction *normal_gf;
      const ParGridFunction *distance_n_gf;
      const ParGridFunction *normal_n_gf;
      const ParGridFunction *distance_np1_gf;
      const ParGridFunction *normal_np1_gf;
     
    public:
      ShiftedDiffusionEnergyNormalVelocityIntegrator(const double hinit, const ParMesh *pmesh, const ParFiniteElementSpace &h1, const ParGridFunction &alpha_gf, double penaltyParameter, const int order_v, const double &globalmax_rho, const double &globalmax_cs, const double &globalmax_viscous_coef, const ParGridFunction &rhoface_gf, const ParGridFunction &viscousface_gf, const ParGridFunction &csface_gf, const ParGridFunction &v_gf, const ParGridFunction &Jac0invface_gf, VectorCoefficient *dist_vec, VectorCoefficient *normal_vec, int nTerms, bool fP = 0) : hinit(hinit), pmesh(pmesh), h1(h1), alpha_gf(alpha_gf), penaltyParameter(penaltyParameter), order_v(order_v), globalmax_rho(globalmax_rho), globalmax_cs(globalmax_cs), globalmax_viscous_coef(globalmax_viscous_coef), rhoface_gf(rhoface_gf), viscousface_gf(viscousface_gf), csface_gf(csface_gf), v_gf(v_gf), Jac0invface_gf(Jac0invface_gf), vD(dist_vec), vN(normal_vec), distance_gf(NULL), normal_gf(NULL), distance_n_gf(NULL), normal_n_gf(NULL), distance_np1_gf(NULL), normal_np1_gf(NULL), nTerms(nTerms), fullPenalty(fP), Vnpt_gf(NULL), Vn_gf(NULL), Vnp1_gf(NULL), c_N(0), c_NP1(0) { }
      virtual void AssembleRHSElementVect(const FiniteElement &el,
					  const FiniteElement &el2,
					  FaceElementTransformations &Tr,
					  Vector &elvect);
      virtual void AssembleRHSElementVect(const FiniteElement &el,
					  ElementTransformation &Tr,
					  Vector &elvect) {}
      virtual void SetVelocityGridFunctionAtNewState(const ParGridFunction * velocity_NPT, const ParGridFunction * velocity_N, const ParGridFunction * velocity_NP1){
	Vnpt_gf = velocity_NPT;
	Vn_gf = velocity_N;
    	Vnp1_gf = velocity_NP1;
      }
      virtual void SetDistanceGridFunctionAtNewState(const ParGridFunction * distance_NPT, const ParGridFunction * distance_N, const ParGridFunction * distance_NP1){
	distance_gf = distance_NPT;
	distance_n_gf = distance_N;
    	distance_np1_gf = distance_NP1;
      }
      virtual void SetNormalGridFunctionAtNewState(const ParGridFunction * normal_NPT, const ParGridFunction * normal_N, const ParGridFunction * normal_NP1){
	normal_gf = normal_NPT;
	normal_n_gf = normal_N;
    	normal_np1_gf = normal_NP1;
     }
      virtual void SetCoefficients(double c_N_, double c_NP1_){
	c_N = c_N_;
	c_NP1 = c_NP1_;
      }
      
    };

    
  } // namespace hydrodynamics
  
} // namespace mfem

#endif // MFEM_LAGHOS_ASSEMBLY
