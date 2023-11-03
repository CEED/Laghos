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

using namespace std;
using namespace mfem;

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

// Performs full assembly for the normal velocity mass matrix operator.
class NormalVelocityMassIntegrator : public BilinearFormIntegrator
{
private:
   const double h0;
   double penaltyParameter;
   double perimeter;
   const double globalmax_rho;

public:
   NormalVelocityMassIntegrator(const double h0, double penaltyParameter, double perimeter,
                                const double globalmax_rho)
      : h0(h0), penaltyParameter(penaltyParameter),
        perimeter(perimeter), globalmax_rho(globalmax_rho) {  }

   virtual void AssembleFaceMatrix(const FiniteElement &fe,
                                   const FiniteElement &fe2,
                                   FaceElementTransformations &Tr,
                                   DenseMatrix &elmat);
};

// Velocity RHS boundary integrator.
class VelocityPenaltyBLFI : public LinearFormIntegrator
{
private:
   const double h0;
   double penaltyParameter;
   double perimeter;
   const ParGridFunction &rhoface_gf;
   const ParGridFunction &v_gf;
   const ParGridFunction &pface_gf;
   const ParGridFunction &csface_gf;

public:
   VelocityPenaltyBLFI(const double h0, double penaltyParameter,
                       double perimeter,
                       const ParGridFunction &rhoface_gf,
                       const ParGridFunction &v_gf,
                       const ParGridFunction &pface_gf,
                       const ParGridFunction &csface_gf)
      : h0(h0), penaltyParameter(penaltyParameter), perimeter(perimeter),
        rhoface_gf(rhoface_gf), v_gf(v_gf),
        pface_gf(pface_gf), csface_gf(csface_gf) { }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect);
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect)
   { MFEM_ABORT("Should not be used."); }
};

// Energy RHS boundary integrator.
class EnergyPenaltyBLFI : public LinearFormIntegrator
{
private:
   const double h0;
   double penaltyParameter;
   double perimeter;
   const ParGridFunction &rhoface_gf;
   const ParGridFunction &csface_gf;
   const ParGridFunction &v_gf;
   const ParGridFunction &pface_gf;
   const ParGridFunction *Vnpt_gf;

public:
   EnergyPenaltyBLFI(const double h0, double penaltyParameter,
                     double perimeter,
                     const ParGridFunction &rhoface_gf,
                     const ParGridFunction &v_gf,
                     const ParGridFunction &pface_gf,
                     const ParGridFunction &csface_gf)
      : h0(h0), penaltyParameter(penaltyParameter), perimeter(perimeter),
        rhoface_gf(rhoface_gf), csface_gf(csface_gf), v_gf(v_gf),
        pface_gf(pface_gf), Vnpt_gf(NULL) {  }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect);
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect)
   { MFEM_ABORT("Should not be used."); }

   virtual void SetVelocityGridFunctionAtNewState(const ParGridFunction *v_new)
   { Vnpt_gf = v_new; }
};

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_LAGHOS_ASSEMBLY
