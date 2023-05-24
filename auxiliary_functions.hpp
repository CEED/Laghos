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
#include "marking.hpp"
#include "sbm_aux.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{
  namespace hydrodynamics
  {
    
    // Container for all data needed at quadrature points.
    struct QuadratureData
    {
      // Initial length scale. This represents a notion of local mesh size.
      // We assume that all initial zones have similar size.
      double h0;

      // Estimate of the minimum time step over all quadrature points. This is
      // recomputed at every time step to achieve adaptive time stepping.
      double dt_est;
  
      QuadratureData()
      { }
    };

    double factorial(int nTerms);
     
    void LengthScaleAndCompression(const DenseMatrix &sgrad_v,
				   ElementTransformation &T,
				   const DenseMatrix &Jac0inv, double h0,
				   double &h, double &mu);
    
    void UpdateDensity(const ParGridFunction &rho0DetJ0_gf, const ParGridFunction &alpha, ParGridFunction &rho_gf);
  
    void UpdatePressure(const ParGridFunction &gamma_gf, const ParGridFunction &e_gf, const ParGridFunction &rho_gf, ParGridFunction &p_gf);

    void UpdateSoundSpeed(const ParGridFunction &gamma_gf, const ParGridFunction &e_gf, ParGridFunction &cs_gf);
    
    void UpdatePenaltyParameter(double &globalmax_rho, double &globalmax_cs, double &globalmax_viscous_coef, const ParGridFunction &rho_gf, const ParGridFunction &cs_gf, const ParGridFunction &v, const ParGridFunction &Jac0invface_gf, ParGridFunction &viscous_gf, VectorCoefficient * dist_vec,  const double h0, const bool use_viscosity, const bool use_vorticity, const bool useEmbedded, const double penaltyParameter);
      
    void ComputeMaterialProperty(const double gamma,
				   const double rho, const double e,
				   double &p, double &cs);
    void ComputeMaterialProperties(int nvalues, const double gamma[],
				   const double rho[], const double e[],
				   double p[], double cs[]);
      
    void ComputeStress(const double p, const int dim, DenseMatrix &stress);
  
    void ComputeViscousStress(ElementTransformation &T, const ParGridFunction &v, const DenseMatrix &Jac0inv, const double h0,  const bool use_viscosity, const bool use_vorticity, const double rho, const double sound_speed, const int dim, DenseMatrix &stress);
    
    void ConvertVectorToDenseMatrix(const int dim, const Vector & vec, DenseMatrix &mat);

    double smooth_step_01(double x, double eps);
    
    class Jac0InvVectorFunctionCoefficient : public VectorCoefficient
    {
    protected:
      int dim;
    public:
      Jac0InvVectorFunctionCoefficient(int dim, int dof)
	: dim(dim), VectorCoefficient(dof) { }
      
      using VectorCoefficient::Eval;
      virtual void Eval(Vector &V, ElementTransformation &T,
			const IntegrationPoint &ip)
      {
	
	T.SetIntPoint(&ip);
	DenseMatrixInverse Jinv(T.Jacobian());
	DenseMatrix Jinv_dummy(dim);
	Jinv_dummy = 0.0;
	Jinv.GetInverseMatrix(Jinv_dummy);
	
	if (dim == 2){
	  V(0) = Jinv_dummy(0,0);
	  V(1) = Jinv_dummy(0,1);
	  V(2) = Jinv_dummy(1,0);
	  V(3) = Jinv_dummy(1,1);
	}
	else {
	  V(0) = Jinv_dummy(0,0);
	  V(1) = Jinv_dummy(0,1);
	  V(2) = Jinv_dummy(0,2);
	  V(3) = Jinv_dummy(1,0);
	  V(4) = Jinv_dummy(1,1);
	  V(5) = Jinv_dummy(1,2);
	  V(6) = Jinv_dummy(2,0);
	  V(7) = Jinv_dummy(2,1);
	  V(8) = Jinv_dummy(2,2);
	}
      } 
    };
      
  } // namespace hydrodynamics
  
} // namespace mfem
#endif // MFEM_USE_MPI

#endif // MFEM_LAGHOS
