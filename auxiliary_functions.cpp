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

#include "auxiliary_functions.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{
  namespace hydrodynamics
  {
        void LengthScaleAndCompression(const DenseMatrix &sgrad_v,
				   ElementTransformation &T,
				   const DenseMatrix &Jac0inv, double h0,
				   double &h, double &mu)
    {
      const int dim = sgrad_v.Height();
      
      double eig_val_data[3], eig_vec_data[9];
      if (dim == 1)
	{
	  eig_val_data[0] = sgrad_v(0, 0);
	  eig_vec_data[0] = 1.;
	}
      else { sgrad_v.CalcEigenvalues(eig_val_data, eig_vec_data); }
      
      DenseMatrix Jpi(dim);
      // Computes the initial->physical transformation Jacobian.
      Mult(T.Jacobian(), Jac0inv, Jpi);
      Vector compr_dir(eig_vec_data, dim), ph_dir(dim);
      Jpi.Mult(compr_dir, ph_dir);
      
      // Change of the initial mesh size in the compression direction.
      h = h0 * ph_dir.Norml2() / compr_dir.Norml2();
      // Measure of maximal compression.
      mu = eig_val_data[0];
    }

    void UpdateDensity(const Vector &rho0DetJ0, const ParGridFunction &alpha, ParGridFunction &rho_gf)
    {
      ParFiniteElementSpace *p_fespace = rho_gf.ParFESpace();
      const int NE = p_fespace->GetParMesh()->GetNE();
      const ParMesh * pmesh = p_fespace->GetParMesh(); 
      rho_gf = 0.0;
      // Compute L2 pressure at the quadrature points, element by element.
      for (int e = 0; e < NE; e++)
	{
	  if ( (pmesh->GetAttribute(e) == ShiftedFaceMarker::SBElementType::INSIDE) || (pmesh->GetAttribute(e) == ShiftedFaceMarker::SBElementType::CUT)){
	    // The points (and their numbering) coincide with the nodes of p.
	    const IntegrationRule &ir = p_fespace->GetFE(e)->GetNodes();
	    const int nqp = ir.GetNPoints();
	    
	    ElementTransformation &Tr = *p_fespace->GetElementTransformation(e);
	    for (int q = 0; q < nqp; q++)
	      {
		const IntegrationPoint &ip = ir.IntPoint(q);
		Tr.SetIntPoint(&ip);
		double volumeFraction = alpha.GetValue(Tr, ip);
		const double rho = rho0DetJ0(e * nqp + q) / (Tr.Weight() * volumeFraction);
		rho_gf(e * nqp + q) = rho;
	      }
	  }
	}
    }
    
    void UpdatePressure(const ParGridFunction &gamma_gf, const ParGridFunction &e_gf, const ParGridFunction &rho_gf, ParGridFunction &p_gf)
    {
      ParFiniteElementSpace *p_fespace = p_gf.ParFESpace();
      const int NE = p_fespace->GetParMesh()->GetNE();
      const ParMesh * pmesh = p_fespace->GetParMesh(); 
      p_gf = 0.0;      
      // Compute L2 pressure at the quadrature points, element by element.
      for (int e = 0; e < NE; e++)
	{
	  if ( (pmesh->GetAttribute(e) == ShiftedFaceMarker::SBElementType::INSIDE) || (pmesh->GetAttribute(e) == ShiftedFaceMarker::SBElementType::CUT) ){
	    // The points (and their numbering) coincide with the nodes of p.
	    const IntegrationRule &ir = p_fespace->GetFE(e)->GetNodes();
	    const int nqp = ir.GetNPoints();
	    
	    ElementTransformation &Tr = *p_fespace->GetElementTransformation(e);
	    for (int q = 0; q < nqp; q++)
	      {
		const IntegrationPoint &ip = ir.IntPoint(q);
		Tr.SetIntPoint(&ip);
		const double gamma_val = gamma_gf.GetValue(Tr, ip);
		const double e_val = fmax(0.0,e_gf.GetValue(Tr, ip));
		const double rho_val = rho_gf.GetValue(Tr, ip);
		p_gf(e * nqp + q) = (gamma_val - 1.0) * rho_val * e_val;
	      }
	  }
	}
    }
    
    void UpdateSoundSpeed(const ParGridFunction &gamma_gf, const ParGridFunction &e_gf, ParGridFunction &cs_gf)
    {
      ParFiniteElementSpace *p_fespace = cs_gf.ParFESpace();
      const int NE = p_fespace->GetParMesh()->GetNE();
      const ParMesh * pmesh = p_fespace->GetParMesh(); 
      cs_gf = 0.0;
      // Compute L2 pressure at the quadrature points, element by element.
      for (int e = 0; e < NE; e++)
	{
	  if ( (pmesh->GetAttribute(e) == ShiftedFaceMarker::SBElementType::INSIDE) || (pmesh->GetAttribute(e) == ShiftedFaceMarker::SBElementType::CUT) ){
	    // The points (and their numbering) coincide with the nodes of p.
	    const IntegrationRule &ir = p_fespace->GetFE(e)->GetNodes();
	    const int nqp = ir.GetNPoints();
	    
	    ElementTransformation &Tr = *p_fespace->GetElementTransformation(e);
	    for (int q = 0; q < nqp; q++)
	      {
		const IntegrationPoint &ip = ir.IntPoint(q);
		Tr.SetIntPoint(&ip);
		double gamma_val = gamma_gf.GetValue(Tr, ip);
		double e_val = fmax(0.0,e_gf.GetValue(Tr, ip));
		cs_gf(e * nqp + q) = sqrt(gamma_val * (gamma_val-1.0) * e_val);
	      }
	  }
	}
    }
    
    void UpdateDensityGL(const Vector &rho0DetJ0, const ParGridFunction &alpha, ParGridFunction &rho_gf)
    {
      ParFiniteElementSpace *p_fespace = rho_gf.ParFESpace();
      const int NE = p_fespace->GetParMesh()->GetNE();
      const ParMesh * pmesh = p_fespace->GetParMesh(); 
      rho_gf = 0.0;
      // Compute L2 pressure at the quadrature points, element by element.
      for (int e = 0; e < NE; e++)
	{
	  if ( (pmesh->GetAttribute(e) == ShiftedFaceMarker::SBElementType::INSIDE) || (pmesh->GetAttribute(e) == ShiftedFaceMarker::SBElementType::CUT) ){
	    // The points (and their numbering) coincide with the nodes of p.
	    const IntegrationRule &ir = p_fespace->GetFE(e)->GetNodes();
	    const int nqp = ir.GetNPoints();
	    
	    ElementTransformation &Tr = *p_fespace->GetElementTransformation(e);
	    for (int q = 0; q < nqp; q++)
	      {
		const IntegrationPoint &ip = ir.IntPoint(q);
		Tr.SetIntPoint(&ip);
		double volumeFraction = alpha.GetValue(Tr, ip);
		const double rho = rho0DetJ0(e * nqp + q) / (Tr.Weight() * volumeFraction);
		rho_gf(e * nqp + q) = rho;
	      }
	  }
	}
    }
    
    void UpdatePressureGL(const ParGridFunction &gamma_gf, const ParGridFunction &e_gf, const ParGridFunction &rho_gf, ParGridFunction &p_gf)
    {
      ParFiniteElementSpace *p_fespace = p_gf.ParFESpace();
      const int NE = p_fespace->GetParMesh()->GetNE();
      const ParMesh * pmesh = p_fespace->GetParMesh();
      p_gf = 0.0;
      // Compute L2 pressure at the quadrature points, element by element.
      for (int e = 0; e < NE; e++)
	{
	  if ( (pmesh->GetAttribute(e) == ShiftedFaceMarker::SBElementType::INSIDE) || (pmesh->GetAttribute(e) == ShiftedFaceMarker::SBElementType::CUT) ){
	    // The points (and their numbering) coincide with the nodes of p.
	    const IntegrationRule &ir = p_fespace->GetFE(e)->GetNodes();
	    const int nqp = ir.GetNPoints();
	    
	    ElementTransformation &Tr = *p_fespace->GetElementTransformation(e);
	    for (int q = 0; q < nqp; q++)
	      {
		const IntegrationPoint &ip = ir.IntPoint(q);
		Tr.SetIntPoint(&ip);
		//	std::cout << " ip.x " << ip.x << " ip.y " << ip.y << std::endl;
		const double gamma_val = gamma_gf.GetValue(Tr, ip);
		const double e_val = fmax(0.0,e_gf.GetValue(Tr, ip));
		const double rho_val = rho_gf.GetValue(Tr, ip);
		p_gf(e * nqp + q) = (gamma_val - 1.0) * rho_val * e_val;
	      }
	  }
	}
    }
    
    void UpdateSoundSpeedGL(const ParGridFunction &gamma_gf, const ParGridFunction &e_gf, ParGridFunction &cs_gf)
    {
      ParFiniteElementSpace *p_fespace = cs_gf.ParFESpace();
      const int NE = p_fespace->GetParMesh()->GetNE();
      const ParMesh * pmesh = p_fespace->GetParMesh(); 
      cs_gf = 0.0;
      // Compute L2 pressure at the quadrature points, element by element.
      for (int e = 0; e < NE; e++)
	{
	  if ( (pmesh->GetAttribute(e) == ShiftedFaceMarker::SBElementType::INSIDE) || (pmesh->GetAttribute(e) == ShiftedFaceMarker::SBElementType::CUT)){	 
	    // The points (and their numbering) coincide with the nodes of p.
	    const IntegrationRule &ir = p_fespace->GetFE(e)->GetNodes();
	    const int nqp = ir.GetNPoints();
	    
	    ElementTransformation &Tr = *p_fespace->GetElementTransformation(e);
	    for (int q = 0; q < nqp; q++)
	      {
		const IntegrationPoint &ip = ir.IntPoint(q);
		Tr.SetIntPoint(&ip);
		double gamma_val = gamma_gf.GetValue(Tr, ip);
		double e_val = fmax(0.0,e_gf.GetValue(Tr, ip));
		cs_gf(e * nqp + q) = sqrt(gamma_val * (gamma_val-1.0) * e_val);
	      }
	  }
	}
    }

    void UpdatePenaltyParameter(double &globalmax_rho, double &globalmax_cs, double &globalmax_viscous_coef, const ParGridFunction &rho_gf, const ParGridFunction &cs_gf, const ParGridFunction &v, const ParGridFunction &Jac0invface_gf, ParGridFunction &viscous_gf,  VectorCoefficient * dist_vec, const double h0, const bool use_viscosity, const bool use_vorticity, const bool useEmbedded, const double penaltyParameter)
    {
      ParFiniteElementSpace *p_fespace = cs_gf.ParFESpace();
      const int NE = p_fespace->GetParMesh()->GetNE();
      const ParMesh * pmesh = p_fespace->GetParMesh();
      int dim = pmesh->Dimension();
    
      double max_viscous_coef = 0.0;
      double max_cs = 0.0;
      double max_rho = 0.0;
      viscous_gf = 0.0;
      // Compute L2 pressure at the quadrature points, element by element.
      for (int e = 0; e < NE; e++)
	{
	  if ( (pmesh->GetAttribute(e) == ShiftedFaceMarker::SBElementType::INSIDE) || (pmesh->GetAttribute(e) == ShiftedFaceMarker::SBElementType::CUT) ){
	    // The points (and their numbering) coincide with the nodes of p.
	    const IntegrationRule &ir = p_fespace->GetFE(e)->GetNodes();
	    const int nqp = ir.GetNPoints();
	    
	    ElementTransformation &Tr = *p_fespace->GetElementTransformation(e);
	    
	    for (int q = 0; q < nqp; q++)
	      {
		const IntegrationPoint &ip = ir.IntPoint(q);
		Tr.SetIntPoint(&ip);
		const double detJ = (Tr.Jacobian()).Det();
		const DenseMatrix & Jac = Tr.Jacobian();
		double rho_vals = rho_gf.GetValue(Tr,ip);
		double sound_speed = cs_gf.GetValue(Tr,ip);
		max_rho = std::max(max_rho, rho_vals);
		max_cs = std::max(max_cs, sound_speed);
		/////
		Vector D_el1(dim);
		D_el1 = 0.0;
		double normD = 0.0;
		if (useEmbedded){
		  dist_vec->Eval(D_el1, Tr, ip);
		}

		Vector Jac0inv_vec(dim*dim);
		Jac0inv_vec = 0.0;
		Jac0invface_gf.GetVectorValue(Tr.ElementNo,ip,Jac0inv_vec);
		
		DenseMatrix Jac0inv(dim);
		if (dim == 2){
		  Jac0inv(0,0) = Jac0inv_vec(0);
		  Jac0inv(0,1) = Jac0inv_vec(1);
		  Jac0inv(1,0) = Jac0inv_vec(2);
		  Jac0inv(1,1) = Jac0inv_vec(3);
		}
		else {
		  Jac0inv(0,0) = Jac0inv_vec(0);
		  Jac0inv(0,1) = Jac0inv_vec(1);
		  Jac0inv(0,2) = Jac0inv_vec(2);
		  Jac0inv(1,0) = Jac0inv_vec(3);
		  Jac0inv(1,1) = Jac0inv_vec(4);
		  Jac0inv(1,2) = Jac0inv_vec(5);
		  Jac0inv(2,0) = Jac0inv_vec(6);
		  Jac0inv(2,1) = Jac0inv_vec(7);
		  Jac0inv(2,2) = Jac0inv_vec(8);
		}

		
		//		for (int j = 0; j < dim; j++){
		//	  normD += D_el1(j) * D_el1(j);
		//}
		//		normD = std::pow(normD,0.5);
		////
		if (use_viscosity){
		  double visc_coeff = 0.0;
		  DenseMatrix Jpi(dim), sgrad_v(dim), Jinv(dim);
		  // Compression-based length scale at the point. The first
		  // eigenvector of the symmetric velocity lgradient gives the
		  // direction of maximal compression. This is used to define the
		  // relative change of the initial length scale.
		  v.GetVectorGradient(Tr, sgrad_v);
		  
		  double vorticity_coeff = 1.0;
		  if (use_vorticity)
		    {
		      const double grad_norm = sgrad_v.FNorm();
		      const double div_v = fabs(sgrad_v.Trace());
		      vorticity_coeff = (grad_norm > 0.0) ? div_v / grad_norm : 1.0;
		    }
		  
		  sgrad_v.Symmetrize();
		  double eig_val_data[3], eig_vec_data[9];
		  if (dim==1)
		    {
		      eig_val_data[0] = sgrad_v(0, 0);
		      eig_vec_data[0] = 1.;
		    }
		  else { sgrad_v.CalcEigenvalues(eig_val_data, eig_vec_data); }
		  Vector compr_dir(eig_vec_data, dim);
		  mfem::Mult(Tr.Jacobian(), Jac0inv, Jpi);
		  Vector ph_dir(dim); Jpi.Mult(compr_dir, ph_dir);
		  // Change of the initial mesh size in the compression direction.
		  const double h = h0 * ph_dir.Norml2() / compr_dir.Norml2();
		  
		  // Measure of maximal compression.
		  const double mu = eig_val_data[0];
		  visc_coeff = 2.0 * rho_vals * h * h * fabs(mu);
		  
		  // The following represents a "smooth" version of the statement
		  // "if (mu < 0) visc_coeff += 0.5 rho h sound_speed".  Note that
		  // eps must be scaled appropriately if a different unit system is
		  // being used.
		  
		  const double eps = 1e-12;
		  visc_coeff += 0.5 * rho_vals * h * sound_speed * vorticity_coeff * (1.0 - smooth_step_01(mu - 2.0 * eps, eps));
		  viscous_gf(e * nqp + q) = visc_coeff;
		  max_viscous_coef = std::max(max_viscous_coef, visc_coeff);
		}
	      }
	  }
	}
      globalmax_viscous_coef = 0.0;
      globalmax_rho = 0.0;
      globalmax_cs = 0.0;
      MPI_Allreduce(&max_viscous_coef, &globalmax_viscous_coef, 1, MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
      MPI_Allreduce(&max_rho, &globalmax_rho, 1, MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
      MPI_Allreduce(&max_cs, &globalmax_cs, 1, MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
      
    }
 

    void ComputeMaterialProperty(const double gamma,
				 const double rho, const double e,
				 double &p, double &cs)
    {
      p  = (gamma - 1.0) * rho * e;
      cs = sqrt(gamma * (gamma-1.0) * e);
    }

    void ComputeMaterialProperties(int nvalues, const double gamma[],
				   const double rho[], const double e[],
				   double p[], double cs[])
    {
      for (int v = 0; v < nvalues; v++)
	{
	  p[v]  = (gamma[v] - 1.0) * rho[v] * e[v];
	  cs[v] = sqrt(gamma[v] * (gamma[v]-1.0) * e[v]);
	}
    }

    void ComputeStress(const double p, const int dim, DenseMatrix &stress)
    {
      stress = 0.0;
      for (int d = 0; d < dim; d++) { stress(d, d) = -p;}
    }

    void ComputeViscousStress(ElementTransformation &T, const ParGridFunction &v, const DenseMatrix &Jac0inv,  const double h0, const bool use_viscosity, const bool use_vorticity, const double rho, const double sound_speed, const int dim, DenseMatrix &stress)
    {
      if (use_viscosity)
	{
	  // Jacobians of reference->physical transformations for all quadrature points
	  // in the batch.
	  const DenseMatrix &Jpr = T.Jacobian();
	  double visc_coeff;
	  visc_coeff = 0.0;	
	  DenseMatrix Jpi(dim), sgrad_v(dim), Jinv(dim);
	  Jpi = 0.0;
	  sgrad_v = 0.0;

	  CalcInverse(Jpr, Jinv);
	
	  v.GetVectorGradient(T, sgrad_v);
	
	  double vorticity_coeff = 1.0;
	  if (use_vorticity)
	    {
	      const double grad_norm = sgrad_v.FNorm();
	      const double div_v = fabs(sgrad_v.Trace());
	      vorticity_coeff = (grad_norm > 0.0) ? div_v / grad_norm : 1.0;
	    }
	
	  sgrad_v.Symmetrize();
	  double eig_val_data[3], eig_vec_data[9];
	  if (dim==1)
	    {
	      eig_val_data[0] = sgrad_v(0, 0);
	      eig_vec_data[0] = 1.;
	    }
	  else { sgrad_v.CalcEigenvalues(eig_val_data, eig_vec_data); }
	  Vector compr_dir(eig_vec_data, dim);
	  // Computes the initial->physical transformation Jacobian.
	  mfem::Mult(Jpr, Jac0inv, Jpi);
	  Vector ph_dir(dim); Jpi.Mult(compr_dir, ph_dir);
	  // Change of the initial mesh size in the compression direction.
	  const double h = h0 * ph_dir.Norml2() / compr_dir.Norml2();
	  // Measure of maximal compression.
	  const double mu = eig_val_data[0];
	  visc_coeff = 2.0 * rho * h * h * fabs(mu);
	  // The following represents a "smooth" version of the statement
	  // "if (mu < 0) visc_coeff += 0.5 rho h sound_speed".  Note that
	  // eps must be scaled appropriately if a different unit system is
	  // being used.
	  const double eps = 1e-12;
	  visc_coeff += 0.5 * rho * h * sound_speed * vorticity_coeff *
	    (1.0 - smooth_step_01(mu - 2.0 * eps, eps));
	  stress.Add(visc_coeff, sgrad_v);
	}
    }

    void ComputeViscousStressGL(ElementTransformation &T, const ParGridFunction &v, const QuadratureDataGL &qdata, const int qdata_quad_index, const bool use_viscosity, const bool use_vorticity, const double rho, const double sound_speed, const int dim, DenseMatrix &stress)
    {
      if (use_viscosity)
	{
	  // Jacobians of reference->physical transformations for all quadrature points
	  // in the batch.
	  const DenseMatrix &Jpr = T.Jacobian();
	  double visc_coeff;
	  visc_coeff = 0.0;	
	  DenseMatrix Jpi(dim), sgrad_v(dim), Jinv(dim);
	  Jpi = 0.0;
	  sgrad_v = 0.0;

	  CalcInverse(Jpr, Jinv);
	
	  v.GetVectorGradient(T, sgrad_v);
	
	  double vorticity_coeff = 1.0;
	  if (use_vorticity)
	    {
	      const double grad_norm = sgrad_v.FNorm();
	      const double div_v = fabs(sgrad_v.Trace());
	      vorticity_coeff = (grad_norm > 0.0) ? div_v / grad_norm : 1.0;
	    }
	
	  sgrad_v.Symmetrize();
	  double eig_val_data[3], eig_vec_data[9];
	  if (dim==1)
	    {
	      eig_val_data[0] = sgrad_v(0, 0);
	      eig_vec_data[0] = 1.;
	    }
	  else { sgrad_v.CalcEigenvalues(eig_val_data, eig_vec_data); }
	  Vector compr_dir(eig_vec_data, dim);
	  // Computes the initial->physical transformation Jacobian.
	  mfem::Mult(Jpr, qdata.Jac0inv(qdata_quad_index), Jpi);
	  Vector ph_dir(dim); Jpi.Mult(compr_dir, ph_dir);
	  // Change of the initial mesh size in the compression direction.
	  const double h = qdata.h0 * ph_dir.Norml2() / compr_dir.Norml2();
	  // Measure of maximal compression.
	  const double mu = eig_val_data[0];
	  visc_coeff = 2.0 * rho * h * h * fabs(mu);
	  // The following represents a "smooth" version of the statement
	  // "if (mu < 0) visc_coeff += 0.5 rho h sound_speed".  Note that
	  // eps must be scaled appropriately if a different unit system is
	  // being used.
	  const double eps = 1e-12;
	  visc_coeff += 0.5 * rho * h * sound_speed * vorticity_coeff *
	    (1.0 - smooth_step_01(mu - 2.0 * eps, eps));
	  stress.Add(visc_coeff, sgrad_v);
	}
    }
 
    // Smooth transition between 0 and 1 for x in [-eps, eps].
    double smooth_step_01(double x, double eps)
    {
      const double y = (x + eps) / (2.0 * eps);
      if (y < 0.0) { return 0.0; }
      if (y > 1.0) { return 1.0; }
      return (3.0 - 2.0 * y) * y * y;
    }
  } // namespace hydrodynamics
  
}
#endif // MFEM_USE_MPI
