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
    void UpdateDensity(const Vector &rho0DetJ0, ParGridFunction &rho_gf)
    {
      ParFiniteElementSpace *p_fespace = rho_gf.ParFESpace();
      const int NE = p_fespace->GetParMesh()->GetNE();
      const ParMesh * pmesh = p_fespace->GetParMesh(); 
      rho_gf = 0.0;
      // Compute L2 pressure at the quadrature points, element by element.
      for (int e = 0; e < NE; e++)
	{
	  if ((pmesh->GetAttribute(e) == ShiftedFaceMarker::SBElementType::INSIDE) ||  (pmesh->GetAttribute(e) == ShiftedFaceMarker::SBElementType::GHOST)){	 
	    // The points (and their numbering) coincide with the nodes of p.
	    const IntegrationRule &ir = p_fespace->GetFE(e)->GetNodes();
	    const int nqp = ir.GetNPoints();
	    
	    ElementTransformation &Tr = *p_fespace->GetElementTransformation(e);
	    for (int q = 0; q < nqp; q++)
	      {
		const IntegrationPoint &ip = ir.IntPoint(q);
		Tr.SetIntPoint(&ip);
		const double rho = rho0DetJ0(e * nqp + q) / Tr.Weight();
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
	  if ((pmesh->GetAttribute(e) == ShiftedFaceMarker::SBElementType::INSIDE) ||  (pmesh->GetAttribute(e) == ShiftedFaceMarker::SBElementType::GHOST)){	 
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
	  if ((pmesh->GetAttribute(e) == ShiftedFaceMarker::SBElementType::INSIDE) ||  (pmesh->GetAttribute(e) == ShiftedFaceMarker::SBElementType::GHOST)){	 
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
    
    void UpdateDensityGL(const Vector &rho0DetJ0, ParGridFunction &rho_gf)
    {
      ParFiniteElementSpace *p_fespace = rho_gf.ParFESpace();
      const int NE = p_fespace->GetParMesh()->GetNE();
      const ParMesh * pmesh = p_fespace->GetParMesh(); 
      rho_gf = 0.0;
      // Compute L2 pressure at the quadrature points, element by element.
      for (int e = 0; e < NE; e++)
	{
	  if ((pmesh->GetAttribute(e) == ShiftedFaceMarker::SBElementType::INSIDE) ||  (pmesh->GetAttribute(e) == ShiftedFaceMarker::SBElementType::GHOST)){	 
	    // The points (and their numbering) coincide with the nodes of p.
	    const IntegrationRule &ir = p_fespace->GetFE(e)->GetNodes();
	    const int nqp = ir.GetNPoints();
	    
	    ElementTransformation &Tr = *p_fespace->GetElementTransformation(e);
	    for (int q = 0; q < nqp; q++)
	      {
		const IntegrationPoint &ip = ir.IntPoint(q);
		Tr.SetIntPoint(&ip);
		const double rho = rho0DetJ0(e * nqp + q) / Tr.Weight();
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
	  if ((pmesh->GetAttribute(e) == ShiftedFaceMarker::SBElementType::INSIDE) ||  (pmesh->GetAttribute(e) == ShiftedFaceMarker::SBElementType::GHOST)){	 
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
    
    void UpdateSoundSpeedGL(const ParGridFunction &gamma_gf, const ParGridFunction &e_gf, ParGridFunction &cs_gf)
    {
      ParFiniteElementSpace *p_fespace = cs_gf.ParFESpace();
      const int NE = p_fespace->GetParMesh()->GetNE();
      const ParMesh * pmesh = p_fespace->GetParMesh(); 
      cs_gf = 0.0;
      // Compute L2 pressure at the quadrature points, element by element.
      for (int e = 0; e < NE; e++)
	{
	  if ((pmesh->GetAttribute(e) == ShiftedFaceMarker::SBElementType::INSIDE) ||  (pmesh->GetAttribute(e) == ShiftedFaceMarker::SBElementType::GHOST)){	 
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

    void UpdatePenaltyParameterGL(ParGridFunction &penaltyScaling_gf, const ParGridFunction &rho_gf, const ParGridFunction &cs_gf, const ParGridFunction &v, VectorCoefficient * dist_vec, const QuadratureDataGL &qdata, const double h0, const bool use_viscosity, const bool use_vorticity, const bool useEmbedded, const double penaltyParameter)
    {
      ParFiniteElementSpace *p_fespace = cs_gf.ParFESpace();
      const int NE = p_fespace->GetParMesh()->GetNE();
      const ParMesh * pmesh = p_fespace->GetParMesh();
      int dim = pmesh->Dimension();
      penaltyScaling_gf = 0.0;
      double max_standard_coef = 0.0;
      double max_viscous_coef = 0.0;
      double min_h = 10000.0;
      double max_cs = 0.0;
      double max_rho = 0.0;
      double max_mu = 0.0;
      double max_vorticity = 0.0;
      double max_h = 0.0;
      double max_smooth_step = 0.0;
      // Compute L2 pressure at the quadrature points, element by element.
      for (int e = 0; e < NE; e++)
	{
	  if ((pmesh->GetAttribute(e) == ShiftedFaceMarker::SBElementType::INSIDE) ||  (pmesh->GetAttribute(e) == ShiftedFaceMarker::SBElementType::GHOST)){	 
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
		max_standard_coef = std::max(max_standard_coef, rho_vals * sound_speed);
		penaltyScaling_gf(e * nqp + q) = penaltyParameter * rho_vals * sound_speed;
		/////
		Vector D_el1(dim);
		D_el1 = 0.0;
		double normD = 0.0;
		if (useEmbedded){
		  dist_vec->Eval(D_el1, Tr, ip);
		}
		//		for (int j = 0; j < dim; j++){
		//	  normD += D_el1(j) * D_el1(j);
		//}
	    //		normD = std::pow(normD,0.5);
		////
		double visc_coeff = 0.0;
		DenseMatrix Jpi(dim), sgrad_v(dim), Jinv(dim);
		if (use_viscosity)
		  {
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
		    max_vorticity = std::max(max_vorticity, vorticity_coeff);
		        
		    sgrad_v.Symmetrize();
		    double eig_val_data[3], eig_vec_data[9];
		    if (dim==1)
		      {
			eig_val_data[0] = sgrad_v(0, 0);
			eig_vec_data[0] = 1.;
		      }
		    else { sgrad_v.CalcEigenvalues(eig_val_data, eig_vec_data); }
		    Vector compr_dir(eig_vec_data, dim);
		    mfem::Mult(Tr.Jacobian(), qdata.Jac0inv(e*nqp + q), Jpi);
		    Vector ph_dir(dim); Jpi.Mult(compr_dir, ph_dir);
		    // Change of the initial mesh size in the compression direction.
		    const double h = h0 * ph_dir.Norml2() / compr_dir.Norml2();
		    // Measure of maximal compression.
		    const double mu = eig_val_data[0];
		    visc_coeff = 2.0 * rho_vals * (h+normD) * (h+normD) * fabs(mu);
		    
		    // The following represents a "smooth" version of the statement
		    // "if (mu < 0) visc_coeff += 0.5 rho h sound_speed".  Note that
		    // eps must be scaled appropriately if a different unit system is
		    // being used.
		   
		    const double eps = 1e-12;
		    visc_coeff += 0.5 * rho_vals * (h+normD) * sound_speed * vorticity_coeff * (1.0 - smooth_step_01(mu - 2.0 * eps, eps));
		    
		    max_viscous_coef = std::max(max_viscous_coef, visc_coeff / (h+normD));
		    min_h = std::min(min_h, h);
		    max_mu = std::max(max_mu, std::fabs(mu));
		    max_h = std::max(max_h, h);
		    max_smooth_step = std::max(max_smooth_step, 1.0 - smooth_step_01(mu - 2.0 * eps, eps));
		    penaltyScaling_gf(e * nqp + q) += penaltyParameter * visc_coeff * (1.0 /  (h+normD));
		  }
	      }
	  }
	}

      double globalmax_standard_coef = 0.0;
      double globalmax_viscous_coef = 0.0;
      double globalmin_h = 0.0;
      double globalmax_mu = 0.0;
      double globalmax_rho = 0.0;
      double globalmax_cs = 0.0;
      double globalmax_h = 0.0;
      double globalmax_smooth_step = 0.0;
      double globalmax_vorticity = 0.0;
     
      MPI_Allreduce(&max_standard_coef, &globalmax_standard_coef, 1, MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
      MPI_Allreduce(&max_viscous_coef, &globalmax_viscous_coef, 1, MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
      MPI_Allreduce(&min_h, &globalmin_h, 1, MPI_DOUBLE, MPI_MIN, pmesh->GetComm());
      MPI_Allreduce(&max_mu, &globalmax_mu, 1, MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
      MPI_Allreduce(&max_rho, &globalmax_rho, 1, MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
      MPI_Allreduce(&max_cs, &globalmax_cs, 1, MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
      MPI_Allreduce(&max_h, &globalmax_h, 1, MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
      MPI_Allreduce(&max_smooth_step, &globalmax_smooth_step, 1, MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
      MPI_Allreduce(&max_vorticity, &globalmax_vorticity, 1, MPI_DOUBLE, MPI_MAX, pmesh->GetComm());

      penaltyScaling_gf = penaltyParameter * (globalmax_standard_coef + globalmax_viscous_coef);
      //  penaltyScaling_gf = penaltyParameter * (globalmax_rho * globalmax_cs + globalmax_viscous_coef);
      //     penaltyScaling_gf = penaltyParameter * (globalmax_rho * globalmax_cs + globalmax_mu / globalmin_h);
      // penaltyScaling_gf = penaltyParameter * (globalmax_rho * globalmax_cs + (0.5 * globalmax_rho * globalmax_h * globalmax_cs * globalmax_vorticity * globalmax_smooth_step + 2.0 * globalmax_rho * globalmax_h * globalmax_h * fabs(globalmax_mu) )/ globalmin_h );
      //     std::cout << " val " << (globalmax_standard_coef + globalmax_viscous_coef) << " visc " << globalmax_viscous_coef << std::endl;
      // std::cout << " old " << (globalmax_rho * globalmax_cs + globalmax_mu / globalmin_h) << std::endl;
      
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

    void ComputeViscousStress(ElementTransformation &T, const ParGridFunction &v, const QuadratureData &qdata, const int qdata_quad_index, const bool use_viscosity, const bool use_vorticity, const double rho, const double sound_speed, const int dim, DenseMatrix &stress)
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

