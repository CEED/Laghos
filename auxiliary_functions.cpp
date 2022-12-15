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
      
      // Compute L2 pressure at the quadrature points, element by element.
      for (int e = 0; e < NE; e++)
	{
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
    
    void UpdatePressure(const ParGridFunction &gamma_gf, const ParGridFunction &e_gf, const ParGridFunction &rho_gf, ParGridFunction &p_gf)
    {
      ParFiniteElementSpace *p_fespace = p_gf.ParFESpace();
      const int NE = p_fespace->GetParMesh()->GetNE();
      
      // Compute L2 pressure at the quadrature points, element by element.
      for (int e = 0; e < NE; e++)
	{
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

    void UpdateSoundSpeed(const ParGridFunction &gamma_gf, const ParGridFunction &e_gf, ParGridFunction &cs_gf)
    {
      ParFiniteElementSpace *p_fespace = cs_gf.ParFESpace();
      const int NE = p_fespace->GetParMesh()->GetNE();
      
      // Compute L2 pressure at the quadrature points, element by element.
      for (int e = 0; e < NE; e++)
	{
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

    void UpdateFaceDensity(const IntegrationRule &b_ir, const Vector &rho0DetJ0, ParGridFunction &rho_gf)
    {
      ParFiniteElementSpace *p_fespace = rho_gf.ParFESpace();
      const int NBE = p_fespace->GetParMesh()->GetNBE();
      const int nqp_face = b_ir.GetNPoints();
      
      // Compute L2 pressure at the quadrature points, element by element.
      for (int e = 0; e < NBE; e++)
	{
	  FaceElementTransformations *eltrans = p_fespace->GetParMesh()->GetBdrFaceTransformations(e);
	  const int faceElemNo = eltrans->ElementNo;

	  // The points (and their numbering) coincide with the nodes of p.
	  const IntegrationRule &ir = p_fespace->GetFE(e)->GetNodes();
	  const int nqp = ir.GetNPoints();
	  
	  for (int q = 0; q < nqp_face; q++)
	    {
	      const IntegrationPoint &ip_f = b_ir.IntPoint(q);
	      // Compute el1 quantities.
	      // Set the integration point in the face and the neighboring elements
	      eltrans->SetAllIntPoints(&ip_f);
	      const IntegrationPoint &eip = eltrans->GetElement1IntPoint();
	      ElementTransformation &Trans_el1 = eltrans->GetElement1Transformation();
	      Trans_el1.SetIntPoint(&eip);
	      const int elementNo = Trans_el1.ElementNo;
	      const double rho = rho0DetJ0(faceElemNo * nqp_face + q) / Trans_el1.Weight();
	      rho_gf(elementNo * nqp + q) = rho;
	    }
	}
    }
    
    void UpdateFacePressure(const IntegrationRule &b_ir, const ParGridFunction &gamma_gf, const ParGridFunction &e_gf, const ParGridFunction &rho_gf, ParGridFunction &p_gf)
    {
      ParFiniteElementSpace *p_fespace = p_gf.ParFESpace();
      const int NBE = p_fespace->GetParMesh()->GetNBE();
      const int nqp_face = b_ir.GetNPoints();
      
      // Compute L2 pressure at the quadrature points, element by element.
      for (int e = 0; e < NBE; e++)
	{
	  FaceElementTransformations *eltrans = p_fespace->GetParMesh()->GetBdrFaceTransformations(e);
	  const int faceElemNo = eltrans->ElementNo;
	  // The points (and their numbering) coincide with the nodes of p.
	  const IntegrationRule &ir = p_fespace->GetFE(e)->GetNodes();
	  const int nqp = ir.GetNPoints();
	  
	  for (int q = 0; q < nqp_face; q++)
	    {
	      const IntegrationPoint &ip_f = b_ir.IntPoint(q);
	      // Compute el1 quantities.
	      // Set the integration point in the face and the neighboring elements
	      eltrans->SetAllIntPoints(&ip_f);
	      const IntegrationPoint &eip = eltrans->GetElement1IntPoint();
	      ElementTransformation &Trans_el1 = eltrans->GetElement1Transformation();
	      Trans_el1.SetIntPoint(&eip);
	      const int elementNo = Trans_el1.ElementNo;
	      const double gamma_val = gamma_gf.GetValue(Trans_el1, eip);
	      const double e_val = fmax(0.0,e_gf.GetValue(Trans_el1, eip));
	      const double rho_val = rho_gf.GetValue(Trans_el1, eip);
	      p_gf(elementNo * nqp + q) = (gamma_val - 1.0) * rho_val * e_val;
	    }
	}
    }

    void UpdateFaceSoundSpeed(const IntegrationRule &b_ir, const ParGridFunction &gamma_gf, const ParGridFunction &e_gf, ParGridFunction &cs_gf)
    {
      ParFiniteElementSpace *p_fespace = cs_gf.ParFESpace();
      const int NBE = p_fespace->GetParMesh()->GetNBE();
      const int nqp_face = b_ir.GetNPoints();
      
      // Compute L2 pressure at the quadrature points, element by element.
      for (int e = 0; e < NBE; e++)
	{
	  FaceElementTransformations *eltrans = p_fespace->GetParMesh()->GetBdrFaceTransformations(e);
	  const int faceElemNo = eltrans->ElementNo;	
	  // The points (and their numbering) coincide with the nodes of p.
	  const IntegrationRule &ir = p_fespace->GetFE(e)->GetNodes();
	  const int nqp = ir.GetNPoints();
	  
	  for (int q = 0; q < nqp_face; q++)
	    {
	      const IntegrationPoint &ip_f = b_ir.IntPoint(q);
	      // Compute el1 quantities.
	      // Set the integration point in the face and the neighboring elements
	      eltrans->SetAllIntPoints(&ip_f);
	      const IntegrationPoint &eip = eltrans->GetElement1IntPoint();
	      ElementTransformation &Trans_el1 = eltrans->GetElement1Transformation();
	      Trans_el1.SetIntPoint(&eip);
	      const int elementNo = Trans_el1.ElementNo;
	      double gamma_val = gamma_gf.GetValue(Trans_el1, eip);
	      double e_val = fmax(0.0,e_gf.GetValue(Trans_el1, eip));
	      cs_gf(elementNo * nqp + q) = sqrt(gamma_val * (gamma_val-1.0) * e_val);
	    }
	}
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

