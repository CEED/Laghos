// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project (17-SC-20-SC)
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include "laghos_assembly.hpp"
#include <unordered_map>

namespace mfem
{

  namespace hydrodynamics
  {

    void DensityIntegrator::AssembleRHSElementVect(const FiniteElement &fe,
						   ElementTransformation &Tr,
						   Vector &elvect)
    {
      const int nqp = IntRule->GetNPoints();
      Vector shape(fe.GetDof());
      elvect.SetSize(fe.GetDof());
      elvect = 0.0;
      for (int q = 0; q < nqp; q++)
	{
	  const IntegrationPoint &ip = IntRule->IntPoint(q);
	  fe.CalcShape(ip, shape);
	  Tr.SetIntPoint (&ip);
	  // Note that rhoDetJ = rho0DetJ0.
	  shape *= qdata.rho0DetJ0(Tr.ElementNo*nqp + q) * ip.weight;
	  elvect += shape;
	}
    }

    void ForceIntegrator::AssembleRHSElementVect(const FiniteElement &el,
						 ElementTransformation &Tr,
						 Vector &elvect)
    {
      const int e = Tr.ElementNo;
      const int nqp = IntRule->GetNPoints();
      const int dim = el.GetDim();
      const int h1dofs_cnt = el.GetDof();
      elvect.SetSize(h1dofs_cnt*dim);
      elvect = 0.0;
      DenseMatrix vshape(h1dofs_cnt, dim);
      for (int q = 0; q < nqp; q++)
	{
	  const IntegrationPoint &ip = IntRule->IntPoint(q);
	  Tr.SetIntPoint(&ip);
	  const DenseMatrix &Jpr = Tr.Jacobian();
	  DenseMatrix Jinv(dim);
	  Jinv = 0.0;
	  CalcInverse(Jpr, Jinv);
	  const int eq = e*nqp + q;
	  // Form stress:grad_shape at the current point.
	  el.CalcDShape(ip, vshape);
	  double pressure = p_gf.GetValue(Tr,ip);
	  double sound_speed = cs_gf.GetValue(Tr,ip);
	  DenseMatrix stress(dim), stressJiT(dim);
	  stressJiT = 0.0;
	  stress = 0.0;
	  const double rho = qdata.rho0DetJ0(eq) / Tr.Weight();
	  ComputeStress(pressure,dim,stress);
	  ComputeViscousStress(Tr, v_gf, qdata, eq, use_viscosity, use_vorticity, rho, sound_speed, dim, stress);
	  MultABt(stress, Jinv, stressJiT);
	  stressJiT *= ip.weight * Jpr.Det();
	  
	  for (int i = 0; i < h1dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  double loc_force = 0.0;
		  for (int gd = 0; gd < dim; gd++) // Gradient components.
		    {
		      elvect(i + vd * h1dofs_cnt) -= stressJiT(vd,gd) * vshape(i,gd);
		    }
		}
	    }
	}
    }

    void EnergyForceIntegrator::AssembleRHSElementVect(const FiniteElement &el,
						       ElementTransformation &Tr,
						       Vector &elvect)
    {
      if (Vnpt_gf != NULL){
	const int e = Tr.ElementNo;
	const int nqp = IntRule->GetNPoints();
	// phi test space
	const int dim = el.GetDim();
	const int l2dofs_cnt = el.GetDof();
	elvect.SetSize(l2dofs_cnt);
	elvect = 0.0;
	////
	
	// v trial space
	ParFiniteElementSpace *v_fespace = Vnpt_gf->ParFESpace();
	ElementTransformation &TrV = *v_fespace->GetElementTransformation(e);
	///
	Vector te_shape(l2dofs_cnt);
	te_shape = 0.0;
	for (int q = 0; q < nqp; q++)
	  {
	    te_shape = 0.0;
	    const IntegrationPoint &ip = IntRule->IntPoint(q);
	    Tr.SetIntPoint(&ip);
	    el.CalcShape(ip, te_shape);
	    const int eq = e*nqp + q;
	    
	    // Form stress:grad_shape at the current point.
	    const DenseMatrix &Jpr = Tr.Jacobian();
	    DenseMatrix Jinv(dim);
	    Jinv = 0.0;
	    CalcInverse(Jpr, Jinv);
	    double pressure = p_gf.GetValue(Tr,ip);
	    double sound_speed = cs_gf.GetValue(Tr,ip);
	    DenseMatrix stress(dim), stressJiT(dim);
	    stressJiT = 0.0;
	    stress = 0.0;
	    const double rho = qdata.rho0DetJ0(eq) / Tr.Weight();
	    ComputeStress(pressure,dim,stress);
	    ComputeViscousStress(Tr, v_gf, qdata, eq, use_viscosity, use_vorticity, rho, sound_speed, dim, stress);
	    stress *= ip.weight * Jpr.Det();
	    //
	    TrV.SetIntPoint(&ip);
	    DenseMatrix vGradShape;
	    Vnpt_gf->GetVectorGradient(TrV, vGradShape);
	  
	    // Calculating grad v
	    double gradVContractedStress = 0.0;	  
	    for (int s = 0; s < dim; s++){
	      for (int k = 0; k < dim; k++){
		gradVContractedStress += stress(s,k) * vGradShape(s,k);
	      }
	    }
	    
	    for (int i = 0; i < l2dofs_cnt; i++)
	      {
		elvect(i) += gradVContractedStress * te_shape(i);
	      }
	  }
      }  
      else{
	const int l2dofs_cnt = el.GetDof();
	elvect.SetSize(l2dofs_cnt);
	elvect = 0.0;
  }
    }

    void VelocityBoundaryForceIntegrator::AssembleRHSElementVect(const FiniteElement &el,
							     FaceElementTransformations &Tr,
							     Vector &elvect)
    {
      const int nqp_face = IntRule->GetNPoints();
      const int dim = el.GetDim();
      const int h1dofs_cnt = el.GetDof();
      elvect.SetSize(h1dofs_cnt*dim);
      elvect = 0.0;

      DenseMatrix loc_force(h1dofs_cnt, dim);
      Vector te_shape(h1dofs_cnt);
      te_shape = 0.0;
  
      for (int q = 0; q  < nqp_face; q++)
	{
	  const IntegrationPoint &ip_f = IntRule->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip = Tr.GetElement1IntPoint();
	  ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
	  Trans_el1.SetIntPoint(&eip);
	  const int elementNo = Trans_el1.ElementNo;

	  Vector nor;
	  nor.SetSize(dim);
	  nor = 0.0;
	  if (dim == 1)
	    {
	      nor(0) = 2*eip.x - 1.0;
	    }
	  else
	    {
	      CalcOrtho(Tr.Jacobian(), nor);
	    }        

	  el.CalcShape(eip, te_shape);
	  loc_force = 0.0;
	  double pressure = pface_gf.GetValue(Trans_el1,eip);
	  DenseMatrix stress(dim);
	  stress = 0.0;
	  ComputeStress(pressure,dim,stress);

	  // evaluation of the normal stress at the face quadrature points
	  Vector weightedNormalStress(dim);
	  weightedNormalStress = 0.0;
	  
	  // Quadrature data for partial assembly of the force operator.
	  stress.Mult( nor, weightedNormalStress);
	  
	  for (int i = 0; i < h1dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  elvect(i + vd * h1dofs_cnt) += weightedNormalStress(vd) * te_shape(i) * ip_f.weight;
		}
	    }
	}
    }

    void VelocityBoundaryForceIntegrator::AssembleRHSElementVect(
								 const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
    {
      mfem_error("DGDirichletLFIntegrator::AssembleRHSElementVect");
    }


    void EnergyBoundaryForceIntegrator::AssembleRHSElementVect(const FiniteElement &el,
							       FaceElementTransformations &Tr,
							       Vector &elvect)
    {
      if (Vnpt_gf != NULL){
	const int nqp_face = IntRule->GetNPoints();
	const int dim = el.GetDim();
	const int l2dofs_cnt = el.GetDof();
	elvect.SetSize(l2dofs_cnt);
	elvect = 0.0;
	Vector te_shape(l2dofs_cnt);
	te_shape = 0.0;
	
	for (int q = 0; q  < nqp_face; q++)
	  {
	    te_shape = 0.0;
	    const IntegrationPoint &ip_f = IntRule->IntPoint(q);
	    // Set the integration point in the face and the neighboring elements
	    Tr.SetAllIntPoints(&ip_f);
	    const IntegrationPoint &eip = Tr.GetElement1IntPoint();
	    ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
	    Trans_el1.SetIntPoint(&eip);
	    const int elementNo = Trans_el1.ElementNo;
	    
	    Vector nor;
	    nor.SetSize(dim);
	    nor = 0.0;
	    if (dim == 1)
	      {
		nor(0) = 2*eip.x - 1.0;
	      }
	    else
	      {
		CalcOrtho(Tr.Jacobian(), nor);
	      }        
	    el.CalcShape(eip, te_shape);
	    double pressure = pface_gf.GetValue(Trans_el1,eip);
	    DenseMatrix stress(dim);
	    stress = 0.0;
	    ComputeStress(pressure,dim,stress);
	    
	    // evaluation of the normal stress at the face quadrature points
	    Vector weightedNormalStress(dim);
	    weightedNormalStress = 0.0;
	    
	    // Quadrature data for partial assembly of the force operator.
	    stress.Mult( nor, weightedNormalStress);
	    
	    double normalStressProjNormal = 0.0;
	    double nor_norm = 0.0;
	    for (int s = 0; s < dim; s++){
	      normalStressProjNormal += weightedNormalStress(s) * nor(s);
	      nor_norm += nor(s) * nor(s);
	    }
	    nor_norm = sqrt(nor_norm);
	    normalStressProjNormal = normalStressProjNormal/nor_norm;
	    
	  Vector vShape;
	  Vnpt_gf->GetVectorValue(elementNo, eip, vShape);
	  double vDotn = 0.0;
	  for (int s = 0; s < dim; s++)
	    {
	      vDotn += vShape(s) * nor(s)/nor_norm;
	    }
	  for (int i = 0; i < l2dofs_cnt; i++)
	    {
	      elvect(i) -= normalStressProjNormal * te_shape(i) * ip_f.weight * vDotn;
	    }
	  }
      }
      else{
	const int l2dofs_cnt = el.GetDof();
	elvect.SetSize(l2dofs_cnt);
	elvect = 0.0;
      }
    }
    
    void EnergyBoundaryForceIntegrator::AssembleRHSElementVect(
							       const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
    {
      mfem_error("DGDirichletLFIntegrator::AssembleRHSElementVect");
    }

    void NormalVelocityMassIntegrator::AssembleFaceMatrix(const FiniteElement &fe,
							  const FiniteElement &fe2,
							  FaceElementTransformations &Tr,
							  DenseMatrix &elmat)
    {
      const int nqp_face = IntRule->GetNPoints();
      const int dim = fe.GetDim();
      const int h1dofs_cnt = fe.GetDof();
      elmat.SetSize(h1dofs_cnt*dim);
      elmat = 0.0;
      Vector shape(h1dofs_cnt), loc_force2(h1dofs_cnt * dim);;
      const int Elem1No = Tr.ElementNo;
      shape = 0.0;
      DenseMatrix loc_force1(h1dofs_cnt, dim);
      Vector Vloc_force(loc_force1.Data(), h1dofs_cnt*dim);
      for (int q = 0; q  < nqp_face; q++)
	{
	  const int eq = Elem1No*nqp_face + q;
	     
	  const IntegrationPoint &ip_f = IntRule->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip = Tr.GetElement1IntPoint();
	  Vector nor;
	  nor.SetSize(dim);
	  nor = 0.0;
	  if (dim == 1)
	    {
	      nor(0) = 2*eip.x - 1.0;
	    }
	  else
	    {
	      CalcOrtho(Tr.Jacobian(), nor);
	    }        
	  fe.CalcShape(eip, shape);
	  double nor_norm = 0.0;
	  for (int s = 0; s < dim; s++){
	    nor_norm += nor(s) * nor(s);
	  }
	  nor_norm = sqrt(nor_norm);
      
	  for (int i = 0; i < h1dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  for (int j = 0; j < h1dofs_cnt; j++)
		    {
		      for (int md = 0; md < dim; md++) // Velocity components.
			{	      
			  elmat(i + vd * h1dofs_cnt, j + md * h1dofs_cnt) += shape(i) * shape(j) * nor(vd) * (nor(md)/nor_norm) * qdata.normalVelocityPenaltyScaling * ip_f.weight;
			}
		    }
		}
	    }
	}
    }
  
  } // namespace hydrodynamics

} // namespace mfem

