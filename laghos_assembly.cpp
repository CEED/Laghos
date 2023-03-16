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
    double factorial(int nTerms){
      double factorial = 1.0;	
      for (int s = 1; s <= nTerms; s++){
	factorial = factorial*s;
      }
      return factorial;
    }

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
      Vector shape(h1dofs_cnt);
      const int Elem1No = Tr.ElementNo;
      shape = 0.0;
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

    void ShiftedVelocityBoundaryForceIntegrator::AssembleRHSElementVect(const FiniteElement &el,
									const FiniteElement &el2,
									FaceElementTransformations &Tr,
									Vector &elvect)      
    {
      if (Tr.Attribute == 12 ){
	const int dim = el.GetDim();      
	const IntegrationRule *ir = IntRule;
	if (ir == NULL)
	  {
	    // a simple choice for the integration order; is this OK?
	    const int order = 5 * max(el.GetOrder(), 1);
	    ir = &IntRules.Get(Tr.GetGeometryType(), order);
	  }
	const int nqp_face = IntRule->GetNPoints();
	
	const int h1dofs_cnt = el.GetDof();
	elvect.SetSize(2*h1dofs_cnt*dim);
	elvect = 0.0;
	Vector te_shape(h1dofs_cnt);
	te_shape = 0.0;
	
	for (int q = 0; q  < nqp_face; q++)
	  {
	    const IntegrationPoint &ip_f = ir->IntPoint(q);
	    // Set the integration point in the face and the neighboring elements
	    Tr.SetAllIntPoints(&ip_f);
	    const IntegrationPoint &eip = Tr.GetElement1IntPoint();
	    ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
	    Trans_el1.SetIntPoint(&eip);
	    
	    Vector nor;
	    nor.SetSize(dim);
	    nor = 0.0;
	    CalcOrtho(Tr.Jacobian(), nor);

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
	    
	    for (int i = 0; i < h1dofs_cnt; i++)
	      {
		for (int vd = 0; vd < dim; vd++) // Velocity components.
		  {
		    elvect(i + vd * h1dofs_cnt) += weightedNormalStress(vd) * te_shape(i) * ip_f.weight;
		  }
	      }
	  }
      }
     else if (Tr.Attribute == 21 ){
	const int dim = el2.GetDim();      
	const IntegrationRule *ir = IntRule;
	if (ir == NULL)
	  {
	    // a simple choice for the integration order; is this OK?
	    const int order = 5 * max(el2.GetOrder(), 1);
	    ir = &IntRules.Get(Tr.GetGeometryType(), order);
	  }
	const int nqp_face = IntRule->GetNPoints();
	
	const int h1dofs_cnt = el2.GetDof();
	int h1dofs_offset = el2.GetDof()*dim;
	
	elvect.SetSize(2*h1dofs_cnt*dim);
	elvect = 0.0;
	Vector te_shape(h1dofs_cnt);
	te_shape = 0.0;
	
	for (int q = 0; q  < nqp_face; q++)
	  {
	    const IntegrationPoint &ip_f = ir->IntPoint(q);
	    // Set the integration point in the face and the neighboring elements
	    Tr.SetAllIntPoints(&ip_f);
	    const IntegrationPoint &eip = Tr.GetElement2IntPoint();
	    ElementTransformation &Trans_el2 = Tr.GetElement2Transformation();
	    Trans_el2.SetIntPoint(&eip);
	    
	    Vector nor;
	    nor.SetSize(dim);
	    nor = 0.0;
	    CalcOrtho(Tr.Jacobian(), nor);
	    nor *= -1.0;
	    el2.CalcShape(eip, te_shape);
	    double pressure = pface_gf.GetValue(Trans_el2,eip);
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
		    elvect(i + vd * h1dofs_cnt + h1dofs_offset) += weightedNormalStress(vd) * te_shape(i) * ip_f.weight;
		  }
	      }
	  }
      }
      else{
	const int dim = el.GetDim();
	const int dofs_cnt = el.GetDof();
	elvect.SetSize(2*dofs_cnt*dim);
	elvect = 0.0;
      }
    }

    void ShiftedEnergyBoundaryForceIntegrator::AssembleRHSElementVect(const FiniteElement &el,
								      const FiniteElement &el2,
								      FaceElementTransformations &Tr,
								      Vector &elvect)
    {
      if (Vnpt_gf != NULL){ 
	if (Tr.Attribute == 12 ){
	  const int dim = el.GetDim();
	  DenseMatrix identity(dim);
	  identity = 0.0;
	  for (int s = 0; s < dim; s++){
	    identity(s,s) = 1.0;
	  }

	  const IntegrationRule *ir = IntRule;
	  if (ir == NULL)
	    {
	      // a simple choice for the integration order; is this OK?
	      const int order = 5 * max(el.GetOrder(), 1);
	      ir = &IntRules.Get(Tr.GetGeometryType(), order);
	    }
	  const int nqp_face = ir->GetNPoints();
	  const int l2dofs_cnt = el.GetDof();
	  elvect.SetSize(l2dofs_cnt*2);
	  elvect = 0.0;
	  Vector te_shape(l2dofs_cnt);
	  te_shape = 0.0;

	  ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
	  const int elementNo = Trans_el1.ElementNo;
	  ParFiniteElementSpace * pfes = Vnpt_gf->ParFESpace();
	  Vector loc_data;
	  int nbr_el_no = elementNo - pfes->GetParMesh()->GetNE();
	  if (nbr_el_no >= 0)
	    {
	      const Vector &faceNbrData = Vnpt_gf->FaceNbrData(); 
	      Array<int> dofs;
	      DofTransformation * doftrans = pfes->GetFaceNbrElementVDofs(nbr_el_no, dofs);
	      faceNbrData.GetSubVector(dofs, loc_data);
	      if (doftrans)
		{
		  doftrans->InvTransformPrimal(loc_data);
		}
	    }
	  else{
	    Array<int> vdofs;
	    DofTransformation * doftrans = pfes->GetElementVDofs(elementNo, vdofs);
	    Vnpt_gf->GetSubVector(vdofs, loc_data);
	    if (doftrans)
	      {
		doftrans->InvTransformPrimal(loc_data);
	      }
	  }
 
	  for (int q = 0; q  < nqp_face; q++)
	    {
	      te_shape = 0.0;
	      const IntegrationPoint &ip_f = ir->IntPoint(q);
	      // Set the integration point in the face and the neighboring elements
	      Tr.SetAllIntPoints(&ip_f);
	      const IntegrationPoint &eip = Tr.GetElement1IntPoint();
	      Trans_el1.SetIntPoint(&eip);
	    
	      Vector nor;
	      nor.SetSize(dim);
	      nor = 0.0;
	      CalcOrtho(Tr.Jacobian(), nor);
	      el.CalcShape(eip, te_shape);
	      double pressure = pface_gf.GetValue(Trans_el1,eip);
	      DenseMatrix stress(dim);
	      stress = 0.0;
	      ComputeStress(pressure,dim,stress);
	    
	      // evaluation of the normal stress at the face quadrature points
	      Vector weightedNormalStress(dim);
	      weightedNormalStress = 0.0;

	      double nor_norm = 0.0;
	      for (int s = 0; s < dim; s++){
		nor_norm += nor(s) * nor(s);
	      }
	      nor_norm = sqrt(nor_norm);
	    
	      int h1dofs_cnt = 0.0;
	      Vector velocity_shape, gradUResD_el1;
	      DenseMatrix nodalGrad_el1, gradUResDirD_el1, taylorExp_el1;
	      if (nbr_el_no >= 0){
		const FiniteElement *FElem = pfes->GetFaceNbrFE(nbr_el_no);
		h1dofs_cnt = FElem->GetDof();
		velocity_shape.SetSize(h1dofs_cnt);
		velocity_shape = 0.0;
		FElem->CalcShape(eip, velocity_shape);
		FElem->ProjectGrad(*FElem,Trans_el1,nodalGrad_el1);
	      }
	      else {
		const FiniteElement *FElem = pfes->GetFE(elementNo);
		h1dofs_cnt = FElem->GetDof();
		velocity_shape.SetSize(h1dofs_cnt);
		velocity_shape = 0.0;
		FElem->CalcShape(eip, velocity_shape);
		FElem->ProjectGrad(*FElem,Trans_el1,nodalGrad_el1);
	      }
	      
	      gradUResD_el1.SetSize(h1dofs_cnt);
	      gradUResDirD_el1.SetSize(h1dofs_cnt);
	      taylorExp_el1.SetSize(h1dofs_cnt);
	      
	      gradUResD_el1 = 0.0;
	      gradUResDirD_el1 = 0.0;
	      taylorExp_el1 = 0.0;

	      /////
	      Vector D_el1(dim);
	      Vector tN_el1(dim);
	      D_el1 = 0.0;
	      tN_el1 = 0.0; 
	      vD->Eval(D_el1, Trans_el1, eip);
	      vN->Eval(tN_el1, Trans_el1, eip);
	      /////
	      double sign = 0.0;
	      double normD = 0.0;
	      for (int a = 0; a < dim; a++){ 
		normD += D_el1(a) * D_el1(a);
	      }
	      normD  = std::pow(normD,0.5);
	      for( int b = 0; b < dim; b++){
		sign += D_el1(b) * tN_el1(b) / normD;  
	      }
	      if (sign < 0.0){
		D_el1 = 0.0;
		for( int b = 0; b < dim; b++){
		  tN_el1 = nor(b) / nor_norm;  
		}
	      }
	      
	      for (int k = 0; k < h1dofs_cnt; k++){
		for (int s = 0; s < h1dofs_cnt; s++){
		  for (int j = 0; j < dim; j++){
		    gradUResDirD_el1(s,k) += nodalGrad_el1(k + j * h1dofs_cnt, s) * D_el1(j);
		  }
		}
	      }

	      DenseMatrix tmp_el1(h1dofs_cnt);
	      DenseMatrix dummy_tmp_el1(h1dofs_cnt);
	      tmp_el1 = gradUResDirD_el1;
	      taylorExp_el1 = gradUResDirD_el1;
	      dummy_tmp_el1 = 0.0;
	      for (int k = 0; k < h1dofs_cnt; k++){
		for (int s = 0; s < h1dofs_cnt; s++){
		  gradUResD_el1(k) += taylorExp_el1(k,s) * velocity_shape(s);  
		}
	      }
	      
	      for ( int p = 1; p < nTerms; p++){
		dummy_tmp_el1 = 0.0;
		taylorExp_el1 = 0.0;
		for (int k = 0; k < h1dofs_cnt; k++){
		  for (int s = 0; s < h1dofs_cnt; s++){
		    for (int r = 0; r < h1dofs_cnt; r++){
		      taylorExp_el1(k,s) += tmp_el1(k,r) * gradUResDirD_el1(r,s) * (1.0/factorial(p+1));
		      dummy_tmp_el1(k,s) += tmp_el1(k,r) * gradUResDirD_el1(r,s);
		    }
		  }
		}
		tmp_el1 = dummy_tmp_el1;
		for (int k = 0; k < h1dofs_cnt; k++){
		  for (int s = 0; s < h1dofs_cnt; s++){
		    gradUResD_el1(k) += taylorExp_el1(k,s) * velocity_shape(s);  
		  }
		}
	      }
	      
	      ////
	      velocity_shape += gradUResD_el1;
	      ////
	      
	      Vector vShape(dim);
	      vShape = 0.0;
	      for (int k = 0; k < dim; k++){
		for (int s = 0; s < h1dofs_cnt; s++){
		  vShape(k) += velocity_shape(s) * loc_data(s + h1dofs_cnt * k);
		}
	      }

	      double nTildaDotN = 0.0;
	      for (int s = 0; s < dim; s++){
		nTildaDotN += nor(s) * tN_el1(s) / nor_norm;
	      }

	      // Quadrature data for partial assembly of the force operator.
	      // stress.Mult( nor, weightedNormalStress);
	      stress.Mult( tN_el1, weightedNormalStress);

	      double normalStressProjNormal = 0.0;
	      for (int s = 0; s < dim; s++){
		//normalStressProjNormal += weightedNormalStress(s) * nor(s) / (nor_norm * nor_norm);
		normalStressProjNormal += weightedNormalStress(s) * tN_el1(s);
	      }

	      double tangentStressProjTangent = 0.0;
	      for (int s = 0; s < dim; s++){
		for (int k = 0; k < dim; k++){
		  tangentStressProjTangent += stress(s,k) * (identity(s,k) - tN_el1(s) * tN_el1(k));
		}
	      }

	      double vDotn = 0.0;
	      for (int s = 0; s < dim; s++)
		{
		  //  vDotn += vShape(s) * nor(s) / nor_norm;
		  vDotn += vShape(s) * tN_el1(s);
		}
	      
	      double vDotTildaDotTangent = 0.0;
	      for (int s = 0; s < dim; s++)
		{
		  for (int k = 0; k < dim; k++)
		    {
		      vDotTildaDotTangent += vShape(s) * (nor(k) / nor_norm) * (identity(s,k) - tN_el1(s) * tN_el1(k));
		    }
		}

	      for (int i = 0; i < l2dofs_cnt; i++)
		{
		  //  elvect(i) -= normalStressProjNormal * te_shape(i) * ip_f.weight * vDotn * nor_norm;
		  elvect(i) -= normalStressProjNormal * te_shape(i) * ip_f.weight * vDotn * nTildaDotN * nor_norm;
		  //  elvect(i) -= tangentStressProjTangent * te_shape(i) * ip_f.weight * vDotTildaDotTangent * nor_norm;
		}
	    }
	}
	else if (Tr.Attribute == 21 ){
	  const int dim = el2.GetDim();
	  DenseMatrix identity(dim);
	  identity = 0.0;
	  for (int s = 0; s < dim; s++){
	    identity(s,s) = 1.0;
	  }

	  const IntegrationRule *ir = IntRule;
	  if (ir == NULL)
	    {
	      // a simple choice for the integration order; is this OK?
	      const int order = 5 * max(el2.GetOrder(), 1);
	      ir = &IntRules.Get(Tr.GetGeometryType(), order);
	    }
	  const int nqp_face = ir->GetNPoints();
	  const int l2dofs_cnt = el2.GetDof();
	  int l2dofs_offset = el2.GetDof();
	  elvect.SetSize(l2dofs_cnt*2);
	  elvect = 0.0;
	  Vector te_shape(l2dofs_cnt);
	  te_shape = 0.0;

	  ElementTransformation &Trans_el2 = Tr.GetElement2Transformation();
	  const int elementNo = Trans_el2.ElementNo;
	  ParFiniteElementSpace * pfes = Vnpt_gf->ParFESpace();
	  Vector loc_data;
	  int nbr_el_no = elementNo - pfes->GetParMesh()->GetNE();
	  if (nbr_el_no >= 0)
	    {
	      const Vector &faceNbrData = Vnpt_gf->FaceNbrData(); 
	      Array<int> dofs;
	      DofTransformation * doftrans = pfes->GetFaceNbrElementVDofs(nbr_el_no, dofs);
	      faceNbrData.GetSubVector(dofs, loc_data);
	      if (doftrans)
		{
		  doftrans->InvTransformPrimal(loc_data);
		}
	    }
	  else{
	    Array<int> vdofs;
	    DofTransformation * doftrans = pfes->GetElementVDofs(elementNo, vdofs);
	    Vnpt_gf->GetSubVector(vdofs, loc_data);
	    if (doftrans)
	      {
		doftrans->InvTransformPrimal(loc_data);
	      }
	  }
 
	  for (int q = 0; q  < nqp_face; q++)
	    {
	      te_shape = 0.0;
	      const IntegrationPoint &ip_f = ir->IntPoint(q);
	      // Set the integration point in the face and the neighboring elements
	      Tr.SetAllIntPoints(&ip_f);
	      const IntegrationPoint &eip = Tr.GetElement2IntPoint();
	      Trans_el2.SetIntPoint(&eip);
	    
	      Vector nor;
	      nor.SetSize(dim);
	      nor = 0.0;
	      CalcOrtho(Tr.Jacobian(), nor);
	      nor *= -1.0;
	      el2.CalcShape(eip, te_shape);
	      double pressure = pface_gf.GetValue(Trans_el2,eip);
	      DenseMatrix stress(dim);
	      stress = 0.0;
	      ComputeStress(pressure,dim,stress);
	    
	      // evaluation of the normal stress at the face quadrature points
	      Vector weightedNormalStress(dim);
	      weightedNormalStress = 0.0;

	      double nor_norm = 0.0;
	      for (int s = 0; s < dim; s++){
		nor_norm += nor(s) * nor(s);
	      }
	      nor_norm = sqrt(nor_norm);

	      int h1dofs_cnt = 0.0;
	      Vector velocity_shape, gradUResD_el2;
	      DenseMatrix nodalGrad_el2, gradUResDirD_el2, taylorExp_el2;
	      if (nbr_el_no >= 0){
		const FiniteElement *FElem = pfes->GetFaceNbrFE(nbr_el_no);
		h1dofs_cnt = FElem->GetDof();
		velocity_shape.SetSize(h1dofs_cnt);
		velocity_shape = 0.0;
		FElem->CalcShape(eip, velocity_shape);
		FElem->ProjectGrad(*FElem,Trans_el2,nodalGrad_el2);
	      }
	      else {
		const FiniteElement *FElem = pfes->GetFE(elementNo);
		h1dofs_cnt = FElem->GetDof();
		velocity_shape.SetSize(h1dofs_cnt);
		velocity_shape = 0.0;
		FElem->CalcShape(eip, velocity_shape);
		FElem->ProjectGrad(*FElem,Trans_el2,nodalGrad_el2);
	      }

	      gradUResD_el2.SetSize(h1dofs_cnt);
	      gradUResDirD_el2.SetSize(h1dofs_cnt);
	      taylorExp_el2.SetSize(h1dofs_cnt);
	      
	      gradUResD_el2 = 0.0;
	      gradUResDirD_el2 = 0.0;
	      taylorExp_el2 = 0.0;

	      /////
	      Vector D_el2(dim);
	      Vector tN_el2(dim);
	      D_el2 = 0.0;
	      tN_el2 = 0.0;
	      vD->Eval(D_el2, Trans_el2, eip);
	      vN->Eval(tN_el2, Trans_el2, eip);
	      /////

	      double sign = 0.0;
	      double normD = 0.0;
	      for (int a = 0; a < dim; a++){ 
		normD += D_el2(a) * D_el2(a);
	      }
	    
	      normD  = std::pow(normD,0.5);
	      for( int b = 0; b < dim; b++){
		sign += D_el2(b) * tN_el2(b) / normD;  
	      }
	      if (sign < 0.0){
		D_el2 = 0.0;
		for( int b = 0; b < dim; b++){
		  tN_el2 = nor(b) / nor_norm;  
		}
	      }
	     
	      for (int k = 0; k < h1dofs_cnt; k++){
		for (int s = 0; s < h1dofs_cnt; s++){
		  for (int j = 0; j < dim; j++){
		    gradUResDirD_el2(s,k) += nodalGrad_el2(k + j * h1dofs_cnt, s) * D_el2(j);
		  }
		}
	      }

	      DenseMatrix tmp_el2(h1dofs_cnt);
	      DenseMatrix dummy_tmp_el2(h1dofs_cnt);
	      tmp_el2 = gradUResDirD_el2;
	      taylorExp_el2 = gradUResDirD_el2;
	      dummy_tmp_el2 = 0.0;
	      for (int k = 0; k < h1dofs_cnt; k++){
		for (int s = 0; s < h1dofs_cnt; s++){
		  gradUResD_el2(k) += taylorExp_el2(k,s) * velocity_shape(s);  
		}
	      }
	      
	      for ( int p = 1; p < nTerms; p++){
		dummy_tmp_el2 = 0.0;
		taylorExp_el2 = 0.0;
		for (int k = 0; k < h1dofs_cnt; k++){
		  for (int s = 0; s < h1dofs_cnt; s++){
		    for (int r = 0; r < h1dofs_cnt; r++){
		      taylorExp_el2(k,s) += tmp_el2(k,r) * gradUResDirD_el2(r,s) * (1.0/factorial(p+1));
		      dummy_tmp_el2(k,s) += tmp_el2(k,r) * gradUResDirD_el2(r,s);
		    }
		  }
		}
		tmp_el2 = dummy_tmp_el2;
		for (int k = 0; k < h1dofs_cnt; k++){
		  for (int s = 0; s < h1dofs_cnt; s++){
		    gradUResD_el2(k) += taylorExp_el2(k,s) * velocity_shape(s);  
		  }
		}
	      }
	      
	      ////
	      velocity_shape += gradUResD_el2;
	      //

	      Vector vShape(dim);
	      vShape = 0.0;
	      for (int k = 0; k < dim; k++){
		for (int s = 0; s < h1dofs_cnt; s++){
		  vShape(k) += velocity_shape(s) * loc_data(s + h1dofs_cnt * k);
		}
	      }

	      double nTildaDotN = 0.0;
	      for (int s = 0; s < dim; s++){
		nTildaDotN += nor(s) * tN_el2(s) / nor_norm;
	      }

	      // Quadrature data for partial assembly of the force operator.
	      //  stress.Mult( nor, weightedNormalStress);
	      stress.Mult( tN_el2, weightedNormalStress);
	    
	      double normalStressProjNormal = 0.0;
	      for (int s = 0; s < dim; s++){
		// normalStressProjNormal += weightedNormalStress(s) * nor(s) / (nor_norm * nor_norm);
		normalStressProjNormal += weightedNormalStress(s) * tN_el2(s);
	      }
	      
	      double tangentStressProjTangent = 0.0;
	      for (int s = 0; s < dim; s++){
		for (int k = 0; k < dim; k++){
		  tangentStressProjTangent += stress(s,k) * (identity(s,k) - tN_el2(s) * tN_el2(k));
		}
	      }

	      double vDotn = 0.0;
	      for (int s = 0; s < dim; s++)
		{
		  vDotn += vShape(s) * tN_el2(s);
		  // vDotn += vShape(s) * nor(s) / nor_norm;
		}

	      double vDotTildaDotTangent = 0.0;
	      for (int s = 0; s < dim; s++)
		{
		  for (int k = 0; k < dim; k++)
		    {
		      vDotTildaDotTangent += vShape(s) * (nor(k) / nor_norm) * (identity(s,k) - tN_el2(s) * tN_el2(k));
		    }
		}
	      
	      for (int i = 0; i < l2dofs_cnt; i++)
		{
		  //	  elvect(i + l2dofs_offset) -= normalStressProjNormal * te_shape(i) * ip_f.weight * vDotn * nor_norm;

		  elvect(i + l2dofs_offset) -= normalStressProjNormal * te_shape(i) * ip_f.weight * vDotn * nTildaDotN * nor_norm;

		  //	  elvect(i + l2dofs_offset) -= tangentStressProjTangent * te_shape(i) * ip_f.weight * vDotTildaDotTangent * nor_norm;
	
		}
	    }
	}
	else{
	  const int dim = el.GetDim();
	  const int dofs_cnt = el.GetDof();
	  elvect.SetSize(2*dofs_cnt);
	  elvect = 0.0;
	}
      }
      else{
	const int l2dofs_cnt = el.GetDof();
	elvect.SetSize(2*l2dofs_cnt);
	elvect = 0.0;
      }
    }

    void ShiftedNormalVelocityMassIntegrator::AssembleFaceMatrix(const FiniteElement &fe,
								 const FiniteElement &fe2,
								 FaceElementTransformations &Tr,
								 DenseMatrix &elmat)
    {
      if (Tr.Attribute == 12 ){	
	const int dim = fe.GetDim();
	DenseMatrix identity(dim);
	identity = 0.0;
	for (int s = 0; s < dim; s++){
	  identity(s,s) = 1.0;
	}
	const IntegrationRule *ir = IntRule;
	if (ir == NULL)
	  {
	    // a simple choice for the integration order; is this OK?
	    const int order = 5 * max(fe.GetOrder(), 1);
	    ir = &IntRules.Get(Tr.GetGeometryType(), order);
	  }
	
	const int nqp_face = ir->GetNPoints();
	const int h1dofs_cnt = fe.GetDof();
	elmat.SetSize(2*h1dofs_cnt*dim);
	elmat = 0.0;
	
	Vector shape(h1dofs_cnt), shape_test(h1dofs_cnt), nor(dim), gradUResD_el1(h1dofs_cnt), test_gradUResD_el1(h1dofs_cnt);
	ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
	DenseMatrix nodalGrad_el1, gradUResDirD_el1(h1dofs_cnt), taylorExp_el1(h1dofs_cnt);
	fe.ProjectGrad(fe,Trans_el1,nodalGrad_el1);
	
	nor = 0.0;
	shape = 0.0;
	shape_test = 0.0;
	gradUResDirD_el1 = 0.0; 
	gradUResD_el1 = 0.0;
	taylorExp_el1 = 0.0;
	test_gradUResD_el1 = 0.0;
	  
	for (int q = 0; q < nqp_face; q++)
	  {
	    nor = 0.0;
	    shape = 0.0;
	    shape_test = 0.0;
	    gradUResDirD_el1 = 0.0;
	    gradUResD_el1 = 0.0;
	    taylorExp_el1 = 0.0;
	    test_gradUResD_el1 = 0.0;
	  
	    const IntegrationPoint &ip_f = ir->IntPoint(q);
	    // Set the integration point in the face and the neighboring elements
	    Tr.SetAllIntPoints(&ip_f);
	    const IntegrationPoint &eip = Tr.GetElement1IntPoint();	    
	    CalcOrtho(Tr.Jacobian(), nor);

	    double nor_norm = 0.0;
	    for (int s = 0; s < dim; s++){
	      nor_norm += nor(s) * nor(s);
	    }
	    nor_norm = sqrt(nor_norm);

	    fe.CalcShape(eip, shape);
	    fe.CalcShape(eip, shape_test);

	    /////
	    Vector D_el1(dim);
	    Vector tN_el1(dim);
	    D_el1 = 0.0;
	    tN_el1 = 0.0; 	    
	    vD->Eval(D_el1, Trans_el1, eip);
	    vN->Eval(tN_el1, Trans_el1, eip);
	    /////
	    double sign = 0.0;
	    double normD = 0.0;
	    for (int a = 0; a < dim; a++){ 
	      normD += D_el1(a) * D_el1(a);
	    }
	    
	    normD  = std::pow(normD,0.5);
	    for( int b = 0; b < dim; b++){
	      sign += D_el1(b) * tN_el1(b) / normD;  
	    }
	    if (sign < 0.0){
	      std::cout << " shit " << std::endl;
	      D_el1 = 0.0;
	      for( int b = 0; b < dim; b++){
		tN_el1 = nor(b) / nor_norm;  
	      }
	    }
	     
	    for (int k = 0; k < h1dofs_cnt; k++){
	      for (int s = 0; s < h1dofs_cnt; s++){
		for (int j = 0; j < dim; j++){
		  gradUResDirD_el1(s,k) += nodalGrad_el1(k + j * h1dofs_cnt, s) * D_el1(j);
		}
	      }
	    }

	    DenseMatrix tmp_el1(h1dofs_cnt);
	    DenseMatrix dummy_tmp_el1(h1dofs_cnt);
	    tmp_el1 = gradUResDirD_el1;
	    taylorExp_el1 = gradUResDirD_el1;
	    dummy_tmp_el1 = 0.0;
	    for (int k = 0; k < h1dofs_cnt; k++){
	      for (int s = 0; s < h1dofs_cnt; s++){
		gradUResD_el1(k) += taylorExp_el1(k,s) * shape(s);  
	      }
	    }
	    Vector test_gradUResD_el1(h1dofs_cnt);
	    test_gradUResD_el1 = gradUResD_el1;
	    
	    for ( int p = 1; p < nTerms; p++){
	      dummy_tmp_el1 = 0.0;
	      taylorExp_el1 = 0.0;
	      for (int k = 0; k < h1dofs_cnt; k++){
		for (int s = 0; s < h1dofs_cnt; s++){
		  for (int r = 0; r < h1dofs_cnt; r++){
		    taylorExp_el1(k,s) += tmp_el1(k,r) * gradUResDirD_el1(r,s) * (1.0/factorial(p+1));
		    dummy_tmp_el1(k,s) += tmp_el1(k,r) * gradUResDirD_el1(r,s);
		  }
		}
	      }
	      tmp_el1 = dummy_tmp_el1;
	      for (int k = 0; k < h1dofs_cnt; k++){
		for (int s = 0; s < h1dofs_cnt; s++){
		  gradUResD_el1(k) += taylorExp_el1(k,s) * shape(s);  
		}
	      }
	    }
	    
	    ////
	    shape += gradUResD_el1;
	    //
	    
	    if (fullPenalty){
	      shape_test += gradUResD_el1;
	    }
	    else{
	      shape_test += test_gradUResD_el1;
	    }

	    double nTildaDotN = 0.0;
	    for (int s = 0; s < dim; s++){
	      nTildaDotN += nor(s) * tN_el1(s) / nor_norm;
	    }
	    for (int i = 0; i < h1dofs_cnt; i++)
	      {
		for (int vd = 0; vd < dim; vd++) // Velocity components.
		  {
		    for (int j = 0; j < h1dofs_cnt; j++)
		      {
			for (int md = 0; md < dim; md++) // Velocity components.
			  {	      
			    //		    elmat(i + vd * h1dofs_cnt, j + md * h1dofs_cnt) += shape_test(i) * shape(j) * nor(vd) * (nor(md)/nor_norm) * qdata.normalVelocityPenaltyScaling * ip_f.weight;
			    elmat(i + vd * h1dofs_cnt, j + md * h1dofs_cnt) += shape_test(i) * shape(j) * nor_norm * tN_el1(vd) * tN_el1(md) * qdata.normalVelocityPenaltyScaling * ip_f.weight * nTildaDotN * nTildaDotN;
			    //  elmat(i + vd * h1dofs_cnt, j + md * h1dofs_cnt) += shape_test(i) * shape(j) * nor_norm * (identity(vd,md) - tN_el1(vd) * tN_el1(md)) * qdata.normalVelocityPenaltyScaling * ip_f.weight * (1.0 - nTildaDotN * nTildaDotN);			
	
			  }
		      }
		  }
	      }
	  }
      }
      else if (Tr.Attribute == 21 ){	
	const int dim = fe2.GetDim();
	DenseMatrix identity(dim);
	identity = 0.0;
	for (int s = 0; s < dim; s++){
	  identity(s,s) = 1.0;
	}

	const IntegrationRule *ir = IntRule;
	if (ir == NULL)
	  {
	    // a simple choice for the integration order; is this OK?
	    const int order = 5 * max(fe2.GetOrder(), 1);
	    ir = &IntRules.Get(Tr.GetGeometryType(), order);
	  }
	
	const int nqp_face = ir->GetNPoints();
	const int h1dofs_cnt = fe2.GetDof();
	int h1dofs_offset = fe2.GetDof()*dim;
	elmat.SetSize(2*h1dofs_cnt*dim);
	elmat = 0.0;

	Vector shape(h1dofs_cnt), shape_test(h1dofs_cnt), nor(dim), gradUResD_el2(h1dofs_cnt), test_gradUResD_el2(h1dofs_cnt);
	ElementTransformation &Trans_el2 = Tr.GetElement2Transformation();
	DenseMatrix nodalGrad_el2, gradUResDirD_el2(h1dofs_cnt), taylorExp_el2(h1dofs_cnt);
	fe2.ProjectGrad(fe2, Trans_el2, nodalGrad_el2);
	
	nor = 0.0;
	shape = 0.0;
	shape_test = 0.0;
	gradUResDirD_el2 = 0.0; 
	gradUResD_el2 = 0.0;
	taylorExp_el2 = 0.0;
	test_gradUResD_el2 = 0.0;
	  
	for (int q = 0; q < nqp_face; q++)
	  {

	    nor = 0.0;
	    shape = 0.0;
	    shape_test = 0.0;
	    gradUResDirD_el2 = 0.0;
	    gradUResD_el2 = 0.0;
	    taylorExp_el2 = 0.0;
	    test_gradUResD_el2 = 0.0;
	  
	    const IntegrationPoint &ip_f = ir->IntPoint(q);
	    // Set the integration point in the face and the neighboring elements
	    Tr.SetAllIntPoints(&ip_f);
	    const IntegrationPoint &eip = Tr.GetElement2IntPoint();
	    
	    nor = 0.0;
	    CalcOrtho(Tr.Jacobian(), nor);
	    nor *= -1.0;

	    double nor_norm = 0.0;
	    for (int s = 0; s < dim; s++){
	      nor_norm += nor(s) * nor(s);
	    }
	    nor_norm = sqrt(nor_norm);

	    fe2.CalcShape(eip, shape);
	    fe2.CalcShape(eip, shape_test);

	    /////
	    Vector D_el2(dim);
	    Vector tN_el2(dim);
	    D_el2 = 0.0;
	    tN_el2 = 0.0;  
	    vD->Eval(D_el2, Trans_el2, eip);
	    vN->Eval(tN_el2, Trans_el2, eip);
	    /////
	    double sign = 0.0;
	    double normD = 0.0;
	    for (int a = 0; a < dim; a++){ 
	      normD += D_el2(a) * D_el2(a);
	    }
	    
	    normD  = std::pow(normD,0.5);
	    for( int b = 0; b < dim; b++){
	      sign += D_el2(b) * tN_el2(b) / normD;  
	    }
	    if (sign < 0.0){
	      std::cout << " shit 2 " << std::endl;
	      D_el2 = 0.0;
	      for( int b = 0; b < dim; b++){
		tN_el2 = nor(b) / nor_norm;  
	      }
	    }
	    
	    for (int k = 0; k < h1dofs_cnt; k++){
	      for (int s = 0; s < h1dofs_cnt; s++){
		for (int j = 0; j < dim; j++){
		  gradUResDirD_el2(s,k) += nodalGrad_el2(k + j * h1dofs_cnt, s) * D_el2(j);
		}
	      }
	    }

	    DenseMatrix tmp_el2(h1dofs_cnt);
	    DenseMatrix dummy_tmp_el2(h1dofs_cnt);
	    tmp_el2 = gradUResDirD_el2;
	    taylorExp_el2 = gradUResDirD_el2;
	    dummy_tmp_el2 = 0.0;
	    for (int k = 0; k < h1dofs_cnt; k++){
	      for (int s = 0; s < h1dofs_cnt; s++){
		gradUResD_el2(k) += taylorExp_el2(k,s) * shape(s);  
	      }
	    }
	    Vector test_gradUResD_el2(h1dofs_cnt);
	    test_gradUResD_el2 = gradUResD_el2;
	    
	    for ( int p = 1; p < nTerms; p++){
	      dummy_tmp_el2 = 0.0;
	      taylorExp_el2 = 0.0;
	      for (int k = 0; k < h1dofs_cnt; k++){
		for (int s = 0; s < h1dofs_cnt; s++){
		  for (int r = 0; r < h1dofs_cnt; r++){
		    taylorExp_el2(k,s) += tmp_el2(k,r) * gradUResDirD_el2(r,s) * (1.0/factorial(p+1));
		    dummy_tmp_el2(k,s) += tmp_el2(k,r) * gradUResDirD_el2(r,s);
		  }
		}
	      }
	      tmp_el2 = dummy_tmp_el2;
	      for (int k = 0; k < h1dofs_cnt; k++){
		for (int s = 0; s < h1dofs_cnt; s++){
		  gradUResD_el2(k) += taylorExp_el2(k,s) * shape(s);  
		}
	      }
	    }
	    
	    ////
	    shape += gradUResD_el2;
	    //
	    
	    if (fullPenalty){
	      shape_test += gradUResD_el2;
	    }
	    else{
	      shape_test += test_gradUResD_el2;
	    }
	    
	    double nTildaDotN = 0.0;
	    for (int s = 0; s < dim; s++){
	      nTildaDotN += nor(s) * tN_el2(s) / nor_norm;
	    }
	    
	    for (int i = 0; i < h1dofs_cnt; i++)
	      {
		for (int vd = 0; vd < dim; vd++) // Velocity components.
		  {
		    for (int j = 0; j < h1dofs_cnt; j++)
		      {
			for (int md = 0; md < dim; md++) // Velocity components.
			  {	      
			    //		    elmat(i + vd * h1dofs_cnt + h1dofs_offset, j + md * h1dofs_cnt + h1dofs_offset) += shape_test(i) * shape(j) * nor(vd) * (nor(md)/nor_norm) * qdata.normalVelocityPenaltyScaling * ip_f.weight;
			    elmat(i + vd * h1dofs_cnt + h1dofs_offset, j + md * h1dofs_cnt + h1dofs_offset) += shape_test(i) * shape(j) * nor_norm * tN_el2(vd) * tN_el2(md) * qdata.normalVelocityPenaltyScaling * ip_f.weight * nTildaDotN * nTildaDotN;
			    //    elmat(i + vd * h1dofs_cnt + h1dofs_offset, j + md * h1dofs_cnt + h1dofs_offset) += shape_test(i) * shape(j) * nor_norm * (identity(vd,md) - tN_el2(vd) * tN_el2(md)) * qdata.normalVelocityPenaltyScaling * ip_f.weight * (1.0 - nTildaDotN * nTildaDotN);

			  }
		      }
		  }
	      }
	  }
      }
      else{
	const int dim = fe.GetDim();
	const int h1dofs_cnt = fe.GetDof();
	elmat.SetSize(2*h1dofs_cnt*dim);
	elmat = 0.0;    
      }
    }
    
    
  } // namespace hydrodynamics
  
} // namespace mfem
