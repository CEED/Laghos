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
	  const int eq = elementNo*nqp_face + q;
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
	  double sound_speed = csface_gf.GetValue(Trans_el1,eip);
	  Vector Jac0inv_vec(dim*dim);
	  Jac0inv_vec = 0.0;
	  Jac0invface_gf.GetVectorValue(elementNo,eip,Jac0inv_vec);

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

	  double nor_norm = 0.0;
	  for (int s = 0; s < dim; s++){
	    nor_norm += nor(s) * nor(s);
	  }
	  nor_norm = sqrt(nor_norm);
	
	  const DenseMatrix &Jpr = Trans_el1.Jacobian();
	  // std::cout << " Dt " << Jpr.Det() << std::endl;
	  DenseMatrix Jpi(dim);
	  Jpi = 0.0;
	  mfem::Mult(Jpr, Jac0inv, Jpi);
	  Vector tn(dim), tN(dim);
	  tn = 0.0;
	  tN = 0.0;
	  tn = nor;
	  tn /= nor_norm;
	  Jpi.MultTranspose(tn,tN);
	  double origNormalProd = 0.0;
	  for (int s = 0; s < dim; s++){
	    origNormalProd += tN(s) * tN(s);
	  }
	  origNormalProd = std::pow(origNormalProd,0.5);
	  tN *= 1.0/origNormalProd;
	  /* std::cout << " current " << std::endl;
	  tn.Print();
	  std::cout << " origin " << std::endl;
	  tN.Print();*/
	
	  DenseMatrix stress(dim);
	  stress = 0.0;
	  //  const double rho = qdata.rho0DetJ0(eq) / Tr.Weight();
	  const double rho = rho0DetJ0face_gf.GetValue(Trans_el1,eip);
	    
	  ComputeStress(pressure,dim,stress);
	  //  ComputeViscousStressGL(Trans_el1, v_gf, qdata, eq, use_viscosity, use_vorticity, rho, sound_speed, dim, stress);
	
	  // evaluation of the normal stress at the face quadrature points
	  Vector weightedNormalStress(dim);
	  weightedNormalStress = 0.0;
	  
	  // Quadrature data for partial assembly of the force operator.
	  stress.Mult( tn, weightedNormalStress);
	  // stress.Mult( tN, weightedNormalStress);
	  
	  for (int i = 0; i < h1dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  elvect(i + vd * h1dofs_cnt) += weightedNormalStress(vd) * te_shape(i) * ip_f.weight * nor_norm;
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
	    const int eq = elementNo*nqp_face + q;
	
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
	    double sound_speed = csface_gf.GetValue(Trans_el1,eip);

	    Vector Jac0inv_vec(dim*dim);
	    Jac0inv_vec = 0.0;
	    Jac0invface_gf.GetVectorValue(elementNo,eip,Jac0inv_vec);

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

	    double nor_norm = 0.0;
	    for (int s = 0; s < dim; s++){
	      nor_norm += nor(s) * nor(s);
	    }
	    nor_norm = sqrt(nor_norm);
	
	    const DenseMatrix &Jpr = Trans_el1.Jacobian();
	    // std::cout << " Dt " << Jpr.Det() << std::endl;
	    DenseMatrix Jpi(dim);
	    Jpi = 0.0;
	    mfem::Mult(Jpr, Jac0inv, Jpi);
	    Vector tn(dim), tN(dim);
	    tn = 0.0;
	    tN = 0.0;
	    tn = nor;
	    tn /= nor_norm;
	    Jpi.MultTranspose(tn,tN);
	    double origNormalProd = 0.0;
	    for (int s = 0; s < dim; s++){
	      origNormalProd += tN(s) * tN(s);
	    }
	    origNormalProd = std::pow(origNormalProd,0.5);
	    tN *= 1.0/origNormalProd;
	    /* std::cout << " current " << std::endl;
	    tn.Print();
	    std::cout << " origin " << std::endl;
	    tN.Print();*/	 
	    
	    DenseMatrix stress(dim);
	    stress = 0.0;
	    // const double rho = qdata.rho0DetJ0(eq) / Tr.Weight();
	    const double rho = rho0DetJ0face_gf.GetValue(Trans_el1,eip);
	    
	    ComputeStress(pressure,dim,stress);
	    //  ComputeViscousStressGL(Trans_el1, v_gf, qdata, eq, use_viscosity, use_vorticity, rho, sound_speed, dim, stress);
	
	    // evaluation of the normal stress at the face quadrature points
	    Vector weightedNormalStress(dim);
	    weightedNormalStress = 0.0;
	    
	    // Quadrature data for partial assembly of the force operator.
	    stress.Mult( tn, weightedNormalStress);
	    // stress.Mult( tN, weightedNormalStress);

	    double normalStressProjNormal = 0.0;
	    for (int s = 0; s < dim; s++){
	      // normalStressProjNormal += weightedNormalStress(s) * tN(s);
	      normalStressProjNormal += weightedNormalStress(s) * tn(s);
	    }
	    normalStressProjNormal = normalStressProjNormal*nor_norm;
	    
	    Vector vShape;
	    Vnpt_gf->GetVectorValue(elementNo, eip, vShape);
	    double vDotn = 0.0;
	    for (int s = 0; s < dim; s++)
	      {
		//	vDotn += vShape(s) * tN(s);
		vDotn += vShape(s) * tn(s);
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
      shape = 0.0;
      std::cout << " f " << nqp_face << std::endl;
      for (int q = 0; q  < nqp_face; q++)
	{
	  const IntegrationPoint &ip_f = IntRule->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip = Tr.GetElement1IntPoint();
	  ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
	  
	  Trans_el1.SetIntPoint(&eip);
	  const int elementNo = Trans_el1.ElementNo;
	  const int eq = elementNo*nqp_face + q;
	 
	  // std::cout << " Aip.x " << eip.x << " Aip.y " << eip.y << std::endl;
	 
	  Vector nor;
	  nor.SetSize(dim);
	  nor = 0.0;
	  CalcOrtho(Tr.Jacobian(), nor);
	  //  Vector x;
	  //  Trans_el1.Transform(eip, x);
	  // std::cout << " x(0) " << x(0) << " x(1) " << x(1) << std::endl;
	  double density = rhoface_gf.GetValue(Trans_el1,eip);
	  // double density = rho0DetJ0face_gf.GetValue(Trans_el1,eip);
	  
	  Vector Jac0inv_vec(dim*dim);
	  Jac0inv_vec = 0.0;
	  Jac0invface_gf.GetVectorValue(elementNo,eip,Jac0inv_vec);
	  
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
	  
	  double nor_norm = 0.0;
	  for (int s = 0; s < dim; s++){
	    nor_norm += nor(s) * nor(s);
	  }
	  nor_norm = sqrt(nor_norm);
	  
	  const DenseMatrix &Jpr = Trans_el1.Jacobian();
	  // std::cout << " Dt " << Jpr.Det() << std::endl;
	  DenseMatrix Jpi(dim);
	  Jpi = 0.0;
	  mfem::Mult(Jpr, Jac0inv, Jpi);
	  Vector tn(dim), tN(dim);
	  tn = 0.0;
	  tN = 0.0;
	  tn = nor;
	  tn /= nor_norm;
	  double proj = 0.0;
	  for (int s = 0; s < dim; s++){
	    proj += std::abs(tn(s));
	  }
	  std::cout << " proj " << proj << std::endl;
	  Jpi.MultTranspose(tn,tN);
	  double origNormalProd = 0.0;
	  for (int s = 0; s < dim; s++){
	    origNormalProd += tN(s) * tN(s);
	  }
	  origNormalProd = std::pow(origNormalProd,0.5);
	  tN *= 1.0/origNormalProd;
	  Vector transip;
	  Trans_el1.Transform(eip, transip);
	  // transip.Print();
	  // tN.Print();
	  /* std::cout << " current " << std::endl;
	  tn.Print();
	  std::cout << " origin " << std::endl;
	  tN.Print();*/
	  // std::cout << " orig " << origNormalProd << std::endl;
	  // std::cout << " val_pr " << 1.0/origNormalProd_tn << " val_pi " << 1.0/origNormalProd << " vol " << Tr.Elem1->Weight() << std::endl; 
	  //  std::cout << " tN(0) " << tN(0) << " tN(1) " << tN(1) << std::endl;
	  double penaltyVal = 0.0;
	  double aMax = (globalmax_rho/globalmax_viscous_coef) * globalmax_cs * 4.0;
	  // std::cout << " aM " << aMax << std::endl;
	  //  double aMax = (globalmax_rho/globalmax_viscous_coef) * globalmax_cs * globalmax_cs;
	  //  penaltyVal = penaltyParameter * globalmax_rho * (1.0 + ((globalmax_viscous_coef/globalmax_rho)*(1.0/globalmax_cs * globalmax_cs) + (globalmax_rho/globalmax_viscous_coef) * globalmax_cs * globalmax_cs) ) *  (Tr.Elem1->Weight() / nor_norm) * std::pow(1.0/origNormalProd,2.0);
	  //  penaltyVal = penaltyParameter * globalmax_rho *  (Tr.Elem1->Weight() / nor_norm) * (1.0 + aMax + 1.0/aMax) * std::pow(1.0/origNormalProd,2.0*order_v);
	  //   std::cout << " amxa " << aMax << " nCn " << 1.0/origNormalProd << " pen " << penaltyVal << std::endl;
	  //  penaltyVal =  4.0 * std::pow(penaltyParameter * (1.0 * 1.0/aMax + aMax),proj) /* * std::pow(penaltyParameter,proj)*/ * globalmax_rho /* * ( nor_norm / Tr.Elem1->Weight()) */ ;
	  penaltyVal = 4.0 * penaltyParameter * globalmax_rho /* * ( nor_norm / Tr.Elem1->Weight()) */ ;
	
	  //	  std::cout << " penV " << penaltyVal << std::endl;
	  // penaltyVal = 4 * globalmax_rho * std::pow(penaltyParameter, 1.0*(1+aMax+1.0/aMax))/* * origNormalProd*/;
	  //	    std::cout << " dens " << density << " abs " << std::abs(density) << std::endl;
	  //  std::cout << " val " << std::pow(1.0/origNormalProd,2.0) << std::endl;
	  // std::cout << " nCn " << std::pow(1.0/origNormalProd,2.0) << std::endl;
	  //std::cout << " pen " << penaltyVal << std::endl;
	  fe.CalcShape(eip, shape);
	  std::cout << " pen " << penaltyVal << std::endl;
	  for (int i = 0; i < h1dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  for (int j = 0; j < h1dofs_cnt; j++)
		    {
		      for (int md = 0; md < dim; md++) // Velocity components.
			{	      
			  elmat(i + vd * h1dofs_cnt, j + md * h1dofs_cnt) += shape(i) * shape(j) * tn(vd) * tn(md) * penaltyVal * ip_f.weight * nor_norm;
			  //  elmat(i + vd * h1dofs_cnt, j + md * h1dofs_cnt) += shape(i) * shape(j) * tN(vd) * tN(md) * penaltyVal * ip_f.weight * nor_norm;
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
      if (Tr.Attribute == 77 ){
	const int dim = el.GetDim();      
	const int nqp_face = IntRule->GetNPoints();
    
	const int h1dofs_cnt = el.GetDof();
	int h1dofs_offset = el2.GetDof()*dim;
	
	elvect.SetSize(2*h1dofs_cnt*dim);
	elvect = 0.0;
	Vector te_shape_el1(h1dofs_cnt), te_shape_el2(h1dofs_cnt), nor(dim);
	te_shape_el1 = 0.0;
	te_shape_el2 = 0.0;
	nor = 0.0;
	for (int q = 0; q  < nqp_face; q++)
	  {
	    te_shape_el1 = 0.0;
	    te_shape_el2 = 0.0;
	    nor = 0.0;
	    
	    const IntegrationPoint &ip_f = IntRule->IntPoint(q);
	    // Set the integration point in the face and the neighboring elements
	    Tr.SetAllIntPoints(&ip_f);
	    const IntegrationPoint &eip_el1 = Tr.GetElement1IntPoint();
	    const IntegrationPoint &eip_el2 = Tr.GetElement2IntPoint();	    
	    ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
	    ElementTransformation &Trans_el2 = Tr.GetElement2Transformation();
	    
	    CalcOrtho(Tr.Jacobian(), nor);

	    double volumeFraction_el1 = alpha_gf.GetValue(Trans_el1, eip_el1);
	    double volumeFraction_el2 = alpha_gf.GetValue(Trans_el2, eip_el2);
	    double sum_volFrac = volumeFraction_el1 + volumeFraction_el2;
	    double gamma_1 =  volumeFraction_el1/sum_volFrac;
	    double gamma_2 =  volumeFraction_el2/sum_volFrac;

	    if (gamma_1 > 0.99){
	   
	      el.CalcShape(eip_el1, te_shape_el1);
	      double pressure = pface_gf.GetValue(Trans_el1,eip_el1);
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
		      elvect(i + vd * h1dofs_cnt) += weightedNormalStress(vd) * te_shape_el1(i) * ip_f.weight;
		    }
		}
	    }
	    else {
	      nor *= -1.0;
	      el2.CalcShape(eip_el2, te_shape_el2);
	      double pressure = pface_gf.GetValue(Trans_el2,eip_el2);
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
		      elvect(i + vd * h1dofs_cnt + h1dofs_offset) += weightedNormalStress(vd) * te_shape_el2(i) * ip_f.weight;
		    }
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
	if (Tr.Attribute == 77 ){
	  const int dim = el.GetDim();
	  const int nqp_face = IntRule->GetNPoints();

	  const int l2dofs_cnt = el.GetDof();
	  int l2dofs_offset = el2.GetDof();
	  elvect.SetSize(l2dofs_cnt*2);
	  elvect = 0.0;
	  Vector te_shape_el1(l2dofs_cnt), te_shape_el2(l2dofs_cnt), nor(dim);
	  te_shape_el1 = 0.0;
	  te_shape_el2 = 0.0;
	  nor = 0.0;
	  
	  ParFiniteElementSpace * pfes = Vnpt_gf->ParFESpace();
	  ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
	  const int elementNo_el1 = Trans_el1.ElementNo;
	  Vector loc_data_el1;
	  Array<int> vdofs_el1;
	  DofTransformation * doftrans_el1 = pfes->GetElementVDofs(elementNo_el1, vdofs_el1);
	  Vnpt_gf->GetSubVector(vdofs_el1, loc_data_el1);
	  if (doftrans_el1)
	  {
	    doftrans_el1->InvTransformPrimal(loc_data_el1);
	  }
	  
	  ElementTransformation &Trans_el2 = Tr.GetElement2Transformation();
	  const int elementNo_el2 = Trans_el2.ElementNo;
	  Vector loc_data_el2;
	  int nbr_el_no = elementNo_el2 - pfes->GetParMesh()->GetNE();
	  if (nbr_el_no >= 0)
	    {
	      int height = pfes->GetVSize();
	      Array<int> vdofs_el2;      
	      pfes->GetFaceNbrElementVDofs(nbr_el_no, vdofs_el2);
	      
	      for (int j = 0; j < vdofs_el2.Size(); j++)
		{
		  if (vdofs_el2[j] >= 0)
		    {
		      vdofs_el2[j] += height;
		    }
		  else
		    {
		      vdofs_el2[j] -= height;
		    }
		}
	      Vnpt_gf->GetSubVector(vdofs_el2, loc_data_el2);
	    }
	  else{
	    Array<int> vdofs_el2;
	    DofTransformation * doftrans_el2 = pfes->GetElementVDofs(elementNo_el2, vdofs_el2); 
	    Vnpt_gf->GetSubVector(vdofs_el2, loc_data_el2);
	  }
	  
	  
	  
	  for (int q = 0; q  < nqp_face; q++)
	    {
	      te_shape_el1 = 0.0;
	      te_shape_el2 = 0.0;
	      nor = 0.0;
	      
	      const IntegrationPoint &ip_f = IntRule->IntPoint(q);
	      // Set the integration point in the face and the neighboring elements
	      Tr.SetAllIntPoints(&ip_f);
	      const IntegrationPoint &eip_el1 = Tr.GetElement1IntPoint();
	      const IntegrationPoint &eip_el2 = Tr.GetElement2IntPoint();
	    
	      double volumeFraction_el1 = alpha_gf.GetValue(Trans_el1, eip_el1);
	      double volumeFraction_el2 = alpha_gf.GetValue(Trans_el2, eip_el2);
	      double sum_volFrac = volumeFraction_el1 + volumeFraction_el2;
	      double gamma_1 =  volumeFraction_el1/sum_volFrac;
	      double gamma_2 =  volumeFraction_el2/sum_volFrac;
	    
	      CalcOrtho(Tr.Jacobian(), nor);

	      if (gamma_1 > 0.99){
		el.CalcShape(eip_el1, te_shape_el1);
		double pressure = pface_gf.GetValue(Trans_el1,eip_el1);
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
		const FiniteElement *FElem = pfes->GetFE(elementNo_el1);
		h1dofs_cnt = FElem->GetDof();
		velocity_shape.SetSize(h1dofs_cnt);
		velocity_shape = 0.0;
		FElem->CalcShape(eip_el1, velocity_shape);
		FElem->ProjectGrad(*FElem,Trans_el1,nodalGrad_el1);
		
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
		vD->Eval(D_el1, Trans_el1, eip_el1);
		vN->Eval(tN_el1, Trans_el1, eip_el1);
		/////
		
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
		    vShape(k) += velocity_shape(s) * loc_data_el1(s + h1dofs_cnt * k);
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
		  //	normalStressProjNormal += weightedNormalStress(s) * nor(s) / (nor_norm * nor_norm);
		  normalStressProjNormal += weightedNormalStress(s) * tN_el1(s);
		}
		
		double vDotn = 0.0;
		for (int s = 0; s < dim; s++)
		  {
		    //  vDotn += vShape(s) * nor(s) / nor_norm;
		    vDotn += vShape(s) * tN_el1(s);
		  }
		
		for (int i = 0; i < l2dofs_cnt; i++)
		  {
		    elvect(i) -= normalStressProjNormal * te_shape_el1(i) * ip_f.weight * vDotn * nTildaDotN * nor_norm;
		  }
	      }
	      else {
		nor *= -1.0;
		el2.CalcShape(eip_el2, te_shape_el2);
		double pressure = pface_gf.GetValue(Trans_el2,eip_el2);
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
		  FElem->CalcShape(eip_el2, velocity_shape);
		  FElem->ProjectGrad(*FElem,Trans_el2,nodalGrad_el2);
		}
		else {
		  const FiniteElement *FElem = pfes->GetFE(elementNo_el2);
		  h1dofs_cnt = FElem->GetDof();
		  velocity_shape.SetSize(h1dofs_cnt);
		  velocity_shape = 0.0;
		  FElem->CalcShape(eip_el2, velocity_shape);
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
		vD->Eval(D_el2, Trans_el2, eip_el2);
		vN->Eval(tN_el2, Trans_el2, eip_el2);
		/////
		
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
		    vShape(k) += velocity_shape(s) * loc_data_el2(s + h1dofs_cnt * k);
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
		  //	normalStressProjNormal += weightedNormalStress(s) * nor(s) / (nor_norm * nor_norm);
		  normalStressProjNormal += weightedNormalStress(s) * tN_el2(s);
		}
		double vDotn = 0.0;
		for (int s = 0; s < dim; s++)
		  {
		    vDotn += vShape(s) * tN_el2(s);
		    //  vDotn += vShape(s) * nor(s) / nor_norm;
		  }
		
		for (int i = 0; i < l2dofs_cnt; i++)
		  {
		    elvect(i + l2dofs_offset) -= normalStressProjNormal * te_shape_el2(i) * ip_f.weight * vDotn * nTildaDotN * nor_norm;
		    
		  }
	      }	  
	    }
	}
	else{
	  const int l2dofs_cnt = el.GetDof();
	  elvect.SetSize(2*l2dofs_cnt);
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
      if (Tr.Attribute == 77 ){
	const int dim = fe.GetDim();
	DenseMatrix identity(dim);
	identity = 0.0;
	for (int s = 0; s < dim; s++){
	  identity(s,s) = 1.0;
	}
	const int nqp_face = IntRule->GetNPoints();
	const int h1dofs_cnt = fe.GetDof();
	elmat.SetSize(2*h1dofs_cnt*dim);
	elmat = 0.0;
	
	Vector shape_el1(h1dofs_cnt), shape_test_el1(h1dofs_cnt), nor(dim), gradUResD_el1(h1dofs_cnt), test_gradUResD_el1(h1dofs_cnt);
	Vector shape_el2(h1dofs_cnt), shape_test_el2(h1dofs_cnt), gradUResD_el2(h1dofs_cnt), test_gradUResD_el2(h1dofs_cnt);
	ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
	ElementTransformation &Trans_el2 = Tr.GetElement2Transformation();
	DenseMatrix nodalGrad_el1, gradUResDirD_el1(h1dofs_cnt), taylorExp_el1(h1dofs_cnt);
	fe.ProjectGrad(fe,Trans_el1,nodalGrad_el1);
	DenseMatrix nodalGrad_el2, gradUResDirD_el2(h1dofs_cnt), taylorExp_el2(h1dofs_cnt);
	fe2.ProjectGrad(fe2, Trans_el2, nodalGrad_el2);

	int h1dofs_offset = fe2.GetDof()*dim;

	for (int q = 0; q < nqp_face; q++)
	  {
	    nor = 0.0;
	    
	    shape_el1 = 0.0;
	    shape_test_el1 = 0.0;
	    shape_el2 = 0.0;
	    shape_test_el2 = 0.0;

	    gradUResDirD_el1 = 0.0;
	    gradUResD_el1 = 0.0;
	    taylorExp_el1 = 0.0;
	    test_gradUResD_el1 = 0.0;

	    gradUResDirD_el2 = 0.0;
	    gradUResD_el2 = 0.0;
	    taylorExp_el2 = 0.0;
	    test_gradUResD_el2 = 0.0;

	    const IntegrationPoint &ip_f = IntRule->IntPoint(q);
	    // Set the integration point in the face and the neighboring elements
	    Tr.SetAllIntPoints(&ip_f);
	    const IntegrationPoint &eip_el1 = Tr.GetElement1IntPoint();
	    const IntegrationPoint &eip_el2 = Tr.GetElement2IntPoint();
	    
	    CalcOrtho(Tr.Jacobian(), nor);

	    double nor_norm = 0.0;
	    for (int s = 0; s < dim; s++){
	      nor_norm += nor(s) * nor(s);
	    }
	    nor_norm = sqrt(nor_norm);

	    double volumeFraction_el1 = alpha_gf.GetValue(Trans_el1, eip_el1);
	    double volumeFraction_el2 = alpha_gf.GetValue(Trans_el2, eip_el2);
	    double sum_volFrac = volumeFraction_el1 + volumeFraction_el2;
	    double gamma_1 =  volumeFraction_el1/sum_volFrac;
	    double gamma_2 =  volumeFraction_el2/sum_volFrac;
	    
	    if (gamma_1 > 0.99){
	      /////
	      Vector D_el1(dim);
	      Vector tN_el1(dim);
	      D_el1 = 0.0;
	      tN_el1 = 0.0; 	    
	      vD->Eval(D_el1, Trans_el1, eip_el1);
	      vN->Eval(tN_el1, Trans_el1, eip_el1);
	      /////
	      double normD = 0.0;
	      for (int s = 0; s < dim; s++){
		normD += D_el1(s) * D_el1(s);
	      }
	      normD = std::sqrt(normD);
	      // normD = 0.0;
	      const DenseMatrix &Jpr_el1 = Trans_el1.Jacobian();

	      double density = rhoface_gf.GetValue(Trans_el1,eip_el1);
	      double sound_speed = csface_gf.GetValue(Trans_el1,eip_el1);
	      double viscCoeff = viscousface_gf.GetValue(Trans_el1,eip_el1);
	      
	      Vector tOrig_el1(dim), tn(dim);
	      tOrig_el1 = 0.0;
	      tn = 0.0;
	      tn = nor;
	      tn /= nor_norm;
	      //  Jpr_el1.MultTranspose(tn,tOrig_el1);
	      Jpr_el1.MultTranspose(tN_el1,tOrig_el1);
	      double origNormalProd_el1 = 0.0;
	      for (int s = 0; s < dim; s++){
		origNormalProd_el1 += tOrig_el1(s) * tOrig_el1(s);
	      }
	      origNormalProd_el1 = std::pow(origNormalProd_el1,0.5);
	      tOrig_el1 *= 1.0/origNormalProd_el1;
	      //  std::cout << " tnX " << tn(0) << " tnY " << tn(1) <<  " x " << tOrig_el1(0) << " y " << tOrig_el1(1) << std::endl; 
	      fe.CalcShape(eip_el1, shape_el1);
	      fe.CalcShape(eip_el1, shape_test_el1);
	      
	      double nTildaDotN = 0.0;
	      for (int s = 0; s < dim; s++){
		nTildaDotN += nor(s) * tN_el1(s) / nor_norm;
	      }
	      double proj = 0.0;
	      for (int s = 0; s < dim; s++){
		proj += std::abs(tn(s));
	      }

	      double penaltyVal = 0.0;
	      if ( (globalmax_viscous_coef != 0.0) && (sound_speed > 0.0) && (viscCoeff > 0.0)){
		//	double aMax = (density/viscCoeff) * sound_speed *  (Tr.Elem1->Weight() / nor_norm + normD);
		//double aMax = (globalmax_rho/globalmax_viscous_coef) * globalmax_cs *  (Tr.Elem2->Weight() / nor_norm + normD);
		double aMax = (globalmax_rho/globalmax_viscous_coef) * globalmax_cs * 4.0;
		//	penaltyVal = penaltyParameter * globalmax_rho * (1.0 + ((globalmax_viscous_c-oef/globalmax_rho)*(1.0/globalmax_cs * globalmax_cs) + (globalmax_rho/globalmax_viscous_coef) * globalmax_cs * globalmax_cs) ) * (Tr.Elem1->Weight() / nor_norm) * std::pow(1.0/origNormalProd_el1,2.0-2.0*std::max(std::fabs(normD*nor_norm/Tr.Elem1->Weight()),1.0));
		//	penaltyVal = penaltyParameter * globalmax_rho * (1.0 + aMax + 1.0/aMax) * (Tr.Elem1->Weight() / nor_norm + normD) * std::pow(1.0/origNormalProd_el1,2.0-2.0*std::max(std::fabs(normD*nor_norm/Tr.Elem1->Weight()),1.0));
		//	penaltyVal = penaltyParameter * density * (1.0 + std::pow(aMax,1.0) + std::pow(1.0/aMax,1.0)) * (Tr.Elem1->Weight() / nor_norm + normD) * std::pow(1.0/origNormalProd_el1,2.0*order_v-2.0*std::min(std::fabs(normD*nor_norm/Tr.Elem1->Weight()),1.0));
		penaltyVal =  4.0 * std::pow(penaltyParameter * (1.0 * 1.0/aMax + aMax),proj) /* * std::pow(penaltyParameter,proj)*/ * globalmax_rho /* * ( nor_norm / Tr.Elem1->Weight()) */ ;

		//
		//	std::cout << " amxa " << aMax << " nCn " << 1.0/origNormalProd_el1 << " pen " << penaltyVal << std::endl;
	      }
	      else {
		//	penaltyVal = penaltyParameter * globalmax_rho * (Tr.Elem1->Weight() / nor_norm + normD) * std::pow(1.0/origNormalProd_el1,2.0-2.0*std::max(std::fabs(normD*nor_norm/Tr.Elem1->Weight()),1.0));
		//	penaltyVal = penaltyParameter * density * (Tr.Elem1->Weight() / nor_norm + normD) * std::pow(1.0/origNormalProd_el1,2.0*order_v-2.0*std::min(std::fabs(normD*nor_norm/Tr.Elem1->Weight()),1.0));
		penaltyVal =  4.0 * std::pow(penaltyParameter,proj) /* * std::pow(penaltyParameter,proj)*/ * globalmax_rho /* * ( nor_norm / Tr.Elem1->Weight()) */ ;

	      }
	      //  std::cout << " pen " << penaltyVal << " sound " << sound_speed << std::endl; 
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
		gradUResD_el1(k) += taylorExp_el1(k,s) * shape_el1(s);  
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
		  gradUResD_el1(k) += taylorExp_el1(k,s) * shape_el1(s);  
		}
	      }
	    }
	    
	    ////
	    shape_el1 += gradUResD_el1;
	    //
	    
	    if (fullPenalty){
	      shape_test_el1 += gradUResD_el1;
	    }
	    else{
	      shape_test_el1 += test_gradUResD_el1;
	    }

	      /*      DenseMatrix h1_grads(h1dofs_cnt, dim);
	      h1_grads = 0.0;
	      Trans_el1.SetIntPoint(&eip_el1);
	      // Compute grad_psi in the first element.

	      fe.CalcPhysDShape(Trans_el1, h1_grads);
	      for (int j = 0; j < h1dofs_cnt; j++){
		for (int s = 0; s < dim; s++){
		  gradUResD_el1(j) += h1_grads(j,s) * D_el1(s);
		}
	      }
	      shape_el1 += gradUResD_el1;
	      shape_test_el1 = shape_el1;*/
	      
	    for (int i = 0; i < h1dofs_cnt; i++)
	      {
		for (int vd = 0; vd < dim; vd++) // Velocity components.
		  {
		    for (int j = 0; j < h1dofs_cnt; j++)
		      {
			for (int md = 0; md < dim; md++) // Velocity components.
			  {	      
			    elmat(i + vd * h1dofs_cnt, j + md * h1dofs_cnt) += shape_test_el1(i) * shape_el1(j) * nor_norm * tN_el1(vd) * tN_el1(md) * penaltyVal * ip_f.weight  * nTildaDotN * nTildaDotN;
	
			  }
		      }
		  }
	      }
	    
	    }
	    else {
	      nor *= -1.0;
	      /////
	      Vector D_el2(dim);
	      Vector tN_el2(dim);
	      D_el2 = 0.0;
	      tN_el2 = 0.0;  
	      vD->Eval(D_el2, Trans_el2, eip_el2);
	      vN->Eval(tN_el2, Trans_el2, eip_el2);
	      /////

	      double normD = 0.0;
	      for (int s = 0; s < dim; s++){
		normD += D_el2(s) * D_el2(s);
	      }
	      normD = std::sqrt(normD);
	      //  normD = 0.0;
	      Vector tn(dim);
	      tn = 0.0;
	      tn = nor;
	      tn /= nor_norm;
	      const DenseMatrix &Jpr_el2 = Trans_el2.Jacobian();

	      double density = rhoface_gf.GetValue(Trans_el2,eip_el2);
	      double sound_speed = csface_gf.GetValue(Trans_el2,eip_el2);
	      double viscCoeff = viscousface_gf.GetValue(Trans_el2,eip_el2);
	
	      Vector tOrig_el2(dim);
	      tOrig_el2 = 0.0;
	      //  Jpr_el2.MultTranspose(tn,tOrig_el2);
	      Jpr_el2.MultTranspose(tN_el2,tOrig_el2);
	      double origNormalProd_el2 = 0.0;
	      for (int s = 0; s < dim; s++){
		origNormalProd_el2 += tOrig_el2(s) * tOrig_el2(s);
	      }
	      origNormalProd_el2 = std::pow(origNormalProd_el2,0.5);
	      tOrig_el2 *= 1.0/origNormalProd_el2;

	      fe2.CalcShape(eip_el2, shape_el2);
	      fe2.CalcShape(eip_el2, shape_test_el2);
	      
	      double nTildaDotN = 0.0;
	      for (int s = 0; s < dim; s++){
		nTildaDotN += nor(s) * tN_el2(s) / nor_norm;
	      }
	      double proj = 0.0;
	      for (int s = 0; s < dim; s++){
		proj += std::abs(tn(s));
	      }

	      double penaltyVal = 0.0;
	      if ( (globalmax_viscous_coef != 0.0) && (sound_speed > 0.0) && (viscCoeff > 0.0)){
		//	double aMax = (density/viscCoeff) * sound_speed *  (Tr.Elem2->Weight() / nor_norm + normD);
		double aMax = (globalmax_rho/globalmax_viscous_coef) * globalmax_cs * 4.0;

		//	double aMax = (globalmax_rho/globalmax_viscous_coef) * globalmax_cs *  (Tr.Elem2->Weight() / nor_norm + normD);

		//	penaltyVal = penaltyParameter * globalmax_rho * (1.0 + ((globalmax_viscous_coef/globalmax_rho)*(1.0/globalmax_cs * globalmax_cs) + (globalmax_rho/globalmax_viscous_coef) * globalmax_cs * globalmax_cs) ) * (Tr.Elem2->Weight() / nor_norm) * std::pow(1.0/origNormalProd_el2,2.0-2.0*std::max(std::fabs(normD*nor_norm/Tr.Elem2->Weight()),1.0));
		//		penaltyVal = penaltyParameter * density * (1.0 + std::pow(aMax,1.0) + std::pow(1.0/aMax,1.0)) * (Tr.Elem2->Weight() / nor_norm + normD) * std::pow(1.0/origNormalProd_el2,2.0*order_v-2.0*std::min(std::fabs(normD*nor_norm/Tr.Elem2->Weight()),1.0));
		penaltyVal =  4.0 * std::pow(penaltyParameter * (1.0 * 1.0/aMax + aMax),proj) /* * std::pow(penaltyParameter,proj)*/ * globalmax_rho /* * ( nor_norm / Tr.Elem1->Weight()) */ ;

	      }
	      else {
		//	penaltyVal = penaltyParameter * density * (Tr.Elem2->Weight() / nor_norm + normD) * std::pow(1.0/origNormalProd_el2,2.0*order_v-2.0*std::min(std::fabs(normD*nor_norm/Tr.Elem2->Weight()),1.0));
		penaltyVal =  4.0 * std::pow(penaltyParameter,proj) /* * std::pow(penaltyParameter,proj)*/ * globalmax_rho /* * ( nor_norm / Tr.Elem1->Weight()) */ ;
	      }
	      // std::cout << " pen " << penaltyVal << " sound " << sound_speed << std::endl; 
	    
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
		  gradUResD_el2(k) += taylorExp_el2(k,s) * shape_el2(s);  
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
		  gradUResD_el2(k) += taylorExp_el2(k,s) * shape_el2(s);  
		}
	      }
	    }
	    
	    ////
	    shape_el2 += gradUResD_el2;
	    //
	    
	    if (fullPenalty){
	      shape_test_el2 += gradUResD_el2;
	    }
	    else{
	      shape_test_el2 += test_gradUResD_el2;
	      }
	      
	    /* DenseMatrix h1_grads(h1dofs_cnt, dim);
	      h1_grads = 0.0;
	      Trans_el2.SetIntPoint(&eip_el2);
	      // Compute grad_psi in the first element.
	      fe2.CalcPhysDShape(Trans_el2, h1_grads);
	      for (int j = 0; j < h1dofs_cnt; j++){
		for (int s = 0; s < dim; s++){
		  gradUResD_el2(j) += h1_grads(j,s) * D_el2(s);
		}
	      }
	      shape_el2 += gradUResD_el2;
	      shape_test_el2 = shape_el2;
	    */
	    for (int i = 0; i < h1dofs_cnt; i++)
	      {
		for (int vd = 0; vd < dim; vd++) // Velocity components.
		  {
		    for (int j = 0; j < h1dofs_cnt; j++)
		      {
			for (int md = 0; md < dim; md++) // Velocity components.
			  {	      
			    elmat(i + vd * h1dofs_cnt + h1dofs_offset, j + md * h1dofs_cnt + h1dofs_offset) += shape_test_el2(i) * shape_el2(j) * nor_norm * tN_el2(vd) * tN_el2(md) * penaltyVal * ip_f.weight * nTildaDotN * nTildaDotN;		    
			  }
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
