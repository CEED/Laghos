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

    void shift_shape(const ParFiniteElementSpace &pfes_e_const,
		     const ParFiniteElementSpace &pfes_p,
		     int e_id,
		     const IntegrationPoint &ip, const Vector &dist,
		     int nterms, Vector &shape_shift)
    {
      auto pfes_e = const_cast<ParFiniteElementSpace *>(&pfes_e_const);
      const int NE = pfes_e->GetNE();
      const FiniteElement &el_e =
	(e_id < NE) ? *pfes_e->GetFE(e_id) : *pfes_e->GetFaceNbrFE(e_id - NE);
      const FiniteElement &el_p =
	(e_id < NE) ? *pfes_p.GetFE(e_id) : *pfes_p.GetFaceNbrFE(e_id - NE);
      const int dim = pfes_e->GetMesh()->Dimension(),
	dof_e = el_e.GetDof(), dof_p = el_p.GetDof();

      IsoparametricTransformation el_tr;
      if (e_id < NE)
	{
	  pfes_e->GetElementTransformation(e_id, &el_tr);
	}
      else
	{
	  pfes_e->GetParMesh()->GetFaceNbrElementTransformation(e_id - NE, &el_tr);
	}
     
      DenseMatrix grad_phys;
      DenseMatrix Transfer_pe;
      el_p.Project(el_e, el_tr, Transfer_pe);
      el_p.ProjectGrad(el_p, el_tr, grad_phys);

      Vector s(dim*dof_p), t(dof_p);
      for (int j = 0; j < dof_e; j++)
	{
	  // Shape function transformed into the p space.
	  Vector u_shape_e(dof_e), u_shape_p(dof_p);
	  u_shape_e = 0.0;
	  u_shape_e(j) = 1.0;
	  Transfer_pe.Mult(u_shape_e, u_shape_p);

	  t = u_shape_p;
	  int factorial = 1;
	  for (int i = 1; i < nterms + 1; i++)
	    {
	      factorial = factorial*i;
	      grad_phys.Mult(t, s);
	      for (int j = 0; j < dof_p; j++)
		{
		  t(j) = 0.0;
		  for(int d = 0; d < dim; d++)
		    {
		      t(j) = t(j) + s(j + d * dof_p) * dist(d);
		    }
		}
	      u_shape_p.Add(1.0/double(factorial), t);
	    }

	  el_tr.SetIntPoint(&ip);
	  el_p.CalcPhysShape(el_tr, t);
	  shape_shift(j) = t * u_shape_p;
	}
    }


    void get_shifted_value(const ParGridFunction &g, int e_id,
			   const IntegrationPoint &ip, const Vector &dist,
			   int nterms, Vector &shifted_vec)
    {
      auto pfes = const_cast<ParFiniteElementSpace *>(g.ParFESpace());
      const int NE = pfes->GetNE();
      const FiniteElement &el =
	(e_id < NE) ? *pfes->GetFE(e_id) : *pfes->GetFaceNbrFE(e_id - NE);
      const int dim = pfes->GetMesh()->Dimension(), dof = el.GetDof();
      const int vdim = pfes->GetVDim();
      /* ElementTransformation *el_tr = NULL;
      if (e_id < NE)
	{
	  el_tr = pfes->GetElementTransformation(e_id);
	}*/
      IsoparametricTransformation el_tr;
      if (e_id < NE)
	{
	  pfes->GetElementTransformation(e_id, &el_tr);
	}
      else
	{
	  pfes->GetParMesh()->GetFaceNbrElementTransformation(e_id - NE, &el_tr);
	}
      DenseMatrix grad_phys;
      el.ProjectGrad(el, el_tr, grad_phys);
      
      Array<int> vdofs;
      Vector u(dof), s(dim*dof), t(dof), g_loc(vdim*dof);
      if (e_id < NE)
	{
	  g.FESpace()->GetElementVDofs(e_id, vdofs);
	  g.GetSubVector(vdofs, g_loc);
	}
      else
	{
	  g.ParFESpace()->GetFaceNbrElementVDofs(e_id - NE, vdofs);
	  g.FaceNbrData().GetSubVector(vdofs, g_loc);
	}
      
      for (int c = 0; c < vdim; c++)
	{
	  u.SetDataAndSize(g_loc.GetData() + c*dof, dof);
	  
	  t = u;
	  int factorial = 1;
	  for (int i = 1; i < nterms + 1; i++)
	    {
	      factorial = factorial*i;
	      grad_phys.Mult(t, s);
	      for (int j = 0; j < dof; j++)
		{
		  t(j) = 0.0;
		  for(int d = 0; d < dim; d++)
		    {
		      t(j) = t(j) + s(j + d * dof) * dist(d);
		    }
		}
	      u.Add(1.0/double(factorial), t);
	    }
	  
	  el_tr.SetIntPoint(&ip);
	  el.CalcPhysShape(el_tr, t);
	  shifted_vec(c) = t * u;
	}
    }

    
    void WeightedVectorMassIntegrator::AssembleElementMatrix(const FiniteElement &el,
							     ElementTransformation &Trans,
							     DenseMatrix &elmat)
    {
      const int dim = el.GetDim();
      int dof = el.GetDof();
      
      elmat.SetSize (dof*dim);
      elmat = 0.0;
      
      Vector shape(dof);
      shape = 0.0;
      
      const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el, Trans);
      
      for (int q = 0; q < ir->GetNPoints(); q++)
	{
	  const IntegrationPoint &ip = ir->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Trans.SetIntPoint(&ip);
	  el.CalcShape(ip, shape);

	  double volumeFraction = alpha.GetValue(Trans, ip);
	  double density = rho_gf.GetValue(Trans, ip);

	  for (int i = 0; i < dof; i++)
	    {
	      for (int k = 0; k < dim; k++)
		{
		  for (int j = 0; j < dof; j++)
		    {
		      elmat(i + k * dof, j + k * dof) += shape(i) * shape(j) * ip.weight * volumeFraction * Trans.Weight() * density;
		    }
		}
	    }
	}
    }

    const IntegrationRule &WeightedVectorMassIntegrator::GetRule(
								 const FiniteElement &trial_fe,
								 const FiniteElement &test_fe,
								 ElementTransformation &Trans)
    {
      int order = Trans.OrderGrad(&trial_fe) + test_fe.GetOrder() + Trans.OrderJ();
      return IntRules.Get(trial_fe.GetGeomType(), 2*order);
    }

    
    void WeightedMassIntegrator::AssembleElementMatrix(const FiniteElement &el,
						       ElementTransformation &Trans,
						       DenseMatrix &elmat)
    {
      const int dim = el.GetDim();
      int dof = el.GetDof();
      
      elmat.SetSize (dof);
      elmat = 0.0;
      
      Vector shape(dof);
      shape = 0.0;
      
      const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el, Trans);
      
      for (int q = 0; q < ir->GetNPoints(); q++)
	{
	  const IntegrationPoint &ip = ir->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Trans.SetIntPoint(&ip);
	  el.CalcShape(ip, shape);

	  double volumeFraction = alpha.GetValue(Trans, ip);
	  double density = rho_gf.GetValue(Trans, ip);

	  for (int i = 0; i < dof; i++)
	    {
	      for (int j = 0; j < dof; j++)
		{
		  elmat(i, j) += shape(i) * shape(j) * ip.weight * volumeFraction * Trans.Weight() * density;
		}
	    }
	}
    }
  

    const IntegrationRule &WeightedMassIntegrator::GetRule(
							   const FiniteElement &trial_fe,
							   const FiniteElement &test_fe,
							   ElementTransformation &Trans)
    {
      int order = Trans.OrderGrad(&trial_fe) + test_fe.GetOrder() + Trans.OrderJ();
      return IntRules.Get(trial_fe.GetGeomType(), 2*order);
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
	  double rho0DetJ0 = rho0DetJ0_gf.GetValue(Tr, ip);

	  // Note that rhoDetJ = rho0DetJ0.
	  shape *= rho0DetJ0 * ip.weight;
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
	  double volumeFraction = alpha.GetValue(Tr, ip);

	  Vector Jac0inv_vec(dim*dim);
	  Jac0inv_vec = 0.0;
	  Jac0inv_gf.GetVectorValue(Tr.ElementNo,ip,Jac0inv_vec);
	  DenseMatrix Jac0inv(dim);
	  ConvertVectorToDenseMatrix(dim, Jac0inv_vec, Jac0inv);
	  
	  const double rho = rho_gf.GetValue(Tr,ip);
	  ComputeStress(pressure,dim,stress);
	  ComputeViscousStress(Tr, v_gf, Jac0inv, h0, use_viscosity, use_vorticity, rho, sound_speed, dim, stress);
	  MultABt(stress, Jinv, stressJiT);
	  stressJiT *= ip.weight * Jpr.Det();
	  
	  for (int i = 0; i < h1dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  for (int gd = 0; gd < dim; gd++) // Gradient components.
		    {
		      elvect(i + vd * h1dofs_cnt) -= stressJiT(vd,gd) * vshape(i,gd) * volumeFraction;
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
	    double volumeFraction = alpha.GetValue(Tr, ip);

	    Vector Jac0inv_vec(dim*dim);
	    Jac0inv_vec = 0.0;
	    Jac0inv_gf.GetVectorValue(Tr.ElementNo,ip,Jac0inv_vec);
	    DenseMatrix Jac0inv(dim);
	    ConvertVectorToDenseMatrix(dim, Jac0inv_vec, Jac0inv);
	    const double rho = rho_gf.GetValue(Tr,ip);

	    ComputeStress(pressure,dim,stress);
	    ComputeViscousStress(Tr, v_gf, Jac0inv, h0, use_viscosity, use_vorticity, rho, sound_speed, dim, stress);
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
		elvect(i) += gradVContractedStress * te_shape(i) * volumeFraction;
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
	  Jac0invface_gf.GetVectorValue(Trans_el1.ElementNo,eip,Jac0inv_vec);
	  DenseMatrix Jac0inv(dim);
	  ConvertVectorToDenseMatrix(dim, Jac0inv_vec, Jac0inv);
	  
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
	    Jac0invface_gf.GetVectorValue(Trans_el1.ElementNo,eip,Jac0inv_vec);
	    DenseMatrix Jac0inv(dim);
	    ConvertVectorToDenseMatrix(dim, Jac0inv_vec, Jac0inv);
	    

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
	  
	  Vector Jac0inv_vec(dim*dim);
	  Jac0inv_vec = 0.0;
	  Jac0invface_gf.GetVectorValue(elementNo,eip,Jac0inv_vec);
	  DenseMatrix Jac0inv(dim);
	  ConvertVectorToDenseMatrix(dim, Jac0inv_vec, Jac0inv);
	  
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

	  DenseMatrix v_grad_q1(dim);
	  v_gf.GetVectorGradient(Trans_el1, v_grad_q1);
	  // As in the volumetric viscosity.
	  v_grad_q1.Symmetrize();
	  double h_1, mu_1;
	  
	  LengthScaleAndCompression(v_grad_q1, Trans_el1, Jac0inv,
				    h0, h_1, mu_1);
	  double density_el1 = rhoface_gf.GetValue(Trans_el1,eip);

	   // OLD //
	  penaltyVal = 4.0 * penaltyParameter * globalmax_rho /* * ( nor_norm / Tr.Elem1->Weight()) */ ;
	  //////
	  // NEW //
	  // penaltyVal = 4.0  * penaltyParameter * density_el1 * h_1 /* * origNormalProd*/ /* * (qdata.h0 * qdata.h0 / h_1)*/ ;
	  //////
	  // penaltyVal = 4.0 * penaltyParameter * globalmax_rho /* * ( nor_norm / Tr.Elem1->Weight()) */ ;
	  ///
	  
	
	  //	  std::cout << " penV " << penaltyVal << std::endl;
	  // penaltyVal = 4 * globalmax_rho * std::pow(penaltyParameter, 1.0*(1+aMax+1.0/aMax))/* * origNormalProd*/;
	  //	    std::cout << " dens " << density << " abs " << std::abs(density) << std::endl;
	  //  std::cout << " val " << std::pow(1.0/origNormalProd,2.0) << std::endl;
	  // std::cout << " nCn " << std::pow(1.0/origNormalProd,2.0) << std::endl;
	  //std::cout << " pen " << penaltyVal << std::endl;
	  fe.CalcShape(eip, shape);
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

    
    void DiffusionNormalVelocityIntegrator::AssembleRHSElementVect(const FiniteElement &el,
								   FaceElementTransformations &Tr,
								   Vector &elvect)
    {
      const int nqp_face = IntRule->GetNPoints();
      const int dim = el.GetDim();
      const int h1dofs_cnt = el.GetDof();
      elvect.SetSize(h1dofs_cnt*dim);
      elvect = 0.0;
      Vector shape(h1dofs_cnt);
      shape = 0.0;
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
	  CalcOrtho(Tr.Jacobian(), nor);
	  
	  Vector Jac0inv_vec(dim*dim);
	  Jac0inv_vec = 0.0;
	  Jac0invface_gf.GetVectorValue(elementNo,eip,Jac0inv_vec);
	  DenseMatrix Jac0inv(dim);
	  ConvertVectorToDenseMatrix(dim, Jac0inv_vec, Jac0inv);
	  
	  double nor_norm = 0.0;
	  for (int s = 0; s < dim; s++){
	    nor_norm += nor(s) * nor(s);
	  }
	  nor_norm = sqrt(nor_norm);
	  
	  double penaltyVal = 0.0;

	  DenseMatrix v_grad_q1(dim);
	  v_gf.GetVectorGradient(Trans_el1, v_grad_q1);
	  // As in the volumetric viscosity.
	  v_grad_q1.Symmetrize();
	  double h_1, mu_1;
	  
	  LengthScaleAndCompression(v_grad_q1, Trans_el1, Jac0inv,
				    h0, h_1, mu_1);
	  double density_el1 = rhoface_gf.GetValue(Trans_el1,eip);

	  Vector vShape;
	  v_gf.GetVectorValue(elementNo, eip, vShape);
	  double vDotn = 0.0;
	  for (int s = 0; s < dim; s++)
	    {
	      vDotn += vShape(s) * nor(s)/nor_norm;
	    }

	  double cs_el1 = csface_gf.GetValue(Trans_el1,eip);
	  
	  // NEW //
	  penaltyVal = penaltyParameter * density_el1 * cs_el1;
	  ///
	  el.CalcShape(eip, shape);
	  for (int i = 0; i < h1dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  elvect(i + vd * h1dofs_cnt) -= shape(i) * vDotn * nor(vd) * penaltyVal * ip_f.weight;
		  //	  std::cout << " val " << vDotn << std::endl;
		}
	    }
	}
    }
    
    void DiffusionNormalVelocityIntegrator::AssembleRHSElementVect(
								   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
    {
      mfem_error("DGDirichletLFIntegrator::AssembleRHSElementVect");
    }

    
    void DiffusionEnergyNormalVelocityIntegrator::AssembleRHSElementVect(const FiniteElement &el,
									 FaceElementTransformations &Tr,
									 Vector &elvect)
    {
      if (Vnpt_gf != NULL){
	const int nqp_face = IntRule->GetNPoints();
	const int dim = el.GetDim();
	const int l2dofs_cnt = el.GetDof();
	elvect.SetSize(l2dofs_cnt);
	elvect = 0.0;
	Vector shape(l2dofs_cnt);
	shape = 0.0;
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
	    CalcOrtho(Tr.Jacobian(), nor);
	  
	    Vector Jac0inv_vec(dim*dim);
	    Jac0inv_vec = 0.0;
	    Jac0invface_gf.GetVectorValue(elementNo,eip,Jac0inv_vec);
	    DenseMatrix Jac0inv(dim);
	    ConvertVectorToDenseMatrix(dim, Jac0inv_vec, Jac0inv);
	  
	    double nor_norm = 0.0;
	    for (int s = 0; s < dim; s++){
	      nor_norm += nor(s) * nor(s);
	    }
	    nor_norm = sqrt(nor_norm);
	  
	    double penaltyVal = 0.0;

	    DenseMatrix v_grad_q1(dim);
	    v_gf.GetVectorGradient(Trans_el1, v_grad_q1);
	    // As in the volumetric viscosity.
	    v_grad_q1.Symmetrize();
	    double h_1, mu_1;
	  
	    LengthScaleAndCompression(v_grad_q1, Trans_el1, Jac0inv,
				      h0, h_1, mu_1);
	    double density_el1 = rhoface_gf.GetValue(Trans_el1,eip);

	    Vector vShape;
	    v_gf.GetVectorValue(elementNo, eip, vShape);
	    double vDotn = 0.0;
	    for (int s = 0; s < dim; s++)
	      {
		vDotn += vShape(s) * nor(s)/nor_norm;
	      }
	    double cs_el1 = csface_gf.GetValue(Trans_el1,eip);
	  
	    // NEW //
	    penaltyVal = penaltyParameter * density_el1 * cs_el1;
	    ///
	    el.CalcShape(eip, shape);
	    for (int i = 0; i < l2dofs_cnt; i++)
	      {
		elvect(i) += shape(i) * vDotn * vDotn * penaltyVal * ip_f.weight * nor_norm;
		//		std::cout << " energ val " << elvect(i) << std::endl;
	      }
	  }
      }
      else{
	const int l2dofs_cnt = el.GetDof();
	elvect.SetSize(l2dofs_cnt);
	elvect = 0.0;
      }
    }
    
    void DiffusionEnergyNormalVelocityIntegrator::AssembleRHSElementVect(
								   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
    {
      mfem_error("DGDirichletLFIntegrator::AssembleRHSElementVect");
    }


    
    void ShiftedVelocityBoundaryForceIntegrator::AssembleRHSElementVect(const FiniteElement &el,
									const FiniteElement &el2,
									FaceElementTransformations &Tr,
									Vector &elvect)      
    {
      if ( (Tr.Attribute == 77) || (Tr.Attribute == 11) ){
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
	    
	    el.CalcShape(eip_el1, te_shape_el1);
	    el2.CalcShape(eip_el2, te_shape_el2);
	    
	    double pressure_el1 = pface_gf.GetValue(Trans_el1,eip_el1);
	    double pressure_el2 = pface_gf.GetValue(Trans_el2,eip_el2);
	    
	    DenseMatrix stress_el1(dim), stress_el2(dim);
	    stress_el1 = 0.0;
	    stress_el2 = 0.0;
	    
	    ComputeStress(pressure_el1,dim,stress_el1);
	    ComputeStress(pressure_el2,dim,stress_el2);
	    
	    for (int i = 0; i < h1dofs_cnt; i++)
	      {
		for (int vd = 0; vd < dim; vd++) // Velocity components.
		  {
		    for (int md = 0; md < dim; md++) // Velocity components.
		      {		
			elvect(i + vd * h1dofs_cnt) += (stress_el1(vd,md) * gamma_1 + stress_el2(vd,md) * gamma_2 ) * (volumeFraction_el1 * nor(md) - volumeFraction_el2 * nor(md) ) * gamma_1 * te_shape_el1(i) * ip_f.weight;
			elvect(i + vd * h1dofs_cnt + h1dofs_cnt * dim) += (stress_el1(vd,md) * gamma_1 + stress_el2(vd,md) * gamma_2 ) * (volumeFraction_el1 * nor(md) - volumeFraction_el2 * nor(md) ) * gamma_2 * te_shape_el2(i) * ip_f.weight;
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
	if ( (Tr.Attribute == 77) || (Tr.Attribute == 11) ){
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
	  
	  ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
	 
	  ElementTransformation &Trans_el2 = Tr.GetElement2Transformation();
	  
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

	      // std::cout << " g amma " <<  gamma_1 << " gamma " << gamma_2 << std::endl;
	      el.CalcShape(eip_el1, te_shape_el1);
	      el2.CalcShape(eip_el2, te_shape_el2);
      
	      double pressure_el1 = pface_gf.GetValue(Trans_el1,eip_el1);
	      double pressure_el2 = pface_gf.GetValue(Trans_el2,eip_el2);
	     
	      DenseMatrix stress_el1(dim), stress_el2(dim);
	      stress_el1 = 0.0;
	      stress_el2 = 0.0;
	      ComputeStress(pressure_el1,dim,stress_el1);
	      ComputeStress(pressure_el2,dim,stress_el2);

	      /////////////

	      /////
	      Vector D_el1(dim);
	      Vector tN_el1(dim);
	      D_el1 = 0.0;
	      tN_el1 = 0.0;
	      // if (Tr.Attribute == 77){
	      vD->Eval(D_el1, Trans_el1, eip_el1);
		// }
	      vN->Eval(tN_el1, Trans_el1, eip_el1);

	      double nTildaDotN = 0.0;
	      for (int s = 0; s < dim; s++){
		nTildaDotN += nor(s) * tN_el1(s) / nor_norm;
	      }
		
	      Vector gradv_d_el1(dim);
	      gradv_d_el1 = 0.0;
	      get_shifted_value(*Vnpt_gf, Trans_el1.ElementNo, eip_el1, D_el1, nTerms, gradv_d_el1);
	      Vector v_vals_el1(dim);
	      v_vals_el1 = 0.0;
	      Vnpt_gf->GetVectorValue(Trans_el1, eip_el1, v_vals_el1);
	      // gradv_d_el1 += v_vals_el1;
	
	      double vDotn_el1 = 0.0;
	      for (int s = 0; s < dim; s++)
	       {
		 vDotn_el1 += gradv_d_el1(s) * tN_el1(s);
	       }
	      ////////////////	
	      /////
	      Vector D_el2(dim);
	      Vector tN_el2(dim);
	      D_el2 = 0.0;
	      tN_el2 = 0.0;
	      // if (Tr.Attribute == 77){
		vD->Eval(D_el2, Trans_el2, eip_el2);
		// }
	      vN->Eval(tN_el2, Trans_el2, eip_el2);
	      /////
	      
	      Vector gradv_d_el2(dim);
	      gradv_d_el2 = 0.0;
	      //	      std::cout << " first elem " << Trans_el1.ElementNo << " second elem " << Trans_el2.ElementNo << std::endl;
	      get_shifted_value(*Vnpt_gf, Trans_el2.ElementNo, eip_el2, D_el2, nTerms, gradv_d_el2);
	      Vector v_vals_el2(dim);
	      v_vals_el2 = 0.0;
	      Vnpt_gf->GetVectorValue(Trans_el2, eip_el2, v_vals_el2);
	      // gradv_d_el2 += v_vals_el2;
	      
	      double vDotn_el2 = 0.0;
	      for (int s = 0; s < dim; s++)
	       {
		 vDotn_el2 += gradv_d_el2(s) * tN_el1(s);
	       }
			 		 
	      ///////	
	      for (int i = 0; i < l2dofs_cnt; i++)
		{
		  for (int vd = 0; vd < dim; vd++)
		    {
		      for (int md = 0; md < dim; md++)
			{
			  elvect(i) -= gamma_1 * stress_el1(vd,md) * te_shape_el1(i) * (volumeFraction_el1 * nor(vd) - volumeFraction_el2 * nor(vd) ) * (gamma_1 * vDotn_el1 + gamma_2 * vDotn_el2) * ip_f.weight * tN_el1(md);
			  elvect(i+l2dofs_cnt) -= gamma_2 * stress_el2(vd,md) * te_shape_el2(i) * (volumeFraction_el1 * nor(vd) - volumeFraction_el2 * nor(vd) ) * (gamma_1 * vDotn_el1 + gamma_2 * vDotn_el2) * ip_f.weight * tN_el1(md);

	      /*  for (int i = 0; i < l2dofs_cnt; i++)
		{
		  elvect(i) -= gamma_1 * pressure_el1 * te_shape_el1(i) * (volumeFraction_el1 - volumeFraction_el2 ) * (gamma_1 * vDotn_el1 + gamma_2 * vDotn_el2) * ip_f.weight * std::abs(nTildaDotN) * nor_norm;
		  elvect(i+l2dofs_cnt) -= gamma_2 * pressure_el2 * te_shape_el2(i) * (volumeFraction_el1 - volumeFraction_el2 ) * (gamma_1 * vDotn_el1 + gamma_2 * vDotn_el2) * ip_f.weight * std::abs(nTildaDotN) * nor_norm;
		}*/
			  
			}
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
      if ( (Tr.Attribute == 77) || (Tr.Attribute == 11) ){
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
	
	Vector shape_el1(h1dofs_cnt), shape_test_el1(h1dofs_cnt), nor(dim);
	Vector shape_el2(h1dofs_cnt), shape_test_el2(h1dofs_cnt);
	ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
	ElementTransformation &Trans_el2 = Tr.GetElement2Transformation();

	int h1dofs_offset = fe2.GetDof()*dim;

	for (int q = 0; q < nqp_face; q++)
	  {
	    nor = 0.0;
	    
	    shape_el1 = 0.0;
	    shape_test_el1 = 0.0;
	    shape_el2 = 0.0;
	    shape_test_el2 = 0.0;

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

	    Vector tn(dim);
	    tn = 0.0;
	    tn = nor;
	    tn /= nor_norm;
	    
	    double volumeFraction_el1 = alpha_gf.GetValue(Trans_el1, eip_el1);
	    double volumeFraction_el2 = alpha_gf.GetValue(Trans_el2, eip_el2);
	    double sum_volFrac = volumeFraction_el1 + volumeFraction_el2;
	    double gamma_1 =  volumeFraction_el1/sum_volFrac;
	    double gamma_2 =  volumeFraction_el2/sum_volFrac;
	    
	    /////
	    Vector D_el1(dim);
	    Vector tN_el1(dim);
	    D_el1 = 0.0;
	    tN_el1 = 0.0;
	    // if (Tr.Attribute == 77){
	      vD->Eval(D_el1, Trans_el1, eip_el1);
	      // }
	    vN->Eval(tN_el1, Trans_el1, eip_el1);
	    /////
	    
	    /////
	      Vector D_el2(dim);
	      Vector tN_el2(dim);
	      D_el2 = 0.0;
	      tN_el2 = 0.0;
	      // if (Tr.Attribute == 77){
	      vD->Eval(D_el2, Trans_el2, eip_el2);
		// }
	      vN->Eval(tN_el2, Trans_el2, eip_el2);
	      /////
	      
	    double nTildaDotN = 0.0;
	    for (int s = 0; s < dim; s++){
	      nTildaDotN += nor(s) * tN_el1(s) / nor_norm;
	    }


	    double penaltyVal = 0.0;
	    penaltyVal = 4.0 * penaltyParameter * globalmax_rho ;
	    
	    Vector Jac0inv_vec_el1(dim*dim),Jac0inv_vec_el2(dim*dim);
	    Jac0inv_vec_el1 = 0.0;
	    Jac0inv_vec_el2 = 0.0;
	    
	    Jac0invface_gf.GetVectorValue(Trans_el1.ElementNo,eip_el1,Jac0inv_vec_el1);
	    Jac0invface_gf.GetVectorValue(Trans_el2.ElementNo,eip_el2,Jac0inv_vec_el2);
	    
	    DenseMatrix Jac0inv_el1(dim), Jac0inv_el2(dim);
	    ConvertVectorToDenseMatrix(dim, Jac0inv_vec_el1, Jac0inv_el1);
	    ConvertVectorToDenseMatrix(dim, Jac0inv_vec_el2, Jac0inv_el2);
	    
	    DenseMatrix v_grad_q1(dim), v_grad_q2(dim);
	    v_gf.GetVectorGradient(Trans_el1, v_grad_q1);
	    v_gf.GetVectorGradient(Trans_el2, v_grad_q2);
	    
	    // As in the volumetric viscosity.
	    v_grad_q1.Symmetrize();
	    v_grad_q2.Symmetrize();
	    double h_1, h_2, mu_1, mu_2;
	    
	    LengthScaleAndCompression(v_grad_q1, Trans_el1, Jac0inv_el1,
				      hinit, h_1, mu_1);
	    LengthScaleAndCompression(v_grad_q2, Trans_el2, Jac0inv_el2,
				      hinit, h_2, mu_2);
	    double density_el1 = rhoface_gf.GetValue(Trans_el1,eip_el1);
	    double density_el2 = rhoface_gf.GetValue(Trans_el2,eip_el2);
	    // penaltyVal = penaltyParameter * (gamma_1 * h_1 * density_el1  + gamma_2 * h_2 * density_el2 );
	    
	    fe.CalcShape(eip_el1, shape_el1);
	    shift_shape(h1, h1, Trans_el1.ElementNo, eip_el1, D_el1, nTerms, shape_el1);
	    
	    //////////
	    
	    fe2.CalcShape(eip_el2, shape_el2);	    
	    shift_shape(h1, h1, Trans_el2.ElementNo, eip_el2, D_el2, nTerms, shape_el2);
	    
	    for (int i = 0; i < h1dofs_cnt; i++)
	      {
		for (int vd = 0; vd < dim; vd++) // Velocity components.
		  {
		    for (int j = 0; j < h1dofs_cnt; j++)
		      {
			for (int md = 0; md < dim; md++) // Velocity components.
			  {	      
			    elmat(i + vd * h1dofs_cnt, j + md * h1dofs_cnt) += gamma_1 * gamma_1 * shape_el1(i) * shape_el1(j) * nor_norm * tN_el1(vd) * tN_el1(md) * penaltyVal * ip_f.weight  * nTildaDotN * nTildaDotN * std::abs(volumeFraction_el1 - volumeFraction_el2);
			    elmat(i + vd * h1dofs_cnt, j + md * h1dofs_cnt + dim * h1dofs_cnt) += gamma_1 * gamma_2 * shape_el1(i) * shape_el2(j) * nor_norm * tN_el1(vd) * tN_el2(md) * penaltyVal * ip_f.weight  * nTildaDotN * nTildaDotN * std::abs(volumeFraction_el1 - volumeFraction_el2);
			    elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + md * h1dofs_cnt) += gamma_2 * gamma_1 * shape_el2(i) * shape_el1(j) * nor_norm * tN_el2(vd) * tN_el1(md) * penaltyVal * ip_f.weight  * nTildaDotN * nTildaDotN * std::abs(volumeFraction_el1 - volumeFraction_el2);
			    elmat(i + vd * h1dofs_cnt + dim * h1dofs_cnt, j + md * h1dofs_cnt + dim * h1dofs_cnt) += gamma_2 * gamma_2 * shape_el2(i) * shape_el2(j) * nor_norm * tN_el2(vd) * tN_el2(md) * penaltyVal * ip_f.weight * nTildaDotN * nTildaDotN * std::abs(volumeFraction_el1 - volumeFraction_el2);
	
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

    void ShiftedDiffusionNormalVelocityIntegrator::AssembleRHSElementVect(const FiniteElement &el,
								      const FiniteElement &el2,
								      FaceElementTransformations &Tr,
								      Vector &elvect)
    {
      if ( (Tr.Attribute == 77) || (Tr.Attribute == 11) ){
	const int dim = el.GetDim();
	DenseMatrix identity(dim);
	identity = 0.0;
	for (int s = 0; s < dim; s++){
	  identity(s,s) = 1.0;
	}
	const int nqp_face = IntRule->GetNPoints();
	const int h1dofs_cnt = el.GetDof();
	elvect.SetSize(2*h1dofs_cnt*dim);
	elvect = 0.0;
	
	Vector shape_el1(h1dofs_cnt), shape_test_el1(h1dofs_cnt), nor(dim);
	Vector shape_el2(h1dofs_cnt), shape_test_el2(h1dofs_cnt);
	ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
	ElementTransformation &Trans_el2 = Tr.GetElement2Transformation();

	int h1dofs_offset = el2.GetDof()*dim;

	for (int q = 0; q < nqp_face; q++)
	  {
	    nor = 0.0;
	    
	    shape_el1 = 0.0;
	    shape_test_el1 = 0.0;
	    shape_el2 = 0.0;
	    shape_test_el2 = 0.0;

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

	    Vector tn(dim);
	    tn = 0.0;
	    tn = nor;
	    tn /= nor_norm;
	    
	    double volumeFraction_el1 = alpha_gf.GetValue(Trans_el1, eip_el1);
	    double volumeFraction_el2 = alpha_gf.GetValue(Trans_el2, eip_el2);
	    double sum_volFrac = volumeFraction_el1 + volumeFraction_el2;
	    double gamma_1 =  volumeFraction_el1/sum_volFrac;
	    double gamma_2 =  volumeFraction_el2/sum_volFrac;
	    
	    /////
	    Vector D_el1(dim);
	    Vector tN_el1(dim);
	    D_el1 = 0.0;
	    tN_el1 = 0.0;
	    // if (Tr.Attribute == 77){
	    vD->Eval(D_el1, Trans_el1, eip_el1);
	    // }
	    vN->Eval(tN_el1, Trans_el1, eip_el1);
	    /////
	    
	    /////
	    Vector D_el2(dim);
	    Vector tN_el2(dim);
	    D_el2 = 0.0;
	    tN_el2 = 0.0;
	    // if (Tr.Attribute == 77){
	    vD->Eval(D_el2, Trans_el2, eip_el2);
	    // }
	    vN->Eval(tN_el2, Trans_el2, eip_el2);
	    /////
	      
	    double nTildaDotN = 0.0;
	    for (int s = 0; s < dim; s++){
	      nTildaDotN += nor(s) * tN_el1(s) / nor_norm;
	    }

	    /////////////////////
	    double penaltyVal = 0.0;
	    // penaltyVal = 4.0 * penaltyParameter * globalmax_rho ;
	    
	    Vector Jac0inv_vec_el1(dim*dim),Jac0inv_vec_el2(dim*dim);
	    Jac0inv_vec_el1 = 0.0;
	    Jac0inv_vec_el2 = 0.0;
	    
	    Jac0invface_gf.GetVectorValue(Trans_el1.ElementNo,eip_el1,Jac0inv_vec_el1);
	    Jac0invface_gf.GetVectorValue(Trans_el2.ElementNo,eip_el2,Jac0inv_vec_el2);
	    
	    DenseMatrix Jac0inv_el1(dim), Jac0inv_el2(dim);
	    ConvertVectorToDenseMatrix(dim, Jac0inv_vec_el1, Jac0inv_el1);
	    ConvertVectorToDenseMatrix(dim, Jac0inv_vec_el2, Jac0inv_el2);
	    
	    DenseMatrix v_grad_q1(dim), v_grad_q2(dim);
	    v_gf.GetVectorGradient(Trans_el1, v_grad_q1);
	    v_gf.GetVectorGradient(Trans_el2, v_grad_q2);
	    
	    // As in the volumetric viscosity.
	    v_grad_q1.Symmetrize();
	    v_grad_q2.Symmetrize();
	    double h_1, h_2, mu_1, mu_2;
	    
	    LengthScaleAndCompression(v_grad_q1, Trans_el1, Jac0inv_el1,
				      hinit, h_1, mu_1);
	    LengthScaleAndCompression(v_grad_q2, Trans_el2, Jac0inv_el2,
				      hinit, h_2, mu_2);
	    double density_el1 = rhoface_gf.GetValue(Trans_el1,eip_el1);
	    double density_el2 = rhoface_gf.GetValue(Trans_el2,eip_el2);
	    double cs_el1 = csface_gf.GetValue(Trans_el1,eip_el1);
	    double cs_el2 = csface_gf.GetValue(Trans_el2,eip_el2);
	    
	    penaltyVal = penaltyParameter * (gamma_1 * h_1 * cs_el1  + gamma_2 * h_2 * cs_el2 );
	    /////////////////////////////
	    Vector gradv_d_el1(dim);
	    gradv_d_el1 = 0.0;
	    get_shifted_value(v_gf, Trans_el1.ElementNo, eip_el1, D_el1, nTerms, gradv_d_el1);

	    Vector gradv_d_el2(dim);
	    gradv_d_el2 = 0.0;
	    get_shifted_value(v_gf, Trans_el2.ElementNo, eip_el2, D_el2, nTerms, gradv_d_el2);
	    
	    el.CalcShape(eip_el1, shape_el1);
	    shift_shape(h1, h1, Trans_el1.ElementNo, eip_el1, D_el1, nTerms, shape_el1);
	    
	    double vDotn_el1 = 0.0;
	    for (int s = 0; s < dim; s++)
	      {
		vDotn_el1 += gradv_d_el1(s) * tN_el1(s);
	      }
	    
	    el2.CalcShape(eip_el2, shape_el2);	    
	    shift_shape(h1, h1, Trans_el2.ElementNo, eip_el2, D_el2, nTerms, shape_el2);

	    double vDotn_el2 = 0.0;
	    for (int s = 0; s < dim; s++)
	      {
		vDotn_el2 += gradv_d_el2(s) * tN_el2(s);
	      }
	    
	
	    for (int i = 0; i < h1dofs_cnt; i++)
	      {
		for (int vd = 0; vd < dim; vd++) // Velocity components.
		  {
		    elvect(i + vd * h1dofs_cnt) -= gamma_1 * shape_el1(i) * (gamma_1 * vDotn_el1 + gamma_2 * vDotn_el2) * nor_norm * tN_el1(vd) * penaltyVal * ip_f.weight  * nTildaDotN * nTildaDotN * std::abs(volumeFraction_el1 - volumeFraction_el2);
		    elvect(i + vd * h1dofs_cnt + dim * h1dofs_cnt) -= gamma_2 * shape_el2(i) * (gamma_1 * vDotn_el1 + gamma_2 * vDotn_el2) * nor_norm * tN_el2(vd) * penaltyVal * ip_f.weight  * nTildaDotN * nTildaDotN * std::abs(volumeFraction_el1 - volumeFraction_el2);
		  }
	      }
	  }
      }
      else{
	const int dim = el.GetDim();
	const int h1dofs_cnt = el.GetDof();
	elvect.SetSize(2*h1dofs_cnt*dim);
	elvect = 0.0;    
      }
    }
    
    void ShiftedDiffusionEnergyNormalVelocityIntegrator::AssembleRHSElementVect(const FiniteElement &el,
									    const FiniteElement &el2,
									    FaceElementTransformations &Tr,
									    Vector &elvect)
    {
      if (Vnpt_gf != NULL){ 
	if ( (Tr.Attribute == 77) || (Tr.Attribute == 11) ){
	  const int dim = el.GetDim();
	  DenseMatrix identity(dim);
	  identity = 0.0;
	  for (int s = 0; s < dim; s++){
	    identity(s,s) = 1.0;
	  }
	  const int nqp_face = IntRule->GetNPoints();
	  const int l2dofs_cnt = el.GetDof();
	  elvect.SetSize(2*l2dofs_cnt);
	  elvect = 0.0;
	
	  Vector shape_el1(l2dofs_cnt), shape_test_el1(l2dofs_cnt), nor(dim);
	  Vector shape_el2(l2dofs_cnt), shape_test_el2(l2dofs_cnt);
	  ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
	  ElementTransformation &Trans_el2 = Tr.GetElement2Transformation();

	  for (int q = 0; q < nqp_face; q++)
	    {
	      nor = 0.0;
	    
	      shape_el1 = 0.0;
	      shape_test_el1 = 0.0;
	      shape_el2 = 0.0;
	      shape_test_el2 = 0.0;

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

	      Vector tn(dim);
	      tn = 0.0;
	      tn = nor;
	      tn /= nor_norm;
	    
	      double volumeFraction_el1 = alpha_gf.GetValue(Trans_el1, eip_el1);
	      double volumeFraction_el2 = alpha_gf.GetValue(Trans_el2, eip_el2);
	      double sum_volFrac = volumeFraction_el1 + volumeFraction_el2;
	      double gamma_1 =  volumeFraction_el1/sum_volFrac;
	      double gamma_2 =  volumeFraction_el2/sum_volFrac;
	    
	      /////
	      Vector D_el1(dim);
	      Vector tN_el1(dim);
	      D_el1 = 0.0;
	      tN_el1 = 0.0;
	      // if (Tr.Attribute == 77){
	      vD->Eval(D_el1, Trans_el1, eip_el1);
	      // }
	      vN->Eval(tN_el1, Trans_el1, eip_el1);
	      /////
	    
	      /////
	      Vector D_el2(dim);
	      Vector tN_el2(dim);
	      D_el2 = 0.0;
	      tN_el2 = 0.0;
	      // if (Tr.Attribute == 77){
	      vD->Eval(D_el2, Trans_el2, eip_el2);
	      // }
	      vN->Eval(tN_el2, Trans_el2, eip_el2);
	      /////
	      
	      double nTildaDotN = 0.0;
	      for (int s = 0; s < dim; s++){
		nTildaDotN += nor(s) * tN_el1(s) / nor_norm;
	      }

	      /////////////////////
	      double penaltyVal = 0.0;
	      // penaltyVal = 4.0 * penaltyParameter * globalmax_rho ;
	    
	      Vector Jac0inv_vec_el1(dim*dim),Jac0inv_vec_el2(dim*dim);
	      Jac0inv_vec_el1 = 0.0;
	      Jac0inv_vec_el2 = 0.0;
	    
	      Jac0invface_gf.GetVectorValue(Trans_el1.ElementNo,eip_el1,Jac0inv_vec_el1);
	      Jac0invface_gf.GetVectorValue(Trans_el2.ElementNo,eip_el2,Jac0inv_vec_el2);
	    
	      DenseMatrix Jac0inv_el1(dim), Jac0inv_el2(dim);
	      ConvertVectorToDenseMatrix(dim, Jac0inv_vec_el1, Jac0inv_el1);
	      ConvertVectorToDenseMatrix(dim, Jac0inv_vec_el2, Jac0inv_el2);
	    
	      DenseMatrix v_grad_q1(dim), v_grad_q2(dim);
	      v_gf.GetVectorGradient(Trans_el1, v_grad_q1);
	      v_gf.GetVectorGradient(Trans_el2, v_grad_q2);
	    
	      // As in the volumetric viscosity.
	      v_grad_q1.Symmetrize();
	      v_grad_q2.Symmetrize();
	      double h_1, h_2, mu_1, mu_2;
	    
	      LengthScaleAndCompression(v_grad_q1, Trans_el1, Jac0inv_el1,
					hinit, h_1, mu_1);
	      LengthScaleAndCompression(v_grad_q2, Trans_el2, Jac0inv_el2,
					hinit, h_2, mu_2);
	      double density_el1 = rhoface_gf.GetValue(Trans_el1,eip_el1);
	      double density_el2 = rhoface_gf.GetValue(Trans_el2,eip_el2);
	      double cs_el1 = csface_gf.GetValue(Trans_el1,eip_el1);
	      double cs_el2 = csface_gf.GetValue(Trans_el2,eip_el2);
	    
	      penaltyVal = penaltyParameter * (gamma_1 * h_1 * cs_el1  + gamma_2 * h_2 * cs_el2 );
	      /////////////////////////////
	      Vector gradv_d_el1(dim);
	      gradv_d_el1 = 0.0;
	      get_shifted_value(*Vnpt_gf, Trans_el1.ElementNo, eip_el1, D_el1, nTerms, gradv_d_el1);

	      Vector gradv_d_el2(dim);
	      gradv_d_el2 = 0.0;
	      get_shifted_value(*Vnpt_gf, Trans_el2.ElementNo, eip_el2, D_el2, nTerms, gradv_d_el2);
	    
	      double vDotn_el1 = 0.0;
	      for (int s = 0; s < dim; s++)
		{
		  vDotn_el1 += gradv_d_el1(s) * tN_el1(s);
		}
	    
	      double vDotn_el2 = 0.0;
	      for (int s = 0; s < dim; s++)
		{
		  vDotn_el2 += gradv_d_el2(s) * tN_el2(s);
		}
	    
	      el.CalcShape(eip_el1, shape_el1);
	      el2.CalcShape(eip_el2, shape_el2);
	    
	      for (int i = 0; i < l2dofs_cnt; i++)
		{
		  elvect(i) += shape_el1(i) * (gamma_1 * vDotn_el1 + gamma_2 * vDotn_el2) * (gamma_1 * vDotn_el1 + gamma_2 * vDotn_el2) * nor_norm * penaltyVal * ip_f.weight * nTildaDotN * nTildaDotN * std::abs(volumeFraction_el1 - volumeFraction_el2);
		  elvect(i + l2dofs_cnt) += shape_el2(i) * (gamma_1 * vDotn_el1 + gamma_2 * vDotn_el2) * (gamma_1 * vDotn_el1 + gamma_2 * vDotn_el2) * nor_norm * penaltyVal * ip_f.weight * nTildaDotN * nTildaDotN * std::abs(volumeFraction_el1 - volumeFraction_el2);
		}
	    }
	}
	else{
	  const int dim = el.GetDim();
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
  } // namespace hydrodynamics
  
} // namespace mfem
