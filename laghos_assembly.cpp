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
      fe.CalcShape(IntRule->IntPoint(q), shape);
      // Note that rhoDetJ = rho0DetJ0.
      shape *= qdata.rho0DetJ0w(Tr.ElementNo*nqp + q);
      elvect += shape;
   }
}

void ForceIntegrator::AssembleElementMatrix2(const FiniteElement &trial_fe,
                                             const FiniteElement &test_fe,
                                             ElementTransformation &Tr,
                                             DenseMatrix &elmat)
{
   const int e = Tr.ElementNo;
   const int nqp = IntRule->GetNPoints();
   const int dim = trial_fe.GetDim();
   const int h1dofs_cnt = test_fe.GetDof();
   const int l2dofs_cnt = trial_fe.GetDof();
   elmat.SetSize(h1dofs_cnt*dim, l2dofs_cnt);
   elmat = 0.0;
   DenseMatrix vshape(h1dofs_cnt, dim), loc_force(h1dofs_cnt, dim);
   Vector shape(l2dofs_cnt), Vloc_force(loc_force.Data(), h1dofs_cnt*dim);
   for (int q = 0; q < nqp; q++)
   {
      const IntegrationPoint &ip = IntRule->IntPoint(q);
      // Form stress:grad_shape at the current point.
      test_fe.CalcDShape(ip, vshape);
      for (int i = 0; i < h1dofs_cnt; i++)
      {
         for (int vd = 0; vd < dim; vd++) // Velocity components.
         {
            loc_force(i, vd) = 0.0;
            for (int gd = 0; gd < dim; gd++) // Gradient components.
            {
               const int eq = e*nqp + q;
               const double stressJinvT = qdata.stressJinvT(vd)(eq, gd);
	       loc_force(i, vd) +=  stressJinvT * vshape(i,gd);
           }
         }
      }
      trial_fe.CalcShape(ip, shape);
      AddMultVWt(Vloc_force, shape, elmat);
   }
}

void VelocityBoundaryForceIntegrator::AssembleFaceMatrix(const FiniteElement &trial_fe,
                                             const FiniteElement &test_fe,
					     FaceElementTransformations &Tr,
                                             DenseMatrix &elmat)
{
  int elem1_status = 1;
  if (elemStatus.Size() > 0){
    elem1_status = elemStatus[Tr.Elem1No];
  }
  const int nqp_face = IntRule->GetNPoints();
  const int dim = trial_fe.GetDim();
  const int h1dofs_cnt = test_fe.GetDof();
  const int l2dofs_cnt = trial_fe.GetDof();
  elmat.SetSize(h1dofs_cnt*dim, l2dofs_cnt);
  elmat = 0.0;
  
  if (elem1_status == 1){  
    DenseMatrix loc_force(h1dofs_cnt, dim);
    Vector te_shape(h1dofs_cnt),tr_shape(l2dofs_cnt), Vloc_force(loc_force.Data(), h1dofs_cnt*dim);
    const int Elem1No = Tr.ElementNo;
    te_shape = 0.0;
    tr_shape = 0.0;
    Vloc_force = 0.0;
    for (int q = 0; q  < nqp_face; q++)
      {
	const int eq = Elem1No*nqp_face + q;
	
	const IntegrationPoint &ip_f = IntRule->IntPoint(q);
	// Set the integration point in the face and the neighboring elements
	Tr.SetAllIntPoints(&ip_f);
	const IntegrationPoint &eip = Tr.GetElement1IntPoint();
	test_fe.CalcShape(eip, te_shape);
	loc_force = 0.0;
	/*	if ( pmesh->GetBdrAttribute(Elem1No) == 3) {
	  Vector sigmaTildaNDotN(dim);
	  sigmaTildaNDotN = 0.0;
	  Vector x_eip(3);
	  Tr.Transform(eip,x_eip);
	
	  Vector D(dim);
	  Vector tN(dim);
	  D = 0.0;
	  tN = 0.0;
	  analyticalSurface->ComputeDistanceAndNormalAtCoordinates(x_eip,D,tN);
	  
	  for (int vd = 0; vd < dim; vd++) // Velocity components.
	    {
	      double temp = 0.0;
	      for (int md = 0; md < dim; md++){ // Velocity components.
		
		temp += qdata.weightedNormalStress(eq,md) * tN(md);
	      }
	      sigmaTildaNDotN(vd) = temp * tN(vd);    
	    }
	  for (int i = 0; i < h1dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  loc_force(i, vd) += sigmaTildaNDotN(vd) * te_shape(i);
		}
	    }
	  trial_fe.CalcShape(eip, tr_shape);
	  AddMultVWt(Vloc_force,tr_shape,elmat);	  
	}*/
	//	else{
	  for (int i = 0; i < h1dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  loc_force(i, vd) += qdata.weightedNormalStress(eq,vd) * te_shape(i);
		}
	    }
	  trial_fe.CalcShape(eip, tr_shape);
	  AddMultVWt(Vloc_force,tr_shape,elmat);
	  // }
      }
  }
}
    

void EnergyBoundaryForceIntegrator::AssembleFaceMatrix(const FiniteElement &trial_fe,
                                             const FiniteElement &test_fe,
					     FaceElementTransformations &Tr,
                                             DenseMatrix &elmat)
{
  int elem1_status = 1;
  if (elemStatus.Size() > 0){
    elem1_status = elemStatus[Tr.Elem1No];
  }
  const int nqp_face = IntRule->GetNPoints();
  const int dim = trial_fe.GetDim();
  const int h1dofs_cnt = test_fe.GetDof();
  const int l2dofs_cnt = trial_fe.GetDof();
  elmat.SetSize(h1dofs_cnt*dim, l2dofs_cnt);
  elmat = 0.0;
  
  if (elem1_status == 1){ 
    DenseMatrix loc_force(h1dofs_cnt, dim), dshapephys(h1dofs_cnt, dim), dshape(h1dofs_cnt, dim);
    Vector te_shape(h1dofs_cnt),tr_shape(l2dofs_cnt), Vloc_force(loc_force.Data(), h1dofs_cnt*dim), dshapephysdd(h1dofs_cnt);
    const int Elem1No = Tr.ElementNo;
    te_shape = 0.0;
    tr_shape = 0.0;
    Vloc_force = 0.0;
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
	test_fe.CalcShape(eip, te_shape);

	if ( pmesh->GetBdrAttribute(Elem1No) == 3) {
	  dshapephys = 0.0;
	  dshape = 0.0;
	  dshapephysdd = 0.0;
	  
	  const DenseMatrix& quadDist = analyticalSurface->GetQuadratureDistance_BF();
	  const DenseMatrix& quadTrueNorm = analyticalSurface->GetQuadratureTrueNormal_BF();
	  
	  Vector x_eip(3);
	  Tr.Transform(eip,x_eip);
	
	  Vector D(dim);
	  Vector tN(dim);
	  D = 0.0;
	  tN = 0.0;
	  analyticalSurface->ComputeDistanceAndNormalAtCoordinates(x_eip,D,tN);
	  /*for (int s = 0; s < dim; s++){
	    D(s) = quadDist(eq,s);
	    tN(s) = quadTrueNorm(eq,s);
	    }*/

	  test_fe.CalcDShape(eip, dshape);
	  const DenseMatrix &Jinv = (Tr.GetElement1Transformation()).InverseJacobian();
	  Mult(dshape, Jinv, dshapephys);
	
	  //  std::cout << " eq " << eq << " xeip " << x_eip(0) << " yeip " << x_eip(1) << " DistY " << D(1) << std::endl;
	  dshapephys.Mult(D, dshapephysdd); // dphi/dx.D);
	  te_shape += dshapephysdd;

	  /////// SECOND ORDER ////////
	  Vector q_hess_dot_d(h1dofs_cnt);
	  q_hess_dot_d = 0.0;

	  int size = (dim*(dim+1))/2;
	  DenseMatrix physical_hess(h1dofs_cnt,size);
	  physical_hess = 0.0;
	  test_fe.CalcPhysHessian2(Tr.GetElement1Transformation(), physical_hess);
	  
	  DenseMatrix adjusted_physical_hess(h1dofs_cnt,dim*dim);
	  for (int s = 0; s < h1dofs_cnt; s++){ 
	    for (int l = 0; l < dim; l++){
	      for (int g = 0; g <= l; g++){
		adjusted_physical_hess(s,g + l * dim) = physical_hess(s, l + g + (dim - 2) * std::min({l,g,1}));
		adjusted_physical_hess(s,l + g * dim) = physical_hess(s, l + g + (dim - 2) * std::min({l,g,1}));
	      }
	    }
	  }
	  for (int s = 0; s < h1dofs_cnt; s++){
	    for (int l = 0; l < dim; l++){
	      for (int g = 0; g < dim; g++){
		q_hess_dot_d(s) += adjusted_physical_hess(s, g + l*dim) * D(g) * D(l);
	      }
	    }
	  }
	  q_hess_dot_d *= 1.0/2.0;
	  //te_shape += q_hess_dot_d;
	  ///////////////////////////////
	  
	  
	  loc_force = 0.0;
	  double normalStressProjNormal = 0.0;
	  double nor_norm = 0.0;
	  for (int s = 0; s < dim; s++){
	    normalStressProjNormal += qdata.weightedNormalStress(eq,s) * nor(s);
	    nor_norm += nor(s) * nor(s);
	    //  std::cout << " eq " << eq << " xeip " << x_eip(0) << " yeip " << x_eip(1) << " val " << qdata.weightedNormalStress(eq,s) << std::endl;
	  }
	  nor_norm = sqrt(nor_norm);

	  double ntildaDotTrueN = 0;
	  for (int s = 0; s < dim; s++){
	    ntildaDotTrueN += tN(s) * nor(s)/nor_norm;
	  }

	  normalStressProjNormal = normalStressProjNormal/nor_norm;
	  
	  for (int i = 0; i < h1dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  loc_force(i, vd) += normalStressProjNormal * te_shape(i) * tN(vd) * ntildaDotTrueN/*nor(vd)/nor_norm*/;
		}
	    }
	  trial_fe.CalcShape(eip, tr_shape);
	  AddMultVWt(Vloc_force,tr_shape,elmat);
	}
	else{
	  loc_force = 0.0;
	  double normalStressProjNormal = 0.0;
	  double nor_norm = 0.0;
	  for (int s = 0; s < dim; s++){
	    normalStressProjNormal += qdata.weightedNormalStress(eq,s) * nor(s);
	    nor_norm += nor(s) * nor(s);
	  }
	  nor_norm = sqrt(nor_norm);
	  normalStressProjNormal = normalStressProjNormal/nor_norm;
	  
	  for (int i = 0; i < h1dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  loc_force(i, vd) += normalStressProjNormal * te_shape(i) * nor(vd)/nor_norm;
		}
	    }
	  trial_fe.CalcShape(eip, tr_shape);
	  AddMultVWt(Vloc_force,tr_shape,elmat);
	}
      }
  }
}

void NormalVelocityMassIntegrator::AssembleFaceMatrix(const FiniteElement &el1,
						      const FiniteElement &el2,
                                             FaceElementTransformations &Tr,
                                             DenseMatrix &elmat)
{
  int elem1_status = 1;
  if (elemStatus.Size() > 0){
    elem1_status = elemStatus[Tr.Elem1No];
  }
  const int nqp_face = IntRule->GetNPoints();
  const int dim = el1.GetDim();
  const int h1dofs_cnt = el1.GetDof();
  elmat.SetSize(h1dofs_cnt*dim);
  elmat = 0.0;
  // std::cout << " in " << std::endl;
  if (elem1_status == 1){
    Vector shape(h1dofs_cnt),  dshapephysdd(h1dofs_cnt);
    DenseMatrix dshapephys(h1dofs_cnt, dim), dshape(h1dofs_cnt, dim);
 
    const int Elem1No = Tr.ElementNo;
    shape = 0.0;
    //  std::cout << " face Elem " << pmesh->GetBdrAttribute(Elem1No) << std::endl;
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
	el1.CalcShape(eip, shape);

	if ( pmesh->GetBdrAttribute(Elem1No) == 3) {
	  //	  std::cout << " x_before " << eip.x << " y_before " << eip.y << std::endl;

	  dshapephys = 0.0;
	  dshape = 0.0;
	  dshapephysdd = 0.0;
	  
	  const DenseMatrix& quadDist = analyticalSurface->GetQuadratureDistance_BF();
	  const DenseMatrix& quadTrueNorm = analyticalSurface->GetQuadratureTrueNormal_BF();
	  
	  /*	  for (int s = 0; s < dim; s++){
	    D(s) = quadDist(eq,s);
	    tN(s) = quadTrueNorm(eq,s);
	    }*/
	  Vector x_eip(3);
	  Tr.Transform(eip,x_eip);
	  //  ElementTransformation &Trans_el1 = Tr.GetElement1Transformation(); 
	  //Trans_el1.Transform(eip,x_eip);
	  Vector D(dim);
	  Vector tN(dim);
	  D = 0.0;
	  tN = 0.0;
	  analyticalSurface->ComputeDistanceAndNormalAtCoordinates(x_eip,D,tN);
	  //  std::cout << " x " << x_eip(0) << " Y " << x_eip(1) << " distY " << D(1) << std::endl; 	
	  /* el1.CalcDShape(eip, dshape);
	  const DenseMatrix &Jinv = (Tr.GetElement1Transformation()).InverseJacobian();
	  Mult(dshape, Jinv, dshapephys);*/
	  //  const IntegrationPoint &eip_check = (Tr.GetElement1Transformation()).GetIntPoint();
	  //  std::cout << " x " << eip_check.x << " y " << eip_check.y << std::endl;
	    
	  el1.CalcPhysDShape(Tr.GetElement1Transformation(),dshapephys);
	  dshapephys.Mult(D, dshapephysdd); // dphi/dx.D);
	  shape += dshapephysdd;

	  /////// SECOND ORDER ////////
	  Vector q_hess_dot_d(h1dofs_cnt);
	  q_hess_dot_d = 0.0;
	  int size = (dim*(dim+1))/2;
	  DenseMatrix physical_hess(h1dofs_cnt,size);
	  physical_hess = 0.0;
	  el1.CalcPhysHessian2(Tr.GetElement1Transformation(), physical_hess);
	  
	  DenseMatrix adjusted_physical_hess(h1dofs_cnt,dim*dim);
	  for (int s = 0; s < h1dofs_cnt; s++){ 
	    for (int l = 0; l < dim; l++){
	      for (int g = 0; g <= l; g++){
		adjusted_physical_hess(s,g + l * dim) = physical_hess(s, l + g + (dim - 2) * std::min({l,g,1}));
		adjusted_physical_hess(s,l + g * dim) = physical_hess(s, l + g + (dim - 2) * std::min({l,g,1}));
	      }
	    }
	  }
	  for (int s = 0; s < h1dofs_cnt; s++){
	    for (int l = 0; l < dim; l++){
	      for (int g = 0; g < dim; g++){
		q_hess_dot_d(s) += adjusted_physical_hess(s, g + l*dim) * D(g) * D(l);
	      }
	    }
	  }
	  q_hess_dot_d *= 1.0/2.0;
	  //  shape += q_hess_dot_d;
	  ///////////////////////////////
	  
	  double nor_norm = 0.0;
	  for (int s = 0; s < dim; s++){
	    nor_norm += nor(s) * nor(s);
	  }
	  nor_norm = sqrt(nor_norm);
	  double ntildaDotTrueN = 0;
	  for (int s = 0; s < dim; s++){
	    ntildaDotTrueN += tN(s) * nor(s)/nor_norm;
	  }
	  
	  for (int i = 0; i < h1dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  for (int j = 0; j < h1dofs_cnt; j++)
		    {
		      for (int md = 0; md < dim; md++) // Velocity components.
			{	      
			  elmat(i + vd * h1dofs_cnt, j + md * h1dofs_cnt) += shape(i) * shape(j) * tN(vd) * nor_norm * tN(md) * qdata.normalVelocityPenaltyScaling(eq) * ntildaDotTrueN * ntildaDotTrueN;
			}
		    }
		}
	    }
	}
	else{
	  //  std::cout << " face " << pmesh->GetBdrAttribute(Elem1No) << std::endl;
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
			  elmat(i + vd * h1dofs_cnt, j + md * h1dofs_cnt) += shape(i) * shape(j) * nor(vd) * (nor(md)/nor_norm) * qdata.normalVelocityPenaltyScaling(eq);
			}
		    }
		}
	    }
	}
      }
  }
}

  void ShiftedVelocityBoundaryForceIntegrator::AssembleFaceMatrix(const FiniteElement &trial_fe1,
								  const FiniteElement &trial_fe2,
								  const FiniteElement &test_fe1,
								  const FiniteElement &test_fe2,
								  FaceElementTransformations &Trans,
								  DenseMatrix &elmat,
								  Array<int> &trial_vdofs,
							          Array<int> &test_vdofs)
  {
    MPI_Comm comm = pmesh->GetComm();
    int myid;
    MPI_Comm_rank(comm, &myid);
    //  std::cout << " I AM IN ASSEMBLE FACE " << std::endl;
  int elem1 = Trans.Elem1No;
  int elem2 = Trans.Elem2No;
  int elemStatus1 = elemStatus[elem1];
  int elemStatus2 = elemStatus[elem2];
  int NEproc = pmesh->GetNE();
    
  const int nqp_face = IntRule->GetNPoints();
  const int dim = test_fe1.GetDim();
  const int h1dofs_cnt = test_fe1.GetDof();
  const int l2dofs_cnt = trial_fe1.GetDof();
  const int h1dofs2_cnt = test_fe2.GetDof();
  const int l2dofs2_cnt = trial_fe2.GetDof();
  
  trial_vdofs.DeleteAll();
  test_vdofs.DeleteAll();
  
  DenseMatrix loc_force(h1dofs_cnt, dim);
  Vector te_shape(h1dofs_cnt),tr_shape(l2dofs_cnt), Vloc_force(loc_force.Data(), h1dofs_cnt*dim);
  const int e = Trans.ElementNo;
  te_shape = 0.0;
  tr_shape = 0.0;
  Vloc_force = 0.0;
  
  if (faceTags[e] == 5){
    if (elemStatus1 == AnalyticalGeometricShape::SBElementType::INSIDE){

      test_H1.GetElementVDofs(Trans.Elem1No, test_vdofs);	
      trial_L2.GetElementVDofs(Trans.Elem1No, trial_vdofs);	

      elmat.SetSize(h1dofs_cnt*dim, l2dofs_cnt);
      elmat = 0.0;

      DenseMatrix temp_elmat;
      temp_elmat.SetSize(h1dofs_cnt*dim, l2dofs_cnt);
      temp_elmat = 0.0;

      for (int q = 0; q  < nqp_face; q++)
	{
	  const int eq = e*nqp_face + q;
	  const IntegrationPoint &ip_f = IntRule->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Trans.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip = Trans.GetElement1IntPoint();
	  test_fe1.CalcShape(eip, te_shape);
	  loc_force = 0.0;
	  for (int i = 0; i < h1dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  loc_force(i, vd) += qdata.weightedNormalStress(eq,vd) * te_shape(i);
		}
	    }
	  trial_fe1.CalcShape(eip, tr_shape);
	  AddMultVWt(Vloc_force,tr_shape, elmat);
	}
      //  elmat.CopyMN(temp_elmat, 0, 0);
    }
    else if ((elem2 < NEproc) && (elemStatus2 == AnalyticalGeometricShape::SBElementType::INSIDE)){

      elmat.SetSize(h1dofs2_cnt*dim, l2dofs2_cnt);
      elmat = 0.0;

      test_H1.GetElementVDofs (Trans.Elem2No, test_vdofs);	
      trial_L2.GetElementVDofs (Trans.Elem2No, trial_vdofs);	
      
      DenseMatrix temp_elmat;
      temp_elmat.SetSize(h1dofs_cnt*dim, l2dofs_cnt);
      temp_elmat = 0.0;

      int h1dofs_offset = h1dofs_cnt * dim;
      int l2dofs_offset = l2dofs_cnt;
      for (int q = 0; q < nqp_face; q++)
	{
	  const int eq = e*nqp_face + q;
	  
	  const IntegrationPoint &ip_f = IntRule->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Trans.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip = Trans.GetElement2IntPoint();
	  test_fe2.CalcShape(eip, te_shape);
	  loc_force = 0.0;
	  for (int i = 0; i < h1dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  loc_force(i, vd) += qdata.weightedNormalStress(eq,vd) * te_shape(i);
		}
	    }
	  trial_fe2.CalcShape(eip, tr_shape);
	  AddMultVWt(Vloc_force,tr_shape, elmat);
	}
      //  elmat.CopyMN(temp_elmat, h1dofs_offset, l2dofs_offset);
    }
  }
  else{
    test_H1.GetElementVDofs (Trans.Elem1No, test_vdofs);	
    trial_L2.GetElementVDofs (Trans.Elem1No, trial_vdofs);	
    elmat.SetSize(h1dofs_cnt*dim, l2dofs_cnt);
    elmat = 0.0; 
    }
}

void ShiftedEnergyBoundaryForceIntegrator::AssembleFaceMatrix(const FiniteElement &trial_fe1,
							      const FiniteElement &trial_fe2,
							      const FiniteElement &test_fe1,
							      const FiniteElement &test_fe2,
							      FaceElementTransformations &Trans,
							      DenseMatrix &elmat,
							      Array<int> &trial_vdofs,
							      Array<int> &test_vdofs)
{
  const DenseMatrix& quadDist = analyticalSurface->GetQuadratureDistance();
  const DenseMatrix& quadTrueNorm = analyticalSurface->GetQuadratureTrueNormal();
  
  int elem1 = Trans.Elem1No;
  int elem2 = Trans.Elem2No;
  int elemStatus1 = elemStatus[elem1];
  int elemStatus2 = elemStatus[elem2];
  int NEproc = pmesh->GetNE();
  trial_vdofs.DeleteAll();
  test_vdofs.DeleteAll();
  
  const int nqp_face = IntRule->GetNPoints();
  const int dim = trial_fe1.GetDim();
  const int h1dofs_cnt = test_fe1.GetDof();
  const int l2dofs_cnt = trial_fe1.GetDof();
  const int h1dofs2_cnt = test_fe2.GetDof();
  const int l2dofs2_cnt = trial_fe2.GetDof();
 
  DenseMatrix loc_force(h1dofs_cnt, dim), dshapephys(h1dofs_cnt, dim), dshape(h1dofs_cnt, dim), /*Jinv(dim),*/ hessshape(h1dofs_cnt, dim * dim);
  Vector te_shape(h1dofs_cnt), tr_shape(l2dofs_cnt), Vloc_force(loc_force.Data(), h1dofs_cnt*dim),  dshapephysdd(h1dofs_cnt), q_hess_dot_d(h1dofs_cnt);
	
  const int e = Trans.ElementNo;
  if (faceTags[e] == 5){
    if (elemStatus1 == AnalyticalGeometricShape::SBElementType::INSIDE){

      elmat.SetSize(h1dofs_cnt*dim, l2dofs_cnt);
      elmat = 0.0;
      test_H1.GetElementVDofs (Trans.Elem1No, test_vdofs);	
      trial_L2.GetElementVDofs (Trans.Elem1No, trial_vdofs);

      Vloc_force = 0.0;

      for (int q = 0; q  < nqp_face; q++)
	{
	  te_shape = 0.0;	  
	  tr_shape = 0.0;

	  dshapephys = 0.0;
	  dshape = 0.0;
	  dshapephysdd = 0.0;
	  //  Jinv = 0.0;
	  hessshape = 0.0;

	  q_hess_dot_d = 0.0;

	  Vector D(dim);
	  Vector tN(dim);
	  D = 0.0;
	  tN = 0.0;
	  const int eq = e*nqp_face + q;
	  const IntegrationPoint &ip_f = IntRule->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Trans.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip = Trans.GetElement1IntPoint();
	  
	  Vector nor;
	  nor.SetSize(dim);
	  nor = 0.0;
	  if (dim == 1)
	    {
	      nor(0) = 2*eip.x - 1.0;
	    }
	  else
	    {
	      CalcOrtho(Trans.Jacobian(), nor);
	    }

	  test_fe1.CalcShape(eip, te_shape);
	  test_fe1.CalcDShape(eip, dshape);
	  //	  CalcInverse((Trans.Elem1)->Jacobian(), Jinv);
	  const DenseMatrix &Jinv = (Trans.GetElement1Transformation()).InverseJacobian();
	  Mult(dshape, Jinv, dshapephys);

	  int size = (dim*(dim+1))/2;
	  DenseMatrix physical_hess(h1dofs_cnt,size);
	  physical_hess = 0.0;
	  test_fe1.CalcPhysHessian2(Trans.GetElement1Transformation(),physical_hess);
	  DenseMatrix adjusted_physical_hess(h1dofs_cnt,dim*dim);
	  for (int s = 0; s < h1dofs_cnt; s++){ 
	    for (int l = 0; l < dim; l++){
	      for (int g = 0; g <= l; g++){
		adjusted_physical_hess(s,g + l * dim) = physical_hess(s, l + g + (dim - 2) * std::min({l,g,1}));
		adjusted_physical_hess(s,l + g * dim) = physical_hess(s, l + g + (dim - 2) * std::min({l,g,1}));
	      }
	    }
	  }
	  
	  /*	  std::cout << " phy " << std::endl; 
	  physical_hess.Print(std::cout,1);
	  std::cout << " adjusted phy " << std::endl; 
	  adjusted_physical_hess.Print(std::cout,1);
*/
	  
	  loc_force = 0.0;
	  double normalStressProjNormal = 0.0;
	  double nor_norm = 0.0;
	  for (int s = 0; s < dim; s++){
	    normalStressProjNormal += qdata.weightedNormalStress(eq,s) * nor(s);
	    nor_norm += nor(s) * nor(s);
	    D(s) = quadDist(eq,s);
	    tN(s) = quadTrueNorm(eq,s);
	    //    tN(s) = nor(s);
	  }
	  nor_norm = sqrt(nor_norm);
	  //  tN /= nor_norm;
	  normalStressProjNormal = normalStressProjNormal/nor_norm;
	  double ntildaDotTrueN = 0;
	  for (int s = 0; s < dim; s++){
	    ntildaDotTrueN += tN(s) * nor(s)/nor_norm;
	  }
	  dshapephys.Mult(D, dshapephysdd); // dphi/dx.D);
	  
	  for (int s = 0; s < h1dofs_cnt; s++){
	    for (int l = 0; l < dim; l++){
	      for (int g = 0; g < dim; g++){
		q_hess_dot_d(s) += adjusted_physical_hess(s, g + l*dim) * D(g) * D(l);
	      }
	    }
	  }
	  q_hess_dot_d *= 1.0/2.0;

	  te_shape += dshapephysdd;
	  //  te_shape += q_hess_dot_d;
	    
	  for (int i = 0; i < h1dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  loc_force(i, vd) += normalStressProjNormal * te_shape(i) * tN(vd) * ntildaDotTrueN/*nor(vd)/nor_norm*/;
		}
	    }
	  trial_fe1.CalcShape(eip, tr_shape);
	  AddMultVWt(Vloc_force,tr_shape, elmat);
	}
      // elmat.CopyMN(temp_elmat, 0, 0);
    }
    else if ((elem2 < NEproc) && (elemStatus2 == AnalyticalGeometricShape::SBElementType::INSIDE)){

      elmat.SetSize(h1dofs2_cnt*dim, l2dofs2_cnt);
      elmat = 0.0; 
      test_H1.GetElementVDofs (Trans.Elem2No, test_vdofs);	
      trial_L2.GetElementVDofs (Trans.Elem2No, trial_vdofs);	

      Vloc_force = 0.0;

      for (int q = 0; q  < nqp_face; q++)
	{
	  te_shape = 0.0;	  
	  tr_shape = 0.0; 
	  dshapephys = 0.0;
	  dshape = 0.0;
	  dshapephysdd = 0.0;
	  //  Jinv = 0.0;
	  hessshape = 0.0;

	  q_hess_dot_d = 0.0;
	  
	  Vector D(dim);
	  Vector tN(dim);
	  D = 0.0;
	  tN = 0.0;
	  const int eq = e*nqp_face + q;
	  
	  const IntegrationPoint &ip_f = IntRule->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Trans.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip = Trans.GetElement2IntPoint();
	  Vector nor;
	  nor.SetSize(dim);
	  nor = 0.0;
	  if (dim == 1)
	    {
	      nor(0) = 2*eip.x - 1.0;
	    }
	  else
	    {
	      CalcOrtho(Trans.Jacobian(), nor);
	    }
	  nor *= -1.0;

	  test_fe2.CalcShape(eip, te_shape);
	  test_fe2.CalcDShape(eip, dshape);
	  //	  CalcInverse((Trans.Elem2)->Jacobian(), Jinv);
	  const DenseMatrix &Jinv = (Trans.GetElement2Transformation()).InverseJacobian();
	  Mult(dshape, Jinv, dshapephys);

	  int size = (dim*(dim+1))/2; 
	  DenseMatrix physical_hess(h1dofs_cnt,size);
	  physical_hess = 0.0;
	  test_fe2.CalcPhysHessian2(Trans.GetElement2Transformation(),physical_hess);

	  DenseMatrix adjusted_physical_hess(h1dofs_cnt,dim*dim);
	  for (int s = 0; s < h1dofs_cnt; s++){ 
	    for (int l = 0; l < dim; l++){
	      for (int g = 0; g <= l; g++){
		adjusted_physical_hess(s,g + l * dim) = physical_hess(s, l + g + (dim - 2) * std::min({l,g,1}));
		adjusted_physical_hess(s,l + g * dim) = physical_hess(s, l + g + (dim - 2) * std::min({l,g,1}));
	      }
	    }
	  }
	
	  loc_force = 0.0;
	  double normalStressProjNormal = 0.0;
	  double nor_norm = 0.0;
	  for (int s = 0; s < dim; s++){
	    normalStressProjNormal += qdata.weightedNormalStress(eq,s) * nor(s);
	    nor_norm += nor(s) * nor(s);
	    D(s) = quadDist(eq,s);
	    tN(s) = quadTrueNorm(eq,s);
	    //   tN(s) = nor(s);
	  }	  
	  nor_norm = sqrt(nor_norm);
	  //  tN /= nor_norm;
	  normalStressProjNormal = normalStressProjNormal/nor_norm;
	  double ntildaDotTrueN = 0;
	  for (int s = 0; s < dim; s++){
	    ntildaDotTrueN += tN(s) * nor(s)/nor_norm;
	  }
	  dshapephys.Mult(D, dshapephysdd); // dphi/dx.D);

	  for (int s = 0; s < h1dofs_cnt; s++){
	    for (int l = 0; l < dim; l++){
	      for (int g = 0; g < dim; g++){
		q_hess_dot_d(s) += adjusted_physical_hess(s, g + l*dim) * D(g) * D(l);
	      }
	    }
	  }
	  q_hess_dot_d *= 1.0/2.0;

	  te_shape += dshapephysdd;
	  //  te_shape += q_hess_dot_d;
	    
	  for (int i = 0; i < h1dofs2_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  loc_force(i, vd) += normalStressProjNormal * te_shape(i) * tN(vd) * ntildaDotTrueN /*nor(vd)/nor_norm*/;
		}
	    }
	  trial_fe2.CalcShape(eip, tr_shape);
	  AddMultVWt(Vloc_force,tr_shape, elmat);
	}
      //  elmat.CopyMN(temp_elmat, h1dofs_offset, l2dofs_offset);
    }
  }
  else{
    test_H1.GetElementVDofs (Trans.Elem1No, test_vdofs);	
    trial_L2.GetElementVDofs (Trans.Elem1No, trial_vdofs);	
    elmat.SetSize(h1dofs_cnt*dim, l2dofs_cnt);
    elmat = 0.0;
    }
}


  void ShiftedNormalVelocityMassIntegrator::AssembleFaceMatrix(const FiniteElement &el1,
							       const FiniteElement &el2,
							       FaceElementTransformations &Trans,
							       DenseMatrix &elmat, Array<int> &vdofs)
{
  MPI_Comm comm = pmesh->GetComm();
  int myid;
  MPI_Comm_rank(comm, &myid);

  vdofs.DeleteAll();
  const DenseMatrix& quadDist = analyticalSurface->GetQuadratureDistance();
  const DenseMatrix& quadTrueNorm = analyticalSurface->GetQuadratureTrueNormal();
  
  int elem1 = Trans.Elem1No;
  int elem2 = Trans.Elem2No;
  int elemStatus1 = elemStatus[elem1];
  int elemStatus2 = elemStatus[elem2];
  int NEproc = pmesh->GetNE();
  
  const int nqp_face = IntRule->GetNPoints();
  const int dim = el1.GetDim();
  const int h1dofs_cnt = el1.GetDof();
  DenseMatrix  dshapephys(h1dofs_cnt, dim), dshape(h1dofs_cnt, dim), /*Jinv(dim),*/ hessshape(h1dofs_cnt, dim * dim);

  Vector shape(h1dofs_cnt), dshapephysdd(h1dofs_cnt), q_hess_dot_d(h1dofs_cnt);
  const int e = Trans.ElementNo;
 
  //  std::cout << " qnp face shift " << nqp_face << std::endl;
  if (faceTags[e] == 5){
    if (elemStatus1 == AnalyticalGeometricShape::SBElementType::INSIDE){
      elmat.SetSize(h1dofs_cnt*dim,h1dofs_cnt*dim);
      elmat = 0.0;

      H1.GetElementVDofs (Trans.Elem1No, vdofs);

      for (int q = 0; q  < nqp_face; q++)
	{
	  shape = 0.0;	  
	  
	  dshapephys = 0.0;
	  dshape = 0.0;
	  dshapephysdd = 0.0;
	  //Jinv = 0.0;
	  hessshape = 0.0;

	  q_hess_dot_d = 0.0;
	  	  
	  Vector D(dim);
	  Vector tN(dim);
	  D = 0.0;
	  tN = 0.0;
	  const int eq = e*nqp_face + q;
	  const IntegrationPoint &ip_f = IntRule->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Trans.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip = Trans.GetElement1IntPoint();
	  Vector nor;
	  nor.SetSize(dim);
	  nor = 0.0;
	  if (dim == 1)
	    {
	      nor(0) = 2*eip.x - 1.0;
	    }
	  else
	    {
	      CalcOrtho(Trans.Jacobian(), nor);
	    }

	  Trans.SetAllIntPoints(&eip);

	  el1.CalcShape(eip, shape);
	  el1.CalcDShape(eip, dshape);
	  // CalcInverse((Trans.Elem1)->Jacobian(), Jinv);
	  const DenseMatrix &Jinv = (Trans.GetElement1Transformation()).InverseJacobian();
	  Mult(dshape, Jinv, dshapephys);

	  int size = (dim*(dim+1))/2;
	  DenseMatrix physical_hess(h1dofs_cnt,size);
	  physical_hess = 0.0;
	  el1.CalcPhysHessian2(Trans.GetElement1Transformation(),physical_hess);

	  DenseMatrix adjusted_physical_hess(h1dofs_cnt,dim*dim);
	  adjusted_physical_hess = 0.0;
	  for (int s = 0; s < h1dofs_cnt; s++){ 
	    for (int l = 0; l < dim; l++){
	      for (int g = 0; g <= l; g++){
		adjusted_physical_hess(s,g + l * dim) = physical_hess(s, l + g + (dim - 2) * std::min({l,g,1}));
		adjusted_physical_hess(s,l + g * dim) = physical_hess(s, l + g + (dim - 2) * std::min({l,g,1}));
	      }
	    }
	  }
	  
	  /* Vector x_eip(3);
	  Trans.Transform(eip,x_eip);
	  Vector xcheck_eip(3);
	  Trans.Transform(Trans.GetElement1Transformation().GetIntPoint(),xcheck_eip);
	  std::cout << " eipX " << x_eip(0) << " eipY " << x_eip(1) << std::endl;
	  std::cout << " CheckX " << xcheck_eip(0) << " checkY " << xcheck_eip(1) << std::endl;*/
	  
	  //el1.CalcHessian(eip,hessshape);
	  
	  Vector x_eip(3);
	  Trans.Transform(eip,x_eip);
	  
	  double nor_norm = 0.0;
	  for (int s = 0; s < dim; s++){
	    nor_norm += nor(s) * nor(s);
	    D(s) = quadDist(e*nqp_face+q,s);
	    tN(s) = quadTrueNorm(e*nqp_face+q,s);
	    //  tN(s) = nor(s);
	  }
	  nor_norm = sqrt(nor_norm);
	  //  std::cout << " Xeip " << x_eip(0) << " Yeip " << x_eip(1)  << " dY " << D(1) << std::endl;
	  
	  //  tN /= nor_norm;
	  //  std::cout << " DistX " << D(0) << " DistY " << D(1) << std::endl;
	  
	  double ntildaDotTrueN = 0;
	  for (int s = 0; s < dim; s++){
	    ntildaDotTrueN += tN(s) * nor(s)/nor_norm;
	  }
	  dshapephys.Mult(D, dshapephysdd); // dphi/dx.D);
	  //  std::cout << " nTildaDotNElem1 " << ntildaDotTrueN << std::endl;

	  for (int s = 0; s < h1dofs_cnt; s++){
	    for (int l = 0; l < dim; l++){
	      for (int g = 0; g < dim; g++){
		q_hess_dot_d(s) += adjusted_physical_hess(s, g + l*dim) * D(g) * D(l);
	      }
	    }
	  }
	  q_hess_dot_d *= 1.0/2.0;
	  
	  Vector trial_wrk = shape;
	  Vector test_wrk = shape;
	  trial_wrk += dshapephysdd;
	  test_wrk += dshapephysdd;
	  //  trial_wrk += q_hess_dot_d;
	  //  test_wrk += q_hess_dot_d;
	  
	  for (int i = 0; i < h1dofs_cnt; i++)
	    {	      
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  for (int j = 0; j < h1dofs_cnt; j++)
		    {
		      for (int md = 0; md < dim; md++) // Velocity components.
			{	      
			  elmat(i + vd * h1dofs_cnt, j + md * h1dofs_cnt) += test_wrk(i) * trial_wrk(j) * tN(vd) * nor_norm * tN(md) /*nor(vd) * (nor(md)/nor_norm)*/ * qdata.normalVelocityPenaltyScaling(eq) * ntildaDotTrueN * ntildaDotTrueN;
			}
		    }
		}
	    }
	}
    }
    else if ((elem2 < NEproc) && (elemStatus2 == AnalyticalGeometricShape::SBElementType::INSIDE)){
      elmat.SetSize(h1dofs_cnt*dim,h1dofs_cnt*dim);
      elmat = 0.0;

      //  int h1dofs_offset = h1dofs_cnt * dim;
      H1.GetElementVDofs (Trans.Elem2No, vdofs);	
    
      for (int q = 0; q  < nqp_face; q++)
	{
	  shape = 0.0;	  
	  
	  dshapephys = 0.0;
	  dshape = 0.0;
	  dshapephysdd = 0.0;
	  // Jinv = 0.0;
	  hessshape = 0.0;

	  q_hess_dot_d = 0.0;

	  Vector D(dim);
	  Vector tN(dim);
	  D = 0.0;
	  tN = 0.0;
	  const int eq = e*nqp_face + q;
		
	  const IntegrationPoint &ip_f = IntRule->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Trans.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip = Trans.GetElement2IntPoint();
	  Vector nor;
	  nor.SetSize(dim);
	  nor = 0.0;
	  if (dim == 1)
	    {
	      nor(0) = 2*eip.x - 1.0;
	    }
	  else
	    {
	      CalcOrtho(Trans.Jacobian(), nor);
	    }
	  nor *= -1.0;

	  el2.CalcShape(eip, shape);
	  el2.CalcDShape(eip, dshape);
	  // CalcInverse((Trans.Elem2)->Jacobian(), Jinv);
	  const DenseMatrix &Jinv = (Trans.GetElement2Transformation()).InverseJacobian();
	  Mult(dshape, Jinv, dshapephys);

	  int size = (dim*(dim+1))/2;
	  DenseMatrix physical_hess(h1dofs_cnt,size);
	  physical_hess = 0.0;
	  el2.CalcPhysHessian2(Trans.GetElement2Transformation(), physical_hess);
	  
	  DenseMatrix adjusted_physical_hess(h1dofs_cnt,dim*dim);
	  for (int s = 0; s < h1dofs_cnt; s++){ 
	    for (int l = 0; l < dim; l++){
	      for (int g = 0; g <= l; g++){
		adjusted_physical_hess(s,g + l * dim) = physical_hess(s, l + g + (dim - 2) * std::min({l,g,1}));
		adjusted_physical_hess(s,l + g * dim) = physical_hess(s, l + g + (dim - 2) * std::min({l,g,1}));
	      }
	    }
	  }
	
	  double nor_norm = 0.0;
	  for (int s = 0; s < dim; s++){
	    nor_norm += nor(s) * nor(s);
	    D(s) = quadDist(e*nqp_face+q,s);
	    tN(s) = quadTrueNorm(e*nqp_face+q,s);
	    //  tN(s) = nor(s);
	  }
	  //  std::cout << " dX " << D(0) << " dY " << D(1) << std::endl;
	  nor_norm = sqrt(nor_norm);
	  //  tN /= nor_norm;
	  //  std::cout << " NormalX " << tN(0) << " NormalY " << tN(1) << std::endl;

	  double ntildaDotTrueN = 0.0;
	  for (int s = 0; s < dim; s++){
	    ntildaDotTrueN += tN(s) * nor(s)/nor_norm;
	  }
	  dshapephys.Mult(D, dshapephysdd); // dphi/dx.D);
	  //  std::cout << " nTildaDotN " << ntildaDotTrueN << std::endl;

	  
	  for (int s = 0; s < h1dofs_cnt; s++){
	    for (int l = 0; l < dim; l++){
	      for (int g = 0; g < dim; g++){
		q_hess_dot_d(s) += adjusted_physical_hess(s, g + l*dim) * D(g) * D(l);
	      }
	    }
	  }
	  q_hess_dot_d *= 1.0/2.0;

	  Vector trial_wrk = shape;
	  Vector test_wrk = shape;
	  trial_wrk += dshapephysdd;
	  test_wrk += dshapephysdd;
	  //  trial_wrk += q_hess_dot_d;
	  // test_wrk += q_hess_dot_d;
	  
	  for (int i = 0; i < h1dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  for (int j = 0; j < h1dofs_cnt; j++)
		    {
		      for (int md = 0; md < dim; md++) // Velocity components.
			{	      
			  elmat(i + vd * h1dofs_cnt, j + md * h1dofs_cnt) += test_wrk(i) * trial_wrk(j) * tN(vd) * nor_norm * tN(md)/*nor(vd) * (nor(md)/nor_norm)*/ * qdata.normalVelocityPenaltyScaling(eq) * ntildaDotTrueN * ntildaDotTrueN;
			}
		    }
		}
	    }
	}
    }
  }
  else{
    H1.GetElementVDofs (Trans.Elem1No, vdofs);	
    elmat.SetSize(h1dofs_cnt*dim);
    elmat = 0.0;
    
    }

  }

} // namespace hydrodynamics

} // namespace mfem
