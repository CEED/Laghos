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


#include "Line.hpp"

namespace mfem{

  Line::Line(ParFiniteElementSpace &h1_fes, ParFiniteElementSpace &l2_fes): AnalyticalGeometricShape(h1_fes, l2_fes), slope(0), yIntercept(0.51) {
  }

  Line::~Line(){}
  
  void Line::SetupElementStatus(Array<int> &elemStatus, Array<int> &ess_inactive){
    const int max_elem_attr = (pmesh->attributes).Max();
  // Check elements on the current MPI rank
  for (int i = 0; i < H1.GetNE(); i++)
    {
      const FiniteElement *FElem = H1.GetFE(i);
      const IntegrationRule &ir = FElem->GetNodes();
      ElementTransformation &T = *H1.GetElementTransformation(i);
      int count = 0;
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
	const IntegrationPoint &ip = ir.IntPoint(j);
	Vector x(3);
	T.Transform(ip,x);
	double ptOnLine = slope * x(0) + yIntercept;
	if ( x(1) <= ptOnLine){
	  count++;
	}
      }
      if (count == ir.GetNPoints()){
	elemStatus[i] = SBElementType::INSIDE;
	Array<int> dofs;
	H1.GetElementVDofs(i, dofs);
	for (int k = 0; k < dofs.Size(); k++)
	  {
	    ess_inactive[dofs[k]] = 0;	       
	  }
      }
      else if ( (count > 0) && (count < ir.GetNPoints())){
	elemStatus[i] = SBElementType::CUT;
	pmesh->SetAttribute(i, max_elem_attr+1);
      }
      else if (count == 0){
	elemStatus[i] = SBElementType::OUTSIDE;
	pmesh->SetAttribute(i, max_elem_attr+1);
      }
    }

  /* pmesh->ExchangeFaceNbrNodes();
  for (int i = H1.GetNE(); i < (H1.GetNE() + pmesh->GetNSharedFaces()) ; i++){
    FaceElementTransformations *eltrans = pmesh->GetSharedFaceTransformations(i-H1.GetNE());
    if (eltrans != NULL){
      int Elem2No = eltrans->Elem2No;
      int Elem2NbrNo = Elem2No - pmesh->GetNE();
      const FiniteElement *FElem = H1.GetFE(Elem2No);
      const IntegrationRule &ir = FElem->GetNodes();
      ElementTransformation *nbrftr = H1.GetFaceNbrElementTransformation(Elem2NbrNo);
      int count = 0;
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
	const IntegrationPoint &ip = ir.IntPoint(j);
	Vector x(3);
	nbrftr->Transform(ip,x);
	double ptOnLine = slope * x(0) + yIntercept;
	if ( x(1) <= ptOnLine){
	  count++;
	}
      }

      if (count == ir.GetNPoints()){
	elemStatus[i] = SBElementType::INSIDE;
      }
      else if ( (count > 0) && (count < ir.GetNPoints())){
	elemStatus[i] = SBElementType::CUT;
      }
      else if (count == 0){
	elemStatus[i] = SBElementType::OUTSIDE;
      }
      
    }
  }*/
  pmesh->SetAttributes();
  }

  void Line::SetupFaceTags(Array<int> &elemStatus, Array<int> &faceTags, Array<int> &initialBoundaryFaceTags, int maxBTag){
    //elemStatus.Print();
    for (int i = 0; i < H1.GetNF() ; i++){
      FaceElementTransformations *eltrans = pmesh->GetInteriorFaceTransformations(i);
      if (eltrans != NULL){
	const int faceElemNo = eltrans->ElementNo; 
	int Elem1No = eltrans->Elem1No;
	int Elem2No = eltrans->Elem2No;
	int statusElem1 = elemStatus[Elem1No];
	int statusElem2 = elemStatus[Elem2No];
	if ( ( (statusElem1 == SBElementType::INSIDE) && (statusElem2 == SBElementType::CUT) ) ||  ( (statusElem1 == SBElementType::CUT) && (statusElem2 == SBElementType::INSIDE) ) ){
	  faceTags[faceElemNo] = 5;
	}
	else {
	  faceTags[faceElemNo] = 0;
	}
      }
    }

    /*  pmesh->ExchangeFaceNbrNodes();
    for (int i = H1.GetNF(); i < (H1.GetNF() + pmesh->GetNSharedFaces()) ; i++){
      FaceElementTransformations *eltrans = pmesh->GetSharedFaceTransformations(i-H1.GetNF());
      if (eltrans != NULL){
	const int faceElemNo = eltrans->ElementNo;
	int Elem2No = eltrans->Elem2No;
	int Elem2NbrNo = Elem2No - H1.GetNE();
	int Elem1No = eltrans->Elem1No;
	int statusElem1 = elemStatus[Elem1No];
	int statusElem2 = elemStatus[Elem2No];
	if ( ( (statusElem1 == SBElementType::INSIDE) && (statusElem2 == SBElementType::CUT) ) ||  ( (statusElem1 == SBElementType::CUT) && (statusElem2 == SBElementType::INSIDE) ) ){
	  faceTags[faceElemNo] = 5;
	}
	else {
	  faceTags[faceElemNo] = maxBTag + 2;
	}
      }
      }*/
  }
  void Line::ComputeDistanceAndNormalAtQuadraturePoints(const IntegrationRule &b_ir, Array<int> &elemStatus, Array<int> &faceTags, DenseMatrix &quadratureDistance, DenseMatrix &quadratureTrueNormal){
    // MPI_Comm comm = pmesh->GetComm();
    // This code is only for the 1D/FA mode
    const int nqp_face = b_ir.GetNPoints();
    for (int i = 0; i < H1.GetNF() ; i++){
      FaceElementTransformations *eltrans = pmesh->GetInteriorFaceTransformations(i);
      if (eltrans != NULL){
	const int faceElemNo = eltrans->ElementNo; 
	if (faceTags[faceElemNo] == 5){
	  for (int q = 0; q  < nqp_face; q++)
	    {
	      const IntegrationPoint &ip_f = b_ir.IntPoint(q);
	      eltrans->SetAllIntPoints(&ip_f);
	      int Elem1No = eltrans->Elem1No;
	      int Elem2No = eltrans->Elem2No;
	      int statusElem1 = elemStatus[Elem1No];
	      int statusElem2 = elemStatus[Elem2No];
	      Vector x(3);
	      if (statusElem1 == SBElementType::INSIDE){
		const IntegrationPoint &eip = eltrans->GetElement1IntPoint();
		ElementTransformation &Trans_el1 = eltrans->GetElement1Transformation();
		Trans_el1.SetIntPoint(&eip);	
		Trans_el1.Transform(eip,x);
	      }
	      else {
		const IntegrationPoint &eip = eltrans->GetElement2IntPoint();
		ElementTransformation &Trans_el2 = eltrans->GetElement2Transformation();
		Trans_el2.SetIntPoint(&eip);
		Trans_el2.Transform(eip,x);	    
	      }	   
	      double xPtOnLine = x(0) - slope * yIntercept + slope * x(1);
	      double distX = xPtOnLine - x(0);
	      double distY = slope * xPtOnLine + yIntercept - x(1);
	      quadratureDistance(faceElemNo*nqp_face + q,0) = distX;
	      quadratureDistance(faceElemNo*nqp_face + q,1) = distY;
	      double normD = sqrt(distX * distX + distY * distY);
	      quadratureTrueNormal(faceElemNo*nqp_face + q,0) = distX /  normD;
	      quadratureTrueNormal(faceElemNo*nqp_face + q,1) = distY /  normD;
	    }
	}
      }
    }
    /* pmesh->ExchangeFaceNbrNodes();   
    for (int i = H1.GetNF(); i <  (H1.GetNF() + pmesh->GetNSharedFaces()) ; i++){
      FaceElementTransformations *eltrans = pmesh->GetSharedFaceTransformations(i-H1.GetNF());
      if (eltrans != NULL){
	const int faceElemNo = eltrans->ElementNo;
	if (faceTags[faceElemNo] == 5){
	  for (int q = 0; q  < nqp_face; q++)
	    {
	      const IntegrationPoint &ip_f = b_ir.IntPoint(q);
	      eltrans->SetAllIntPoints(&ip_f);
	      int Elem1No = eltrans->Elem1No;
	      int statusElem1 = elemStatus[Elem1No];
	      Vector x(3);
	      if (statusElem1 == SBElementType::INSIDE){
		const IntegrationPoint &eip = eltrans->GetElement1IntPoint();
		ElementTransformation &Trans_el1 = eltrans->GetElement1Transformation();
		Trans_el1.SetIntPoint(&eip);	
		Trans_el1.Transform(eip,x);
	      }
	      else {
		continue;
	      }
	      double xPtOnLine = x(0) - slope * yIntercept + slope * x(1);
	      double distX = xPtOnLine - x(0);
	      double distY = slope * xPtOnLine + yIntercept - x(1);
	      quadratureDistance(i*nqp_face + q,0) = distX;
	      quadratureDistance(i*nqp_face + q,1) = distY;
	      double normD = sqrt(distX * distX + distY * distY);
	      quadratureTrueNormal(i*nqp_face + q,0) = distX /  normD;
	      quadratureTrueNormal(i*nqp_face + q,1) = distY /  normD;
	    }
	}
      }
      }*/
  }
}
