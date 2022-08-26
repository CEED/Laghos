// CopyrighAt (AAcA) 2017, Lawrence Livermore National Security, OALLC. Produced at
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

  Line::Line(ParFiniteElementSpace &h1_fes, ParFiniteElementSpace &l2_fes): AnalyticalGeometricShape(h1_fes, l2_fes), slope(0), yIntercept(1.51) {
  }

  Line::~Line(){}
  
  void Line::SetupElementStatus(Array<int> &elemStatus, Array<int> &ess_inactive){
 
    const int max_elem_attr = (pmesh->attributes).Max();
    int activeCount = 0;
    int inactiveCount = 0;
    int cutCount = 0;
    
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
	//	std::cout << " x " << x(0) << " y " << x(1) << std::endl;
	double ptOnLine = slope * x(0) + yIntercept;
	if ( x(1) <= (ptOnLine)){
	  count++;
	}
      }
      // std::cout << " count " << count << std::endl;
      if (count == ir.GetNPoints()){
	elemStatus[i] = SBElementType::INSIDE;
	Array<int> dofs;
	H1.GetElementVDofs(i, dofs);
	activeCount++;
	for (int k = 0; k < dofs.Size(); k++)
	  {
	    ess_inactive[dofs[k]] = 0;	       
	  }
      }
      else if ( (count > 0) && (count < ir.GetNPoints())){
	elemStatus[i] = SBElementType::CUT;
	cutCount++;
	pmesh->SetAttribute(i, max_elem_attr+1);
      }
      else if (count == 0){
	elemStatus[i] = SBElementType::OUTSIDE;
	inactiveCount++;
	pmesh->SetAttribute(i, max_elem_attr+1);
      }
    }
  //std::cout << " active elemSta " << activeCount << " cut " << cutCount << " inacive " << inactiveCount <<  std::endl;
  //  elemStatus.Print(std::cout,1);
  H1.Synchronize(ess_inactive);
  pmesh->SetAttributes();
  /*std::cout << " eleSt " << std::endl;
  elemStatus.Print();
  std::cout << " ess inac " << std::endl;
  ess_inactive.Print();*/
  }

  void Line::SetupFaceTags(Array<int> &elemStatus, Array<int> &faceTags, Array<int> &ess_inactive, Array<int> &initialBoundaryFaceTags, int maxBTag){
    
    MPI_Comm comm = pmesh->GetComm();
    int myid;
    MPI_Comm_rank(comm, &myid);
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

  }
  void Line::ComputeDistanceAndNormalAtQuadraturePoints(const IntegrationRule &b_ir, Array<int> &elemStatus, Array<int> &faceTags, DenseMatrix &quadratureDistance, DenseMatrix &quadratureTrueNormal, DenseMatrix &quadratureDistance_BF,  DenseMatrix &quadratureTrueNormal_BF){
    // This code is only for the 1D/FA mode
    const int nqp_face = b_ir.GetNPoints();
    for (int i = 0; i < H1.GetNF() ; i++){
      FaceElementTransformations *eltrans = pmesh->GetInteriorFaceTransformations(i);
      if (eltrans != NULL){
	const int faceElemNo = eltrans->ElementNo; 
	if (faceTags[faceElemNo] == 5){
	  //  std::cout << " qnp face " << nqp_face << std::endl;
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
	      // std::cout << " shit shit shit shit shit " << std::endl;
	      	    
	      double normD = sqrt(distX * distX + distY * distY);
	      quadratureTrueNormal(faceElemNo*nqp_face + q,0) = distX /  normD;
	      quadratureTrueNormal(faceElemNo*nqp_face + q,1) = distY /  normD;
	    }
	}
      }
    }
    for (int i = 0; i < H1.GetNBE() ; i++){
      FaceElementTransformations *eltrans = pmesh->GetBdrFaceTransformations(i);
      const int faceElemNo = eltrans->ElementNo; 
      if (pmesh->GetBdrAttribute(faceElemNo) == 3 ){
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
	      //  Trans_el1.SetIntPoint(&eip);	
	      Trans_el1.Transform(eip,x);
	      //  eltrans->Transform(eip,x);
	    }
	    double xPtOnLine = x(0) - slope * yIntercept + slope * x(1);
	    double distX = xPtOnLine - x(0);
	    double distY = slope * xPtOnLine + yIntercept - x(1);
	    quadratureDistance_BF(faceElemNo*nqp_face + q,0) = distX;
	    quadratureDistance_BF(faceElemNo*nqp_face + q,1) = distY;
	    // std::cout << " eq Nu " << faceElemNo*nqp_face + q <<  " X " << x(0) << " Y " << x(1) << " Line " << distY << std::endl;
	    double normD = sqrt(distX * distX + distY * distY);
	    quadratureTrueNormal_BF(faceElemNo*nqp_face + q,0) = distX /  normD;
	    quadratureTrueNormal_BF(faceElemNo*nqp_face + q,1) = distY /  normD;
	  }
      }
    }
  }

  void Line::ComputeDistanceAndNormalAtCoordinates(const Vector &x, Vector &D, Vector &tN){
    double xPtOnLine = x(0) - slope * yIntercept + slope * x(1);
    double distX = xPtOnLine - x(0);
    double distY = slope * xPtOnLine + yIntercept - x(1);
    D(0) = distX;
    D(1) = distY;
    double normD = sqrt(distX * distX + distY * distY);
    tN(0) = distX /  normD;
    tN(1) = distY /  normD;
  }	      
}
