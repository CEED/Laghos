// Copyright (c) 2017, Lawrence Livermore National Security, OALLC. Produced at
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


#include "Circle.hpp"

namespace mfem{

  Circle::Circle(ParMesh* pmesh): AnalyticalGeometricShape(pmesh), radius(0.2), center(2)
 {
   center(0) = 0.5;
   center(1) = 0.5;
  }

  Circle::~Circle()
  {}
  
  void Circle::SetupElementStatus(ParGridFunction& alpha)
  {
    const int max_elem_attr = (pmesh->attributes).Max();
    int activeCount = 0;
    int inactiveCount = 0;
    int cutCount = 0;
    IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);
    alpha = 0.0;
    auto fes = alpha.ParFESpace();
    const IntegrationRule &ir = IntRulesLo.Get(fes->GetFE(0)->GetGeomType(), 20);
    const int NE = alpha.ParFESpace()->GetNE(), nqp = ir.GetNPoints();
    // Check elements on the current MPI rank
    for (int e = 0; e < NE; e++)
      {
	ElementTransformation *Tr = fes->GetElementTransformation(e);
	int count = 0;
	for (int q = 0; q < nqp; q++)
	  {
	    const IntegrationPoint &ip = ir.IntPoint(q);
	    Tr->SetIntPoint(&ip);
	    Vector x(3);
	    Tr->Transform(ip,x);
	    
	    double radiusOfPt = pow(pow(x(0)-center(0),2.0)+pow(x(1)-center(1),2.0),0.5);
	    if ( radiusOfPt >= radius)
	      {
		count++;
	      }
	  }
	if (count == nqp)
	  {
	    activeCount++;
	    alpha(e) = 1.0;
	  }
	else if (count > 0 && count < nqp)
	  {
	    cutCount++;
	    alpha(e) = 0.5;
	    //  pmesh->SetAttribute(e, max_elem_attr+1);
	  }
	else if (count == 0)
	  {
	    inactiveCount++;
	    alpha(e) = 0.0;
	    //	    pmesh->SetAttribute(e, max_elem_attr+1);
	  }
      }
    alpha.ExchangeFaceNbrData();
    // pmesh->SetAttributes();
    // std::cout << " active elemSta " << activeCount << " cut " << cutCount << " inacive " << inactiveCount <<  std::endl;
  }
  
  void Circle::ComputeDistanceAndNormal(const Vector& x_ip, Vector& dist, Vector& tn) const
  {
    dist.SetSize(2);
    dist = 0.0;
    tn.SetSize(2);
    tn = 0.0;

    double r = sqrt(pow(x_ip(0)-center(0),2.0)+pow(x_ip(1)-center(1),2.0));
    if (r > radius)
      {
	dist(0) = ((x_ip(0)-center(0))/r)*(radius-r);
	dist(1) = ((x_ip(1)-center(1))/r)*(radius-r);
	double normD = sqrt(dist(0) * dist(0) + dist(1) * dist(1));
	if (normD > 0.0)
	  {
	    tn(0) = dist(0) / normD;
	    tn(1) = dist(1) / normD;	
	  }
      }
    else
      {
	dist = 0.0;
	tn(0) = (center(0) - x_ip(0)) / radius;
	tn(1) = (center(1) - x_ip(1)) / radius;		
      }
  }
}
