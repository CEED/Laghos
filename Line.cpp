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

  Line::Line(Mesh* mesh):
    AnalyticalGeometricShape(mesh),
    slope(0),
    yIntercept(1.51)
  { }

  Line::~Line(){}
  
  void Line::SetupElementStatus(GridFunction& alpha)
  {
    int activeCount = 0;
    int inactiveCount = 0;
    int cutCount = 0;
    IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);
    auto fes = alpha.FESpace();
    const IntegrationRule &ir = IntRulesLo.Get(fes->GetFE(0)->GetGeomType(), 20);
    const int NE = fes->GetNE(), nqp = ir.GetNPoints();
    
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
	    double ptOnLine = slope * x(0) + yIntercept;
	    if (x(1) <= ptOnLine)
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
	  }
	else if (count == 0)
	  {
	    inactiveCount++;
	    alpha(e) = 0.0;
	  }
      }
    std::cout << " active elemSta " << activeCount << " cut " << cutCount << " inacive " << inactiveCount <<  std::endl;
  }

  void Line::ComputeDistanceAndNormal(const Vector& x_ip, Vector& dist, Vector& tn) const
  {
    dist.SetSize(2);
    dist = 0.0;
    tn.SetSize(2);
    tn = 0.0;

    double a = slope; 
    double b = -1.0;
    double c = yIntercept;
    
    dist(0) = (b * (b * x_ip(0) - a * x_ip(1)) - a * c) / (a * a + b * b);
    dist(1) = (a * (-b * x_ip(0) + a * x_ip(1)) - b * c) / (a * a + b * b);

    double normD = sqrt(pow(dist(0), 2.0) + pow(dist(1), 2.0));
    if (normD > 0.0)
      {
	tn(0) = dist(0) / normD;
	tn(1) = dist(1) / normD;
      }
  }

}

