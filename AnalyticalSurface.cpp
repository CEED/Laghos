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

#include "AnalyticalSurface.hpp"

namespace mfem
{

  AnalyticalSurface::AnalyticalSurface(int geometryType, ParMesh *pmesh):
    geometryType(geometryType),
    pmesh(pmesh),
    L2FEC_0(0, pmesh->Dimension()),
    L2_fes_0(pmesh, &L2FEC_0),
    geometry(NULL)
  {
    alpha.SetSpace(&L2_fes_0),
    alpha = 0.0;
    alpha.ExchangeFaceNbrData();
    
    switch (geometryType)
      {
	//case 1: geometry = new Line(pmesh); break;
      case 2: geometry = new Circle(pmesh); break;
      default:
	out << "Unknown geometry type: " << geometryType << '\n';
	break;
      }
  }

  AnalyticalSurface::~AnalyticalSurface()
  {
    delete geometry;
  }
  
  void AnalyticalSurface::SetupElementStatus()
  {
    geometry->SetupElementStatus(alpha);
  }
  void AnalyticalSurface::ResetData()
  {
    alpha = 0.0;
    alpha.ExchangeFaceNbrData();
  }

  ParGridFunction& AnalyticalSurface::GetAlpha()
  {
    return alpha;
  }
  AnalyticalGeometricShape& AnalyticalSurface::GetAnalyticalGeometricShape()
  {
    return *geometry;
  }

  void AnalyticalSurface::ComputeDistanceAndNormal(const Vector& x_ip, Vector& dist, Vector& tn) const
  {
    geometry->ComputeDistanceAndNormal(x_ip, dist, tn);
  }
  
}
