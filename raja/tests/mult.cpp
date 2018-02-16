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
#include "../raja.hpp"
#include "../../laghos_solver.hpp"

namespace mfem {

  // ***************************************************************************
  bool multTest(ParMesh *pmesh, const int order){
    const int dim = pmesh->Dimension();
    L2_FECollection L2FEC(order, dim);
    RajaFiniteElementSpace fes(pmesh, &L2FEC);
    
    RajaOperator VMassPA;
      
    const RajaOperator* trialP = fes.GetProlongationOperator();
    const RajaOperator* testP  = fes.GetProlongationOperator();
    RajaMassIntegrator &massInteg = *(new RajaMassIntegrator(false));
    
    RajaRAPOperator RAP(*testP,VMassPA,*trialP);
    
    const RajaOperator Rt;
    const RajaOperator A;
    RajaOperator P;
   
    RajaVector Px;
    RajaVector APx;

    RajaVector x;
    RajaVector y;
    
    push(SkyBlue);
    P.Mult(x, Px);
    A.Mult(Px, APx);
    Rt.MultTranspose(APx, y);
    pop();
    
    return true;
  }

} // namespace mfem
