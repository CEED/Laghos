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
#include "raja.hpp"

namespace mfem {

  
  // ***************************************************************************
  void RajaForwardEulerSolver::Init(RajaTimeDependentOperator &_f){
    RajaODESolver::Init(_f);
    dxdt.SetSize(RajaODESolver::f->Width());
  }

  // ***************************************************************************
  void RajaForwardEulerSolver::Step(RajaVector &x,
                                    double &t,
                                    double &dt){
    push();
    RajaODESolver::f->SetTime(t);
    RajaODESolver::f->Mult(x, dxdt);
    x.Add(dt, dxdt);
    t += dt;
    pop();
  }
  

  // ***************************************************************************
  void RajaRK2Solver::Init(RajaTimeDependentOperator &_f){
    RajaODESolver::Init(_f);
    int n = RajaODESolver::f->Width();
    dxdt.SetSize(n);
    x1.SetSize(n);
  }

  // ***************************************************************************
  void RajaRK2Solver::Step(RajaVector &x, double &t, double &dt){
    const double b = 0.5/a;
    RajaODESolver::f->SetTime(t);
    RajaODESolver::f->Mult(x, dxdt);
    add(x, (1. - b)*dt, dxdt, x1);
    x.Add(a*dt, dxdt);
    RajaODESolver::f->SetTime(t + a*dt);
    RajaODESolver::f->Mult(x, dxdt);
    add(x1, b*dt, dxdt, x);
    t += dt;
  }


  // ***************************************************************************
  void RajaRK4Solver::Init(RajaTimeDependentOperator &_f){
    RajaODESolver::Init(_f);
    int n = RajaODESolver::f->Width();
    y.SetSize(n);
    k.SetSize(n);
    z.SetSize(n);
  }

  // ***************************************************************************
  void RajaRK4Solver::Step(RajaVector &x, double &t, double &dt){
    RajaODESolver::f->SetTime(t);
    RajaODESolver::f->Mult(x, k); // k1
    add(x, dt/2, k, y);
    add(x, dt/6, k, z);
    RajaODESolver::f->SetTime(t + dt/2);
    RajaODESolver::f->Mult(y, k); // k2
    add(x, dt/2, k, y);
    z.Add(dt/3, k);
    RajaODESolver::f->Mult(y, k); // k3
    add(x, dt, k, y);
    z.Add(dt/3, k);
    RajaODESolver::f->SetTime(t + dt);
    RajaODESolver::f->Mult(y, k); // k4
    add(z, dt/6, k, x);
    t += dt;
  }

} // mfem
