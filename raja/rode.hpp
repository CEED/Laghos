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
#ifndef LAGHOS_RAJA_ODE
#define LAGHOS_RAJA_ODE

namespace mfem {

  // ***************************************************************************
  class RajaODESolver{
  protected:
    RajaTimeDependentOperator *f;
  public:
    RajaODESolver() : f(NULL) { }
    virtual void Init(RajaTimeDependentOperator &f){
      this->f = &f;
    }
    virtual void Step(RajaVector &x, double &t, double &dt)
    { mfem_error("ODESolver::Step(Vector) is not overloaded!"); }
    virtual void Run(RajaVector &x, double &t, double &dt, double tf){
      while (t < tf) { Step(x, t, dt); }
    }
    virtual ~RajaODESolver() { }
  };

  // ***************************************************************************
  class RajaForwardEulerSolver : public RajaODESolver{
  private:
    RajaVector dxdt;
  public:
    virtual void Init(RajaTimeDependentOperator &_f);
    virtual void Step(RajaVector &x, double &t, double &dt);
  };

  // ***************************************************************************
  class RajaRK2Solver : public RajaODESolver{
  private:
    double a;
    RajaVector dxdt, x1;
  public:
    RajaRK2Solver(const double _a = 2./3.) : a(_a) { }
    virtual void Init(RajaTimeDependentOperator &_f);
    virtual void Step(RajaVector &x, double &t, double &dt);
  };

  // ***************************************************************************
  class RajaRK4Solver : public RajaODESolver{
  private:
    RajaVector y, k, z;
  public:
    virtual void Init(RajaTimeDependentOperator &_f);
    virtual void Step(RajaVector &x, double &t, double &dt);
};

} // mfem

#endif // LAGHOS_RAJA_ODE
