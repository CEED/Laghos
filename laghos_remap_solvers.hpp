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

#ifndef MFEM_LAGHOS_REMAP_SOLVERS
#define MFEM_LAGHOS_REMAP_SOLVERS

#include "mfem.hpp"

namespace mfem
{
namespace ale
{
class TimeDependentGeomConsOperator;
namespace geom_consistent_solvers
{
class GeomConsODESolver : public ODESolver
{
protected:
    TimeDependentGeomConsOperator *f;
    using ODESolver::Init;

public:

    virtual void Init(TimeDependentGeomConsOperator &f);
};

/// The classical forward Euler method
class ForwardEulerSolver : public GeomConsODESolver
{
    Vector dU;
    ParGridFunction flux;
public:
    void Init(TimeDependentGeomConsOperator &f) override;
    void Step(Vector &x, real_t &t, real_t &dt) override;
};

/// General RK-s (s-stage Runge-Kutta) solver
class RKSolver : public GeomConsODESolver
{
    const int s;
    const real_t *a, *b, *c;
    real_t *d;
    Vector *dxs, *Ks;
    ParGridFunction *fs;

    // This function constructs coefficients that transform eq. (2.16) from
   // JLG's paper to an update that only uses the previous limited updates.
   // This function does not depend on the Operator f in any way.
   void ConstructD();

public:
    RKSolver(int s_, const real_t a_[], const real_t b_[], const real_t c_[]);
    void Init(TimeDependentGeomConsOperator &f) override;
    void Step(Vector &x, real_t &t, real_t &dt) override;
    virtual ~RKSolver();
};

/// The classical midpoint method
class RK2Solver : public RKSolver
{
    static const real_t a[], b[], c[];
public:
    RK2Solver() : RKSolver(2, a, b, c) { }
};

/// Third-order, Heun's method
class RK3Solver : public RKSolver
{
    static const real_t a[], b[], c[];
public:
    RK3Solver() : RKSolver(3, a, b, c) { }
};

/// Fourth-order, equidistant rule
class RK4Solver : public RKSolver
{
    static const real_t a[], b[], c[];
public:
    RK4Solver() : RKSolver(4, a, b, c) { }
};

/// Fifth-order, equidistant rule
class RK6Solver : public RKSolver
{
    static const real_t a[], b[], c[];
public:
    RK6Solver() : RKSolver(6, a, b, c) { }
};

} // namespace geom_consistent_solvers
} // namespace ale
} // namespace mfem

#endif // MFEM_LAGHOS_REMAP_SOLVERS
