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

#ifndef MFEM_LAGHOS_TIMEINTEG
#define MFEM_LAGHOS_TIMEINTEG

#include "mfem.hpp"

class ROM_Operator;

namespace mfem
{

namespace hydrodynamics
{

class LagrangianHydroOperator;

class HydroODESolver : public ODESolver
{
protected:
    const bool rom;
    LagrangianHydroOperator *hydro_oper;
    ROM_Operator *rom_oper;

public:
    HydroODESolver(const bool romOnline=false) : hydro_oper(NULL), rom_oper(NULL), rom(romOnline) { }

    virtual void Init(TimeDependentOperator &_f);

    virtual void Step(Vector &S, double &t, double &dt)
    {
        MFEM_ABORT("Time stepping is undefined.");
    }
};

class RK2AvgSolver : public HydroODESolver
{
private:
    ParFiniteElementSpace *H1FESpace, *L2FESpace;

public:
    RK2AvgSolver(const bool romOnline=false, ParFiniteElementSpace *H1FESpace_=NULL, ParFiniteElementSpace *L2FESpace_=NULL) : HydroODESolver(romOnline),
        H1FESpace(H1FESpace_), L2FESpace(L2FESpace_) { }

    virtual void Step(Vector &S, double &t, double &dt);
};

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_LAGHOS_TIMEINTEG
