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
#ifndef LAGHOS_RAJA_TYPEDEFS
#define LAGHOS_RAJA_TYPEDE

// mfem::ode.hpp ***************************************************************
using namespace mfem;
typedef TODESolver<RajaVector>              RajaODESolver;
typedef TForwardEulerSolver<RajaVector>     RajaForwardEulerSolver;
typedef TRK2Solver<RajaVector>              RajaRK2Solver;
typedef TRK3SSPSolver<RajaVector>           RajaRK3SSPSolver;
typedef TRK4Solver<RajaVector>              RajaRK4Solver;
typedef TExplicitRKSolver<RajaVector>       RajaExplicitRKSolver;
typedef TRK6Solver<RajaVector>              RajaRK6Solver;
typedef TRK8Solver<RajaVector>              RajaRK8Solver;
typedef TBackwardEulerSolver<RajaVector>    RajaBackwardEulerSolver;
typedef TImplicitMidpointSolver<RajaVector> RajaImplicitMidpointSolver;
typedef TSDIRK23Solver<RajaVector>          RajaSDIRK23Solver;
typedef TSDIRK34Solver<RajaVector>          RajaSDIRK34Solver;
typedef TSDIRK33Solver<RajaVector>          RajaSDIRK33Solver;

// mfem::solver.hpp ************************************************************
typedef TIterativeSolver<RajaVector>        RajaIterativeSolver;
typedef TCGSolver<RajaVector>               RajaCGSolver;

// mfem::operator.hpp **********************************************************
typedef TTimeDependentOperator<RajaVector>  RajaTimeDependentOperator;
typedef TSolver<RajaVector>                 RajaSolver;
typedef TIdentityOperator<RajaVector>       RajaIdentityOperator;
typedef TTransposeOperator<RajaVector>      RajaTransposeOperator;
typedef TRAPOperator<RajaVector>            RajaRAPOperator;
typedef TTripleProductOperator<RajaVector>  RajaTripleProductOperator;

#endif // LAGHOS_RAJA_TYPEDEF

