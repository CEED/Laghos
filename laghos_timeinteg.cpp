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

#include "laghos_timeinteg.hpp"
#include "laghos_solver.hpp"

using namespace std;

namespace mfem
{

namespace hydrodynamics
{

void RK2AvgSolver::Init(TimeDependentOperator &_f)
{
   ODESolver::Init(_f);

   hydro_oper = dynamic_cast<LagrangianHydroOperator *>(f);
   MFEM_VERIFY(hydro_oper, "RK2AvgSolver expects LagrangianHydroOperator.");
}


void RK2AvgSolver::Step(Vector &S, double &t, double &dt)
{
   const int Vsize = hydro_oper->GetH1VSize();
   Vector V(Vsize), Y(S.Size()), dS_dt(S.Size());

   // The monolithic BlockVector stores the unknown fields as follows:
   // (Position, Velocity, Specific Internal Energy).
   Vector v_dS, v_S, x_dS;
   v_S.SetDataAndSize(S.GetData() + Vsize, Vsize);
   v_dS.SetDataAndSize(dS_dt.GetData() + Vsize, Vsize);
   x_dS.SetDataAndSize(dS_dt.GetData(), Vsize);

   // 1.
   Y = S;
   hydro_oper->SolveVelocity(Y, dS_dt);
   add(0.5 * dt, v_dS, v_S, V);

   std::cout << "Solved velocity" << std::endl;

   // 2.
   hydro_oper->SolveEnergy(Y, V, dS_dt);
   x_dS = V;
   // Y = S + 0.5 * dt * dS_dt;
   add(0.5 * dt, dS_dt, S, Y);
   S = Y; // TODO better way.
   hydro_oper->ResetQuadratureData();
   hydro_oper->SolveVelocity(Y, dS_dt);
   // V = v_S + 0.5 * dt * v_dS;
   add(0.5 * dt, v_dS, v_S, V);

   // 3.
   hydro_oper->SolveEnergy(Y, V, dS_dt);
   x_dS = V;
   // Y = S + dt * dS_dt.
   add(dt, dS_dt, S, Y);
   S = Y; // TODO better way.
   hydro_oper->ResetQuadratureData();
}

} // namespace hydrodynamics

} // namespace mfem
