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
#include "laghos_rom.hpp"

using namespace std;

namespace mfem
{

namespace hydrodynamics
{

void HydroODESolver::Init(TimeDependentOperator &_f)
{
    ODESolver::Init(_f);

    if (rom)
    {
        rom_oper = dynamic_cast<ROM_Operator *>(f);
        MFEM_VERIFY(rom_oper, "HydroSolvers expect ROM_Operator.");
    }
    else
    {
        hydro_oper = dynamic_cast<LagrangianHydroOperator *>(f);
        MFEM_VERIFY(hydro_oper, "HydroSolvers expect LagrangianHydroOperator.");
    }
}

void RK2AvgSolver::Step(Vector &S, double &t, double &dt)
{
    if (rom)
    {
        rom_oper->StepRK2Avg(S, t, dt);
        return;
    }

    const int Vsize = hydro_oper->GetH1VSize();
    Vector V(Vsize), dS_dt(S.Size()), S0(S);

    // The monolithic BlockVector stores the unknown fields as follows:
    // (Position, Velocity, Specific Internal Energy).
    Vector dv_dt, v0, dx_dt;
    v0.SetDataAndSize(S0.GetData() + Vsize, Vsize);
    dv_dt.SetDataAndSize(dS_dt.GetData() + Vsize, Vsize);
    dx_dt.SetDataAndSize(dS_dt.GetData(), Vsize);

    // In each sub-step:
    // - Update the global state Vector S.
    // - Compute dv_dt using S.
    // - Update V using dv_dt.
    // - Compute de_dt and dx_dt using S and V.

    // -- 1.
    // S is S0.
    hydro_oper->UpdateMesh(S);
    hydro_oper->SolveVelocity(S, dS_dt);
    // V = v0 + 0.5 * dt * dv_dt;
    add(v0, 0.5 * dt, dv_dt, V);
    hydro_oper->SolveEnergy(S, V, dS_dt);
    dx_dt = V;

    // -- 2.
    // S = S0 + 0.5 * dt * dS_dt;
    add(S0, 0.5 * dt, dS_dt, S);
    hydro_oper->ResetQuadratureData();
    hydro_oper->UpdateMesh(S);
    hydro_oper->SolveVelocity(S, dS_dt);
    // V = v0 + 0.5 * dt * dv_dt;
    add(v0, 0.5 * dt, dv_dt, V);
    hydro_oper->SolveEnergy(S, V, dS_dt);
    dx_dt = V;

    // -- 3.
    // S = S0 + dt * dS_dt.
    add(S0, dt, dS_dt, S);
    hydro_oper->ResetQuadratureData();

    t += dt;
}

} // namespace hydrodynamics

} // namespace mfem
