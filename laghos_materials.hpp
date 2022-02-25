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

#ifndef MFEM_LAGHOS_MATERIALS
#define MFEM_LAGHOS_MATERIALS

#include "mfem.hpp"

namespace mfem
{

namespace hydrodynamics
{

enum PressureSpace {L2, H1};

class PressureFunction
{
private:
   const int problem;
   PressureSpace p_space;
   const int p_order     = 1;
   const int basis_type  = BasisType::GaussLobatto;

   L2_FECollection p_fec_L2;
   H1_FECollection p_fec_H1;
   ParFiniteElementSpace p_fes_L2, p_fes_H1;
   ParGridFunction p_L2, p_H1;
   // Stores rho0 * det(J0)  at the pressure GF's nodes.
   Vector rho0DetJ0;
   ParGridFunction &gamma_gf;

public:
   PressureFunction(int prob, ParMesh &pmesh, PressureSpace space,
                    ParGridFunction &rho0, ParGridFunction &gamma);

   void UpdatePressure(const ParGridFunction &e);

   ParGridFunction &GetPressure() { return (p_space == L2) ? p_L2 : p_H1; }

   ParGridFunction &ComputePressure(const ParGridFunction &e)
   {
      UpdatePressure(e);
      return GetPressure();
   }
};

// Stores the shifted interface options.
struct MaterialData
{
   ParGridFunction  level_set;        // constant in time (moves with the mesh).

   ParGridFunction  gamma_1, gamma_2; // constant in time (moves with the mesh).
   ParGridFunction  rho0_1, rho0_2;   // not updated - use only at time zero!
   ParGridFunction  e_1, e_2;         // evolved by the ODESolver.
   PressureFunction *p_1, *p_2;       // updated in UpdateQuadratureData().

   MaterialData() : p_1(nullptr), p_2(nullptr) { }

   ~MaterialData()
   {
      delete p_1;
      delete p_2;
   }
};

class InterfaceRhoCoeff : public Coefficient
{
private:
   const ParGridFunction &level_set, &rho_1, &rho_2;

public:
   InterfaceRhoCoeff(const ParGridFunction &ls,
                     const ParGridFunction &r1, const ParGridFunction &r2)
      : level_set(ls), rho_1(r1), rho_2(r2) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_LAGHOS_MATERIALS