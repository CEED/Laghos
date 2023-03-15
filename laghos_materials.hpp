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
   const double gamma;
   const int problem, mat_id;
   const int p_order;
   PressureSpace p_space;

   L2_FECollection p_fec_L2;
   H1_FECollection p_fec_H1;
   ParFiniteElementSpace p_fes_L2, p_fes_H1;
   ParGridFunction p_L2, p_H1;
   // Stores alpha * rho0 * det(J0)  at the pressure GF's nodes.
   Vector rho0DetJ0;

public:
   PressureFunction(int prob, int mid, ParMesh &pmesh,
                    PressureSpace space, int p_ord,
                    ParGridFunction &ind0, ParGridFunction &rho0, double g);

   void UpdateRho0Alpha0(const ParGridFunction &alpha0,
                         const ParGridFunction &rho0);

   void UpdatePressure(const ParGridFunction &ind0,
                       const ParGridFunction &energy);

   ParGridFunction &ComputePressure(const ParGridFunction &ind0,
                                    const ParGridFunction &energy)
   {
      UpdatePressure(ind0, energy);
      return GetPressure();
   }

   ParGridFunction &GetPressure() { return (p_space == L2) ? p_L2 : p_H1; }

   void ExchangeFaceNbrData() { GetPressure().ExchangeFaceNbrData(); }
};

struct MaterialData;

// Updates the alphas.
// Updates ind when mat_data is not null.
void UpdateAlpha(const ParGridFunction &level_set,
                 ParGridFunction &alpha_1, ParGridFunction &alpha_2,
                 MaterialData *mat_data = nullptr,
                 bool pointwise_alpha = false);

// Stores the shifted interface options.
struct MaterialData
{
   ParGridFunction  level_set;        // constant in time (moves with the mesh).

   double           gamma_1, gamma_2; // constant values.
   ParGridFunction  rho0_1, rho0_2;   // not updated - use only at time zero!
   ParGridFunction  e_1, e_2;         // evolved by the ODESolver.
   PressureFunction *p_1, *p_2;       // recomputed in UpdateQuadratureData().
   ParGridFunction  p;                // recomputed by ComputeTotalPressure().
   ParGridFunction  alpha_1, alpha_2; // recomputed in UpdateQuadratureData().
   bool             pointwise_alpha;
   ParGridFunction  ind0_1, ind0_2;   // recomputed in UpdateQuadratureData().
   ParGridFunction  rho0DetJ_1,       // pointwise masses as GridFunctions.
                    rho0DetJ_2;       // not updated.

   // Remap influence:
   // * level set is remapped, then updates alpha_1 and alpha_2 after remap.
   // * rho0_1 and rho0_2 are updated after remap.
   // * e_1 and e_2 are remapped.
   // * the fields inside p_1 and p_2 are updated after remap.
   // * rhoDetJind0_1 and _2 are updated after remap.

   MaterialData() : p_1(nullptr), p_2(nullptr) { }

   void UpdateInitialMasses();

   void ComputeTotalPressure(const ParGridFunction &p1_gf,
                             const ParGridFunction &p2_gf);

   ~MaterialData()
   {
      delete p_1;
      delete p_2;
   }
};

class InterfaceRhoCoeff : public Coefficient
{
private:
   ParGridFunction &alpha_1, &alpha_2, &rho_1, &rho_2;

public:
   InterfaceRhoCoeff(ParGridFunction &a1, ParGridFunction &a2,
                     ParGridFunction &r1, ParGridFunction &r2)
      : alpha_1(a1), alpha_2(a2), rho_1(r1), rho_2(r2) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      return alpha_1.GetValue(T, ip) * rho_1.GetValue(T, ip) +
             alpha_2.GetValue(T, ip) * rho_2.GetValue(T, ip);
   }

   void ExchangeFaceNbrData()
   {
      alpha_1.ExchangeFaceNbrData();
      alpha_2.ExchangeFaceNbrData();
      rho_1.ExchangeFaceNbrData();
      rho_2.ExchangeFaceNbrData();
   }
};

class AlphaRhoCoeff : public Coefficient
{
private:
   const ParGridFunction &alpha, &rho;

public:
   AlphaRhoCoeff(const ParGridFunction &a, const ParGridFunction &r)
      : alpha(a), rho(r) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      return alpha.GetValue(T, ip) * rho.GetValue(T, ip);
   }
};

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_LAGHOS_MATERIALS
