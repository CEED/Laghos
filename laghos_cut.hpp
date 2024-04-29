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

#ifndef MFEM_LAGHOS_CUT
#define MFEM_LAGHOS_CUT

#include "mfem.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{

namespace hydrodynamics
{

// Material-related data.
struct MaterialData
{
   ParGridFunction  level_set;        // constant in time (moves with the mesh).

   double           gamma_1, gamma_2; // constant values.
   ParGridFunction  rho0_1, rho0_2;   // not updated - use only at time zero!
   ParGridFunction  e_1, e_2;         // evolved by the ODESolver.

   MaterialData() { }
   ~MaterialData() { }
};

// Specifies the material interface, depending on the problem number.
class InterfaceCoeff : public Coefficient
{
   private:
   const int problem;
   const ParMesh &pmesh;
   const int glob_NE;

   public:
   InterfaceCoeff(int prob, const ParMesh &pm)
      : problem(prob), pmesh(pm), glob_NE(pm.GetGlobalNE()) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class ElementMarker
{
private:
   const ParGridFunction &ls;

public:
   ElementMarker(const ParGridFunction &ls_gf, ParFiniteElementSpace &mat_fes)
       : ls(ls_gf), mat_attr(&mat_fes) { }

   // LS < 0 everywhere (mat 1) --> attribute  10.
   // mixed                     --> attribute  15.
   // LS > 0 everywhere (mat 2) --> attribute  20.
   int GetMaterialID(int el_id);

   // Piecewise constant material attributes.
   ParGridFunction mat_attr;
};

class CutMassIntegrator: public MassIntegrator
{
private:
   const Array<const IntegrationRule *> &irules;

public:
   CutMassIntegrator(Coefficient &q, const Array<const IntegrationRule *> &irs)
      : MassIntegrator(q), irules(irs) { }

   void AssembleElementMatrix(const FiniteElement &el,
                              ElementTransformation &Trans,
                              DenseMatrix &elmat) override;
};

class CutVectorMassIntegrator: public VectorMassIntegrator
{
   private:
   const Array<const IntegrationRule *> &irules;

   public:
   CutVectorMassIntegrator(Coefficient &q,
                           const Array<const IntegrationRule *> &irs)
       : VectorMassIntegrator(q), irules(irs) { }

   void AssembleElementMatrix(const FiniteElement &el,
                              ElementTransformation &Trans,
                              DenseMatrix &elmat) override;
};

void InitTG2Mat(MaterialData &mat_data);
void InitSod2Mat(MaterialData &mat_data);
void InitTriPoint2Mat(MaterialData &mat_data);

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_LAGHOS_CUT
