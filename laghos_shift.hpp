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

#ifndef MFEM_LAGHOS_SHIFT
#define MFEM_LAGHOS_SHIFT


//#include "../../mfem_shift/mfem.hpp"
#include "mfem.hpp"
#include "laghos_assembly.hpp"

namespace mfem
{

namespace hydrodynamics
{

int material_id(int el_id, const ParGridFunction &g);

void MarkFaceAttributes(ParFiniteElementSpace &pfes);

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

class FaceForceIntegrator : public BilinearFormIntegrator
{
private:
   Vector h1_shape_face, h1_shape, l2_shape;
   const ParGridFunction &p;

   int v_shift_type = 0;
   double scale = 1.0;

  public:
   FaceForceIntegrator(const ParGridFunction &p_gf) : p(p_gf)  { }

   // Goes over only the H1 dofs that are exactly on the interface.
   void AssembleFaceMatrix(const FiniteElement &trial_fe,
                           const FiniteElement &test_fe,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat);

   void SetShiftType(int type) { v_shift_type = type; }
   void SetScale(double s) { scale = s; }
};

class EnergyInterfaceIntegrator : public LinearFormIntegrator
{
private:
   const ParGridFunction &p; ParGridFunction &v;
   int e_shift_type = 0;
   double scale = 1.0;

public:
   EnergyInterfaceIntegrator(const ParGridFunction &p_gf,
                             ParGridFunction &v_gf)
      : p(p_gf), v(v_gf) { }

   using LinearFormIntegrator::AssembleRHSElementVect;
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Trans,
                                       Vector &elvect)
   { MFEM_ABORT("should not be used"); }

   virtual void AssembleRHSElementVect(const FiniteElement &el_1,
                                       const FiniteElement &el_2,
                                       FaceElementTransformations &Trans,
                                       Vector &elvect);

   void SetVelocityReference(const ParGridFunction &v_gf) { v = v_gf; }
   void SetShiftType(int type) { e_shift_type = type; }
   void SetScale(double s) { scale = s; }

};

class VelocityStabilizerLFI : public LinearFormIntegrator
{
private:
   ParGridFunction &v;
   CutFaceQuadratureData &cfqdata;
   double scale = 1.0; //\beta in Nabil's notes
   double *dt;

public:
   VelocityStabilizerLFI(ParGridFunction &v_gf,
                         CutFaceQuadratureData &cfqdata_,
                         double *dt_)
      : v(v_gf), cfqdata(cfqdata_), dt(dt_) { }

   using LinearFormIntegrator::AssembleRHSElementVect;
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Trans,
                                       Vector &elvect)
   { MFEM_ABORT("should not be used"); }

   virtual void AssembleRHSElementVect(const FiniteElement &el_1,
                                       const FiniteElement &el_2,
                                       FaceElementTransformations &Trans,
                                       Vector &elvect);
   void SetVelocityReference(const ParGridFunction &v_gf) { v = v_gf; }
   void SetScale(double s) { scale = s; }
};

class EnergyStabilizerLFI : public LinearFormIntegrator
{
private:
   ParGridFunction &p;
   CutFaceQuadratureData &cfqdata;
   double scale = 1.0;
   double *dt;

public:
   EnergyStabilizerLFI(ParGridFunction &p_gf,
                       CutFaceQuadratureData &cfqdata_,
                       double *dt_)
      : p(p_gf), cfqdata(cfqdata_), dt(dt_) { }

   using LinearFormIntegrator::AssembleRHSElementVect;
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Trans,
                                       Vector &elvect)
   { MFEM_ABORT("should not be used"); }

   virtual void AssembleRHSElementVect(const FiniteElement &el_1,
                                       const FiniteElement &el_2,
                                       FaceElementTransformations &Trans,
                                       Vector &elvect);
   void SetScale(double s) { scale = s; }
};

void InitSod2Mat(ParGridFunction &rho, ParGridFunction &v,
                 ParGridFunction &e, ParGridFunction &gamma);

void InitWaterAir(ParGridFunction &rho, ParGridFunction &v,
                  ParGridFunction &e, ParGridFunction &gamma);

void InitTriPoint2Mat(ParGridFunction &rho, ParGridFunction &v,
                      ParGridFunction &e, ParGridFunction &gamma);

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_LAGHOS_SHIFT
