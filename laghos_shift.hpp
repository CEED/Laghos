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

#include "mfem.hpp"
#include "laghos_assembly.hpp"
#include "laghos_materials.hpp"

namespace mfem
{

namespace hydrodynamics
{

class SIMarker
{
private:
   const ParGridFunction &ls;

public:
   SIMarker(const ParGridFunction &ls_gf, ParFiniteElementSpace &mat_fes)
      : ls(ls_gf), mat_attr(&mat_fes) { }

   // LS < 0 everywhere (mat 1) --> attribute  10.
   // mixed                     --> attribute  15.
   // LS > 0 everywhere (mat 2) --> attribute  20.
   int GetMaterialID(int el_id);

   // Mixed / mat 1 --> attribute 10.
   // Mixed / mat 2 --> attribute 20.
   // all other     --> attribute 0.
   void MarkFaceAttributes();

   void GetFaceAttributeGF(ParGridFunction &fa_gf);

   // Piecewise constant material attributes.
   ParGridFunction mat_attr;
};

// Stores the shifted interface options.
struct SIOptions
{
   PressureSpace p_space = PressureSpace::L2;
   bool pointwise_alpha = false;

   int distance_type = 0;
   int num_lap = 7;

   int num_taylor = 1;

   bool use_mixed_elem = false;

   int v_shift_type = 0;
   double v_shift_scale = 1.0;
   double v_cut_scale   = 1.0;
   bool v_shift_diffusion = false;
   double v_shift_diffusion_scale = 1.0;

   int e_shift_type = 0;
   double e_shift_scale = 1.0;
   bool e_shift_diffusion = false;
   int e_shift_diffusion_type = 0;
   double e_shift_diffusion_scale = 1.0;
};

// Specifies the material interface, depending on the problem number.
class InterfaceCoeff : public Coefficient
{
private:
   const int problem;
   const ParMesh &pmesh;
   const int glob_NE;
   const bool pure_test;

public:
   InterfaceCoeff(int prob, const ParMesh &pm, bool pure)
      : problem(prob), pmesh(pm), glob_NE(pm.GetGlobalNE()), pure_test(pure) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class MomentumInterfaceIntegrator : public LinearFormIntegrator
{
private:
   const MaterialData &mat_data;
   VectorCoefficient &dist;

public:
   int    num_taylor = 1;
   int    v_shift_type = 0;
   double v_shift_scale = 1.0;
   bool   use_mixed_elem = false;

   MomentumInterfaceIntegrator(const MaterialData &mdata, VectorCoefficient &d)
      : mat_data(mdata), dist(d) { }

   using LinearFormIntegrator::AssembleRHSElementVect;
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Trans,
                                       Vector &elvect)
   { MFEM_ABORT("should not be used"); }

   virtual void AssembleRHSElementVect(const FiniteElement &el_1,
                                       const FiniteElement &el_2,
                                       FaceElementTransformations &Trans,
                                       Vector &elvect);
};

class EnergyInterfaceIntegrator : public LinearFormIntegrator
{
private:
   // What's the material that uses it to construct energy RHS.
   const int mat_id;
   const MaterialData &mat_data;
   const QuadratureData &quad_data;
   const ParGridFunction *v = nullptr, *e = nullptr;
   VectorCoefficient &dist;

public:
   int num_taylor = 1;
   int e_shift_type = 0;
   double e_shift_scale = 1.0;
   bool diffusion = false;
   bool problem_visc = false;
   double diffusion_scale = 1.0;
   int diffusion_type = 0;
   bool use_mixed_elem = false;

   EnergyInterfaceIntegrator(int m_id, const MaterialData &mdata,
                             const QuadratureData &qdata,
                             VectorCoefficient &d)
      : mat_id(m_id), mat_data(mdata), quad_data(qdata), dist(d) { }

   using LinearFormIntegrator::AssembleRHSElementVect;
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Trans,
                                       Vector &elvect)
   { MFEM_ABORT("should not be used"); }

   virtual void AssembleRHSElementVect(const FiniteElement &el_L,
                                       const FiniteElement &el_R,
                                       FaceElementTransformations &Trans,
                                       Vector &elvect);
   void SetVandE(const ParGridFunction *vel, const ParGridFunction *en)
   { v = vel; e = en; }
   void UnsetVandE() { v = nullptr; e = nullptr; }
};

class MomentumCutFaceIntegrator : public LinearFormIntegrator
{
private:
   const MaterialData &mat_data;
   VectorCoefficient &dist;

public:
   int    num_taylor = 1;
   double v_cut_scale = 1.0;

   MomentumCutFaceIntegrator(const MaterialData &mdata, VectorCoefficient &d)
      : mat_data(mdata), dist(d) { }

   using LinearFormIntegrator::AssembleRHSElementVect;
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Trans,
                                       Vector &elvect)
   { MFEM_ABORT("should not be used"); }

   virtual void AssembleRHSElementVect(const FiniteElement &el_1,
                                       const FiniteElement &el_2,
                                       FaceElementTransformations &Trans,
                                       Vector &elvect);
};

void PrintCellNumbers(const Vector &xyz, const ParFiniteElementSpace &pfes,
                      std::string text);

class PointExtractor
{
protected:
   const ParGridFunction *g = nullptr;
   // -1 if the point is not in the current MPI task.
   int element_id = -1;
   IntegrationPoint ip;
   std::ofstream fstream;

   int FindIntegrPoint(const int z_id, const Vector &xyz,
                       const IntegrationRule &ir);

public:
   PointExtractor() { }

   // The assumption is that the point coincides with one of the DOFs of the
   // input GridFunction's nodes.
   void SetPoint(int el_id, const Vector &xyz, const ParGridFunction *gf,
                 const IntegrationRule &ir, std::string filename);

   // Sets the point to the quad_id index in ir.
   void SetPoint(int el_id, int quad_id, const ParGridFunction *gf,
                 const IntegrationRule &ir, std::string filename);

   ~PointExtractor() { fstream.close(); }

   virtual double GetValue() const { return g->GetValue(element_id, ip); }
   void WriteValue(double time)
   {
      fstream << time << " " << GetValue() << "\n"; fstream.flush();
   }
};

// Density extractoin at a quad point.
class RhoPointExtractor : public PointExtractor
{
protected:
   const Vector *rho0DetJ0w = nullptr;
   int nqp = -1, q_id = -1;

public:
   RhoPointExtractor() { }

   // Needs the indicator GridFunction.
   void SetPoint(int el_id, int quad_id,
                 const ParGridFunction *ind_gf, const Vector *rdj,
                 const IntegrationRule &ir, std::string filename)
   {
      PointExtractor::SetPoint(el_id, quad_id, ind_gf, ir, filename);
      rho0DetJ0w = rdj;
      nqp = ir.GetNPoints();
      q_id = quad_id;
   }

   virtual double GetValue() const;
};

// Pressure extraction at a quad point.
class PPointExtractor : public RhoPointExtractor
{
protected:
   const ParGridFunction *e_gf = nullptr;
   double gamma = 0.0;

public:
   PPointExtractor() { }

   // Needs the indicator and energy GridFunctions.
   void SetPoint(int el_id, int quad_id, const Vector *rdj, double gam,
                 const ParGridFunction *ind_gf, const ParGridFunction *en_gf,
                 const IntegrationRule &ir, std::string filename)
   {
      RhoPointExtractor::SetPoint(el_id, quad_id, ind_gf, rdj, ir, filename);
      gamma = gam;
      e_gf = en_gf;
   }

   virtual double GetValue() const;
};

// Shifted extraction at a physical location.
class ShiftedPointExtractor : public PointExtractor
{
protected:
   const ParGridFunction *dist = nullptr;

public:
   ShiftedPointExtractor() { }

   void SetPoint(int zone, const Vector &xyz, const ParGridFunction *gf,
                 const ParGridFunction *d,
                 const IntegrationRule &ir, std::string filename)
   {
      PointExtractor::SetPoint(zone, xyz, gf, ir, filename);
      dist = d;
   }

   virtual double GetValue() const;
};

void InitTG2Mat(MaterialData &mat_data);
void InitSod2Mat(MaterialData &mat_data);
void InitWaterAir(MaterialData &mat_data);
void InitTriPoint2Mat(MaterialData &mat_data);
void InitImpact(MaterialData &mat_data);

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_LAGHOS_SHIFT
