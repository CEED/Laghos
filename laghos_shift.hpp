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
   SIMarker(const ParGridFunction &ls_gf) : ls(ls_gf) { }

   // LS < 0 everywhere (mat 1) --> attribute  10.
   // mixed                     --> attribute  15.
   // LS > 0 everywhere (mat 2) --> attribute  20.
   int GetMaterialID(int el_id);

   // Mixed / mat 1 --> attribute 10.
   // Mixed / mat 2 --> attribute 20.
   // all other     --> attribute 0.
   void MarkFaceAttributes(ParFiniteElementSpace &pfes);

   void GetFaceAttributeGF(ParGridFunction &fa_gf);
};

// Stores the shifted interface options.
struct SIOptions
{
   PressureSpace p_space = PressureSpace::L2;

   int v_shift_type = 0;
   bool v_volume_avg = false;
   double v_shift_scale = 1.0;
   bool v_shift_diffusion = false;
   double v_shift_diffusion_scale = 1.0;

   int e_shift_type = 0;
   bool e_volume_cut = false;
   bool e_shift_diffusion = false;
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

// Performs full assemble for the force face terms:
// F_face_ij = < [grad_p.dist] * h1_shape_j * l2_shape_i >.
class FaceForceIntegrator : public BilinearFormIntegrator
{
private:
   Vector h1_shape_face, h1_shape, l2_shape;
   const MaterialData &mat_data;
   const ParGridFunction *v = nullptr;
   VectorCoefficient &dist;

   int v_shift_type = 0;
   double v_shift_scale = 1.0;

   bool diffuse_v = false;
   double diffuse_v_scale = 1.0;
   CutFaceQuadratureData &qdata;

  public:
   FaceForceIntegrator(const MaterialData &mdata, VectorCoefficient &d,
                       CutFaceQuadratureData &cfqdata)
   : mat_data(mdata), dist(d), qdata(cfqdata) { }

   // Goes over all H1 volumetric dofs in both cells around the interface.
   void AssembleFaceMatrix(const FiniteElement &trial_fe,
                           const FiniteElement &test_fe,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat);

   // Goes over only the H1 dofs that are exactly on the interface.
   void AssembleFaceMatrix(const FiniteElement &trial_face_fe,
                           const FiniteElement &test_fe1,
                           const FiniteElement &test_fe2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat);

   void SetShiftType(int type) { v_shift_type = type; }
   void SetScale(double s) { v_shift_scale = s; }
   void SetDiffusion(bool d, double s) { diffuse_v = d; diffuse_v_scale = s; }

   void SetVelocity(const ParGridFunction &vel) { v = &vel; }
   void UnsetVelocity() { v = nullptr; }
};

class EnergyInterfaceIntegrator : public LinearFormIntegrator
{
private:
   // What's the material that uses it to construct energy RHS.
   const int mat_id;
   const MaterialData &mat_data;
   const ParGridFunction *v = nullptr, *e = nullptr;
   VectorCoefficient &dist;

   CutFaceQuadratureData &qdata_face;

public:
   int e_shift_type = 0;
   bool diffusion = false;
   double diffusion_scale = 1.0;

   EnergyInterfaceIntegrator(int m_id, const MaterialData &mdata,
                             VectorCoefficient &d,
                             CutFaceQuadratureData &cfqdata)
      : mat_id(m_id), mat_data(mdata), dist(d), qdata_face(cfqdata) { }

   using LinearFormIntegrator::AssembleRHSElementVect;
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Trans,
                                       Vector &elvect)
   { MFEM_ABORT("should not be used"); }

   virtual void AssembleRHSFaceVect(const FiniteElement &el_1,
                                    const FiniteElement &el_2,
                                    FaceElementTransformations &Trans,
                                    Vector &elvect);
   void SetVandE(const ParGridFunction *vel, const ParGridFunction *en)
   { v = vel; e = en; }
   void UnsetVandE() { v = nullptr; e = nullptr; }
};

void PrintCellNumbers(const Vector &xyz, const ParFiniteElementSpace &pfes,
                      std::string text);

class PointExtractor
{
protected:
   std::ofstream fstream;

   const ParGridFunction &g;
   // -1 if the point is not in the current MPI task.
   int z_id;
   IntegrationPoint ip;

   int FindIntegrPoint(const int z_id, const Vector &xyz,
                       const IntegrationRule &ir);

public:
   // The assumption is that the point concides with one of the DOFs of the
   // input GridFunction's nodes.
   PointExtractor(int zone, Vector &xyz, const ParGridFunction &gf,
                  const IntegrationRule &ir, std::string filename);

   ~PointExtractor() { fstream.close(); }

   virtual double GetValue() const { return g.GetValue(z_id, ip); }
   void WriteValue(double time);
};

class ShiftedPointExtractor : public PointExtractor
{
protected:
   const ParGridFunction &dist;

public:
   ShiftedPointExtractor(int zone, Vector &xyz,
                         const ParGridFunction &gf, const ParGridFunction &d,
                         const IntegrationRule &ir, std::string filename)
      : PointExtractor(zone, xyz, gf, ir, filename), dist(d) { }

   virtual double GetValue() const;
};

void InitTG2Mat(MaterialData &mat_data);
void InitSod2Mat(MaterialData &mat_data);
void InitWaterAir(MaterialData &mat_data);
void InitTriPoint2Mat(MaterialData &mat_data);

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_LAGHOS_SHIFT
