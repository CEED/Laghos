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

// Performs full assemble for the force face terms:
// F_face_ij = < [grad_p.dist] * h1_shape_j * l2_shape_i >.
class FaceForceIntegrator : public BilinearFormIntegrator
{
private:
   Vector h1_shape_face, h1_shape, l2_shape;
   const ParGridFunction &p;
   VectorCoefficient &dist;
   Array<int> nordir;

   int v_shift_type = 0;
   double scale = 1.0;

   bool diffuse = false;

  public:
   FaceForceIntegrator(const ParGridFunction &p_gf,
                       VectorCoefficient &d, Array<int> nordir_)
   : p(p_gf), dist(d), nordir(nordir_) { }

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
   void SetScale(double s) { scale = s; }
};

class EnergyInterfaceIntegrator : public LinearFormIntegrator
{
private:
   const ParGridFunction &p, &v;
   VectorCoefficient &dist;
   double *dt;
   int e_shift_type = 0;

public:
   EnergyInterfaceIntegrator(const ParGridFunction &p_gf,
                             const ParGridFunction &v_gf,
                             VectorCoefficient &d,
                             double *dt_)
      : p(p_gf), v(v_gf), dist(d), dt(dt_) { }

   using LinearFormIntegrator::AssembleRHSElementVect;
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Trans,
                                       Vector &elvect)
   { MFEM_ABORT("should not be used"); }

   virtual void AssembleRHSFaceVect(const FiniteElement &el_1,
                                    const FiniteElement &el_2,
                                    FaceElementTransformations &Trans,
                                    Vector &elvect);

   void SetShiftType(int type) { e_shift_type = type; }
};

void PrintCellNumbers(const Vector &xyz, const ParFiniteElementSpace &pfes);

class PointExtractor
{
protected:
   std::ofstream fstream;

   const ParGridFunction &g;
   // -1 if the point is not in the current MPI task.
   int dof_id;

public:
   // The assumption is that the point concides with one of the DOFs of the
   // input GridFunction's nodes.
   PointExtractor(int z_id, Vector &xyz, const ParGridFunction &gf,
                  std::string filename);

   ~PointExtractor() { fstream.close(); }

   virtual double GetValue() const { return g(dof_id); }
   void WriteValue(double time);
};

class ShiftedPointExtractor : public PointExtractor
{
protected:
   const ParGridFunction &dist;
   int zone_id, dist_dof_id;

public:
   ShiftedPointExtractor(int z_id, Vector &xyz, const ParGridFunction &gf,
                         const ParGridFunction &d, std::string filename);

   virtual double GetValue() const;
   void PrintDofId() const { std::cout << dof_id << std::endl; }
};

void InitSod2Mat(ParGridFunction &rho, ParGridFunction &v,
                 ParGridFunction &e, ParGridFunction &gamma);

void InitWaterAir(ParGridFunction &rho, ParGridFunction &v,
                  ParGridFunction &e, ParGridFunction &gamma);

void InitTriPoint2Mat(ParGridFunction &rho, ParGridFunction &v,
                      ParGridFunction &e, ParGridFunction &gamma);
} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_LAGHOS_TMOP
