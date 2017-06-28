// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-XXXXXX. All Rights
// reserved. See file LICENSE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project (17-SC-20-SC)
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation’s exascale computing imperative.

#ifndef MFEM_LAGHOS
#define MFEM_LAGHOS

#include "mfem.hpp"


#ifdef MFEM_USE_MPI

#include <memory>
#include <iostream>
#include <fstream>

namespace mfem
{

namespace miniapps
{

/// Visualize the given parallel grid function, using a GLVis server on the
/// specified host and port. Set the visualization window title, and optionally,
/// its geometry.
void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    ParGridFunction &gf, const char *title,
                    int x = 0, int y = 0, int w = 400, int h = 400,
                    bool vec = false);

} // namespace miniapps


namespace hydrodynamics
{

// These are defined in laghos.cpp
double rho0(const Vector &);
void v0(const Vector &, Vector &);
double e0(const Vector &);

struct Tensors1D
{
   // Values of the one-dimensional shape functions and gradients at all 1D
   // quadrature points. All sizes are (dofs1D_cnt x quads1D_cnt).
   DenseMatrix HQshape1D, HQgrad1D, LQshape1D;

   Tensors1D(int H1order, int L2order, int nqp1D);
};

struct QuadratureData
{
   // TODO: use QuadratureFunctions?

   // At each quadrature point, the stress and the Jacobian are (dim x dim)
   // matrices. They must be recomputed in every time step.
   DenseTensor stress, Jac;

   // Reference to physical Jacobian for the initial mesh. These are computed
   // only at time zero and stored here.
   DenseTensor Jac0inv;

   // TODO: have this only when PA is on.
   // Quadrature data used for partial assembly of the force operator. It must
   // be recomputed in every time step.
   DenseTensor stressJinvT;

   // At time zero, we compute and store rho0 * det(J0) at the chosen quadrature
   // points. Then at any other time, we compute rho = rho0 * det(J0) / det(J),
   // representing the notion of pointwise mass conservation.
   Vector rho0DetJ0;

   // TODO: have this only when PA is on.
   // Quadrature data used for partial assembly of the mass matrices, namely
   // (rho * detJ * qp_weight) at each quadrature point. These are computed only
   // at time zero and stored here.
   Vector rhoDetJw;

   // Initial length scale. This represents a notion of local mesh size. We
   // assume that all initial zones have similar size.
   double h0;

   // Estimate of the minimum time step over all quadrature points. This is
   // recomputed at every time step to achieve adaptive time stepping.
   double dt_est;

   QuadratureData(int dim, int nzones, int quads_per_zone)
      : stress(dim, dim, nzones * quads_per_zone),
        Jac(dim, dim, nzones * quads_per_zone),
        Jac0inv(dim, dim, nzones * quads_per_zone),
        stressJinvT(nzones * quads_per_zone, dim, dim),
        rho0DetJ0(nzones * quads_per_zone),
        rhoDetJw(nzones * quads_per_zone) { }
};


class ForceIntegrator : public BilinearFormIntegrator
{
private:
   const QuadratureData &quad_data;

public:
   ForceIntegrator(QuadratureData &quad_data_) : quad_data(quad_data_) { }

   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

class DensityIntegrator : public LinearFormIntegrator
{
private:
   const QuadratureData &quad_data;

public:
   DensityIntegrator(QuadratureData &quad_data_) : quad_data(quad_data_) { }

   virtual void AssembleRHSElementVect(const FiniteElement &fe,
                                       ElementTransformation &Tr,
                                       Vector &elvect);
};

// Partial assembly of the LagrangianHydroOperator::Force matrix.
class ForcePAOperator : public Operator
{
private:
   const int dim, nzones;

   QuadratureData *quad_data;
   ParFiniteElementSpace &H1FESpace, &L2FESpace;

   void MultQuad(const Vector &vecL2, Vector &vecH1) const;
   void MultHex(const Vector &vecL2, Vector &vecH1) const;

   void MultTransposeQuad(const Vector &vecH1, Vector &vecL2) const;
   void MultTransposeHex(const Vector &vecH1, Vector &vecL2) const;

public:
   ForcePAOperator(QuadratureData *quad_data_,
                   ParFiniteElementSpace &h1fes, ParFiniteElementSpace &l2fes);

   virtual void Mult(const Vector &vecL2, Vector &vecH1) const;

   // Here vecH1 has components for each dimension.
   virtual void MultTranspose(const Vector &vecH1, Vector &vecL2) const;

   ~ForcePAOperator() { }
};

class MassPAOperator : public Operator
{
private:
   const int dim, nzones;

   QuadratureData *quad_data;
   ParFiniteElementSpace &FESpace;

   Array<int> *ess_tdofs;

   mutable ParGridFunction xg, yg;

   void MultQuad(const Vector &x, Vector &y) const;
   void MultHex(const Vector &x, Vector &y) const;

public:
   MassPAOperator(QuadratureData *quad_data_, ParFiniteElementSpace &fes);

   // Can be used for both velocity and specific internal energy.
   // For the case of velocity, we only work with one component at a time.
   virtual void Mult(const Vector &x, Vector &y) const;

   void EliminateRHS(Array<int> &dofs, Vector &b)
   {
      ess_tdofs = &dofs;
      for (int i = 0; i < dofs.Size(); i++) { b(dofs[i]) = 0.0; }
   }
};

class LagrangianHydroOperator : public TimeDependentOperator
{
protected:
   ParFiniteElementSpace &H1FESpace;
   ParFiniteElementSpace &L2FESpace;
   mutable ParFiniteElementSpace H1compFESpace;

   Array<int> &ess_tdofs;

   const int dim, zones_cnt, l2dofs_cnt, h1dofs_cnt, source_type;
   const double cfl, gamma;
   const bool use_viscosity, p_assembly;

   // Velocity mass matrix and local inverses of the energy mass matrices. These
   // are constant in time, due to the pointwise mass conservation property.
   mutable ParBilinearForm Mv;
   DenseTensor Me_inv;

   // Integration rule for all assemblies.
   const IntegrationRule &integ_rule;

   // Data associated with each quadrature point in the mesh. These values are
   // recomputed at each time step.
   mutable QuadratureData quad_data;
   mutable bool quad_data_is_current;

   // Force matrix that combines the kinematic and thermodynamic spaces. It is
   // assembled in each time step and then it's used to compute the final
   // right-hand sides for momentum and specific internal energy.
   mutable MixedBilinearForm Force;

   // Same as above, but done through partial assembly.
   ForcePAOperator ForcePA;

   void UpdateQuadratureData(const Vector &S) const;

public:
   LagrangianHydroOperator(int size, ParFiniteElementSpace &h1_fes,
                           ParFiniteElementSpace &l2_fes,
                           Array<int> &essential_tdofs, ParGridFunction &rho0,
                           int source_type_, double cfl_,
                           double gamma_, bool visc, bool pa);

   // Solve for dx_dt, dv_dt and de_dt.
   virtual void Mult(const Vector &S, Vector &dS_dt) const;

   // Calls UpdateQuadratureData to compute the new quad_data.dt_est.
   double GetTimeStepEstimate(const Vector &S) const;
   void ResetTimeStepEstimate() const;
   void ResetQuadratureData() const { quad_data_is_current = false; }

   // The density values, which are stored only at some quadrature points, are
   // projected as a ParGridFunction.
   void ComputeDensity(ParGridFunction &rho);

   ~LagrangianHydroOperator();
};

class TaylorCoefficient : public Coefficient
{
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      Vector x(2);
      T.Transform(ip, x);
      return 3.0 / 8.0 * M_PI * ( cos(3.0*M_PI*x(0)) * cos(M_PI*x(1)) -
                                  cos(M_PI*x(0))     * cos(3.0*M_PI*x(1)) );
   }
};

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_LAGHOS
