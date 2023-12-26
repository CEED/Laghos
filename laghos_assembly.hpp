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

#ifndef MFEM_LAGHOS_ASSEMBLY
#define MFEM_LAGHOS_ASSEMBLY

#include "mfem.hpp"
#include "general/forall.hpp"
#include "linalg/dtensor.hpp"

namespace mfem
{

namespace hydrodynamics
{

// Container for all data needed at quadrature points.
struct QuadratureData
{
   // Reference to physical Jacobian for the initial mesh.
   // These are computed only at time zero and stored here.
   DenseTensor Jac0inv;

   // Quadrature data used for full/partial assembly of the force operator.
   // At each quadrature point, it combines the stress, inverse Jacobian,
   // determinant of the Jacobian and the integration weight.
   // It must be recomputed in every time step.
   DenseTensor stressJinvT;

   // Quadrature data for the boundary force;
   DenseTensor be_force_data;
   // Quadrature data for the boundary mass matrix;
   DenseTensor be_mass_data;

   // Quadrature data used for full/partial assembly of the mass matrices.
   // At time zero, we compute and store (rho0 * det(J0) * qp_weight) at each
   // quadrature point. Note the at any other time, we can compute
   // rho = rho0 * det(J0) / det(J), representing the notion of pointwise mass
   // conservation.
   Vector rho0DetJ0w;
   Vector rho0DetJ0_be;

   // Initial length scale. This represents a notion of local mesh size.
   // We assume that all initial zones have similar size.
   double h0;

   // Estimate of the minimum time step over all quadrature points. This is
   // recomputed at every time step to achieve adaptive time stepping.
   double dt_est;

   QuadratureData(int dim, int NE, int quads_per_el, int NBE, int quads_per_be)
      : Jac0inv(dim, dim, NE * quads_per_el),
        stressJinvT(NE * quads_per_el, dim, dim),
        be_force_data(NBE, quads_per_be, dim),
        be_mass_data(dim, dim, NBE * quads_per_be),
        rho0DetJ0w(NE * quads_per_el),
        rho0DetJ0_be(NBE * quads_per_be) { }
};

class BdrForceCoefficient : public VectorCoefficient
{
private:
   const QuadratureData &qdata;

public:
   BdrForceCoefficient(const QuadratureData &qd)
      : VectorCoefficient(qd.Jac0inv.SizeI()), qdata(qd) { }

   void Eval(Vector &V, ElementTransformation &Tr_f,
             const IntegrationPoint &ip_f) override
   {
     for (int d = 0; d < vdim; d++)
     {
        V(d) = qdata.be_force_data(Tr_f.ElementNo, ip_f.index, d);
     }
   }
};

class BdrMassCoefficient : public MatrixCoefficient
{
private:
   const QuadratureData &qdata;
   const int nqp_per_face;

public:
   BdrMassCoefficient(const QuadratureData &qd)
      : MatrixCoefficient(qd.Jac0inv.SizeI()),
        qdata(qd), nqp_per_face(qdata.be_force_data.SizeJ()) { }

   void Eval(DenseMatrix &K, ElementTransformation &Tr_f,
             const IntegrationPoint &ip) override
   {
      K.Set(1.0, qdata.be_mass_data(Tr_f.ElementNo * nqp_per_face + ip.index));
   }
};
// This class is used only for visualization. It assembles (rho, phi) in each
// zone, which is used by LagrangianHydroOperator::ComputeDensity to do an L2
// projection of the density.
class DensityIntegrator : public LinearFormIntegrator
{
   using LinearFormIntegrator::AssembleRHSElementVect;
private:
   const QuadratureData &qdata;

public:
   DensityIntegrator(QuadratureData &qdata) : qdata(qdata) { }
   virtual void AssembleRHSElementVect(const FiniteElement &fe,
                                       ElementTransformation &Tr,
                                       Vector &elvect);
};

// Performs full assembly for the force operator.
class ForceIntegrator : public BilinearFormIntegrator
{
private:
   const QuadratureData &qdata;
public:
   ForceIntegrator(QuadratureData &qdata) : qdata(qdata) { }
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Tr,
                                       DenseMatrix &elmat);
};

// Performs partial assembly for the force operator.
class ForcePAOperator : public Operator
{
private:
   const int dim, NE;
   const QuadratureData &qdata;
   const ParFiniteElementSpace &H1, &L2;
   const Operator *H1R, *L2R;
   const IntegrationRule &ir1D;
   const int D1D, Q1D, L1D, H1sz, L2sz;
   const DofToQuad *L2D2Q, *H1D2Q;
   mutable Vector X, Y;
public:
   ForcePAOperator(const QuadratureData&,
                   ParFiniteElementSpace&,
                   ParFiniteElementSpace&,
                   const IntegrationRule&);
   virtual void Mult(const Vector&, Vector&) const;
   virtual void MultTranspose(const Vector&, Vector&) const;
};

// Performs partial assembly for the velocity mass matrix.
class MassPAOperator : public Operator
{
private:
   const MPI_Comm comm;
   const int dim, NE, vsize;
   ParBilinearForm pabf;
   int ess_tdofs_count;
   Array<int> ess_tdofs;
   OperatorPtr mass;
public:
   MassPAOperator(ParFiniteElementSpace&, const IntegrationRule&, Coefficient&);
   virtual void Mult(const Vector&, Vector&) const;
   void MultFull(const Vector &x, Vector &y) const { mass->Mult(x, y); }
   virtual void SetEssentialTrueDofs(Array<int>&);
   virtual void EliminateRHS(Vector&) const;
   const ParBilinearForm &GetBF() const { return pabf; }
};

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_LAGHOS_ASSEMBLY
