// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
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
// testbed platforms, in support of the nation's exascale computing imperative.

#include "laghos_assembly.hpp"

using namespace mfem::future;

namespace mfem
{

namespace hydrodynamics
{

void DensityIntegrator::AssembleRHSElementVect(const FiniteElement &fe,
                                               ElementTransformation &Tr,
                                               Vector &elvect)
{
   const int nqp = IntRule->GetNPoints();
   Vector shape(fe.GetDof());
   elvect.SetSize(fe.GetDof());
   elvect = 0.0;
   for (int q = 0; q < nqp; q++)
   {
      fe.CalcShape(IntRule->IntPoint(q), shape);
      // Note that rhoDetJ = rho0DetJ0.
      shape *= qdata.rho0DetJ0w(Tr.ElementNo*nqp + q);
      elvect += shape;
   }
}

void ForceIntegrator::AssembleElementMatrix2(const FiniteElement &trial_fe,
                                             const FiniteElement &test_fe,
                                             ElementTransformation &Tr,
                                             DenseMatrix &elmat)
{
   const int e = Tr.ElementNo;
   const int nqp = IntRule->GetNPoints();
   const int dim = trial_fe.GetDim();
   const int h1dofs_cnt = test_fe.GetDof();
   const int l2dofs_cnt = trial_fe.GetDof();
   elmat.SetSize(h1dofs_cnt*dim, l2dofs_cnt);
   elmat = 0.0;
   DenseMatrix vshape(h1dofs_cnt, dim), loc_force(h1dofs_cnt, dim);
   Vector shape(l2dofs_cnt), Vloc_force(loc_force.Data(), h1dofs_cnt*dim);
   for (int q = 0; q < nqp; q++)
   {
      const IntegrationPoint &ip = IntRule->IntPoint(q);
      // Form stress:grad_shape at the current point.
      test_fe.CalcDShape(ip, vshape);
      for (int i = 0; i < h1dofs_cnt; i++)
      {
         for (int vd = 0; vd < dim; vd++) // Velocity components.
         {
            loc_force(i, vd) = 0.0;
            for (int gd = 0; gd < dim; gd++) // Gradient components.
            {
               const int eq = e*nqp + q;
               const double stressJinvT = qdata.stressJinvT(vd)(eq, gd);
               loc_force(i, vd) +=  stressJinvT * vshape(i,gd);
            }
         }
      }
      trial_fe.CalcShape(ip, shape);
      AddMultVWt(Vloc_force, shape, elmat);
   }
}

MassPAOperator::MassPAOperator(ParFiniteElementSpace &pfes,
                               const IntegrationRule &ir,
                               Coefficient &Q) :
   Operator(pfes.GetTrueVSize()),
   comm(pfes.GetParMesh()->GetComm()),
   dim(pfes.GetMesh()->Dimension()),
   NE(pfes.GetMesh()->GetNE()),
   vsize(pfes.GetVSize()),
   pabf(&pfes),
   ess_tdofs_count(0),
   ess_tdofs(0)
{
   pabf.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   pabf.AddDomainIntegrator(new mfem::MassIntegrator(Q, &ir));
   pabf.Assemble();
   pabf.FormSystemMatrix(mfem::Array<int>(), mass);
}

void MassPAOperator::SetEssentialTrueDofs(Array<int> &dofs)
{
   ess_tdofs_count = dofs.Size();
   if (ess_tdofs.Size() == 0)
   {
      int ess_tdofs_sz;
      MPI_Allreduce(&ess_tdofs_count,&ess_tdofs_sz, 1, MPI_INT, MPI_SUM, comm);
      MFEM_ASSERT(ess_tdofs_sz > 0, "ess_tdofs_sz should be positive!");
      ess_tdofs.SetSize(ess_tdofs_sz);
   }
   if (ess_tdofs_count == 0) { return; }
   ess_tdofs = dofs;
}

void MassPAOperator::EliminateRHS(Vector &b) const
{
   if (ess_tdofs_count > 0) { b.SetSubVector(ess_tdofs, 0.0); }
}

void MassPAOperator::Mult(const Vector &x, Vector &y) const
{
   mass->Mult(x, y);
   if (ess_tdofs_count > 0) { y.SetSubVector(ess_tdofs, 0.0); }
}

constexpr int VELOCITY = 0;
constexpr int SPECIFIC_INTERNAL_ENERGY = 5;
constexpr int STRESS_TENSOR = 9;

template <int DIM>
class ForcePAQFunction
{
public:
   MFEM_HOST_DEVICE inline
   auto operator()(
      const tensor<real_t, DIM, DIM> &sigma) const
   {
      return tuple{sigma};
   }
};

template <int DIM>
class ForceTPAQFunction
{
public:
   ForceTPAQFunction() = default;

   MFEM_HOST_DEVICE inline
   auto operator()(
      const tensor<real_t, DIM, DIM> &dvdxi,
      const tensor<real_t, DIM, DIM> &stressJiT) const
   {
      return tuple{ddot(stressJiT, dvdxi)};
   }
};

ForcePAOperator::ForcePAOperator(
   const QuadratureData& qdata,
   ParFiniteElementSpace& H1,
   ParFiniteElementSpace& L2,
   const IntegrationRule& ir) :
   qdata(qdata), H1(H1), L2(L2),
   ir1D(IntRules.Get(Geometry::SEGMENT, ir.GetOrder()))
{
   const int dim = H1.GetMesh()->Dimension();
   v_gf.SetSpace(&H1);
   e_gf.SetSpace(&L2);

   stress_ups = std::make_shared<UniformParameterSpace>(
                   *H1.GetMesh(), ir, dim*dim);

   stressJiT.NewMemoryAndSize(
      qdata.stressJinvT.GetMemory(),
      qdata.stressJinvT.TotalSize(),
      false
   );

   Array<int> all_domain_attr(H1.GetParMesh()->attributes.Max());
   all_domain_attr = 1;

   // F \cdot 1
   {
      tuple io = {Identity<STRESS_TENSOR>{}};
      tuple oo = {Gradient<VELOCITY>{}};

      std::vector in = {FieldDescriptor{STRESS_TENSOR, stress_ups.get()}};
      std::vector aux = {FieldDescriptor{VELOCITY, &H1}};

      force = std::make_shared<DifferentiableOperator>(in, aux, *H1.GetParMesh());
      force->SetMultLevel(DifferentiableOperator::MultLevel::LVECTOR);

      if (dim == 2)
      {
         ForcePAQFunction<2> qf;
         force->AddDomainIntegrator(qf, io, oo, ir, all_domain_attr);
      }
      else if (dim == 3)
      {
         ForcePAQFunction<3> qf;
         force->AddDomainIntegrator(qf, io, oo, ir, all_domain_attr);
      }
   }

   // F^T \cdot v
   {
      tuple io = {Gradient<VELOCITY>{}, Identity<STRESS_TENSOR>{}};
      tuple oo = {Value<SPECIFIC_INTERNAL_ENERGY>{}};

      std::vector in = {FieldDescriptor{STRESS_TENSOR, stress_ups.get()}};
      std::vector aux =
      {
         FieldDescriptor{VELOCITY, &H1},
         FieldDescriptor{SPECIFIC_INTERNAL_ENERGY, &L2}
      };

      forceT = std::make_shared<DifferentiableOperator>(in, aux, *H1.GetParMesh());
      forceT->SetMultLevel(DifferentiableOperator::MultLevel::LVECTOR);

      if (dim == 2)
      {
         ForceTPAQFunction<2> qf;
         forceT->AddDomainIntegrator(qf, io, oo, ir, all_domain_attr);
      }
      else if (dim == 3)
      {
         ForceTPAQFunction<3> qf;
         forceT->AddDomainIntegrator(qf, io, oo, ir, all_domain_attr);
      }
   }
}

void ForcePAOperator::Mult(const Vector &, Vector &y) const
{
   force->SetParameters({&v_gf});
   force->Mult(stressJiT, y);
}

void ForcePAOperator::MultTranspose(const Vector &v, Vector &y) const
{
   forceT->SetParameters({&v, &e_gf});
   forceT->Mult(stressJiT, y);
}

} // namespace hydrodynamics

} // namespace mfem
