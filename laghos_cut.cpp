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

#include "laghos_cut.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{

namespace hydrodynamics
{

double InterfaceCoeff::Eval(ElementTransformation &T,
                            const IntegrationPoint &ip)
{
   Vector x(3);
   T.Transform(ip, x);

   const int dim = pmesh.Dimension();

   // Modes for Taylor-Green (problem 0):
   // 0 - vertical
   // 1 - diagonal
   // 2 - circle
   const int mode_TG = 2;

   switch (problem)
   {
   case 0:
   case 1:
   {
      if (mode_TG == 0)
      {
         // The domain area for the TG is 1.
         const double dx = sqrt(1.0 / glob_NE);

         // The middle of the element after x = 0.5.
         return tanh(x(0) - (0.5 + 0.5 * dx));
      }
      else if (mode_TG == 1)
      {
         return tanh(x(0) - x(1));
      }
      else if (mode_TG == 2)
      {
         double center[3] = {0.5, 0.5, 0.5};
         double rad = 0.0;
         for (int d = 0; d < dim; d++)
         {
            rad += (x(d) - center[d]) * (x(d) - center[d]);
         }
         rad = sqrt(rad + 1e-16);
         return tanh(rad - 0.3);
      }
      else { MFEM_ABORT("wrong TG mode"); return 0.0; }
   }
   default: MFEM_ABORT("error"); return 0.0;
   }
}

int ElementMarker::GetMaterialID(int el_id)
{
   IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);

   const ParFiniteElementSpace &pfes = *ls.ParFESpace();
   const FiniteElement *fe = pfes.GetFE(el_id);
   Vector ls_vals;
   const IntegrationRule &ir = IntRulesLo.Get(fe->GetGeomType(), 20);

   ls.GetValues(el_id, ir, ls_vals);
   ElementTransformation *Tr = pfes.GetMesh()->GetElementTransformation(el_id);
   double volume_1 = 0.0, volume_2 = 0.0, volume = 0.0;
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      Tr->SetIntPoint(&ip);

      volume += ip.weight * Tr->Weight();
      if (ls_vals(q) + 1e-12 < 0.0) { volume_1 += ip.weight * Tr->Weight(); }
      if (ls_vals(q) - 1e-12 > 0.0) { volume_2 += ip.weight * Tr->Weight(); }
   }

   if (volume_1 == 0.0) { return 20; }
   if (volume_2 == 0.0) { return 10; }
   return 15;
}

// Initially the energies are initialized as the single material version, due
// the special Bernstein projection. This only zeroes them in the empty zones.
void InitTG2Mat(MaterialData &mat_data)
{
   ParFiniteElementSpace &pfes = *mat_data.e_1.ParFESpace();
   const int NE    = pfes.GetNE();
   const int ndofs = mat_data.e_1.Size() / NE;

   mat_data.rho0_1  = 0.0;
   mat_data.rho0_2  = 0.0;
   for (int e = 0; e < NE; e++)
   {
      const int attr = pfes.GetParMesh()->GetAttribute(e);

      if (attr == 10)
      {
         for (int i = 0; i < ndofs; i++) { mat_data.e_2(e*ndofs + i) = 0.0; }
      }
      if (attr == 20)
      {
         for (int i = 0; i < ndofs; i++) { mat_data.e_1(e*ndofs + i) = 0.0; }
      }

      if (attr == 10 || attr == 15)
      {
         mat_data.gamma_1 = 5.0 / 3.0;
         for (int i = 0; i < ndofs; i++)
         {
            mat_data.rho0_1(e*ndofs + i) = 1.0;
         }
      }

      if (attr == 15 || attr == 20)
      {
         mat_data.gamma_2 = 5.0 / 3.0;
         for (int i = 0; i < ndofs; i++)
         {
            mat_data.rho0_2(e*ndofs + i) = 1.0;
         }
      }
   }
}

void CutMassIntegrator::AssembleElementMatrix(const FiniteElement &el,
                                              ElementTransformation &Trans,
                                              DenseMatrix &elmat)
{
   IntRule = irules[Trans.ElementNo];
   MassIntegrator::AssembleElementMatrix(el, Trans, elmat);
}

void CutVectorMassIntegrator::AssembleElementMatrix(const FiniteElement &el,
                                                    ElementTransformation &Trans,
                                                    DenseMatrix &elmat)
{
   IntRule = irules[Trans.ElementNo];
   VectorMassIntegrator::AssembleElementMatrix(el, Trans, elmat);
}

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI
