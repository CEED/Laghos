// Copyright (cA) 2017, Lawrence Livermore National Security, LLC. Produced at
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

#include "laghos_solver.hpp"
#include "laghos_mass_integrators.hpp"

using namespace std;
namespace mfem
{

namespace hydrodynamics
{

void InteriorVectorMassIntegrator::AssembleElementMatrix(const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat )
{
   int nd = el.GetDof();
   int vdim = Trans.GetSpaceDim();

   real_t norm;

   elmat.SetSize(nd*vdim);
   shape.SetSize(nd);
   partelmat.SetSize(nd);

   const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), order);
   
   elmat = 0.0;
   for (int s = 0; s < ir->GetNPoints(); s++)
   {
      const IntegrationPoint &ip = ir->IntPoint(s);
      el.CalcShape(ip, shape);

      Trans.SetIntPoint(&ip);
      norm = ip.weight * Trans.Weight();

      MultVVt(shape, partelmat);
      double detJ = Trans.Jacobian().Det();
      double coeff = Q.Eval(Trans, ip) / detJ / ip.weight;
      norm *= coeff;
      partelmat *= norm;

      for (int k = 0; k < vdim; k++)
	{
	  elmat.AddMatrix(partelmat, nd*k, nd*k);
	}
   }
}

void BoundaryVectorMassIntegratorV2::AssembleFaceMatrix(const FiniteElement &el1,
							const FiniteElement &el2,
							FaceElementTransformations &Tr,
							DenseMatrix &elmat)
{
   int nd = el1.GetDof();
   int vdim = Tr.GetSpaceDim() + 1;
   elmat.SetSize(nd*vdim);
   shape.SetSize(nd);
   partelmat.SetSize(nd);

   const IntegrationRule *ir = &IntRules.Get(el1.GetGeomType(), order);
   
   elmat = 0.0;
   for (int s = 0; s < ir->GetNPoints(); s++)
   {
      const IntegrationPoint &ip_f = ir->IntPoint(s);
      // Set the integration point in the face and the neighboring elements
      Tr.SetAllIntPoints(&ip_f);

      double coeff = Q.Eval(Tr, ip_f);
      const IntegrationPoint &eip1 = Tr.GetElement1IntPoint();
      ElementTransformation& Trans_el1 = Tr.GetElement1Transformation();

      Vector nor(vdim);
      CalcOrtho(Tr.Jacobian(), nor);
      double nor_norm = sqrt(nor * nor);
      Vector tn(nor);
      tn /= nor_norm;
      el1.CalcShape(eip1, shape);
      
      double detJ = Trans_el1.Jacobian().Det();
      coeff /= detJ;
      
      MultVVt(shape, partelmat);
      double penalty_mass = std::pow(el1.GetOrder(),2.0) * 1.0 / std::pow(Trans_el1.Weight(), 1.0/vdim) * perimeter * C_I * wall_bc_penalty;

      for (int dx = 0; dx < vdim; dx++)
	{
	  for (int dy = 0; dy < vdim; dy++)
	    {
	      double mcoeff = coeff * tn(dx) * tn(dy) * nor_norm * ip_f.weight * penalty_mass;
	      elmat.AddMatrix(mcoeff, partelmat, nd*dx, nd*dy); 
	    }
	}
   }
}

} // namespace hydrodynamics

} // namespace mfem
