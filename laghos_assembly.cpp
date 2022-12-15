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
#include <unordered_map>

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
	  const IntegrationPoint &ip = IntRule->IntPoint(q);
	  fe.CalcShape(ip, shape);
	  Tr.SetIntPoint (&ip);
	  // Note that rhoDetJ = rho0DetJ0.
	  shape *= qdata.rho0DetJ0(Tr.ElementNo*nqp + q) * ip.weight;
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
      DenseMatrix vshape(h1dofs_cnt, dim);
      Vector shape(l2dofs_cnt);
      for (int q = 0; q < nqp; q++)
	{
	  const IntegrationPoint &ip = IntRule->IntPoint(q);
	  Tr.SetIntPoint(&ip);
	  const DenseMatrix &Jpr = Tr.Jacobian();
	  DenseMatrix Jinv(dim);
	  Jinv = 0.0;
	  CalcInverse(Jpr, Jinv);
	  const int eq = e*nqp + q;
	  // Form stress:grad_shape at the current point.
	  test_fe.CalcDShape(ip, vshape);
	  trial_fe.CalcShape(ip, shape);
	  double pressure = p_gf.GetValue(Tr,ip);
	  double sound_speed = cs_gf.GetValue(Tr,ip);
	  DenseMatrix stress(dim), stressJiT(dim);
	  stressJiT = 0.0;
	  stress = 0.0;
	  const double rho = qdata.rho0DetJ0(eq) / Tr.Weight();
	  ComputeStress(pressure,dim,stress);
	  ComputeViscousStress(Tr, v_gf, qdata, eq, use_viscosity, use_vorticity, rho, sound_speed, dim, stress);
	  MultABt(stress, Jinv, stressJiT);
	  stressJiT *= ip.weight * Jpr.Det();
	  
	  for (int i = 0; i < h1dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  double loc_force = 0.0;
		  for (int gd = 0; gd < dim; gd++) // Gradient components.
		    {
		      loc_force += stressJiT(vd,gd) * vshape(i,gd);
		    }
		  for (int j = 0; j < l2dofs_cnt; j++){
		    elmat(i + vd * h1dofs_cnt, j) += loc_force * shape(j);
		  }
		}
	    }
	}
    }

    void VelocityBoundaryForceIntegrator::AssembleFaceMatrix(const FiniteElement &trial_fe,
							     const FiniteElement &test_fe1,
							     FaceElementTransformations &Tr,
							     DenseMatrix &elmat)
    {
      const int nqp_face = IntRule->GetNPoints();
      const int dim = trial_fe.GetDim();
      const int h1dofs_cnt = test_fe1.GetDof();
      const int l2dofs_cnt = trial_fe.GetDof();
      elmat.SetSize(h1dofs_cnt*dim, l2dofs_cnt);
      elmat = 0.0;

      DenseMatrix loc_force(h1dofs_cnt, dim);
      Vector te_shape(h1dofs_cnt),tr_shape(l2dofs_cnt), Vloc_force(loc_force.Data(), h1dofs_cnt*dim);
      const int Elem1No = Tr.ElementNo;
      te_shape = 0.0;
      tr_shape = 0.0;
      Vloc_force = 0.0;
  
      for (int q = 0; q  < nqp_face; q++)
	{
	  const int eq = Elem1No*nqp_face + q;

	  const IntegrationPoint &ip_f = IntRule->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip = Tr.GetElement1IntPoint();
	  ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
	  Trans_el1.SetIntPoint(&eip);
	  const int elementNo = Trans_el1.ElementNo;

	  Vector nor;
	  nor.SetSize(dim);
	  nor = 0.0;
	  if (dim == 1)
	    {
	      nor(0) = 2*eip.x - 1.0;
	    }
	  else
	    {
	      CalcOrtho(Tr.Jacobian(), nor);
	    }        

	  test_fe1.CalcShape(eip, te_shape);
	  loc_force = 0.0;
	  double pressure = pface_gf.GetValue(Trans_el1,eip);
	  DenseMatrix stress(dim);
	  stress = 0.0;
	  ComputeStress(pressure,dim,stress);

	  // evaluation of the normal stress at the face quadrature points
	  Vector weightedNormalStress(dim);
	  weightedNormalStress = 0.0;
	  
	  // Quadrature data for partial assembly of the force operator.
	  stress.Mult( nor, weightedNormalStress);
	  
	  for (int i = 0; i < h1dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  loc_force(i, vd) += qdata.weightedNormalStress(eq,vd) * te_shape(i);
		  //	  loc_force(i, vd) += weightedNormalStress(vd) * te_shape(i);
		}
	    }
	  trial_fe.CalcShape(eip, tr_shape);
	  AddMultVWt(Vloc_force,tr_shape,elmat);
	}
    }

    void EnergyBoundaryForceIntegrator::AssembleFaceMatrix(const FiniteElement &trial_fe,
							   const FiniteElement &test_fe1,
							   FaceElementTransformations &Tr,
							   DenseMatrix &elmat)
    {
      const int nqp_face = IntRule->GetNPoints();
      const int dim = trial_fe.GetDim();
      const int h1dofs_cnt = test_fe1.GetDof();
      const int l2dofs_cnt = trial_fe.GetDof();
      elmat.SetSize(h1dofs_cnt*dim, l2dofs_cnt);
      elmat = 0.0;
      DenseMatrix loc_force(h1dofs_cnt, dim);
      Vector te_shape(h1dofs_cnt),tr_shape(l2dofs_cnt), Vloc_force(loc_force.Data(), h1dofs_cnt*dim);
      const int Elem1No = Tr.ElementNo;
      te_shape = 0.0;
      tr_shape = 0.0;
      Vloc_force = 0.0;
      for (int q = 0; q  < nqp_face; q++)
	{
	  const int eq = Elem1No*nqp_face + q;
	     
	  const IntegrationPoint &ip_f = IntRule->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip = Tr.GetElement1IntPoint();
	  Vector nor;
	  nor.SetSize(dim);
	  nor = 0.0;
	  if (dim == 1)
	    {
	      nor(0) = 2*eip.x - 1.0;
	    }
	  else
	    {
	      CalcOrtho(Tr.Jacobian(), nor);
	    }        
	  test_fe1.CalcShape(eip, te_shape);
	  loc_force = 0.0;
	  double normalStressProjNormal = 0.0;
	  double nor_norm = 0.0;
	  for (int s = 0; s < dim; s++){
	    normalStressProjNormal += qdata.weightedNormalStress(eq,s) * nor(s);
	    nor_norm += nor(s) * nor(s);
	  }
	  nor_norm = sqrt(nor_norm);
	  normalStressProjNormal = normalStressProjNormal/nor_norm;
      
	  for (int i = 0; i < h1dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  loc_force(i, vd) += normalStressProjNormal * te_shape(i) * nor(vd)/nor_norm;
		}
	    }
	  trial_fe.CalcShape(eip, tr_shape);
	  AddMultVWt(Vloc_force,tr_shape,elmat);
	}
    }


    void NormalVelocityMassIntegrator::AssembleFaceMatrix(const FiniteElement &fe,
							  const FiniteElement &fe2,
							  FaceElementTransformations &Tr,
							  DenseMatrix &elmat)
    {
      const int nqp_face = IntRule->GetNPoints();
      const int dim = fe.GetDim();
      const int h1dofs_cnt = fe.GetDof();
      elmat.SetSize(h1dofs_cnt*dim);
      elmat = 0.0;
      Vector shape(h1dofs_cnt), loc_force2(h1dofs_cnt * dim);;
      const int Elem1No = Tr.ElementNo;
      shape = 0.0;
      DenseMatrix loc_force1(h1dofs_cnt, dim);
      Vector Vloc_force(loc_force1.Data(), h1dofs_cnt*dim);
      for (int q = 0; q  < nqp_face; q++)
	{
	  const int eq = Elem1No*nqp_face + q;
	     
	  const IntegrationPoint &ip_f = IntRule->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Tr.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip = Tr.GetElement1IntPoint();
	  Vector nor;
	  nor.SetSize(dim);
	  nor = 0.0;
	  if (dim == 1)
	    {
	      nor(0) = 2*eip.x - 1.0;
	    }
	  else
	    {
	      CalcOrtho(Tr.Jacobian(), nor);
	    }        
	  fe.CalcShape(eip, shape);
	  double nor_norm = 0.0;
	  for (int s = 0; s < dim; s++){
	    nor_norm += nor(s) * nor(s);
	  }
	  nor_norm = sqrt(nor_norm);
      
	  for (int i = 0; i < h1dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  for (int j = 0; j < h1dofs_cnt; j++)
		    {
		      for (int md = 0; md < dim; md++) // Velocity components.
			{	      
			  elmat(i + vd * h1dofs_cnt, j + md * h1dofs_cnt) += shape(i) * shape(j) * nor(vd) * (nor(md)/nor_norm) * qdata.normalVelocityPenaltyScaling * ip_f.weight;
			}
		    }
		}
	    }
	}
    }
  
  } // namespace hydrodynamics

} // namespace mfem

