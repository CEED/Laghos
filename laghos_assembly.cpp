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
void WeightedVectorMassIntegrator::AssembleElementMatrix(const FiniteElement &el,
                                                         ElementTransformation &Trans,
                                                         DenseMatrix &elmat)
{
   const int dim = el.GetDim();
   int dof = el.GetDof();

   elmat.SetSize (dof*dim);
   elmat = 0.0;

   Vector shape(dof);
   shape = 0.0;

   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el, Trans);

   for (int q = 0; q < ir->GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir->IntPoint(q);
      // Set the integration point in the face and the neighboring elements
      Trans.SetIntPoint(&ip);
      el.CalcShape(ip, shape);

      double volumeFraction = alpha.GetValue(Trans, ip);
      double density = rho_gf.GetValue(Trans, ip);

      for (int i = 0; i < dof; i++)
      {
         for (int k = 0; k < dim; k++)
         {
            for (int j = 0; j < dof; j++)
            {
               elmat(i + k * dof, j + k * dof) += shape(i) * shape(j) * ip.weight * volumeFraction * Trans.Weight() * density;
            }
         }
      }
   }
}

const IntegrationRule &WeightedVectorMassIntegrator::GetRule(
      const FiniteElement &trial_fe,
      const FiniteElement &test_fe,
      ElementTransformation &Trans)
{
   int order = Trans.OrderGrad(&trial_fe) + test_fe.GetOrder() + Trans.OrderJ();
   return IntRules.Get(trial_fe.GetGeomType(), 2*order);
}


void WeightedMassIntegrator::AssembleElementMatrix(const FiniteElement &el,
                                                   ElementTransformation &Trans,
                                                   DenseMatrix &elmat)
{
   const int dim = el.GetDim();
   int dof = el.GetDof();

   elmat.SetSize (dof);
   elmat = 0.0;

   Vector shape(dof);
   shape = 0.0;

   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el, Trans);

   for (int q = 0; q < ir->GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir->IntPoint(q);
      // Set the integration point in the face and the neighboring elements
      Trans.SetIntPoint(&ip);
      el.CalcShape(ip, shape);

      double volumeFraction = alpha.GetValue(Trans, ip);
      double density = rho_gf.GetValue(Trans, ip);

      for (int i = 0; i < dof; i++)
      {
         for (int j = 0; j < dof; j++)
         {
            elmat(i, j) += shape(i) * shape(j) * ip.weight * volumeFraction * Trans.Weight() * density;
         }
      }
   }
}


const IntegrationRule &WeightedMassIntegrator::GetRule(
      const FiniteElement &trial_fe,
      const FiniteElement &test_fe,
      ElementTransformation &Trans)
{
   int order = Trans.OrderGrad(&trial_fe) + test_fe.GetOrder() + Trans.OrderJ();
   return IntRules.Get(trial_fe.GetGeomType(), 2*order);
}

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
      double rho0DetJ0 = rho0DetJ0_gf.GetValue(Tr, ip);

      // Note that rhoDetJ = rho0DetJ0.
      shape *= rho0DetJ0 * ip.weight;
      elvect += shape;
   }
}

void SourceForceIntegrator::AssembleRHSElementVect(const FiniteElement &fe,
                                                   ElementTransformation &Tr,
                                                   Vector &elvect)
{
   const int dim = fe.GetDim();
   elvect.SetSize(fe.GetDof()*dim);
   elvect = 0.0;

   if (accel_src_gf != NULL){
      const int nqp = IntRule->GetNPoints();
      Vector shape(fe.GetDof());
      int dofs = fe.GetDof();
      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = IntRule->IntPoint(q);
         fe.CalcShape(ip, shape);
         Tr.SetIntPoint (&ip);
         double rho = rho_gf.GetValue(Tr, ip);
         Vector accel_vec;
         accel_src_gf->GetVectorValue(Tr.ElementNo, ip, accel_vec);
         for (int i = 0; i < dofs; i++){
            for (int vd = 0; vd < dim; vd++){
               elvect(i + vd * dofs) += shape(i) * rho * ip.weight * accel_vec(vd) * Tr.Weight();
            }
         }
      }
   }
}

void ForceIntegrator::AssembleRHSElementVect(const FiniteElement &el,
                                             ElementTransformation &Tr,
                                             Vector &elvect)
{
   const int e = Tr.ElementNo;
   const int nqp = IntRule->GetNPoints();
   const int dim = el.GetDim();
   const int h1dofs_cnt = el.GetDof();
   elvect.SetSize(h1dofs_cnt*dim);
   elvect = 0.0;
   DenseMatrix vshape(h1dofs_cnt, dim);
   // std::cout << " num " << nqp << std::endl;
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
      el.CalcDShape(ip, vshape);
      double pressure = p_gf.GetValue(Tr,ip);
      double sound_speed = cs_gf.GetValue(Tr,ip);
      DenseMatrix stress(dim), stressJiT(dim);
      stressJiT = 0.0;
      stress = 0.0;
      double volumeFraction = alpha.GetValue(Tr, ip);

      Vector Jac0inv_vec(dim*dim);
      Jac0inv_vec = 0.0;
      Jac0inv_gf.GetVectorValue(Tr.ElementNo,ip,Jac0inv_vec);
      DenseMatrix Jac0inv(dim);
      ConvertVectorToDenseMatrix(dim, Jac0inv_vec, Jac0inv);

      const double rho = rho_gf.GetValue(Tr,ip);
      ComputeStress(pressure,dim,stress);
      ComputeViscousStress(Tr, v_gf, Jac0inv, h0, use_viscosity, use_vorticity, rho, sound_speed, dim, stress);
      MultABt(stress, Jinv, stressJiT);
      stressJiT *= ip.weight * Jpr.Det();

      for (int i = 0; i < h1dofs_cnt; i++)
      {
         for (int vd = 0; vd < dim; vd++) // Velocity components.
         {
            for (int gd = 0; gd < dim; gd++) // Gradient components.
            {
               elvect(i + vd * h1dofs_cnt) -= stressJiT(vd,gd) * vshape(i,gd) * volumeFraction;
            }
         }
      }
   }
}

void EnergyForceIntegrator::AssembleRHSElementVect(const FiniteElement &el,
                                                   ElementTransformation &Tr,
                                                   Vector &elvect)
{
   if (Vnpt_gf != NULL){
      const int e = Tr.ElementNo;
      const int nqp = IntRule->GetNPoints();
      // phi test space
      const int dim = el.GetDim();
      const int l2dofs_cnt = el.GetDof();
      elvect.SetSize(l2dofs_cnt);
      elvect = 0.0;
      ////

      // v trial space
      ParFiniteElementSpace *v_fespace = Vnpt_gf->ParFESpace();
      ElementTransformation &TrV = *v_fespace->GetElementTransformation(e);
      ///
      Vector te_shape(l2dofs_cnt);
      te_shape = 0.0;
      for (int q = 0; q < nqp; q++)
      {
         te_shape = 0.0;
         const IntegrationPoint &ip = IntRule->IntPoint(q);
         Tr.SetIntPoint(&ip);
         el.CalcShape(ip, te_shape);
         const int eq = e*nqp + q;

         // Form stress:grad_shape at the current point.
         const DenseMatrix &Jpr = Tr.Jacobian();
         DenseMatrix Jinv(dim);
         Jinv = 0.0;
         CalcInverse(Jpr, Jinv);
         double pressure = p_gf.GetValue(Tr,ip);
         double sound_speed = cs_gf.GetValue(Tr,ip);
         DenseMatrix stress(dim), stressJiT(dim);
         stressJiT = 0.0;
         stress = 0.0;
         double volumeFraction = alpha.GetValue(Tr, ip);

         Vector Jac0inv_vec(dim*dim);
         Jac0inv_vec = 0.0;
         Jac0inv_gf.GetVectorValue(Tr.ElementNo,ip,Jac0inv_vec);
         DenseMatrix Jac0inv(dim);
         ConvertVectorToDenseMatrix(dim, Jac0inv_vec, Jac0inv);
         const double rho = rho_gf.GetValue(Tr,ip);

         ComputeStress(pressure,dim,stress);
         ComputeViscousStress(Tr, v_gf, Jac0inv, h0, use_viscosity, use_vorticity, rho, sound_speed, dim, stress);
         stress *= ip.weight * Jpr.Det();
         //
         TrV.SetIntPoint(&ip);
         DenseMatrix vGradShape;
         Vnpt_gf->GetVectorGradient(TrV, vGradShape);

         // Calculating grad v
         double gradVContractedStress = 0.0;
         for (int s = 0; s < dim; s++){
            for (int k = 0; k < dim; k++){
               gradVContractedStress += stress(s,k) * vGradShape(s,k);
            }
         }

         for (int i = 0; i < l2dofs_cnt; i++)
         {
            elvect(i) += gradVContractedStress * te_shape(i) * volumeFraction;
         }
      }
   }
   else{
      const int l2dofs_cnt = el.GetDof();
      elvect.SetSize(l2dofs_cnt);
      elvect = 0.0;
   }
}

void VelocityBoundaryForceIntegrator::AssembleRHSElementVect(const FiniteElement &el,
                                                             FaceElementTransformations &Tr,
                                                             Vector &elvect)
{
   const int nqp_face = IntRule->GetNPoints();
   const int dim = el.GetDim();
   const int h1dofs_cnt = el.GetDof();
   elvect.SetSize(h1dofs_cnt*dim);
   elvect = 0.0;
   Vector te_shape(h1dofs_cnt);
   te_shape = 0.0;
   for (int q = 0; q  < nqp_face; q++)
   {
      const IntegrationPoint &ip_f = IntRule->IntPoint(q);
      // Set the integration point in the face and the neighboring elements
      Tr.SetAllIntPoints(&ip_f);

      const IntegrationPoint &eip = Tr.GetElement1IntPoint();
      ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
      Trans_el1.SetIntPoint(&eip);
      const int elementNo = Trans_el1.ElementNo;
      const int eq = elementNo*nqp_face + q;
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

      el.CalcShape(eip, te_shape);
      double pressure = pface_gf.GetValue(Trans_el1,eip);
      double sound_speed = csface_gf.GetValue(Trans_el1,eip);

      double nor_norm = 0.0;
      for (int s = 0; s < dim; s++){
         nor_norm += nor(s) * nor(s);
      }
      nor_norm = sqrt(nor_norm);

      Vector tn(dim);
      tn = 0.0;
      tn = nor;
      tn /= nor_norm;

      DenseMatrix stress(dim);
      stress = 0.0;
      const double rho = rho0DetJ0face_gf.GetValue(Trans_el1,eip);

      ComputeStress(pressure,dim,stress);

      // evaluation of the normal stress at the face quadrature points
      Vector weightedNormalStress(dim);
      weightedNormalStress = 0.0;

      // Quadrature data for partial assembly of the force operator.
      stress.Mult( tn, weightedNormalStress);

      for (int i = 0; i < h1dofs_cnt; i++)
      {
         for (int vd = 0; vd < dim; vd++) // Velocity components.
         {
            elvect(i + vd * h1dofs_cnt) += weightedNormalStress(vd) * te_shape(i) * ip_f.weight * nor_norm;
         }
      }
   }
}

void VelocityBoundaryForceIntegrator::AssembleRHSElementVect(
      const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("DGDirichletLFIntegrator::AssembleRHSElementVect");
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
   Vector shape(h1dofs_cnt);
   shape = 0.0;
   for (int q = 0; q  < nqp_face; q++)
   {
      const IntegrationPoint &ip_f = IntRule->IntPoint(q);
      // Set the integration point in the face and the neighboring elements
      Tr.SetAllIntPoints(&ip_f);
      const IntegrationPoint &eip = Tr.GetElement1IntPoint();

      ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();
      Trans_el1.SetIntPoint(&eip);
      const double alpha_0 = penaltyParameter * perimeter /
                             std::pow(Trans_el1.Weight(), 1.0/dim);

      Vector nor(dim);
      CalcOrtho(Tr.Jacobian(), nor);

      double nor_norm = 0.0;
      for (int s = 0; s < dim; s++) { nor_norm += nor(s) * nor(s); }
      nor_norm = sqrt(nor_norm);

      Vector tn(nor);
      tn /= nor_norm;
      double penaltyVal = 0.0;

      penaltyVal = alpha_0 * globalmax_rho * perimeter;

      fe.CalcShape(eip, shape);
      for (int i = 0; i < h1dofs_cnt; i++)
      {
         for (int vd = 0; vd < dim; vd++) // Velocity components.
         {
            for (int j = 0; j < h1dofs_cnt; j++)
            {
               for (int md = 0; md < dim; md++) // Velocity components.
               {
                  elmat(i + vd * h1dofs_cnt, j + md * h1dofs_cnt) +=
                     shape(i) * shape(j) * tn(vd) * tn(md) * penaltyVal * ip_f.weight * nor_norm;
               }
            }
         }
      }
   }
}


void DiffusionNormalVelocityIntegrator::AssembleRHSElementVect(const FiniteElement &el,
                                                               FaceElementTransformations &Tr,
                                                               Vector &elvect)
{
   const int nqp_face = IntRule->GetNPoints();
   const int dim = el.GetDim();
   const int h1dofs_cnt = el.GetDof();
   elvect.SetSize(h1dofs_cnt*dim);
   elvect = 0.0;
   Vector shape(h1dofs_cnt);
   shape = 0.0;
   for (int q = 0; q  < nqp_face; q++)
   {
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
      CalcOrtho(Tr.Jacobian(), nor);

      double nor_norm = 0.0;
      for (int s = 0; s < dim; s++) { nor_norm += nor(s) * nor(s); }
      nor_norm = sqrt(nor_norm);

      Vector tn(nor);
      tn /= nor_norm;

      double density_el1 = rhoface_gf.GetValue(Trans_el1,eip);

      Vector vShape;
      v_gf.GetVectorValue(elementNo, eip, vShape);
      double vDotn = 0.0;
      for (int s = 0; s < dim; s++)
      {
         vDotn += vShape(s) * nor(s)/nor_norm;
      }

      double cs_el1 = csface_gf.GetValue(Trans_el1,eip);

      double penaltyVal =  penaltyParameter * density_el1 * cs_el1;

      el.CalcShape(eip, shape);
      double pressure = pface_gf.GetValue(Trans_el1,eip);
      DenseMatrix stress(dim);
      stress = 0.0;
      ComputeStress(pressure,dim,stress);
      // evaluation of the normal stress at the face quadrature points
      Vector weightedNormalStress(dim);
      weightedNormalStress = 0.0;
      // Quadrature data for partial assembly of the force operator.
      stress.Mult(tn, weightedNormalStress);

      for (int i = 0; i < h1dofs_cnt; i++)
      {
         for (int vd = 0; vd < dim; vd++) // Velocity components.
         {
            elvect(i + vd * h1dofs_cnt) -= shape(i) * vDotn * tn(vd) * penaltyVal * ip_f.weight * nor_norm;
            elvect(i + vd * h1dofs_cnt) += weightedNormalStress(vd) * shape(i) * ip_f.weight * nor_norm;
         }
      }
   }
}

void EnergyPenaltyBLFI::AssembleRHSElementVect(const FiniteElement &el,
                                               FaceElementTransformations &Tr,
                                               Vector &elvect)
{
   const int l2dofs_cnt = el.GetDof();

   if (Vnpt_gf == nullptr)
   {
      elvect.SetSize(l2dofs_cnt);
      elvect = 0.0;
      return;
   }

   const int nqp_face = IntRule->GetNPoints();
   const int dim = el.GetDim();
   elvect.SetSize(l2dofs_cnt);
   elvect = 0.0;
   Vector shape(l2dofs_cnt);
   shape = 0.0;
   for (int q = 0; q  < nqp_face; q++)
   {
      const IntegrationPoint &ip_f = IntRule->IntPoint(q);
      // Set the integration point in the face and the neighboring elements
      Tr.SetAllIntPoints(&ip_f);
      const IntegrationPoint &eip = Tr.GetElement1IntPoint();
      ElementTransformation &Trans_el1 = Tr.GetElement1Transformation();

      Trans_el1.SetIntPoint(&eip);

      Vector nor(dim);
      nor = 0.0;
      CalcOrtho(Tr.Jacobian(), nor);

      double nor_norm = 0.0;
      for (int s = 0; s < dim; s++) { nor_norm += nor(s) * nor(s); }
      nor_norm = sqrt(nor_norm);

      Vector tn(nor);
      tn /= nor_norm;

      Vector vShape_npt;
      Vnpt_gf->GetVectorValue(Trans_el1.ElementNo, eip, vShape_npt);
      double vDotn_npt = 0.0;
      for (int s = 0; s < dim; s++)
      {
         vDotn_npt += vShape_npt(s) * nor(s)/nor_norm;
      }
      Vector vShape;
      v_gf.GetVectorValue(Trans_el1.ElementNo, eip, vShape);
      double vDotn = 0.0;
      for (int s = 0; s < dim; s++)
      {
         vDotn += vShape(s) * nor(s)/nor_norm;
      }

      double density_el1 = rhoface_gf.GetValue(Trans_el1,eip);
      double cs_el1 = csface_gf.GetValue(Trans_el1,eip);
      double penaltyVal = penaltyParameter * density_el1 * cs_el1;

      el.CalcShape(eip, shape);
      double pressure = pface_gf.GetValue(Trans_el1,eip);
      DenseMatrix stress(dim);
      stress = 0.0;
      ComputeStress(pressure,dim,stress);
      // evaluation of the normal stress at the face quadrature points
      Vector weightedNormalStress(dim);
      weightedNormalStress = 0.0;
      stress.Mult(tn, weightedNormalStress);

      double normalStressProjNormal = 0.0;
      for (int s = 0; s < dim; s++){
         normalStressProjNormal += weightedNormalStress(s) * tn(s);
      }
      normalStressProjNormal = normalStressProjNormal*nor_norm;

      for (int i = 0; i < l2dofs_cnt; i++)
      {
         elvect(i) += shape(i) * vDotn * vDotn_npt * penaltyVal * ip_f.weight * nor_norm;
         elvect(i) -= normalStressProjNormal * shape(i) * ip_f.weight * vDotn_npt;
      }
   }
}

} // namespace hydrodynamics
} // namespace mfem
