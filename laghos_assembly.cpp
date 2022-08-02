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

void VelocityBoundaryForceIntegrator::AssembleFaceMatrix(const FiniteElement &trial_fe,
                                             const FiniteElement &test_fe1,
					     FaceElementTransformations &Tr,
                                             DenseMatrix &elmat)
{
  int elem1_status = 1;
  if (elemStatus.Size() > 0){
    elem1_status = elemStatus[Tr.Elem1No];
  }
  const int nqp_face = IntRule->GetNPoints();
  const int dim = trial_fe.GetDim();
  const int h1dofs_cnt = test_fe1.GetDof();
  const int l2dofs_cnt = trial_fe.GetDof();
  elmat.SetSize(h1dofs_cnt*dim, l2dofs_cnt);
  elmat = 0.0;
  
  if (elem1_status == 1){  
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
	test_fe1.CalcShape(eip, te_shape);
	loc_force = 0.0;
	for (int i = 0; i < h1dofs_cnt; i++)
	  {
	    for (int vd = 0; vd < dim; vd++) // Velocity components.
	      {
		loc_force(i, vd) += qdata.weightedNormalStress(eq,vd) * te_shape(i);
	      }
	  }
	trial_fe.CalcShape(eip, tr_shape);
	AddMultVWt(Vloc_force,tr_shape,elmat);
      }
  }
}
    

void EnergyBoundaryForceIntegrator::AssembleFaceMatrix(const FiniteElement &trial_fe,
                                             const FiniteElement &test_fe1,
					     FaceElementTransformations &Tr,
                                             DenseMatrix &elmat)
{
  int elem1_status = 1;
  if (elemStatus.Size() > 0){
    elem1_status = elemStatus[Tr.Elem1No];
  }
  const int nqp_face = IntRule->GetNPoints();
  const int dim = trial_fe.GetDim();
  const int h1dofs_cnt = test_fe1.GetDof();
  const int l2dofs_cnt = trial_fe.GetDof();
  elmat.SetSize(h1dofs_cnt*dim, l2dofs_cnt);
  elmat = 0.0;
  
  if (elem1_status == 1){ 
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
}

void NormalVelocityMassIntegrator::AssembleFaceMatrix(const FiniteElement &fe,
						      const FiniteElement &fe2,
                                             FaceElementTransformations &Tr,
                                             DenseMatrix &elmat)
{
  int elem1_status = 1;
  if (elemStatus.Size() > 0){
    elem1_status = elemStatus[Tr.Elem1No];
  }
  const int nqp_face = IntRule->GetNPoints();
  const int dim = fe.GetDim();
  const int h1dofs_cnt = fe.GetDof();
  elmat.SetSize(h1dofs_cnt*dim);
  elmat = 0.0;
  
  if (elem1_status == 1){
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
			elmat(i + vd * h1dofs_cnt, j + md * h1dofs_cnt) += shape(i) * shape(j) * nor(vd) * (nor(md)/nor_norm) * qdata.normalVelocityPenaltyScaling(eq);
		      }
		  }
	      }
	  }
      }
  }
}
  
  void ShiftedVelocityBoundaryForceIntegrator::AssembleFaceMatrix(const FiniteElement &trial_fe1,
								  const FiniteElement &trial_fe2,
								  const FiniteElement &test_fe1,
								  const FiniteElement &test_fe2,
								  FaceElementTransformations &Trans,
								  DenseMatrix &elmat)
  {
    MPI_Comm comm = pmesh->GetComm();
    int myid;
    MPI_Comm_rank(comm, &myid);
    //  std::cout << " I AM IN ASSEMBLE FACE " << std::endl;
  int elem1 = Trans.Elem1No;
  int elem2 = Trans.Elem2No;
  int elemStatus1 = elemStatus[elem1];
  int elemStatus2 = elemStatus[elem2];
  int NEproc = pmesh->GetNE();
    
  const int nqp_face = IntRule->GetNPoints();
  const int dim = test_fe1.GetDim();
  const int h1dofs_cnt = test_fe1.GetDof();
  const int l2dofs_cnt = trial_fe1.GetDof();
  const int h1dofs2_cnt = test_fe2.GetDof();
  const int l2dofs2_cnt = trial_fe2.GetDof();

  elmat.SetSize((h1dofs_cnt+h1dofs2_cnt)*dim, l2dofs_cnt+l2dofs2_cnt);
  elmat = 0.0;
  DenseMatrix loc_force(h1dofs_cnt, dim);
  Vector te_shape(h1dofs_cnt),tr_shape(l2dofs_cnt), Vloc_force(loc_force.Data(), h1dofs_cnt*dim);
  const int e = Trans.ElementNo;
  te_shape = 0.0;
  tr_shape = 0.0;
  Vloc_force = 0.0;
  
  if (faceTags[e] == 5){
    //   std::cout << " myid " << myid << " face " << std::endl;
    if (elemStatus1 == AnalyticalGeometricShape::SBElementType::INSIDE){
      //  std::cout << " myid " << myid << " i am in one " << std::endl;
      DenseMatrix temp_elmat;
      temp_elmat.SetSize(h1dofs_cnt*dim, l2dofs_cnt);
      temp_elmat = 0.0;

      for (int q = 0; q  < nqp_face; q++)
	{
	  const int eq = e*nqp_face + q;
	  const IntegrationPoint &ip_f = IntRule->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Trans.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip = Trans.GetElement1IntPoint();
	  test_fe1.CalcShape(eip, te_shape);
	  loc_force = 0.0;
	  for (int i = 0; i < h1dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  loc_force(i, vd) += qdata.weightedNormalStress(eq,vd) * te_shape(i);
		}
	    }
	  trial_fe1.CalcShape(eip, tr_shape);
	  AddMultVWt(Vloc_force,tr_shape,temp_elmat);
	}
      elmat.CopyMN(temp_elmat, 0, 0);
    }
    else if ((elem2 < NEproc) && (elemStatus2 == AnalyticalGeometricShape::SBElementType::INSIDE)){
      //      std::cout << " myid " << myid << " i am in two " << std::endl;	 
      DenseMatrix temp_elmat;
      temp_elmat.SetSize(h1dofs_cnt*dim, l2dofs_cnt);
      temp_elmat = 0.0;

      int h1dofs_offset = h1dofs_cnt * dim;
      int l2dofs_offset = l2dofs_cnt;
      for (int q = 0; q < nqp_face; q++)
	{
	  const int eq = e*nqp_face + q;
	  
	  const IntegrationPoint &ip_f = IntRule->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Trans.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip = Trans.GetElement2IntPoint();
	  test_fe2.CalcShape(eip, te_shape);
	  loc_force = 0.0;
	  for (int i = 0; i < h1dofs_cnt; i++)
	    {
	  for (int vd = 0; vd < dim; vd++) // Velocity components.
	    {
	      loc_force(i, vd) += qdata.weightedNormalStress(eq,vd) * te_shape(i);
	    }
	    }
	  trial_fe2.CalcShape(eip, tr_shape);
	  AddMultVWt(Vloc_force,tr_shape,temp_elmat);
	}
      elmat.CopyMN(temp_elmat, h1dofs_offset, l2dofs_offset);
    }
  }
}

void ShiftedEnergyBoundaryForceIntegrator::AssembleFaceMatrix(const FiniteElement &trial_fe1,
							      const FiniteElement &trial_fe2,
							      const FiniteElement &test_fe1,
							      const FiniteElement &test_fe2,
							      FaceElementTransformations &Trans,
							      DenseMatrix &elmat)
{
  const DenseMatrix& quadDist = analyticalSurface->GetQuadratureDistance();
  const DenseMatrix& quadTrueNorm = analyticalSurface->GetQuadratureTrueNormal();
  
  int elem1 = Trans.Elem1No;
  int elem2 = Trans.Elem2No;
  int elemStatus1 = elemStatus[elem1];
  int elemStatus2 = elemStatus[elem2];
  int NEproc = pmesh->GetNE();
  
  const int nqp_face = IntRule->GetNPoints();
  const int dim = trial_fe1.GetDim();
  const int h1dofs_cnt = test_fe1.GetDof();
  const int l2dofs_cnt = trial_fe1.GetDof();
  const int h1dofs2_cnt = test_fe2.GetDof();
  const int l2dofs2_cnt = trial_fe2.GetDof();
  elmat.SetSize((h1dofs_cnt+h1dofs2_cnt)*dim, l2dofs_cnt+l2dofs2_cnt);
  //  elmat.SetSize(h1dofs_cnt*dim, l2dofs_cnt);
  elmat = 0.0;
 
  DenseMatrix loc_force(h1dofs_cnt, dim), dshapephys(h1dofs_cnt, dim);
  Vector te_shape(h1dofs_cnt), tr_shape(l2dofs_cnt), Vloc_force(loc_force.Data(), h1dofs_cnt*dim), dshapedn(h1dofs_cnt), dshapephysdd(h1dofs_cnt);
  const int e = Trans.ElementNo;
  te_shape = 0.0;
  tr_shape = 0.0;
  Vloc_force = 0.0;
  if (faceTags[e] == 5){
    if (elemStatus1 == AnalyticalGeometricShape::SBElementType::INSIDE){
      Array<DenseMatrix *> dkphi_dxk;
      DenseMatrix grad_phys;
      Vector Factorial;
      Array<DenseMatrix *> grad_phys_dir;
      test_fe1.ProjectGrad(test_fe1, *Trans.Elem1, grad_phys);
      DenseMatrix grad_work;
      grad_phys_dir.SetSize(dim); // NxN matrices for derivative in each direction
      for (int i = 0; i < dim; i++)
      {
         grad_phys_dir[i] = new DenseMatrix(h1dofs_cnt, h1dofs_cnt);
         grad_phys_dir[i]->CopyRows(grad_phys, i*h1dofs_cnt, (i+1)*h1dofs_cnt-1);
      }
      DenseMatrix grad_phys_work = grad_phys;
      grad_phys_work.SetSize(h1dofs_cnt, h1dofs_cnt*dim);

      dkphi_dxk.SetSize(1);
      int sz1 = pow(dim, 1);
      dkphi_dxk[0] = new DenseMatrix(h1dofs_cnt, h1dofs_cnt*sz1*dim);
      int loc_col_per_dof = sz1;
      int tot_col_per_dof = loc_col_per_dof*dim;
      for (int k = 0; k < dim; k++)
	{
	  grad_work.SetSize(h1dofs_cnt, h1dofs_cnt*sz1);
	  // grad_work[k] has derivative in kth direction for each DOF.
	  // grad_work[0] has d^2phi/dx^2 and d^2phi/dxdy terms and
	  // grad_work[1] has d^2phi/dydx and d^2phi/dy2 terms for each dof
	  Mult(*grad_phys_dir[k], grad_phys_work, grad_work);
            // Now we must place columns for each dof together so that they are
            // in order: d^2phi/dx^2, d^2phi/dxdy, d^2phi/dydx, d^2phi/dy2.
	  for (int j = 0; j < h1dofs_cnt; j++)
            {
	      for (int d = 0; d < loc_col_per_dof; d++)
		{
                  Vector col;
                  grad_work.GetColumn(j*loc_col_per_dof+d, col);
                  dkphi_dxk[0]->SetCol(j*tot_col_per_dof+k*loc_col_per_dof+d, col);
		}
            }
	}

      for (int i = 0; i < grad_phys_dir.Size(); i++)
      {
         delete grad_phys_dir[i];
      }

      Factorial.SetSize(1);
      Factorial(0) = 2;

      DenseMatrix q_hess_dn(dim, h1dofs_cnt);
      Vector q_hess_dn_work(q_hess_dn.GetData(), h1dofs_cnt*dim);
      Vector q_hess_dot_d(h1dofs_cnt);

      q_hess_dn = 0.0;
      q_hess_dn_work = 0.0;
      q_hess_dot_d = 0.0;
	  
      DenseMatrix temp_elmat;
      temp_elmat.SetSize(h1dofs_cnt*dim, l2dofs_cnt);
      temp_elmat = 0.0;

      for (int q = 0; q  < nqp_face; q++)
	{
	  
	  Vector D(dim);
	  Vector tN(dim);
	  D = 0.0;
	  tN = 0.0;
	  const int eq = e*nqp_face + q;
	  const IntegrationPoint &ip_f = IntRule->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Trans.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip = Trans.GetElement1IntPoint();
	  Vector nor;
	  nor.SetSize(dim);
	  nor = 0.0;
	  if (dim == 1)
	    {
	      nor(0) = 2*eip.x - 1.0;
	    }
	  else
	    {
	      CalcOrtho(Trans.Jacobian(), nor);
	    }
	  dshapephys = 0.0;
	  te_shape = 0.0;
	  test_fe1.CalcShape(eip, te_shape);
	  test_fe1.CalcPhysDShape(*(Trans.Elem1), dshapephys);

	  loc_force = 0.0;
	  double normalStressProjNormal = 0.0;
	  double nor_norm = 0.0;
	  for (int s = 0; s < dim; s++){
	    normalStressProjNormal += qdata.weightedNormalStress(eq,s) * nor(s);
	    nor_norm += nor(s) * nor(s);
	    D(s) = quadDist(e*nqp_face+q,s);
	    tN(s) = quadTrueNorm(e*nqp_face+q,s);
	    //    tN(s) = nor(s);
	  }
	  nor_norm = sqrt(nor_norm);
	  //  tN /= nor_norm;
	  normalStressProjNormal = normalStressProjNormal/nor_norm;
	  double ntildaDotTrueN = 0;
	  for (int s = 0; s < dim; s++){
	    ntildaDotTrueN += tN(s) * nor(s)/nor_norm;
	  }
	  dshapephys.Mult(D, dshapephysdd); // dphi/dx.D);
	  q_hess_dot_d = 0.;
	  int sz1 = pow(dim, 1);
	  DenseMatrix T1(dim, h1dofs_cnt*sz1);
	  Vector T1_wrk(T1.GetData(), dim*h1dofs_cnt*sz1);
	  dkphi_dxk[0]->MultTranspose(te_shape, T1_wrk);
	  
	  DenseMatrix T2;
	  Vector T2_wrk;
	  for (int j = 0; j < 1; j++)
	    {
	      int sz2 = pow(dim, -j);
	      T2.SetSize(dim, h1dofs_cnt*sz2);
	      T2_wrk.SetDataAndSize(T2.GetData(), dim*h1dofs_cnt*sz2);
	      T1.MultTranspose(D, T2_wrk);
	      T1 = T2;
	    }
	  Vector q_hess_dot_d_work(h1dofs_cnt);
	  T1.MultTranspose(D, q_hess_dot_d_work);
	  q_hess_dot_d_work *= 1./Factorial(0);
	  q_hess_dot_d += q_hess_dot_d_work;

	  te_shape += dshapephysdd;
	  // te_shape += q_hess_dot_d;
	    
	  for (int i = 0; i < h1dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  loc_force(i, vd) += normalStressProjNormal * te_shape(i) * tN(vd) * ntildaDotTrueN/*nor(vd)/nor_norm*/;
		}
	    }
	  trial_fe1.CalcShape(eip, tr_shape);
	  AddMultVWt(Vloc_force,tr_shape,temp_elmat);
	}
      elmat.CopyMN(temp_elmat, 0, 0);
    }
    else if ((elem2 < NEproc) && (elemStatus2 == AnalyticalGeometricShape::SBElementType::INSIDE)){
      Array<DenseMatrix *> dkphi_dxk;
      DenseMatrix grad_phys;
      Vector Factorial;
      Array<DenseMatrix *> grad_phys_dir;
      test_fe2.ProjectGrad(test_fe2, *Trans.Elem2, grad_phys);
      DenseMatrix grad_work;
      grad_phys_dir.SetSize(dim); // NxN matrices for derivative in each direction
      for (int i = 0; i < dim; i++)
      {
         grad_phys_dir[i] = new DenseMatrix(h1dofs_cnt, h1dofs_cnt);
         grad_phys_dir[i]->CopyRows(grad_phys, i*h1dofs_cnt, (i+1)*h1dofs_cnt-1);
      }
      DenseMatrix grad_phys_work = grad_phys;
      grad_phys_work.SetSize(h1dofs_cnt, h1dofs_cnt*dim);

      dkphi_dxk.SetSize(1);
      int sz1 = pow(dim, 1);
      dkphi_dxk[0] = new DenseMatrix(h1dofs_cnt, h1dofs_cnt*sz1*dim);
      int loc_col_per_dof = sz1;
      int tot_col_per_dof = loc_col_per_dof*dim;
      for (int k = 0; k < dim; k++)
	{
	  grad_work.SetSize(h1dofs_cnt, h1dofs_cnt*sz1);
	  // grad_work[k] has derivative in kth direction for each DOF.
	  // grad_work[0] has d^2phi/dx^2 and d^2phi/dxdy terms and
	  // grad_work[1] has d^2phi/dydx and d^2phi/dy2 terms for each dof
	  Mult(*grad_phys_dir[k], grad_phys_work, grad_work);
            // Now we must place columns for each dof together so that they are
            // in order: d^2phi/dx^2, d^2phi/dxdy, d^2phi/dydx, d^2phi/dy2.
	  for (int j = 0; j < h1dofs_cnt; j++)
            {
	      for (int d = 0; d < loc_col_per_dof; d++)
		{
                  Vector col;
                  grad_work.GetColumn(j*loc_col_per_dof+d, col);
                  dkphi_dxk[0]->SetCol(j*tot_col_per_dof+k*loc_col_per_dof+d, col);
		}
            }
	}

      for (int i = 0; i < grad_phys_dir.Size(); i++)
      {
         delete grad_phys_dir[i];
      }

      Factorial.SetSize(1);
      Factorial(0) = 2;      
      DenseMatrix temp_elmat;
      temp_elmat.SetSize(h1dofs_cnt*dim, l2dofs_cnt);
      temp_elmat = 0.0;

      int h1dofs_offset = h1dofs_cnt * dim;
      int l2dofs_offset = l2dofs_cnt;

      DenseMatrix q_hess_dn(dim, h1dofs_cnt);
      Vector q_hess_dn_work(q_hess_dn.GetData(), h1dofs_cnt*dim);
      Vector q_hess_dot_d(h1dofs_cnt);
      q_hess_dn = 0.0;
      q_hess_dn_work = 0.0;
      q_hess_dot_d = 0.0;
      
      for (int q = 0; q  < nqp_face; q++)
	{	   
	  Vector D(dim);
	  Vector tN(dim);
	  D = 0.0;
	  tN = 0.0;
	  const int eq = e*nqp_face + q;
	  
	  const IntegrationPoint &ip_f = IntRule->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Trans.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip = Trans.GetElement2IntPoint();
	  Vector nor;
	  nor.SetSize(dim);
	  nor = 0.0;
	  if (dim == 1)
	    {
	      nor(0) = 2*eip.x - 1.0;
	    }
	  else
	    {
	      CalcOrtho(Trans.Jacobian(), nor);
	    }
	  nor *= -1.0;
	  dshapephys = 0.0;
	  te_shape = 0.0;

	  test_fe2.CalcShape(eip, te_shape);
	  test_fe2.CalcPhysDShape(*(Trans.Elem2), dshapephys);

	  loc_force = 0.0;
	  double normalStressProjNormal = 0.0;
	  double nor_norm = 0.0;
	  for (int s = 0; s < dim; s++){
	    normalStressProjNormal += qdata.weightedNormalStress(eq,s) * nor(s);
	    nor_norm += nor(s) * nor(s);
	    D(s) = quadDist(e*nqp_face+q,s);
	    tN(s) = quadTrueNorm(e*nqp_face+q,s);
	    //   tN(s) = nor(s);
	  }
	  //  std::cout << " distanceX " << D(0) <<  " distanceY " << D(1) << std::endl;
	  
	  nor_norm = sqrt(nor_norm);
	  //  tN /= nor_norm;
	  normalStressProjNormal = normalStressProjNormal/nor_norm;
	  double ntildaDotTrueN = 0;
	  for (int s = 0; s < dim; s++){
	    ntildaDotTrueN += tN(s) * nor(s)/nor_norm;
	  }
	  dshapephys.Mult(D, dshapephysdd); // dphi/dx.D);
	  q_hess_dot_d = 0.;
	  int sz1 = pow(dim, 1);
	  DenseMatrix T1(dim, h1dofs_cnt*sz1);
	  Vector T1_wrk(T1.GetData(), dim*h1dofs_cnt*sz1);
	  dkphi_dxk[0]->MultTranspose(te_shape, T1_wrk);
	  
	  DenseMatrix T2;
	  Vector T2_wrk;
	  for (int j = 0; j < 1; j++)
	    {
	      int sz2 = pow(dim, -j);
	      T2.SetSize(dim, h1dofs_cnt*sz2);
	      T2_wrk.SetDataAndSize(T2.GetData(), dim*h1dofs_cnt*sz2);
	      T1.MultTranspose(D, T2_wrk);
	      T1 = T2;
	    }
	  Vector q_hess_dot_d_work(h1dofs_cnt);
	  T1.MultTranspose(D, q_hess_dot_d_work);
	  q_hess_dot_d_work *= 1./Factorial(0);
	  q_hess_dot_d += q_hess_dot_d_work;

	  te_shape += dshapephysdd;
	  //  te_shape += q_hess_dot_d;

	  for (int i = 0; i < h1dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  loc_force(i, vd) += normalStressProjNormal * te_shape(i) * tN(vd) * ntildaDotTrueN /*nor(vd)/nor_norm*/;
		}
	    }
	  trial_fe2.CalcShape(eip, tr_shape);
	  AddMultVWt(Vloc_force,tr_shape,temp_elmat);
	}
      elmat.CopyMN(temp_elmat, h1dofs_offset, l2dofs_offset);
    }
  }
}


  void ShiftedNormalVelocityMassIntegrator::AssembleFaceMatrix(const FiniteElement &fe1,
							     const FiniteElement &fe2,
							     FaceElementTransformations &Trans,
							     DenseMatrix &elmat)
{
  MPI_Comm comm = pmesh->GetComm();
  int myid;
  MPI_Comm_rank(comm, &myid);

  const DenseMatrix& quadDist = analyticalSurface->GetQuadratureDistance();
  const DenseMatrix& quadTrueNorm = analyticalSurface->GetQuadratureTrueNormal();
  
  int elem1 = Trans.Elem1No;
  int elem2 = Trans.Elem2No;
  int elemStatus1 = elemStatus[elem1];
  int elemStatus2 = elemStatus[elem2];
  int NEproc = pmesh->GetNE();
  
  const int nqp_face = IntRule->GetNPoints();
  const int dim = fe1.GetDim();
  const int h1dofs_cnt = fe1.GetDof();
  elmat.SetSize(h1dofs_cnt*dim*2,h1dofs_cnt*dim*2);
  //  elmat.SetSize(h1dofs_cnt*dim);
  elmat = 0.0;
  Vector shape(h1dofs_cnt), loc_force2(h1dofs_cnt * dim), dshapedn(h1dofs_cnt), dshapephysdd(h1dofs_cnt);
 
  const int e = Trans.ElementNo;
  shape = 0.0;
  DenseMatrix loc_force1(h1dofs_cnt, dim), dshapephys(h1dofs_cnt, dim);
  Vector Vloc_force(loc_force1.Data(), h1dofs_cnt*dim);
  // elmat.Print(std::cout,1);
  // these are not thread-safe!
  Vector Factorial;
  Factorial.SetSize(1);
  Factorial(0) = 2.0;
  
  if (faceTags[e] == 5){
    if (elemStatus1 == AnalyticalGeometricShape::SBElementType::INSIDE){
      Array<DenseMatrix *> dkphi_dxk;
      DenseMatrix grad_phys;
      Array<DenseMatrix *> grad_phys_dir;
      fe1.ProjectGrad(fe1, *Trans.Elem1, grad_phys);
      DenseMatrix grad_work;
      grad_phys_dir.SetSize(dim); // NxN matrices for derivative in each direction
      for (int i = 0; i < dim; i++)
      {
         grad_phys_dir[i] = new DenseMatrix(h1dofs_cnt, h1dofs_cnt);
         grad_phys_dir[i]->CopyRows(grad_phys, i*h1dofs_cnt, (i+1)*h1dofs_cnt-1);
      }
      DenseMatrix grad_phys_work = grad_phys;
      grad_phys_work.SetSize(h1dofs_cnt, h1dofs_cnt*dim);

      dkphi_dxk.SetSize(1);
      int sz1 = pow(dim, 1);
      dkphi_dxk[0] = new DenseMatrix(h1dofs_cnt, h1dofs_cnt*sz1*dim);
      int loc_col_per_dof = sz1;
      int tot_col_per_dof = loc_col_per_dof*dim;
      for (int k = 0; k < dim; k++)
	{
	  grad_work.SetSize(h1dofs_cnt, h1dofs_cnt*sz1);
	  // grad_work[k] has derivative in kth direction for each DOF.
	  // grad_work[0] has d^2phi/dx^2 and d^2phi/dxdy terms and
	  // grad_work[1] has d^2phi/dydx and d^2phi/dy2 terms for each dof
	  Mult(*grad_phys_dir[k], grad_phys_work, grad_work);
            // Now we must place columns for each dof together so that they are
            // in order: d^2phi/dx^2, d^2phi/dxdy, d^2phi/dydx, d^2phi/dy2.
	  for (int j = 0; j < h1dofs_cnt; j++)
            {
	      for (int d = 0; d < loc_col_per_dof; d++)
		{
                  Vector col;
                  grad_work.GetColumn(j*loc_col_per_dof+d, col);
                  dkphi_dxk[0]->SetCol(j*tot_col_per_dof+k*loc_col_per_dof+d, col);
		}
            }
	}

      for (int i = 0; i < grad_phys_dir.Size(); i++)
      {
         delete grad_phys_dir[i];
      }
      
      DenseMatrix q_hess_dn(dim, h1dofs_cnt);
      Vector q_hess_dn_work(q_hess_dn.GetData(), h1dofs_cnt*dim);
      Vector q_hess_dot_d(h1dofs_cnt);
      
      q_hess_dn = 0.0;
      q_hess_dn_work = 0.0;
      q_hess_dot_d = 0.0;
      
      //  std::cout << " neq face in assemble " << nqp_face << std::endl;
      for (int q = 0; q  < nqp_face; q++)
	{
	  Vector D(dim);
	  Vector tN(dim);
	  D = 0.0;
	  tN = 0.0;
	  const int eq = e*nqp_face + q;
	  //  std::cout << " show me El1 " <<  qdata.normalVelocityPenaltyScaling(eq) << std::endl;
	  const IntegrationPoint &ip_f = IntRule->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Trans.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip = Trans.GetElement1IntPoint();
	  Vector nor;
	  nor.SetSize(dim);
	  nor = 0.0;
	  if (dim == 1)
	    {
	      nor(0) = 2*eip.x - 1.0;
	    }
	  else
	    {
	      CalcOrtho(Trans.Jacobian(), nor);
	    }
	  dshapephys = 0.0;
	  dshapephysdd = 0.0;
	  shape = 0.0;
	  fe1.CalcShape(eip, shape);
	  fe1.CalcPhysDShape(*(Trans.Elem1), dshapephys);
	  
	  double nor_norm = 0.0;
	  for (int s = 0; s < dim; s++){
	    nor_norm += nor(s) * nor(s);
	    D(s) = quadDist(e*nqp_face+q,s);
	    tN(s) = quadTrueNorm(e*nqp_face+q,s);
	    //  tN(s) = nor(s);
	  }
	  nor_norm = sqrt(nor_norm);
	  //  tN /= nor_norm;
	  
	  double ntildaDotTrueN = 0;
	  for (int s = 0; s < dim; s++){
	    ntildaDotTrueN += tN(s) * nor(s)/nor_norm;
	  }
	  dshapephys.Mult(D, dshapephysdd); // dphi/dx.D);
	  q_hess_dot_d = 0.;
	  int sz1 = pow(dim, 1);
	  DenseMatrix T1(dim, h1dofs_cnt*sz1);
	  Vector T1_wrk(T1.GetData(), dim*h1dofs_cnt*sz1);
	  dkphi_dxk[0]->MultTranspose(shape, T1_wrk);
	  
	  DenseMatrix T2;
	  Vector T2_wrk;
	  //  for (int j = 0; j < 1; j++)
	  //   {
	  int sz2 = pow(dim, 0);
	  T2.SetSize(dim, h1dofs_cnt*sz2);
	  T2_wrk.SetDataAndSize(T2.GetData(), dim*h1dofs_cnt*sz2);
	  T1.MultTranspose(D, T2_wrk);
	  T1 = T2;
	      //  }
	  Vector q_hess_dot_d_work(h1dofs_cnt);
	  T1.MultTranspose(D, q_hess_dot_d_work);
	  q_hess_dot_d_work *= 1./Factorial(0);
	  q_hess_dot_d += q_hess_dot_d_work;

	  Vector trial_wrk = shape;
	  Vector test_wrk = shape;
	  trial_wrk += dshapephysdd;
	  //  trial_wrk += q_hess_dot_d;
	  test_wrk += dshapephysdd;
	  //  test_wrk += q_hess_dot_d;
	  
	  for (int i = 0; i < h1dofs_cnt; i++)
	    {	      
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  for (int j = 0; j < h1dofs_cnt; j++)
		    {
		      for (int md = 0; md < dim; md++) // Velocity components.
			{	      
			  elmat(i + vd * h1dofs_cnt, j + md * h1dofs_cnt) += test_wrk(i) * trial_wrk(j) * tN(vd) * nor_norm * tN(md) /*nor(vd) * (nor(md)/nor_norm)*/ * qdata.normalVelocityPenaltyScaling(eq) * ntildaDotTrueN * ntildaDotTrueN;
			}
		    }
		}
	    }
	}
      //  std::cout << " myid " << myid << " print " << std::endl;
      //  temp_elmat.Print(std::cout,1);
      //  elmat = 0.0;
      //  elmat.CopyMN(temp_elmat, 0, 0);
      //  elmat.CopyMN(temp_elmat,h1dofs_cnt*dim*2,h1dofs_cnt*dim*2,0,0);
      //  std::cout << " myid " << myid << " Printing ELEM1 " << std::endl;
      //  elmat.Print(std::cout,1);
    }
    else if ((elem2 < NEproc) && (elemStatus2 == AnalyticalGeometricShape::SBElementType::INSIDE)){
      Array<DenseMatrix *> dkphi_dxk;
      DenseMatrix grad_phys;
      Array<DenseMatrix *> grad_phys_dir;
      fe2.ProjectGrad(fe2, *Trans.Elem2, grad_phys);
      DenseMatrix grad_work;
      grad_phys_dir.SetSize(dim); // NxN matrices for derivative in each direction
      for (int i = 0; i < dim; i++)
      {
         grad_phys_dir[i] = new DenseMatrix(h1dofs_cnt, h1dofs_cnt);
         grad_phys_dir[i]->CopyRows(grad_phys, i*h1dofs_cnt, (i+1)*h1dofs_cnt-1);
      }
      DenseMatrix grad_phys_work = grad_phys;
      grad_phys_work.SetSize(h1dofs_cnt, h1dofs_cnt*dim);

      dkphi_dxk.SetSize(1);
      int sz1 = pow(dim, 1);
      dkphi_dxk[0] = new DenseMatrix(h1dofs_cnt, h1dofs_cnt*sz1*dim);
      int loc_col_per_dof = sz1;
      int tot_col_per_dof = loc_col_per_dof*dim;
      for (int k = 0; k < dim; k++)
	{
	  grad_work.SetSize(h1dofs_cnt, h1dofs_cnt*sz1);
	  // grad_work[k] has derivative in kth direction for each DOF.
	  // grad_work[0] has d^2phi/dx^2 and d^2phi/dxdy terms and
	  // grad_work[1] has d^2phi/dydx and d^2phi/dy2 terms for each dof
	  Mult(*grad_phys_dir[k], grad_phys_work, grad_work);
          // Now we must place columns for each dof together so that they are
	  // in order: d^2phi/dx^2, d^2phi/dxdy, d^2phi/dydx, d^2phi/dy2.
	  for (int j = 0; j < h1dofs_cnt; j++)
            {
	      for (int d = 0; d < loc_col_per_dof; d++)
		{
                  Vector col;
                  grad_work.GetColumn(j*loc_col_per_dof+d, col);
                  dkphi_dxk[0]->SetCol(j*tot_col_per_dof+k*loc_col_per_dof+d, col);
		}
            }
	}

      for (int i = 0; i < grad_phys_dir.Size(); i++)
      {
         delete grad_phys_dir[i];
      }
      
      int h1dofs_offset = h1dofs_cnt * dim;

      DenseMatrix q_hess_dn(dim, h1dofs_cnt);
      Vector q_hess_dn_work(q_hess_dn.GetData(), h1dofs_cnt*dim);
      Vector q_hess_dot_d(h1dofs_cnt);
      
      q_hess_dn = 0.0;
      q_hess_dn_work = 0.0;
      q_hess_dot_d = 0.0;
      
      for (int q = 0; q  < nqp_face; q++)
	{
	  Vector D(dim);
	  Vector tN(dim);
	  D = 0.0;
	  tN = 0.0;
	  const int eq = e*nqp_face + q;
	  //	  std::cout << " show me El2 " <<  qdata.normalVelocityPenaltyScaling(eq) << std::endl;
		
	  const IntegrationPoint &ip_f = IntRule->IntPoint(q);
	  // Set the integration point in the face and the neighboring elements
	  Trans.SetAllIntPoints(&ip_f);
	  const IntegrationPoint &eip = Trans.GetElement2IntPoint();
	  Vector nor;
	  nor.SetSize(dim);
	  nor = 0.0;
	  if (dim == 1)
	    {
	      nor(0) = 2*eip.x - 1.0;
	    }
	  else
	    {
	      CalcOrtho(Trans.Jacobian(), nor);
	    }
	  nor *= -1.0;
	  dshapephys = 0.0;
	  shape = 0.0;
	  fe2.CalcShape(eip, shape);
	  fe2.CalcPhysDShape(*(Trans.Elem2), dshapephys);

	  double nor_norm = 0.0;
	  for (int s = 0; s < dim; s++){
	    nor_norm += nor(s) * nor(s);
	    D(s) = quadDist(e*nqp_face+q,s);
	    tN(s) = quadTrueNorm(e*nqp_face+q,s);
	    //  tN(s) = nor(s);
	  }
	  nor_norm = sqrt(nor_norm);
	  //  tN /= nor_norm;
	  
	  double ntildaDotTrueN = 0.0;
	  for (int s = 0; s < dim; s++){
	    ntildaDotTrueN += tN(s) * nor(s)/nor_norm;
	  }
	  //  std::cout << " ntildaDot " << ntildaDotTrueN << std::endl;
	  dshapephys.Mult(D, dshapephysdd); // dphi/dx.D);
	  q_hess_dot_d = 0.;
	  int sz1 = pow(dim, 1);
	  DenseMatrix T1(dim, h1dofs_cnt*sz1);
	  Vector T1_wrk(T1.GetData(), dim*h1dofs_cnt*sz1);
	  dkphi_dxk[0]->MultTranspose(shape, T1_wrk);
	  
	  DenseMatrix T2;
	  Vector T2_wrk;
	  //	  for (int j = 0; j < 1; j++)
	  //  {
	  int sz2 = pow(dim, 0);
	  T2.SetSize(dim, h1dofs_cnt*sz2);
	  T2_wrk.SetDataAndSize(T2.GetData(), dim*h1dofs_cnt*sz2);
	  T1.MultTranspose(D, T2_wrk);
	  T1 = T2;
	      //  }
	  Vector q_hess_dot_d_work(h1dofs_cnt);
	  T1.MultTranspose(D, q_hess_dot_d_work);
	  q_hess_dot_d_work *= 1./Factorial(0);
	  q_hess_dot_d += q_hess_dot_d_work;

	  Vector trial_wrk = shape;
	  Vector test_wrk = shape;
	  trial_wrk += dshapephysdd;
	  //	  trial_wrk += q_hess_dot_d;
	  test_wrk += dshapephysdd;
	  //  test_wrk += q_hess_dot_d;

	  for (int i = 0; i < h1dofs_cnt; i++)
	    {
	      for (int vd = 0; vd < dim; vd++) // Velocity components.
		{
		  for (int j = 0; j < h1dofs_cnt; j++)
		    {
		      for (int md = 0; md < dim; md++) // Velocity components.
			{	      
			  elmat(i + vd * h1dofs_cnt + h1dofs_offset, j + md * h1dofs_cnt + h1dofs_offset) += test_wrk(i) * trial_wrk(j) * tN(vd) * nor_norm * tN(md)/*nor(vd) * (nor(md)/nor_norm)*/ * qdata.normalVelocityPenaltyScaling(eq) * ntildaDotTrueN * ntildaDotTrueN;
			}
		    }
		}
	    }
	}
      //  elmat.CopyMN(temp_elmat, h1dofs_offset, h1dofs_offset);
      //  std::cout << " myid " << myid << " Printing ELEM2 " << std::endl;
      //  elmat.Print(std::cout,1);
    }
  }
}

} // namespace hydrodynamics

} // namespace mfem
