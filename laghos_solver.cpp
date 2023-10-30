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

#include "general/forall.hpp"
#include "laghos_solver.hpp"
#include "linalg/kernels.hpp"
#include <unordered_map>
#include "laghos_solver.hpp"
#include "sbm_aux.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{

  namespace hydrodynamics
  {

    void VisualizeField(socketstream &sock, const char *vishost, int visport,
			ParGridFunction &gf, const char *title,
			int x, int y, int w, int h, bool vec)
    {
      gf.HostRead();
      ParMesh &pmesh = *gf.ParFESpace()->GetParMesh();
      MPI_Comm comm = pmesh.GetComm();

      int num_procs, myid;
      MPI_Comm_size(comm, &num_procs);
      MPI_Comm_rank(comm, &myid);

      bool newly_opened = false;
      int connection_failed;

      do
	{
	  if (myid == 0)
	    {
	      if (!sock.is_open() || !sock)
		{
		  sock.open(vishost, visport);
		  sock.precision(8);
		  newly_opened = true;
		}
	      sock << "solution\n";
	    }

	  pmesh.PrintAsOne(sock);
	  gf.SaveAsOne(sock);

	  if (myid == 0 && newly_opened)
	    {
	      const char* keys = (gf.FESpace()->GetMesh()->Dimension() == 2)
		? "mAcRjl" : "mmaaAcl";

	      sock << "window_title '" << title << "'\n"
		   << "window_geometry "
		   << x << " " << y << " " << w << " " << h << "\n"
		   << "keys " << keys;
	      if ( vec ) { sock << "vvv"; }
	      sock << std::endl;
	    }

	  if (myid == 0)
	    {
	      connection_failed = !sock && !newly_opened;
	    }
	  MPI_Bcast(&connection_failed, 1, MPI_INT, 0, comm);
	}
      while (connection_failed);
    }

    LagrangianHydroOperator::LagrangianHydroOperator(const int size,
						     const int order_e,
						     const int order_v,
						     const int faceIndex,
						     double &globalmax_rho,
						     double &globalmax_cs,
						     double &globalmax_viscous_coef,
						     ParFiniteElementSpace &h1,
						     ParFiniteElementSpace &l2,
						     ParFiniteElementSpace &p_l2_fes,
						     ParFiniteElementSpace &pface_l2_fes,
						     Coefficient &rho0_coeff,
						     ParGridFunction &rho0_gf,
						     ParGridFunction &rho_gf,
						     ParGridFunction &rhoface_gf,
						     ParGridFunction &gamma_gf,
						     ParGridFunction &p_gf,
						     ParGridFunction &pface_gf,
						     ParGridFunction &v_gf,
						     ParGridFunction &e_gf,
						     ParGridFunction &cs_gf,
						     ParGridFunction &csface_gf,
						     ParGridFunction &viscousface_gf,
						     ParGridFunction &rho0DetJ0_gf,
						     ParGridFunction &rho0DetJ0face_gf,
						     ParGridFunction &Jac0inv_gf,
						     ParGridFunction &Jac0invface_gf,
						     const int source,
						     const double cfl,
						     const bool visc,
						     const bool vort,
						     const double cgt,
						     const int cgiter,
						     double ftz,
						     const int oq,
						     const double penaltyParameter,
						     const double perimeter,
						     const double nitscheVersion) :
      TimeDependentOperator(size),
      H1(h1), L2(l2), P_L2(p_l2_fes), PFace_L2(pface_l2_fes), H1c(H1.GetParMesh(), H1.FEColl(), 1), L2c(L2.GetParMesh(), L2.FEColl(), 1), 
      alpha_fes(NULL), alpha_fec(NULL),
      pmesh(H1.GetParMesh()),
      H1Vsize(H1.GetVSize()),
      L2Vsize(L2.GetVSize()),
      block_offsets(4),
      x_gf(&H1),
      dim(pmesh->Dimension()),
      NE(pmesh->GetNE()),
      l2dofs_cnt(L2.GetFE(0)->GetDof()),
      h1dofs_cnt(H1.GetFE(0)->GetDof()),
      source_type(source), cfl(cfl),
      use_viscosity(visc),
      use_vorticity(vort),
      cg_rel_tol(cgt), cg_max_iter(cgiter),ftz_tol(ftz),penaltyParameter(penaltyParameter),perimeter(perimeter),
      nitscheVersion(nitscheVersion),
      ess_elem(pmesh->attributes.Max()),
      fi(NULL),
      efi(NULL),
      sfi(NULL),
      nvmi(NULL),
      d_nvmi(NULL),
      de_nvmi(NULL),
      mi(NULL),
      vmi(NULL),
      alphaCut(NULL),
      rho0_gf(rho0_gf),
      rho_gf(rho_gf),
      rhoface_gf(rhoface_gf),
      gamma_gf(gamma_gf),
      v_gf(v_gf),
      p_gf(p_gf),
      e_gf(e_gf),
      cs_gf(cs_gf),
      globalmax_rho(globalmax_rho),
      globalmax_cs(globalmax_cs),
      globalmax_viscous_coef(globalmax_viscous_coef),
      pface_gf(pface_gf),
      csface_gf(csface_gf),
      viscousface_gf(viscousface_gf),
      rho0DetJ0_gf(rho0DetJ0_gf),
      rho0DetJ0face_gf(rho0DetJ0face_gf),
      Jac0inv_gf(Jac0inv_gf),
      Jac0invface_gf(Jac0invface_gf),
      Mv(new ParBilinearForm(&H1)), Mv_spmat_copy(),
      Me_mat(new ParBilinearForm(&L2)),
      Me(l2dofs_cnt, l2dofs_cnt, NE),
      Me_inv(l2dofs_cnt, l2dofs_cnt, NE),
      GLIntRules(0, BasisType::GaussLobatto),
      ir(IntRules.Get(pmesh->GetElementBaseGeometry(0),
		      ((oq > 0) ? oq : 3 * H1.GetOrder(0) + L2.GetOrder(0) - 1) )),
      b_ir(GLIntRules.Get((pmesh->GetInteriorFaceTransformations(faceIndex))->GetGeometryType(), ( 3 * H1.GetOrder(0) + L2.GetOrder(0) - 1) )),
      Q1D(int(floor(0.7 + pow(ir.GetNPoints(), 1.0 / dim)))),
      qdata(),
      qdata_is_current(false),
      forcemat_is_assembled(false),
      sourcevec_is_assembled(false),
      energyforcemat_is_assembled(false),
      bv_qdata_is_current(false),
      bvemb_qdata_is_current(false),
      beemb_qdata_is_current(false),
      be_qdata_is_current(false),
      bvdiffusion_forcemat_is_assembled(false),
      bvemb_forcemat_is_assembled(false),
      bediffusion_forcemat_is_assembled(false),
      beemb_forcemat_is_assembled(false),
      Force(&H1),
      EnergyForce(&L2),
      SourceForce(&H1),
      DiffusionVelocityBoundaryForce(&H1),
      DiffusionEnergyBoundaryForce(&L2),
      X(H1c.GetTrueVSize()),
      B(H1c.GetTrueVSize()),
      X_e(L2c.GetTrueVSize()),
      B_e(L2c.GetTrueVSize()),
      one(L2Vsize),
      rhs(H1Vsize),
      b_rhs(H1Vsize),
      e_rhs(L2Vsize),
      be_rhs(L2Vsize),
      C_I_E(0.0),
      C_I_V(0.0)
    {
      block_offsets[0] = 0;
      block_offsets[1] = block_offsets[0] + H1Vsize;
      block_offsets[2] = block_offsets[1] + H1Vsize;
      block_offsets[3] = block_offsets[2] + L2Vsize;
      one = 1.0;

      
      rho_gf.ExchangeFaceNbrData();
      p_gf.ExchangeFaceNbrData();
      cs_gf.ExchangeFaceNbrData();
      rho0DetJ0_gf.ExchangeFaceNbrData();
      Jac0inv_gf.ExchangeFaceNbrData();
      
      rhoface_gf.ExchangeFaceNbrData();
      pface_gf.ExchangeFaceNbrData();
      csface_gf.ExchangeFaceNbrData();
      viscousface_gf.ExchangeFaceNbrData();
      rho0DetJ0face_gf.ExchangeFaceNbrData();
      Jac0invface_gf.ExchangeFaceNbrData();
      
      switch (pmesh->GetElementBaseGeometry(0))
      {
      case Geometry::TRIANGLE:
      case Geometry::TETRAHEDRON:{
         C_I_E = (order_e+1)*(order_e+dim)/dim+1.0;
         C_I_V = (order_v+1)*(order_v+dim)/dim;
         break;
      }
      case Geometry::SQUARE:
      case Geometry::CUBE:{
         C_I_E = order_e*order_e+1.0;
         C_I_V = order_v*order_v;
         break;
      }
      default: MFEM_ABORT("Unknown zone type!");
      }

      // int val  =  (oq > 0) ? oq : 3 * H1.GetOrder(0) + L2.GetOrder(0) - 1;
      // std::cout << " val " << val << std::endl; 
      alpha_fec = new L2_FECollection(0, pmesh->Dimension());
      alpha_fes = new ParFiniteElementSpace(pmesh, alpha_fec);
      alpha_fes->ExchangeFaceNbrData();
      alphaCut = new ParGridFunction(alpha_fes);
      alphaCut->ExchangeFaceNbrData();
      *alphaCut = 1;
    
      const int max_elem_attr = pmesh->attributes.Max();
      ess_elem.SetSize(max_elem_attr);
      ess_elem = 1;
      
      // Values of rho0DetJ0 and Jac0inv at all quadrature points.
      // Initial local mesh size (assumes all mesh elements are the same).
      int Ne, ne = NE;
      double Volume, vol = 0.0;

      const int NQ = ir.GetNPoints();
      
      for (int e = 0; e < NE; e++)
	{	 	  
	  const IntegrationRule &ir_p = P_L2.GetFE(e)->GetNodes();
	  const int gl_nqp = ir_p.GetNPoints(); 
	  ElementTransformation &Tr = *P_L2.GetElementTransformation(e);
	  for (int q = 0; q < gl_nqp; q++)
	    {
	      const IntegrationPoint &ip = ir.IntPoint(q);
	      Tr.SetIntPoint(&ip);
	      // std::cout << " ip.x " << ip.x << " ip.y " << ip.y << " ip.z " << ip.z << std::endl;
	      double volumeFraction = alphaCut->GetValue(Tr, ip);
	      const double rho0DetJ0 = Tr.Weight() * rho0_gf.GetValue(Tr, ip) * volumeFraction;
	      rho0DetJ0_gf(e * gl_nqp + q) = rho0DetJ0;
	    }
	}
      rho0DetJ0_gf.ExchangeFaceNbrData();
      
      for (int e = 0; e < NE; e++)
	{
	  // The points (and their numbering) coincide with the nodes of p.
	  const IntegrationRule &ir_p = PFace_L2.GetFE(e)->GetNodes();
	  const int gl_nqp = ir_p.GetNPoints();
	  ElementTransformation &Tr = *PFace_L2.GetElementTransformation(e);
	  for (int q = 0; q < gl_nqp; q++)
	    {
	      const IntegrationPoint &ip = ir_p.IntPoint(q);
	      Tr.SetIntPoint(&ip);
	      // std::cout << " faceip.x " << ip.x << " faceip.y " << ip.y << " faceip.z " << ip.z << std::endl;
	      const double rho0DetJ0 = Tr.Weight() * rho0_gf.GetValue(Tr, ip);
	      double volumeFraction = alphaCut->GetValue(Tr, ip);
	      rho0DetJ0face_gf(e * gl_nqp + q) = rho0DetJ0 * volumeFraction;
	    }
	}
      rho0DetJ0face_gf.ExchangeFaceNbrData();
      
      for (int e = 0; e < NE; e++) { vol += pmesh->GetElementVolume(e); }
      
      MPI_Allreduce(&vol, &Volume, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
      MPI_Allreduce(&ne, &Ne, 1, MPI_INT, MPI_SUM, pmesh->GetComm());
      switch (pmesh->GetElementBaseGeometry(0))
	{
	case Geometry::SEGMENT: qdata.h0 = Volume / Ne; break;
	case Geometry::SQUARE: qdata.h0 = sqrt(Volume / Ne); break;
	case Geometry::TRIANGLE: qdata.h0 = sqrt(2.0 * Volume / Ne); break;
	case Geometry::CUBE: qdata.h0 = pow(Volume / Ne, 1./3.); break;
	case Geometry::TETRAHEDRON: qdata.h0 = pow(6.0 * Volume / Ne, 1./3.); break;
	default: MFEM_ABORT("Unknown zone type!");
	}
      qdata.h0 /= (double) H1.GetOrder(0);

      //Compute quadrature quantities
      UpdateDensity(rho0DetJ0_gf, *alphaCut, rho_gf);
      UpdatePressure(gamma_gf, e_gf, rho_gf, p_gf);
      UpdateSoundSpeed(gamma_gf, e_gf, cs_gf);
      rho_gf.ExchangeFaceNbrData();
      p_gf.ExchangeFaceNbrData();
      cs_gf.ExchangeFaceNbrData();
      
     //Compute quadrature quantities
     // std::cout << " calling " << std::endl;
      UpdateDensity(rho0DetJ0face_gf, *alphaCut, rhoface_gf);
      UpdatePressure(gamma_gf, e_gf, rhoface_gf, pface_gf);
      UpdateSoundSpeed(gamma_gf, e_gf, csface_gf);
      UpdateGlobalMaxRho(globalmax_rho, rhoface_gf);
      rhoface_gf.ExchangeFaceNbrData();
      pface_gf.ExchangeFaceNbrData();
      csface_gf.ExchangeFaceNbrData();
      viscousface_gf.ExchangeFaceNbrData();

      // Standard local assembly and inversion for energy mass matrices.
      // 'Me' is used in the computation of the internal energy
      // which is used twice: once at the start and once at the end of the run.
      mi = new WeightedMassIntegrator(*alphaCut, rho_gf, &ir);
      Me_mat->AddDomainIntegrator(mi, ess_elem);

      MassIntegrator mi_2(rho0_coeff, &ir);
      for (int e = 0; e < NE; e++)
	{
	  DenseMatrixInverse inv(&Me(e));
	  const FiniteElement &fe = *L2.GetFE(e);
	  ElementTransformation &Tr = *L2.GetElementTransformation(e);
	  mi_2.AssembleElementMatrix(fe, Tr, Me(e));
	  inv.Factor();
	  inv.GetInverseMatrix(Me_inv(e));
	}
      
      // Standard assembly for the velocity mass matrix.
      vmi = new WeightedVectorMassIntegrator(*alphaCut, rho_gf, &ir);
      Mv->AddDomainIntegrator(vmi, ess_elem);

      fi = new ForceIntegrator(qdata.h0, *alphaCut, v_gf, e_gf, p_gf, cs_gf, rho_gf, Jac0inv_gf, use_viscosity, use_vorticity);
      fi->SetIntRule(&ir);
      Force.AddDomainIntegrator(fi, ess_elem);
      // Make a dummy assembly to figure out the sparsity.
      Force.Assemble();

      efi = new EnergyForceIntegrator(qdata.h0, *alphaCut, v_gf, e_gf, p_gf, cs_gf, rho_gf, Jac0inv_gf, use_viscosity, use_vorticity);
      efi->SetIntRule(&ir);
      EnergyForce.AddDomainIntegrator(efi, ess_elem);
      // Make a dummy assembly to figure out the sparsity.
      EnergyForce.Assemble();

      sfi = new SourceForceIntegrator(rho_gf);
      sfi->SetIntRule(&ir);
      SourceForce.AddDomainIntegrator(sfi, ess_elem);
      // Make a dummy assembly to figure out the sparsity.
      SourceForce.Assemble();
      
      nvmi = new NormalVelocityMassIntegrator(qdata.h0, *alphaCut, 2.0 * penaltyParameter * (C_I_V), perimeter, order_v, rhoface_gf, v_gf, csface_gf, Jac0invface_gf, rho0DetJ0face_gf, globalmax_rho, globalmax_cs, globalmax_viscous_coef);

      nvmi->SetIntRule(&b_ir);
      Mv->AddBdrFaceIntegrator(nvmi);

      d_nvmi = new DiffusionNormalVelocityIntegrator(qdata.h0, *alphaCut, 2.0 * penaltyParameter * (C_I_V), order_v, rhoface_gf, v_gf, pface_gf, Jac0invface_gf, rho0DetJ0face_gf, csface_gf, globalmax_rho, globalmax_cs, globalmax_viscous_coef, use_viscosity, use_vorticity);
      d_nvmi->SetIntRule(&b_ir);
      DiffusionVelocityBoundaryForce.AddBdrFaceIntegrator(d_nvmi);
      DiffusionVelocityBoundaryForce.Assemble();
  
      de_nvmi = new DiffusionEnergyNormalVelocityIntegrator(qdata.h0, *alphaCut, 2.0 * penaltyParameter * (C_I_V), order_v, rhoface_gf, v_gf, pface_gf, Jac0invface_gf, rho0DetJ0face_gf, csface_gf, globalmax_rho, globalmax_cs, globalmax_viscous_coef, use_viscosity, use_vorticity);
      de_nvmi->SetIntRule(&b_ir);
      DiffusionEnergyBoundaryForce.AddBdrFaceIntegrator(de_nvmi);
      DiffusionEnergyBoundaryForce.Assemble();
      
      Me_mat->Assemble();
      Mv->Assemble();
      
      Mv_spmat_copy = Mv->SpMat();
       
    }

    LagrangianHydroOperator::~LagrangianHydroOperator() {
    }

    void LagrangianHydroOperator::Mult(const Vector &S, Vector &dS_dt, const Vector &S_init) const
    {
      // Make sure that the mesh positions correspond to the ones in S. This is
      // needed only because some mfem time integrators don't update the solution
      // vector at every intermediate stage (hence they don't change the mesh).
      UpdateMesh(S);
      // The monolithic BlockVector stores the unknown fields as follows:
      // (Position, Velocity, Specific Internal Energy).
      Vector* sptr = const_cast<Vector*>(&S);
      ParGridFunction v;
      const int VsizeH1 = H1.GetVSize();
      v.MakeRef(&H1, *sptr, VsizeH1);
      // Set dx_dt = v (explicit).
      ParGridFunction dx;
      dx.MakeRef(&H1, dS_dt, 0);
      dx = v;
      SolveVelocity(S, dS_dt, S_init,0);
      SolveEnergy(S, v, dS_dt);
      qdata_is_current = false;
      bv_qdata_is_current = false;
      be_qdata_is_current = false;
      bvemb_qdata_is_current = false;
      beemb_qdata_is_current = false;
    }

    void LagrangianHydroOperator::UpdateLevelSet(const Vector &S, const Vector &S_init){
      
      //Compute quadrature quantities
      UpdateDensity(rho0DetJ0_gf, *alphaCut, rho_gf);
      UpdatePressure(gamma_gf, e_gf, rho_gf, p_gf);
      UpdateSoundSpeed(gamma_gf, e_gf, cs_gf);
      rho_gf.ExchangeFaceNbrData();
      p_gf.ExchangeFaceNbrData();
      cs_gf.ExchangeFaceNbrData();
	
      //Compute quadrature quantities
      UpdateDensity(rho0DetJ0face_gf, *alphaCut, rhoface_gf);
      UpdatePressure(gamma_gf, e_gf, rhoface_gf, pface_gf);
      UpdateSoundSpeed(gamma_gf, e_gf, csface_gf);
      UpdateGlobalMaxRho(globalmax_rho, rhoface_gf);
      rhoface_gf.ExchangeFaceNbrData();
      pface_gf.ExchangeFaceNbrData();
      csface_gf.ExchangeFaceNbrData();
      viscousface_gf.ExchangeFaceNbrData();

      v_gf.ExchangeFaceNbrData();
 
    }
    
    void LagrangianHydroOperator::SolveVelocity(const Vector &S,
						Vector &dS_dt,
						const Vector &S_init,
						const double dt) const
    {
     
      AssembleForceMatrix();
      AssembleDiffusionVelocityBoundaryForceMatrix();
     
      // The monolithic BlockVector stores the unknown fields as follows:
      // (Position, Velocity, Specific Internal Energy).
      ParGridFunction dv;
      dv.MakeRef(&H1, dS_dt, H1Vsize);
      dv = 0.0;

      AssembleSourceVector();
      
    
      rhs = 0.0;
      rhs += Force;
      rhs += DiffusionVelocityBoundaryForce;
      rhs += SourceForce;
      
      HypreParMatrix A;
      Mv->FormLinearSystem(ess_tdofs, dv, rhs, A, X, B);
      CGSolver cg(H1.GetParMesh()->GetComm());
      HypreSmoother prec;
      prec.SetType(HypreSmoother::Jacobi, 1);
      cg.SetPreconditioner(prec);
      cg.SetOperator(A);
      cg.SetRelTol(cg_rel_tol);
      cg.SetAbsTol(0.0);
      cg.SetMaxIter(cg_max_iter);
      cg.SetPrintLevel(-1);
      cg.Mult(B, X);
      Mv->RecoverFEMSolution(X, rhs, dv);
    }

    void LagrangianHydroOperator::SolveEnergy(const Vector &S, const Vector &v,
					      Vector &dS_dt) const
    {      
      // Updated Velocity, needed for the energy solve
      Vector* sptr = const_cast<Vector*>(&v);
      ParGridFunction v_updated;
      v_updated.MakeRef(&H1, *sptr, 0);
      v_updated.ExchangeFaceNbrData();

      
      efi->SetVelocityGridFunctionAtNewState(&v_updated);
      AssembleEnergyForceMatrix();
      
      de_nvmi->SetVelocityGridFunctionAtNewState(&v_updated);
      AssembleDiffusionEnergyBoundaryForceMatrix();
      
      // The monolithic BlockVector stores the unknown fields as follows:
      // (Position, Velocity, Specific Internal Energy).
      ParGridFunction de;
      de.MakeRef(&L2, dS_dt, H1Vsize*2);
      de = 0.0;

      // Solve for energy, assemble the energy source if such exists.
      ParLinearForm *e_source = nullptr;
      Array<int> recastEssElem(ess_elem);
      if (source_type == 1) // 2D Taylor-Green.
	{
	  e_source = new ParLinearForm(&L2);
	  TaylorCoefficient coeff;
	  DomainLFIntegrator *d = new DomainLFIntegrator(coeff, &ir);
	  e_source->AddDomainIntegrator(d, recastEssElem);
	  e_source->Assemble();
	  e_source->ParallelAssemble();
	}

      Array<int> h1dofs;
      e_rhs = 0.0;
      e_rhs += EnergyForce;
      e_rhs += DiffusionEnergyBoundaryForce;
     
      Array<int> l2dofs;
   
      if (e_source) { e_rhs += *e_source; }
      Vector loc_rhs(l2dofs_cnt), loc_de(l2dofs_cnt);
      
      for (int e = 0; e < NE; e++)
	{
	  L2.GetElementDofs(e, l2dofs);	 
	    e_rhs.GetSubVector(l2dofs, loc_rhs);
	    Me_inv(e).Mult(loc_rhs, loc_de);
	    de.SetSubVector(l2dofs, loc_de);
	 
	}
      delete e_source;
    }

    void LagrangianHydroOperator::UpdateMesh(const Vector &S) const
    {
      Vector* sptr = const_cast<Vector*>(&S);
      x_gf.MakeRef(&H1, *sptr, 0);
      H1.GetParMesh()->NewNodes(x_gf, false);
      pmesh->ExchangeFaceNbrNodes();  
    }

    double LagrangianHydroOperator::GetTimeStepEstimate(const Vector &S) const
    {
     
      UpdateMesh(S);
      UpdateQuadratureData(S); 
      double glob_dt_est;
      const MPI_Comm comm = H1.GetParMesh()->GetComm();
      MPI_Allreduce(&qdata.dt_est, &glob_dt_est, 1, MPI_DOUBLE, MPI_MIN, comm);
      return glob_dt_est;  
    }

    void LagrangianHydroOperator::ResetTimeStepEstimate() const
    {
      qdata.dt_est = std::numeric_limits<double>::infinity();
    }

    void LagrangianHydroOperator::ComputeDensity(ParGridFunction &rho) const
    {
      rho.SetSpace(&L2);
      DenseMatrix Mrho(l2dofs_cnt);
      Vector rhs(l2dofs_cnt), rho_z(l2dofs_cnt);
      Array<int> dofs(l2dofs_cnt);
      DenseMatrixInverse inv(&Mrho);
      MassIntegrator lmi(&ir);
      DensityIntegrator di(rho0DetJ0_gf);
      di.SetIntRule(&ir);
      for (int e = 0; e < NE; e++)
	{
	  L2.GetElementDofs(e, dofs);
	  const FiniteElement &fe = *L2.GetFE(e);
	  ElementTransformation &eltr = *L2.GetElementTransformation(e);
	  di.AssembleRHSElementVect(fe, eltr, rhs);
	  lmi.AssembleElementMatrix(fe, eltr, Mrho);
	  inv.Factor();
	  inv.Mult(rhs, rho_z);
	  rho.SetSubVector(dofs, rho_z);	  
	}
    }


    double LagrangianHydroOperator::InternalEnergy(const ParGridFunction &gf) const
    {
      double glob_ie = 0.0;
      
      Vector one(l2dofs_cnt), loc_e(l2dofs_cnt);
      one = 1.0;
      Array<int> l2dofs;
      double loc_ie = 0.0;
      for (int e = 0; e < NE; e++)
	{
	  L2.GetElementDofs(e, l2dofs);
	  gf.GetSubVector(l2dofs, loc_e);
	  loc_ie += Me(e).InnerProduct(loc_e, one);
	}
      MPI_Comm comm = H1.GetParMesh()->GetComm();
      MPI_Allreduce(&loc_ie, &glob_ie, 1, MPI_DOUBLE, MPI_SUM, comm);
      
      return glob_ie;
    }

    double LagrangianHydroOperator::KineticEnergy(const ParGridFunction &v) const
    {
      double glob_ke = 0.0;
      // This should be turned into a kernel so that it could be displayed in pa
      double loc_ke = 0.5 * Mv_spmat_copy.InnerProduct(v, v);
      MPI_Allreduce(&loc_ke, &glob_ke, 1, MPI_DOUBLE, MPI_SUM,
		    H1.GetParMesh()->GetComm());
      return glob_ke;
    }

    void LagrangianHydroOperator::OutputSedovRho() const
    {
       MFEM_VERIFY(L2.GetNRanks() == 1, "Use only in serial!");
       MFEM_VERIFY(dim == 2, "Use only in 2D!");

       std::ofstream fstream_rho;
       fstream_rho.open("./sedov_out/rho.out");
       fstream_rho.precision(8);

       const int nqp = ir.GetNPoints();
       Vector pos(dim);
       for (int e = 0; e < NE; e++)
       {
          ElementTransformation &Tr = *L2.GetElementTransformation(e);
          for (int q = 0; q < nqp; q++)
          {
             const IntegrationPoint &ip = ir.IntPoint(q);
             Tr.SetIntPoint(&ip);
             Tr.Transform(ip, pos);

             double r = sqrt(pos(0)*pos(0) + pos(1)*pos(1));

             double rho = rho0DetJ0_gf.GetValue(Tr, ip)
                          / Tr.Weight() / alphaCut->GetValue(Tr, ip);
             fstream_rho << r << " " << rho << "\n";
             fstream_rho.flush();
          }
       }
       fstream_rho.close();
    }

    void LagrangianHydroOperator::UpdateQuadratureData(const Vector &S) const
    {
      if (qdata_is_current) { return; }

      qdata_is_current = true;
      forcemat_is_assembled = false;
      energyforcemat_is_assembled = false;
      sourcevec_is_assembled = false;
      
      // This code is only for the 1D/FA mode
      const int nqp = ir.GetNPoints();
      ParGridFunction x, v, e;
      Vector* sptr = const_cast<Vector*>(&S);
      x.MakeRef(&H1, *sptr, 0);
      v.MakeRef(&H1, *sptr, H1.GetVSize());
      e.MakeRef(&L2, *sptr, 2*H1.GetVSize());
      Vector e_vals;
      DenseMatrix Jpi(dim), sgrad_v(dim), Jinv(dim);

      // Batched computations are needed, because hydrodynamic codes usually
      // involve expensive computations of material properties. Although this
      // miniapp uses simple EOS equations, we still want to represent the batched
      // cycle structure.
      int nzones_batch = 3;
      const int nbatches =  NE / nzones_batch + 1; // +1 for the remainder.
      int nqp_batch = nqp * nzones_batch;
      double *gamma_b = new double[nqp_batch],
	*rho_b = new double[nqp_batch],
	*e_b   = new double[nqp_batch],
	*p_b   = new double[nqp_batch],
	*cs_b  = new double[nqp_batch];
      // Jacobians of reference->physical transformations for all quadrature points
      // in the batch.
      DenseTensor *Jpr_b = new DenseTensor[nzones_batch];    
      for (int b = 0; b < nbatches; b++)
	{
	  int z_id = b * nzones_batch; // Global index over zones.
	  // The last batch might not be full.
	  if (z_id == NE) { break; }
	  else if (z_id + nzones_batch > NE)
	    {
	      nzones_batch = NE - z_id;
	      nqp_batch    = nqp * nzones_batch;
	    }

	  double min_detJ = std::numeric_limits<double>::infinity();
	  for (int z = 0; z < nzones_batch; z++)
	    {
	      ElementTransformation *T = H1.GetElementTransformation(z_id);
	      Jpr_b[z].SetSize(dim, dim, nqp);
	      e.GetValues(z_id, ir, e_vals);
	      for (int q = 0; q < nqp; q++)
		{
		  const IntegrationPoint &ip = ir.IntPoint(q);
		  T->SetIntPoint(&ip);
		  double volumeFraction = alphaCut->GetValue(*T, ip);
		  
		  Jpr_b[z](q) = T->Jacobian();
		  const double detJ = Jpr_b[z](q).Det();
		  min_detJ = fmin(min_detJ, detJ);
		  const int idx = z * nqp + q;
		  double rho0DetJ0 = rho0DetJ0_gf.GetValue(*T, ip);
		  
		  // Assuming piecewise constant gamma that moves with the mesh.
		  gamma_b[idx] = gamma_gf(z_id);		
		  rho_b[idx] = rho0DetJ0 / (detJ * volumeFraction);
		  e_b[idx] = fmax(0.0, e_vals(q));
		}
	      ++z_id;
	    }
	  
	  // Batched computation of material properties.
	  ComputeMaterialProperties(nqp_batch, gamma_b, rho_b, e_b, p_b, cs_b);
	  
	  z_id -= nzones_batch;
	  for (int z = 0; z < nzones_batch; z++)
	    {
		ElementTransformation *T = H1.GetElementTransformation(z_id);
	
		for (int q = 0; q < nqp; q++)
		{
		  const IntegrationPoint &ip = ir.IntPoint(q);
		  T->SetIntPoint(&ip);
		  // Note that the Jacobian was already computed above. We've chosen
		  // not to store the Jacobians for all batched quadrature points.
		  const DenseMatrix &Jpr = Jpr_b[z](q);
		  CalcInverse(Jpr, Jinv);
		  const double detJ = Jpr.Det(), rho = rho_b[z*nqp + q],
		    p = p_b[z*nqp + q], sound_speed = cs_b[z*nqp + q], energy = e_b[z*nqp + q];
		  double visc_coeff = 0.0;
		  if (use_viscosity)
		    {
		      // Compression-based length scale at the point. The first
		      // eigenvector of the symmetric velocity gradient gives the
		      // direction of maximal compression. This is used to define the
		      // relative change of the initial length scale.
		      v.GetVectorGradient(*T, sgrad_v);

		      double vorticity_coeff = 1.0;
		      if (use_vorticity)
			{
			  const double grad_norm = sgrad_v.FNorm();
			  const double div_v = fabs(sgrad_v.Trace());
			  vorticity_coeff = (grad_norm > 0.0) ? div_v / grad_norm : 1.0;
			}

		      sgrad_v.Symmetrize();
		      double eig_val_data[3], eig_vec_data[9];
		      if (dim==1)
			{
			  eig_val_data[0] = sgrad_v(0, 0);
			  eig_vec_data[0] = 1.;
			}
		      else { sgrad_v.CalcEigenvalues(eig_val_data, eig_vec_data); }
		      Vector compr_dir(eig_vec_data, dim);
		      // Computes the initial->physical transformation Jacobian.
		      Vector Jac0inv_vec(dim*dim);
		      Jac0inv_vec = 0.0;
		      Jac0inv_gf.GetVectorValue(T->ElementNo,ip,Jac0inv_vec);
		      DenseMatrix Jac0inv(dim);
		      ConvertVectorToDenseMatrix(dim, Jac0inv_vec, Jac0inv);
	  
		      mfem::Mult(Jpr, Jac0inv, Jpi);
		      Vector ph_dir(dim); Jpi.Mult(compr_dir, ph_dir);
		      // Change of the initial mesh size in the compression direction.
		      const double h = qdata.h0 * ph_dir.Norml2() /
			compr_dir.Norml2();
		      // Measure of maximal compression.
		      const double mu = eig_val_data[0];
		      visc_coeff = 2.0 * rho * h * h * fabs(mu);
		    
		      // The following represents a "smooth" version of the statement
		      // "if (mu < 0) visc_coeff += 0.5 rho h sound_speed".  Note that
		      // eps must be scaled appropriately if a different unit system is
		      // being used.
		      const double eps = 1e-12;
		      visc_coeff += 0.5 * rho * h * sound_speed * vorticity_coeff *
			(1.0 - smooth_step_01(mu - 2.0 * eps, eps));
		    }
		  // Time step estimate at the point. Here the more relevant length
		  // scale is related to the actual mesh deformation; we use the min
		  // singular value of the ref->physical Jacobian. In addition, the
		  // time step estimate should be aware of the presence of shocks.
		  const double h_min =
		    Jpr.CalcSingularvalue(dim-1) / (double) H1.GetOrder(0);
		  const double inv_dt = sound_speed / h_min +
		    2.5 * visc_coeff / rho / h_min / h_min;
		  if (min_detJ < 0.0)
		    {
		      // This will force repetition of the step with smaller dt.
		      qdata.dt_est = 0.0;
		    }
		  else
		    {
		      if (inv_dt>0.0)
			{
			  qdata.dt_est = fmin(qdata.dt_est, cfl*(1.0/inv_dt));
			}
		    }
		}
	      
	      ++z_id;
	    }
	}
      
      delete [] gamma_b;
      delete [] rho_b;
      delete [] e_b;
      delete [] p_b;
      delete [] cs_b;
      delete [] Jpr_b;
    }


    void LagrangianHydroOperator::AssembleForceMatrix() const
    {
      Force = 0.0;
      Force.Assemble();
      forcemat_is_assembled = true;
    }

    void LagrangianHydroOperator::AssembleSourceVector() const
    {
      SourceForce = 0.0;
      ParGridFunction accel_src_gf(&H1);
      if (source_type == 2)
	{
	  RTCoefficient accel_coeff(dim);
	  accel_src_gf.ProjectCoefficient(accel_coeff);
	  accel_src_gf.ExchangeFaceNbrData();

	  // accel_src_gf.Read();
	  sfi->SetAccelerationGridFunction(&accel_src_gf);
	  SourceForce.Assemble();
	}
      sourcevec_is_assembled = true;
    }

    void LagrangianHydroOperator::AssembleEnergyForceMatrix() const
    {
      EnergyForce = 0.0;
      EnergyForce.Assemble();
      energyforcemat_is_assembled = true;
    }

    void LagrangianHydroOperator::AssembleDiffusionVelocityBoundaryForceMatrix() const
    {   
      DiffusionVelocityBoundaryForce = 0.0;
      DiffusionVelocityBoundaryForce.Assemble();

      bvdiffusion_forcemat_is_assembled = true;
    }

    void LagrangianHydroOperator::AssembleDiffusionEnergyBoundaryForceMatrix() const
    {
      DiffusionEnergyBoundaryForce = 0.0;
      DiffusionEnergyBoundaryForce.Assemble();
      bediffusion_forcemat_is_assembled = true;
    }


  } // namespace hydrodynamics

  void HydroODESolver::Init(TimeDependentOperator &tdop)
  {
    ODESolver::Init(tdop);
    hydro_oper = dynamic_cast<hydrodynamics::LagrangianHydroOperator *>(f);
    MFEM_VERIFY(hydro_oper, "HydroSolvers expect LagrangianHydroOperator.");
  }

  void RK2AvgSolver::Init(TimeDependentOperator &tdop)
  {
    HydroODESolver::Init(tdop);
    const Array<int> &block_offsets = hydro_oper->GetBlockOffsets();
    V.SetSize(block_offsets[1], mem_type);
    dS_dt.Update(block_offsets, mem_type);
    dS_dt = 0.0;
    S0.Update(block_offsets, mem_type);
    S_init.Update(block_offsets, mem_type);
  }

  void RK2AvgSolver::Step(Vector &S, double &t, double &dt)
  {
    // storing the initial state at S_init.
    // counter is need to prevent continuous update.
    // S_init will be used to compute the velocity mass matrix which is stays fixed throughout the calculation
    // Velocity mass matrix Cannot be computed like with the strong b.c enforcement scenario
    // since the normal velocity penalty term has to be assembled to that mass matrix
    // and needs to be continuously updated at not be fixed at S_init.
    // So, S_init will preserve the initial state which is solely used to compute the velocity mass matrix
    if (counter == 0){
      S_init.Vector::operator=(S);
      counter++;
    }
    // The monolithic BlockVector stores the unknown fields as follows:
    // (Position, Velocity, Specific Internal Energy).
    S0.Vector::operator=(S);
    Vector &v0 = S0.GetBlock(1);
    Vector &dx_dt = dS_dt.GetBlock(0);
    Vector &dv_dt = dS_dt.GetBlock(1);

    // In each sub-step:
    // - Update the global state Vector S.
    // - Compute dv_dt using S.
    // - Update V using dv_dt.
    // - Compute de_dt and dx_dt using S and V.

    // -- 1.
    // S is S0.
    hydro_oper->UpdateMesh(S);
    hydro_oper->UpdateLevelSet(S, S_init);
   
    hydro_oper->SolveVelocity(S, dS_dt, S_init, dt);
    // V = v0 + 0.5 * dt * dv_dt;
    add(v0, 0.5 * dt, dv_dt, V);
    hydro_oper->SolveEnergy(S, V, dS_dt);
    dx_dt = V;

    // -- 2.
    // S = S0 + 0.5 * dt * dS_dt;
    add(S0, 0.5 * dt, dS_dt, S);
    hydro_oper->ResetQuadratureData();
    hydro_oper->UpdateMesh(S);
    hydro_oper->UpdateLevelSet(S, S_init);   
   
    hydro_oper->SolveVelocity(S, dS_dt, S_init, dt);
    // V = v0 + 0.5 * dt * dv_dt;
    add(v0, 0.5 * dt, dv_dt, V);
    hydro_oper->SolveEnergy(S, V, dS_dt);
    dx_dt = V;

    // -- 3.
    // S = S0 + dt * dS_dt.
    add(S0, dt, dS_dt, S);
    hydro_oper->ResetQuadratureData();
    t += dt;
  }

} // namespace mfem

#endif // MFEM_USE_MPI
