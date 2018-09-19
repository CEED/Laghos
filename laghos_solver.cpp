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

#include "laghos_solver.hpp"
#include "qupdate/qupdate.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{

namespace hydrodynamics
{

void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    ParGridFunction &gf, const char *title,
                    int x, int y, int w, int h, bool vec)
{
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
         sock << "window_title '" << title << "'\n"
              << "window_geometry "
              << x << " " << y << " " << w << " " << h << "\n"
              << "keys maaAcl";
         if ( vec ) { sock << "vvv"; }
         sock << endl;
      }

      if (myid == 0)
      {
         connection_failed = !sock && !newly_opened;
      }
      MPI_Bcast(&connection_failed, 1, MPI_INT, 0, comm);
   }
   while (connection_failed);
}

LagrangianHydroOperator::LagrangianHydroOperator(int size,
                                                 ParFiniteElementSpace &h1_fes,
                                                 ParFiniteElementSpace &l2_fes,
                                                 Array<int> &essential_tdofs,
                                                 ParGridFunction &rho0,
                                                 int source_type_, double cfl_,
                                                 Coefficient *material_,
                                                 bool visc, bool pa,
                                                 double cgt, int cgiter,bool qu) :
#ifdef MFEM_USE_BACKENDS
   TimeDependentOperator(*h1_fes.GetMesh()->GetEngine().MakeLayout(size)),
#else
   TimeDependentOperator(size),
#endif
   H1FESpace(h1_fes), L2FESpace(l2_fes),
   H1compFESpace(h1_fes.GetParMesh(), h1_fes.FEColl(),1),
   ess_tdofs(essential_tdofs),
   dim(h1_fes.GetMesh()->Dimension()),
   nzones(h1_fes.GetMesh()->GetNE()),
   l2dofs_cnt(l2_fes.GetFE(0)->GetDof()),
   h1dofs_cnt(h1_fes.GetFE(0)->GetDof()),
   source_type(source_type_), cfl(cfl_),
   use_viscosity(visc), p_assembly(pa),
#ifdef MFEM_USE_BACKENDS
   has_engine(h1_fes.GetMesh()->HasEngine()),
#else
   has_engine(false),
#endif
   cg_rel_tol(cgt), cg_max_iter(cgiter),
   material_pcf(material_),
   Mv(&h1_fes), Me_inv(l2dofs_cnt, l2dofs_cnt, nzones),
   integ_rule(IntRules.Get(h1_fes.GetMesh()->GetElementBaseGeometry(),
                           3*h1_fes.GetOrder(0) + l2_fes.GetOrder(0) - 1)),
   quad_data(dim, nzones, integ_rule.GetNPoints()),
   quad_data_is_current(false),
   Force(&l2_fes, &h1_fes),
   ForcePA((!has_engine)?
           static_cast<AbcForcePAOperator*>(new  ForcePAOperator(&quad_data, h1_fes, l2_fes)):
           static_cast<AbcForcePAOperator*>(new kForcePAOperator(&quad_data, h1_fes, l2_fes,
                                                                 integ_rule,has_engine))),
   VMassPA((!has_engine)?
           static_cast<AbcMassPAOperator*>(new  MassPAOperator(&quad_data, H1FESpace)):
           static_cast<AbcMassPAOperator*>(new kMassPAOperator(&quad_data, H1compFESpace,
                                                               integ_rule))),
   VMassPA_prec(H1FESpace),
   locEMassPA(&quad_data, l2_fes),
   CG_VMass(H1FESpace.GetParMesh()->GetComm()),
   locCG(),
   timer(),
   VsizeL2(L2FESpace.GetVSize()),
   VsizeH1(H1FESpace.GetVSize()),
   x(VsizeH1),
   v(VsizeH1),
   e(VsizeL2),
   rhs(VsizeH1),
   B(H1compFESpace.GetTrueVSize()),
   X(H1compFESpace.GetTrueVSize()),
   one(VsizeH1),
   e_rhs(VsizeL2),
   rhs_c(H1compFESpace.GetVSize()),
   dv_c(H1compFESpace.GetVSize()),
   kv(H1FESpace.GetVSize()),
   qupdate(qu)
{
   push();
   one = 1.0;
   if (has_engine){
      dbg("has_engine");
      assert(this->InLayout());
      assert(InLayout()->Size()==size);
      x.Resize(H1FESpace.GetVLayout());
      v.Resize(H1FESpace.GetVLayout());
      e.Resize(L2FESpace.GetVLayout());
      rhs.Resize(H1FESpace.GetVLayout());
      one.Resize(L2FESpace.GetVLayout());
      one.Fill(1.0);
      e_rhs.Resize(L2FESpace.GetVLayout());
      rhs_c.Resize(H1compFESpace.GetVLayout());
      dv_c.Resize(H1compFESpace.GetVLayout());
      kv.Resize(H1FESpace.GetVLayout());
      B.Resize(H1compFESpace.GetTrueVLayout());
      X.Resize(H1compFESpace.GetTrueVLayout());
   }
   
   GridFunctionCoefficient rho_coeff(&rho0);

   if (!p_assembly){
      // Standard local assembly and inversion for energy mass matrices.
      DenseMatrix Me(l2dofs_cnt);
      DenseMatrixInverse inv(&Me);
      MassIntegrator mi(rho_coeff, &integ_rule);
      for (int i = 0; i < nzones; i++)
      {
         mi.AssembleElementMatrix(*l2_fes.GetFE(i),
                                  *l2_fes.GetElementTransformation(i), Me);
         inv.Factor();
         inv.GetInverseMatrix(Me_inv(i));
      }
   }

   // Standard assembly for the velocity mass matrix.
   if (!p_assembly)
   {
      VectorMassIntegrator *vmi = new VectorMassIntegrator(rho_coeff, &integ_rule);
      Mv.AddDomainIntegrator(vmi);
      Mv.Assemble();
   }

   // Values of rho0DetJ0 and Jac0inv at all quadrature points.
   const int nqp = integ_rule.GetNPoints();
   Vector rho_vals(nqp);
   for (int i = 0; i < nzones; i++)
   {
      rho0.GetValues(i, integ_rule, rho_vals);
      ElementTransformation *T = h1_fes.GetElementTransformation(i);
      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = integ_rule.IntPoint(q);
         T->SetIntPoint(&ip);

         DenseMatrixInverse Jinv(T->Jacobian());
         Jinv.GetInverseMatrix(quad_data.Jac0inv(i*nqp + q));

         const double rho0DetJ0 = T->Weight() * rho_vals(q);
         quad_data.rho0DetJ0w(i*nqp + q) = rho0DetJ0 *
                                           integ_rule.IntPoint(q).weight;
      }
   }
   
   if (has_engine){
#ifdef __NVCC__
      /*dbg("Jac0inv UseExternalData");
      const int ji_isz = quad_data.Jac0inv.SizeI();
      const int ji_jsz = quad_data.Jac0inv.SizeJ();
      const int ji_ksz = quad_data.Jac0inv.SizeK();
      double *ji_ext_data = (double*)mfem::kernels::kmalloc<double>::operator new(ji_isz*ji_jsz*ji_ksz);
      mfem::kernels::kmemcpy::rHtoD(ji_ext_data,
                                    quad_data.Jac0inv.Data(),
                                    ji_isz*ji_jsz*ji_ksz*sizeof(double));
      quad_data.Jac0inv.UseExternalData(ji_ext_data, ji_isz,ji_jsz,ji_ksz);
      */
      
      dbg("stressJinvT UseExternalData");
      const int si_isz = quad_data.stressJinvT.SizeI();
      const int si_jsz = quad_data.stressJinvT.SizeJ();
      const int si_ksz = quad_data.stressJinvT.SizeK();
      double *si_ext_data = (double*)mfem::kernels::kmalloc<double>::operator new(si_isz*si_jsz*si_ksz);
      mfem::kernels::kmemcpy::rHtoD(si_ext_data,
                                    quad_data.stressJinvT.Data(),
                                    si_isz*si_jsz*si_ksz*sizeof(double));
      quad_data./*d_*/stressJinvT.UseExternalData(si_ext_data, si_isz,si_jsz,si_ksz);
#endif
   }

   // Initial local mesh size (assumes all mesh elements are of the same type).
   double loc_area = 0.0, glob_area;
   int loc_z_cnt = nzones, glob_z_cnt;
   ParMesh *pm = H1FESpace.GetParMesh();
   for (int i = 0; i < nzones; i++) { loc_area += pm->GetElementVolume(i); }
   MPI_Allreduce(&loc_area, &glob_area, 1, MPI_DOUBLE, MPI_SUM, pm->GetComm());
   MPI_Allreduce(&loc_z_cnt, &glob_z_cnt, 1, MPI_INT, MPI_SUM, pm->GetComm());
   switch (pm->GetElementBaseGeometry(0))
   {
      case Geometry::SEGMENT:
         quad_data.h0 = glob_area / glob_z_cnt; break;
      case Geometry::SQUARE:
         quad_data.h0 = sqrt(glob_area / glob_z_cnt); break;
      case Geometry::TRIANGLE:
         quad_data.h0 = sqrt(2.0 * glob_area / glob_z_cnt); break;
      case Geometry::CUBE:
         quad_data.h0 = pow(glob_area / glob_z_cnt, 1.0/3.0); break;
      case Geometry::TETRAHEDRON:
         quad_data.h0 = pow(6.0 * glob_area / glob_z_cnt, 1.0/3.0); break;
      default: MFEM_ABORT("Unknown zone type!");
   }
   quad_data.h0 /= (double) H1FESpace.GetOrder(0);

   // ForceIntegrator assembles element contributions to the global force matrix.
   // This class is used for the full assembly case; it's not used with partial assembly.
   if (!p_assembly)
   {
      ForceIntegrator *fi = new ForceIntegrator(quad_data);
      fi->SetIntRule(&integ_rule);
      Force.AddDomainIntegrator(fi);
      // Make a dummy assembly to figure out the sparsity.
      Force.Assemble(0);
      Force.Finalize(0);
   }

   if (p_assembly)
   {
      // Compute the global 1D reference tensors.
      tensors1D = new Tensors1D(H1FESpace.GetFE(0)->GetOrder(),
                                L2FESpace.GetFE(0)->GetOrder(),
                                int(floor(0.7 + pow(nqp, 1.0 / dim))));
      evaluator = new FastEvaluator(H1FESpace);

      // Setup the preconditioner of the velocity mass operator.
      if (!has_engine){
         Vector d;
         (dim == 2) ? VMassPA->ComputeDiagonal2D(d) : VMassPA->ComputeDiagonal3D(d);
         VMassPA_prec.SetDiagonal(d);
      }
   }
   
   if (has_engine){
      VMassPA->Setup(); // quad_data has been filled, used for SetOperator
      //CG_VMass.SetPreconditioner(VMassPA_prec);
      CG_VMass.SetOperator(*VMassPA);
      CG_VMass.SetRelTol(cg_rel_tol);
      CG_VMass.SetAbsTol(0.0);
      CG_VMass.SetMaxIter(cg_max_iter);
      CG_VMass.SetPrintLevel(-1);
   }
   
   locCG.SetOperator(locEMassPA);
   locCG.iterative_mode = false;
   locCG.SetRelTol(1e-8);
   locCG.SetAbsTol(1e-8 * numeric_limits<double>::epsilon());
   locCG.SetMaxIter(200);
   locCG.SetPrintLevel(0);
   pop();
}

void LagrangianHydroOperator::Mult(const Vector &S, Vector &dS_dt) const
{
   push();
   dbg("\033[7mLagrangianHydroOperator::Mult");
   
   if (has_engine){
      //dbg("has_engine");
      //dbg("dS_dt.Resize");
      //const int dS_dt_sz = dS_dt.Size();
      ///const Engine &engine = H1FESpace.GetParMesh()->GetEngine();
      //dS_dt.SetEngine(engine);
      //dS_dt.Resize(engine.MakeLayout(dS_dt_sz));
   }
   dbg("dS_dt.Fill(0.0)");
   dS_dt.Fill(0.0);
      
   // Make sure that the mesh positions correspond to the ones in S. This is
   // needed only because some mfem time integrators don't update the solution
   // vector at every intermediate stage (hence they don't change the mesh).
   Vector* sptr = (Vector*) &S;
   
   ParGridFunction x;
   x.MakeRef(&H1FESpace, *sptr, 0);
   H1FESpace.GetParMesh()->NewNodes(x, false);

   dbg("UpdateQuadratureData");
   UpdateQuadratureData(S);

   // The monolithic BlockVector stores the unknown fields as follows:
   // - Position
   // - Velocity
   // - Specific Internal Energy
   dbg("v.MakeRefOffset");
   v.MakeRefOffset(*sptr,VsizeH1);
   dbg("v.Size()=%d, VsizeH1=%d",v.Size(),VsizeH1);
   dbg("e.MakeRefOffset");
   e.MakeRefOffset(*sptr,VsizeH1*2);
   dbg("eSize()=%d, VsizeL2=%d",e.Size(),VsizeL2);

   dbg("dx, dv, de");
   ParGridFunction dx, dv, de;
   if (has_engine){
      dbg("Resizing");
      dx.Resize(H1FESpace.GetVLayout());
      dv.Resize(H1FESpace.GetVLayout());
      de.Resize(L2FESpace.GetVLayout());
      de.Pull(false);
   }
   
   dbg("dx");
   dx.MakeRefOffset(dS_dt, 0);
   dbg("dv");
   dv.MakeRefOffset(dS_dt, VsizeH1);
   dbg("de");
   de.MakeRefOffset(dS_dt, VsizeH1*2);

   dbg("Set dx_dt = v (explicit).");
   dx.Assign(v);

   if (!p_assembly)
   {
      Force = 0.0;
      timer.sw_force.Start();
      Force.Assemble();
      timer.sw_force.Stop();
   }

   // Solve for velocity.
   if (p_assembly)
   {
      timer.sw_force.Start();
      dbg("one.Size()=%d",one.Size());
      dbg("rhs.Size()=%d",rhs.Size());
      dbg("ForcePA->Mult(one, rhs)");
      ForcePA->Mult(one, rhs);
      //rhs.Pull();dbg("rhs:\n"); rhs.Print();assert(__FILE__ && __LINE__ && false);

      timer.sw_force.Stop();
      rhs.Axpby(-1.0, rhs, 0.0, one);
      //rhs.Pull();dbg("rhs:\n"); rhs.Print();assert(__FILE__ && __LINE__ && false);

      if (!has_engine)
      {
         Operator *cVMassPA;
         CGSolver cg(H1FESpace.GetParMesh()->GetComm());
         VMassPA->FormLinearSystem(ess_tdofs, dv, rhs, cVMassPA, X, B);
         cg.SetOperator(*cVMassPA);
         cg.SetRelTol(cg_rel_tol); cg.SetAbsTol(0.0);
         cg.SetMaxIter(cg_max_iter);
         cg.SetPrintLevel(0);
         timer.sw_cgH1.Start();
         cg.Mult(B, X);
         timer.sw_cgH1.Stop();
         timer.H1cg_iter += cg.GetNumIterations();
         VMassPA->RecoverFEMSolution(X, rhs, dv);
         delete cVMassPA;
         //dbg("\033[7mdv_c:");dv.Print();assert(__FILE__ && __LINE__ && false);
      }
      else
      {
         // Partial assembly solve for each velocity component
         const int size = H1compFESpace.GetVSize();
         kMassPAOperator *kVMassPA = static_cast<kMassPAOperator*>(VMassPA);

         for (int c = 0; c < dim; c++)
         {
            dbg("\033[7mdim #%d",c);
            dbg("\033[7mrhs_c:");
            rhs_c.MakeRefOffset(rhs, c*size);
            
            //dbg("rhs_c:\n"); rhs_c.Print();//assert(__FILE__ && __LINE__ && false);

            //dbg("\033[7mdv_c:");
            //dv_c.MakeRefOffset(dv, c*size);
            //dv_c.Pull();dbg("dv_c:\n"); dv_c.Print();assert(__FILE__ && __LINE__ && false);
            //dv_c.Push();
            
            mfem::Array<int> c_tdofs;
            const int bdr_attr_max = H1FESpace.GetMesh()->bdr_attributes.Max();
            mfem::Array<int> ess_bdr(bdr_attr_max);

            // Attributes 1/2/3 correspond to fixed-x/y/z boundaries, i.e.,
            // we must enforce v_x/y/z = 0 for the velocity components.
            ess_bdr = 0; ess_bdr[c] = 1;
            dbg("Essential true dofs as if there's only one component.");
            H1compFESpace.GetEssentialTrueDofs(ess_bdr, c_tdofs);
            dbg("done");

            dbg("dv_c.Fill(0.0);");
            dv_c.Fill(0.0);

            // *****************************************************************
            dbg("MultTranspose(rhs_c, B)");
            dbg("rhs_c.Size()=%d",rhs_c.Size());
            dbg("B.Size()=%d",B.Size());
            H1compFESpace.Get_PFESpace()->As<kernels::kFiniteElementSpace>().
               GetProlongationOperator()->MultTranspose(rhs_c, B);
            dbg("MultTranspose done, B.Size()=%d",B.Size());
            //B.Pull();dbg("B:\n"); B.Print();assert(__FILE__ && __LINE__ && false);
            
            dbg("GetRestrictionOperator->Mult");
            H1compFESpace.Get_PFESpace()->As<kernels::kFiniteElementSpace>().
               GetRestrictionOperator()->Mult(dv_c, X);

            dbg("SetEssentialTrueDofs");
            kVMassPA->SetEssentialTrueDofs(c_tdofs);
            dbg("EliminateRHS");
            kVMassPA->EliminateRHS(B);
            dbg("B.Size()=%d",B.Size());

            timer.sw_cgH1.Start();
            dbg("CG_VMass.Mult(B, X);");
            //B.Pull();dbg("B:\n"); B.Print();assert(__FILE__ && __LINE__ && false);
//0 -6.245e-17 0 0 -6.93889e-17 0 0.0757866 -1.11022e-16
//0.0378933 0 -0.0757866 0 -0.0378933 0.151573 -0.151573
// &
//0 -5.55112e-17 0 0.0378933 -1.11022e-16 0 -0.0378933 0
//0.151573 -0.151573 0 0 0 0 0
            //B.Pull();B=1.0;B.Push();
            CG_VMass.Mult(B, X); // linalg/solver.cpp
//0 -8.43076e-16 0 0 -9.36751e-16 0 1.02312 -1.4988e-15
//0.511559 0 -1.02312 0 -0.511559 2.04624 -2.04624
// &
//0 -7.32173e-16 0 0.499799 -1.46435e-15 0 -0.499799 0
//1.9992 -1.9992 0 0 0 0 0
            //X.Pull();dbg("X:\n"); X.Print();assert(__FILE__ && __LINE__ && false);
            //while(true);
            //assert(false);
            
            // then results differ because DOT differs (multi components here)
            timer.sw_cgH1.Stop();
            timer.H1cg_iter += CG_VMass.GetNumIterations();
            dbg("GetProlongationOperator->Mult");
            H1compFESpace.Get_PFESpace()->As<kernels::kFiniteElementSpace>().
               GetProlongationOperator()->Mult(X, dv_c);

            dbg("dv.Assign(dv_c)");
            ParGridFunction dvc;
            dvc.Resize(H1compFESpace.GetVLayout());
            dvc.MakeRefOffset(dS_dt, VsizeH1 + c*size);
            dvc.Assign(dv_c);
         }
         dbg("\033[7mend of for loop");
      } // engine
   }
   else // p_assembly
   {
      timer.sw_force.Start();
      Force.Mult(one, rhs);
      timer.sw_force.Stop();
      rhs.Neg();

      HypreParMatrix A;
      Mv.FormLinearSystem(ess_tdofs, dv, rhs, A, X, B);
      CGSolver cg(H1FESpace.GetParMesh()->GetComm());
      HypreSmoother prec;
      prec.SetType(HypreSmoother::Jacobi, 1);
      cg.SetPreconditioner(prec);
      cg.SetOperator(A);
      cg.SetRelTol(cg_rel_tol); cg.SetAbsTol(0.0);
      cg.SetMaxIter(cg_max_iter);
      cg.SetPrintLevel(0);
      timer.sw_cgH1.Start();
      cg.Mult(B, X);
      timer.sw_cgH1.Stop();
      timer.H1cg_iter += cg.GetNumIterations();
      Mv.RecoverFEMSolution(X, rhs, dv);
   }

   dbg("Solve for energy, assemble the energy source if such exists.");
   LinearForm *e_source = NULL;
   if (source_type == 1) // 2D Taylor-Green.
   {
      dbg("Taylor-Green");
      e_source = new LinearForm(&L2FESpace);
      TaylorCoefficient coeff;
      dbg("DomainLFIntegrator");
      DomainLFIntegrator *d = new DomainLFIntegrator(coeff, &integ_rule);
      e_source->AddDomainIntegrator(d);
      dbg("Assemble");
      e_source->Assemble();
   }
   
   Array<int> l2dofs;
   Vector loc_rhs(l2dofs_cnt), loc_de(l2dofs_cnt);
   
   if (p_assembly)
   {
      timer.sw_force.Start();
      dbg("MultTranspose");
      if (!has_engine) ForcePA->MultTranspose(v, e_rhs);
      else {
         //kv.PushData(v.GetData());
         ForcePA->MultTranspose(/*k*/v, e_rhs);
         dbg("done");
      }
      dbg("e_rhs.Pull()");
      e_rhs.Pull();
      dbg("done");
      timer.sw_force.Stop();

      if (e_source) {
         dbg("e_source");
         e_rhs += *e_source;
      }
      dbg("for zones");
      for (int z = 0; z < nzones; z++)
      {
         L2FESpace.GetElementDofs(z, l2dofs);
         e_rhs.GetSubVector(l2dofs, loc_rhs);
         locEMassPA.SetZoneId(z);
         timer.sw_cgL2.Start();
         locCG.Mult(loc_rhs, loc_de);
         timer.sw_cgL2.Stop();
         timer.L2dof_iter += locCG.GetNumIterations() * l2dofs_cnt;
         dbg("de.SetSubVector");
         de.SetSubVector(l2dofs, loc_de);
      }
      de.Push();
      dbg("done");
   }
   else
   {
      timer.sw_force.Start();
      Force.MultTranspose(v, e_rhs);
      timer.sw_force.Stop();
      if (e_source) { e_rhs += *e_source; }
      for (int z = 0; z < nzones; z++)
      {
         L2FESpace.GetElementDofs(z, l2dofs);
         e_rhs.GetSubVector(l2dofs, loc_rhs);
         timer.sw_cgL2.Start();
         Me_inv(z).Mult(loc_rhs, loc_de);
         timer.sw_cgL2.Stop();
         timer.L2dof_iter += l2dofs_cnt;
         de.SetSubVector(l2dofs, loc_de);
      }
   }
   delete e_source;
   dbg("finished");
   quad_data_is_current = false;
}

double LagrangianHydroOperator::GetTimeStepEstimate(const Vector &S) const
{
   Vector* sptr = (Vector*) &S;
   ParGridFunction x;
   x.MakeRef(&H1FESpace, *sptr, 0);
   H1FESpace.GetParMesh()->NewNodes(x, false);
   UpdateQuadratureData(S);

   double glob_dt_est;
   MPI_Allreduce(&quad_data.dt_est, &glob_dt_est, 1, MPI_DOUBLE, MPI_MIN,
                 H1FESpace.GetParMesh()->GetComm());
   return glob_dt_est;
}

void LagrangianHydroOperator::ResetTimeStepEstimate() const
{
   quad_data.dt_est = numeric_limits<double>::infinity();
}

void LagrangianHydroOperator::ComputeDensity(ParGridFunction &rho)
{
   rho.SetSpace(&L2FESpace);

   DenseMatrix Mrho(l2dofs_cnt);
   Vector rhs(l2dofs_cnt), rho_z(l2dofs_cnt);
   Array<int> dofs(l2dofs_cnt);
   DenseMatrixInverse inv(&Mrho);
   MassIntegrator mi(&integ_rule);
   DensityIntegrator di(quad_data);
   di.SetIntRule(&integ_rule);
   for (int i = 0; i < nzones; i++)
   {
      di.AssembleRHSElementVect(*L2FESpace.GetFE(i),
                                *L2FESpace.GetElementTransformation(i), rhs);
      mi.AssembleElementMatrix(*L2FESpace.GetFE(i),
                               *L2FESpace.GetElementTransformation(i), Mrho);
      inv.Factor();
      inv.Mult(rhs, rho_z);
      L2FESpace.GetElementDofs(i, dofs);
      rho.SetSubVector(dofs, rho_z);
   }
}

void LagrangianHydroOperator::PrintTimingData(bool IamRoot, int steps)
{
   double my_rt[5], rt_max[5];
   my_rt[0] = timer.sw_cgH1.RealTime();
   my_rt[1] = timer.sw_cgL2.RealTime();
   my_rt[2] = timer.sw_force.RealTime();
   my_rt[3] = timer.sw_qdata.RealTime();
   my_rt[4] = my_rt[0] + my_rt[2] + my_rt[3];
   MPI_Reduce(my_rt, rt_max, 5, MPI_DOUBLE, MPI_MAX, 0, H1FESpace.GetComm());

   HYPRE_Int mydata[2], alldata[2];
   mydata[0] = timer.L2dof_iter;
   mydata[1] = timer.quad_tstep;
   MPI_Reduce(mydata, alldata, 2, HYPRE_MPI_INT, MPI_SUM, 0,
              H1FESpace.GetComm());

   if (IamRoot)
   {
      const HYPRE_Int H1gsize = H1FESpace.GlobalTrueVSize(),
                      L2gsize = L2FESpace.GlobalTrueVSize();
      using namespace std;
      cout << endl;
      cout << "CG (H1) total time: " << rt_max[0] << endl;
      cout << "CG (H1) rate (megadofs x cg_iterations / second): "
           << 1e-6 * H1gsize * timer.H1cg_iter / rt_max[0] << endl;
      cout << endl;
      cout << "CG (L2) total time: " << rt_max[1] << endl;
      cout << "CG (L2) rate (megadofs x cg_iterations / second): "
           << 1e-6 * alldata[0] / rt_max[1] << endl;
      cout << endl;
      // The Force operator is applied twice per time step, on the H1 and the L2
      // vectors, respectively.
      cout << "Forces total time: " << rt_max[2] << endl;
      cout << "Forces rate (megadofs x timesteps / second): "
           << 1e-6 * steps * (H1gsize + L2gsize) / rt_max[2] << endl;
      cout << endl;
      cout << "UpdateQuadData total time: " << rt_max[3] << endl;
      cout << "UpdateQuadData rate (megaquads x timesteps / second): "
           << 1e-6 * alldata[1] * integ_rule.GetNPoints() / rt_max[3] << endl;
      cout << endl;
      cout << "Major kernels total time (seconds): " << rt_max[4] << endl;
      cout << "Major kernels total rate (megadofs x time steps / second): "
           << 1e-6 * H1gsize * steps / rt_max[4] << endl;
   }
}

LagrangianHydroOperator::~LagrangianHydroOperator()
{
   delete tensors1D;
   delete ForcePA;
   delete VMassPA;
}

void LagrangianHydroOperator::UpdateQuadratureData(const Vector &S) const
{
//#warning UpdateQuadratureData drand48
   //const size_t mx = S.Size();
   //Vector *S_p = (Vector*) &S;
   //for(size_t k=0;k<mx;k+=1) S_p->GetData()[k] = drand48();
   if (qupdate) {
      QUpdate(dim,nzones,l2dofs_cnt,h1dofs_cnt,use_viscosity,p_assembly,cfl,
              timer,material_pcf,integ_rule,H1FESpace,L2FESpace,S,
              quad_data_is_current,quad_data);
      return;
   }
   StdUpdateQuadratureData(S);
}
   
void LagrangianHydroOperator::StdUpdateQuadratureData(const Vector &S) const
{
   if (quad_data_is_current) { return; }
   push();
   timer.sw_qdata.Start();

   const int nqp = integ_rule.GetNPoints();

   ParGridFunction x, v, e;
   Vector* sptr = (Vector*) &S;
   dbg("x");
   x.MakeRef(&H1FESpace, *sptr, 0);
   dbg("v");
   v.MakeRef(&H1FESpace, *sptr, H1FESpace.GetVSize());
   dbg("e");
   e.MakeRef(&L2FESpace, *sptr, 2*H1FESpace.GetVSize());
   //for(size_t k=0;k<L2FESpace.GetVSize();k+=1) dbg("\te[%d]=%f",k,e.GetData()[k]);
   Vector e_vals, e_loc(l2dofs_cnt), vector_vals(h1dofs_cnt * dim);
   DenseMatrix Jpi(dim), sgrad_v(dim), Jinv(dim), stress(dim), stressJiT(dim),
               vecvalMat(vector_vals.GetData(), h1dofs_cnt, dim);
   DenseTensor grad_v_ref(dim, dim, nqp);
   Array<int> L2dofs, H1dofs;

   // Batched computations are needed, because hydrodynamic codes usually
   // involve expensive computations of material properties. Although this
   // miniapp uses simple EOS equations, we still want to represent the batched
   // cycle structure.
   int nzones_batch = 3;
   const int nbatches =  nzones / nzones_batch + 1; // +1 for the remainder.
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
      if (z_id == nzones) { break; }
      else if (z_id + nzones_batch > nzones)
      {
         nzones_batch = nzones - z_id;
         nqp_batch    = nqp * nzones_batch;
      }

      double min_detJ = numeric_limits<double>::infinity();
      for (int z = 0; z < nzones_batch; z++)
      {
         ElementTransformation *T = H1FESpace.GetElementTransformation(z_id);
         Jpr_b[z].SetSize(dim, dim, nqp);

         if (p_assembly)
         {
            // Energy values at quadrature point.
            L2FESpace.GetElementDofs(z_id, L2dofs);
            e.GetSubVector(L2dofs, e_loc);
            evaluator->GetL2Values(e_loc, e_vals);

            // All reference->physical Jacobians at the quadrature points.
            H1FESpace.GetElementVDofs(z_id, H1dofs);
            x.GetSubVector(H1dofs, vector_vals);
            evaluator->GetVectorGrad(vecvalMat, Jpr_b[z]);
         }
         else { e.GetValues(z_id, integ_rule, e_vals); }
         for (int q = 0; q < nqp; q++)
         {
            const IntegrationPoint &ip = integ_rule.IntPoint(q);
            T->SetIntPoint(&ip);
            if (!p_assembly) { Jpr_b[z](q) = T->Jacobian(); }
            const double detJ = Jpr_b[z](q).Det();
            min_detJ = min(min_detJ, detJ);

            const int idx = z * nqp + q;
            if (material_pcf == NULL) { gamma_b[idx] = 5./3.; } // Ideal gas.
            else { gamma_b[idx] = material_pcf->Eval(*T, ip); }
            rho_b[idx] = quad_data.rho0DetJ0w(z_id*nqp + q) / detJ / ip.weight;
            e_b[idx]   = max(0.0, e_vals(q));
         }
         ++z_id;
      }

      // Batched computation of material properties.
      ComputeMaterialProperties(nqp_batch, gamma_b, rho_b, e_b, p_b, cs_b);

      z_id -= nzones_batch;
      for (int z = 0; z < nzones_batch; z++)
      {
         ElementTransformation *T = H1FESpace.GetElementTransformation(z_id);
         if (p_assembly)
         {
            // All reference->physical Jacobians at the quadrature points.
            H1FESpace.GetElementVDofs(z_id, H1dofs);
            v.GetSubVector(H1dofs, vector_vals);
            evaluator->GetVectorGrad(vecvalMat, grad_v_ref);
         }
         for (int q = 0; q < nqp; q++)
         {
            const IntegrationPoint &ip = integ_rule.IntPoint(q);
            T->SetIntPoint(&ip);
            // Note that the Jacobian was already computed above. We've chosen
            // not to store the Jacobians for all batched quadrature points.
            const DenseMatrix &Jpr = Jpr_b[z](q);
            CalcInverse(Jpr, Jinv);
            const double detJ = Jpr.Det(), rho = rho_b[z*nqp + q],
                         p = p_b[z*nqp + q], sound_speed = cs_b[z*nqp + q];
            //printf("\n\t[%d,%d] %f %f %f %f",b/*z*/,q,detJ,rho,p,sound_speed);
            stress = 0.0;
            for (int d = 0; d < dim; d++) { stress(d, d) = -p; }

            double visc_coeff = 0.0;
            if (use_viscosity)
            {
               // Compression-based length scale at the point. The first
               // eigenvector of the symmetric velocity gradient gives the
               // direction of maximal compression. This is used to define the
               // relative change of the initial length scale.
               if (p_assembly)
               {
                  mfem::Mult(grad_v_ref(q), Jinv, sgrad_v);
               }
               else
               {
                  v.GetVectorGradient(*T, sgrad_v);
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
               mfem::Mult(Jpr, quad_data.Jac0inv(z_id*nqp + q), Jpi);
               Vector ph_dir(dim); Jpi.Mult(compr_dir, ph_dir);
               // Change of the initial mesh size in the compression direction.
               const double h = quad_data.h0 * ph_dir.Norml2() /
                                compr_dir.Norml2();

               // Measure of maximal compression.
               const double mu = eig_val_data[0];
               visc_coeff = 2.0 * rho * h * h * fabs(mu);
               if (mu < 0.0) { visc_coeff += 0.5 * rho * h * sound_speed; }
               stress.Add(visc_coeff, sgrad_v);
            }

            // Time step estimate at the point. Here the more relevant length
            // scale is related to the actual mesh deformation; we use the min
            // singular value of the ref->physical Jacobian. In addition, the
            // time step estimate should be aware of the presence of shocks.
            const double h_min =
               Jpr.CalcSingularvalue(dim-1) / (double) H1FESpace.GetOrder(0);
            const double inv_dt = sound_speed / h_min +
                                  2.5 * visc_coeff / rho / h_min / h_min;
            if (min_detJ < 0.0)
            {
               // This will force repetition of the step with smaller dt.
               quad_data.dt_est = 0.0;
            }
            else
            {
               quad_data.dt_est = min(quad_data.dt_est, cfl * (1.0 / inv_dt) );
            }

            // Quadrature data for partial assembly of the force operator.
            MultABt(stress, Jinv, stressJiT);
            stressJiT *= integ_rule.IntPoint(q).weight * detJ;
            for (int vd = 0 ; vd < dim; vd++)
            {
               for (int gd = 0; gd < dim; gd++)
               {
                  quad_data.stressJinvT(vd)(z_id*nqp + q, gd) =
                     stressJiT(vd, gd);
               }
            }
         }
         ++z_id;
      }
   }
   dbg("\033[7mdt_est=%.15e",quad_data.dt_est);
   //assert(false);
   delete [] gamma_b;
   delete [] rho_b;
   delete [] e_b;
   delete [] p_b;
   delete [] cs_b;
   delete [] Jpr_b;
   quad_data_is_current = true;

   timer.sw_qdata.Stop();
   timer.quad_tstep += nzones;
}

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI
