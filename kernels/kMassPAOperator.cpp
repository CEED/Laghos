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

#include "../laghos_assembly.hpp"

#include "kMassPAOperator.hpp"
#include "backends/kernels/kernels.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{

namespace hydrodynamics
{

// *****************************************************************************
kMassPAOperator::kMassPAOperator(QuadratureData *qd_,
                                 ParFiniteElementSpace &fes_,
                                 const IntegrationRule &ir_) :
      AbcMassPAOperator(fes_.GetVSize()),
      dim(fes_.GetMesh()->Dimension()),
      nzones(fes_.GetMesh()->GetNE()),
      quad_data(qd_),
      fes(fes_),
      ir(ir_),
      ess_tdofs_count(0),
      ess_tdofs(0),
      bilinearForm(NULL) { }

// *****************************************************************************
void kMassPAOperator::Setup()
{
   push(Wheat);
   const mfem::Engine &engine = fes.GetMesh()->GetEngine();
   kernels::KernelsMassIntegrator &massInteg = *(new kernels::KernelsMassIntegrator(engine));
   massInteg.SetIntegrationRule(ir);
   massInteg.SetOperator(quad_data->rho0DetJ0w);
   bilinearForm = new kernels::kBilinearForm(fes.Get_PFESpace().As<kernels::kFiniteElementSpace>());
   dbg("bilinearForm->AddDomainIntegrator");
   bilinearForm->AddDomainIntegrator(&massInteg);
   dbg("bilinearForm->Assemble");
   bilinearForm->Assemble();
   // ?! no constraintList: dealt with 'each velocity component'
   dbg("bilinearForm->FormOperator");
   bilinearForm->FormOperator(Array<int>(), massOperator); // which is a KernelsConstrainedOperator
   dbg("done");
   pop();
}

// *************************************************************************
void kMassPAOperator::SetEssentialTrueDofs(mfem::Array<int> &dofs)
{
   push(Wheat);
   //dbg("\n\033[33;7m[SetEssentialTrueDofs] dofs.Size()=%d\033[m",dofs.Size());
   ess_tdofs_count = dofs.Size();
  
   if (ess_tdofs.Size()==0){
#ifdef MFEM_USE_MPI
      int global_ess_tdofs_count;
      const MPI_Comm comm = fes.GetParMesh()->GetComm();
      MPI_Allreduce(&ess_tdofs_count,&global_ess_tdofs_count,
                    1, MPI_INT, MPI_SUM, comm);
      assert(global_ess_tdofs_count>0);
      //dbg("Resize of %d",global_ess_tdofs_count);
      ess_tdofs.Resize(ess_tdofs_count);
#else
      assert(ess_tdofs_count>0);
      ess_tdofs.Resize(ess_tdofs_count);
#endif
   }else assert(ess_tdofs_count<=ess_tdofs.Size());

   assert(ess_tdofs>0);
  
   if (ess_tdofs_count == 0) { pop(); return; }
  
   {
      dbg("rHtoD");
      assert(ess_tdofs_count>0);
      assert(dofs.GetData());
      kernels::kmemcpy::rHtoD((void*)ess_tdofs.GetData(),
                              dofs.GetData(),
                              ess_tdofs_count*sizeof(int));
      pop();
   }
   pop();
}

// *****************************************************************************
void kMassPAOperator::EliminateRHS(mfem::Vector &b)
{
   push(Wheat);
   if (ess_tdofs_count > 0){
      mfem::Vector mb(fes.GetVLayout());//massOperator->InLayout());
      mb.PushData(b.GetData());
      kernels::Vector rb = mb.Get_PVector()->As<const kernels::Vector>();
      rb.SetSubVector(ess_tdofs, 0.0, ess_tdofs_count);
      b=mb;
   }
   pop();
}

// *************************************************************************
void kMassPAOperator::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
   push();
   /*dbg("\033[32;1;7m****kMassPAOperator::Mult ****\033[m");
   dbg("\033[32;1;7m[kMassPAOperator::Mult] x.Size()=%d\033[m",x.Size());
   dbg("\033[32;1;7m[kMassPAOperator::Mult] y.Size()=%d\033[m",y.Size());
   const std::size_t isz = massOperator->InLayout()->Size();
   const std::size_t osz = massOperator->OutLayout()->Size();
   dbg("\033[32;1;7m[kMassPAOperator::Mult] massOperator->InLayout()=%d\033[m",isz);
   dbg("\033[32;1;7m[kMassPAOperator::Mult] massOperator->OutLayout()=%d\033[m",osz);
   mfem::Vector mx(fes.GetVLayout());//isz);
   mx.Pull(); //mx.PushData(x.GetData());
   mfem::Vector my(fes.GetVLayout());//osz);
   my.Pull(false);*/

   mfem::Vector mx(fes.GetVLayout());
   mx.PushData(x.GetData());
   kernels::Vector kx = mx.Get_PVector()->As<kernels::Vector>();
   
   Vector my(fes.GetVLayout());
   kernels::Vector ky = my.Get_PVector()->As<kernels::Vector>();

   if (ess_tdofs_count)
   {
      kx.SetSubVector(ess_tdofs, 0.0, ess_tdofs_count);
   }
   
   massOperator->Mult(mx, my);
   //ky.KernelsMem().copyFrom(my.GetData(), ky.Size() * sizeof(double));
   my.Push();
   
   if (ess_tdofs_count)
   {
      ky.SetSubVector(ess_tdofs, 0.0, ess_tdofs_count);
   }
   //ky.Fill(0.0);
   
   y = my;
   //dbg("y:\n"); y.Print();assert(__FILE__&&__LINE__&&false);
   
   pop();
}

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI
