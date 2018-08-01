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
   AbcMassPAOperator(*fes_.GetTrueVLayout()),
   dim(fes_.GetMesh()->Dimension()),
   nzones(fes_.GetMesh()->GetNE()),
   quad_data(qd_),
   fes(fes_),
   ir(ir_),
   ess_tdofs_count(0),
   ess_tdofs(0),
   bilinearForm(new kernels::kBilinearForm(fes.Get_PFESpace().As<kernels::kFiniteElementSpace>())) { }

// *****************************************************************************
void kMassPAOperator::Setup()
{
   push(Wheat);
   const mfem::Engine &engine = fes.GetMesh()->GetEngine();
   kernels::KernelsMassIntegrator &massInteg = *(new kernels::KernelsMassIntegrator(engine));
   massInteg.SetIntegrationRule(ir);
   massInteg.SetOperator(quad_data->rho0DetJ0w);
   bilinearForm->AddDomainIntegrator(&massInteg);
   bilinearForm->Assemble();
   bilinearForm->FormOperator(Array<int>(), massOperator);
   pop();
}

// *************************************************************************
void kMassPAOperator::SetEssentialTrueDofs(mfem::Array<int> &dofs)
{
   push(Wheat);
   dbg("ess_tdofs_count=%d, ess_tdofs.Size()=%d & dofs.Size()=%d",ess_tdofs_count, ess_tdofs.Size(), dofs.Size());
   ess_tdofs_count = dofs.Size();
  
   if (ess_tdofs.Size()==0){
      dbg("ess_tdofs.Size()==0");
#ifdef MFEM_USE_MPI
      int global_ess_tdofs_count;
      const MPI_Comm comm = fes.GetParMesh()->GetComm();
      MPI_Allreduce(&ess_tdofs_count,&global_ess_tdofs_count,
                    1, MPI_INT, MPI_SUM, comm);
      assert(global_ess_tdofs_count>0);
      ess_tdofs.Resize(ess_tdofs_count);
#else
      assert(ess_tdofs_count>0);
      ess_tdofs.Resize(ess_tdofs_count);
#endif
   }//else assert(ess_tdofs_count<=ess_tdofs.Size());

   assert(ess_tdofs>0);
  
   if (ess_tdofs_count == 0) { pop(); return; }
  
   {
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
      kernels::Vector kb = b.Get_PVector()->As<kernels::Vector>();
      kb.SetSubVector(ess_tdofs, 0.0, ess_tdofs_count);
   }
   pop();
}

// *************************************************************************
void kMassPAOperator::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
   push();

   mfem::Vector mx(fes.GetTrueVLayout());
   mx.PushData(x.GetData());
   
   kernels::Vector &kx = mx.Get_PVector()->As<kernels::Vector>();
   kernels::Vector &ky = y.Get_PVector()->As<kernels::Vector>();

   if (ess_tdofs_count)
   {
      kx.SetSubVector(ess_tdofs, 0.0, ess_tdofs_count);
   }
   
   massOperator->Mult(mx, y);
   
   if (ess_tdofs_count)
   {
      ky.SetSubVector(ess_tdofs, 0.0, ess_tdofs_count);
   }
      
   pop();
}

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI
