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
#include "kernels.hpp"

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
   AbcMassPAOperator(fes_.GetVSize()/**fes_.GetTrueVLayout()*/),
   dim(fes_.GetMesh()->Dimension()),
   nzones(fes_.GetMesh()->GetNE()),
   quad_data(qd_),
   fes(fes_),
   ir(ir_),
   ess_tdofs_count(0),
   ess_tdofs(0),
   bilinearForm(new mfem::PABilinearForm(&fes_)),//*(new kFiniteElementSpace(static_cast<FiniteElementSpace*>(&fes_))))),
   //bilinearForm(new kernels::kBilinearForm(&fes.Get_PFESpace()->As<kernels::kFiniteElementSpace>())),
   massOperator(/*NULL*/)
{
   push(Wheat);
   pop();
}

// *****************************************************************************
void kMassPAOperator::Setup()
{
//#warning Setup
   assert(false);
   push(Wheat);
   //const mfem::Engine &engine = fes.GetMesh()->GetEngine();
   mfem::PAMassIntegrator *massInteg = new mfem::PAMassIntegrator(/*engine*/);
   massInteg->SetIntegrationRule(ir);
   //massInteg->SetOperator(quad_data->rho0DetJ0w);
   bilinearForm->AddDomainIntegrator(massInteg);
   bilinearForm->Assemble();
   //bilinearForm->FormOperator(mfem::Array<int>(), massOperator);
   //pop();
}

// *************************************************************************
void kMassPAOperator::SetEssentialTrueDofs(mfem::Array<int> &dofs)
{
   push(Wheat);
   ess_tdofs_count = dofs.Size();
   
   if (ess_tdofs.Size()==0){
#ifdef MFEM_USE_MPI
      int global_ess_tdofs_count;
      const MPI_Comm comm = fes.GetParMesh()->GetComm();
      MPI_Allreduce(&ess_tdofs_count,&global_ess_tdofs_count,
                    1, MPI_INT, MPI_SUM, comm);
      assert(global_ess_tdofs_count>0);
      //const mfem::Engine &engine = fes.GetMesh()->GetEngine();
      ess_tdofs.SetSize(global_ess_tdofs_count);//engine.MakeLayout(global_ess_tdofs_count));
      //ess_tdofs.Pull(false);
#else
      assert(ess_tdofs_count>0);
      ess_tdofs.Resize(ess_tdofs_count);
#endif
   } else{
      assert(ess_tdofs_count<=ess_tdofs.Size());
   } 
  
   if (ess_tdofs_count == 0) {
      //pop();
      return;
   }
   assert(dofs.GetData());
   ess_tdofs.Assign(dofs);
   //ess_tdofs.Push();
   //pop();
}

// *****************************************************************************
void kMassPAOperator::EliminateRHS(mfem::Vector &b)
{
   push(Wheat);
   if (ess_tdofs_count > 0){
      //mfem::Vector &kb = b.Get_PVector()->As<kernels::Vector>();
      /*k*/b.SetSubVector(ess_tdofs, 0.0);//, ess_tdofs_count);
   }
   //pop();
}

// *************************************************************************
void kMassPAOperator::Mult(const mfem::Vector &x,
                                 mfem::Vector &y) const
{
   push(Wheat);
   
   if (distX.Size()!=x.Size()) {
      distX.SetSize(x.Size());//fes.GetMesh()->GetEngine().MakeLayout(x.Size()));
   }
   
   assert(distX.Size()==x.Size());
   distX = x;

   //mfem::Vector &kx = distX.Get_PVector()->As<kernels::Vector>();
   //kernels::Vector &ky = y.Get_PVector()->As<kernels::Vector>();

   if (ess_tdofs_count)
   {
      distX.SetSubVector(ess_tdofs, 0.0);//, ess_tdofs_count);
   }
   
   massOperator->Mult(distX, y);

   if (ess_tdofs_count)
   {
      /*k*/y.SetSubVector(ess_tdofs, 0.0);//, ess_tdofs_count);
   }
   //pop();
}

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI
