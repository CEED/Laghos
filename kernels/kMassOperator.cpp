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

#include "kMassOperator.hpp"
#include "backends/raja/raja.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{

namespace hydrodynamics
{

// *****************************************************************************
kMassOperator::kMassOperator(QuadratureData *qd_,
                             ParFiniteElementSpace &fes_,
                             ParFiniteElementSpace &cfes_,
                             const IntegrationRule &ir_) :
      Operator(fes_.GetVSize()),
      dim(fes_.GetMesh()->Dimension()),
      nzones(fes_.GetMesh()->GetNE()),
      quad_data(qd_),
      fes(fes_),
      cfes(cfes_),
      ir(ir_),
      ess_tdofs_count(0),
      ess_tdofs(0),
      bilinearForm(NULL) { }

// *****************************************************************************
void kMassOperator::Setup()
{
   push(Wheat);
   const mfem::Engine &engine = fes.GetMesh()->GetEngine();
   raja::RajaMassIntegrator &massInteg = *(new raja::RajaMassIntegrator(engine));
   massInteg.SetIntegrationRule(ir);
   massInteg.SetOperator(quad_data->rho0DetJ0w);
   #warning forcing RajaBilinearForm to 'component' size
   bilinearForm = new raja::RajaBilinearForm(cfes.Get_PFESpace().As<raja::RajaFiniteElementSpace>());
   bilinearForm->AddDomainIntegrator(&massInteg);
   bilinearForm->Assemble();
   bilinearForm->FormOperator(Array<int>(), massOperator);
   pop();
}

// *************************************************************************
void kMassOperator::SetEssentialTrueDofs(Array<int> &dofs)
{
   push(Wheat);
  dbg("\n\033[33;7m[SetEssentialTrueDofs] dofs.Size()=%d\033[m",dofs.Size());
  ess_tdofs_count = dofs.Size();
  
  if (ess_tdofs.Size()==0){
#ifdef MFEM_USE_MPI
    int global_ess_tdofs_count;
    const MPI_Comm comm = fes.GetParMesh()->GetComm();
    MPI_Allreduce(&ess_tdofs_count,&global_ess_tdofs_count,
                  1, MPI_INT, MPI_SUM, comm);
    assert(global_ess_tdofs_count>0);
    dbg("Resize of %d",global_ess_tdofs_count);
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
    for(int i=0;i<dofs.Size();i+=1) dbg(" %d",dofs[i]);
    raja::rmemcpy::rHtoD((void*)ess_tdofs.GetData(),
                         dofs.GetData(),
                         ess_tdofs_count*sizeof(int));
    dbg("ess_tdofs:\n");
    ess_tdofs.Print();
    pop();
  }
  pop();
}

// *****************************************************************************
void kMassOperator::EliminateRHS(mfem::Vector &b)
{
   push(Wheat);
   raja::Vector rb = b.Get_PVector()->As<const raja::Vector>();

   if (ess_tdofs_count > 0)
      rb.SetSubVector(ess_tdofs, 0.0, ess_tdofs_count);
   pop();
}

// *************************************************************************
void kMassOperator::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
   push();
   dbg("x size= %d",x.Size());
   dbg("y size= %d",y.Size());
   Vector kx(cfes.GetVLayout());
   //kx.PushData(x.GetData());
   kx.Fill(1.0);
   dbg("kx size= %d",kx.Size());
   raja::Vector rx = kx.Get_PVector()->As<raja::Vector>();

   Vector ky(cfes.GetVLayout());
   //ky.PushData(y.GetData());
   dbg("ky size= %d",ky.Size());
   ky.Fill(1.0);
   raja::Vector ry = ky.Get_PVector()->As<raja::Vector>();

   if (ess_tdofs_count)
   {
      rx.SetSubVector(ess_tdofs, 0.0, ess_tdofs_count);
   }
  
   massOperator->Mult(kx, ky);

   if (ess_tdofs_count)
   {
      ry.SetSubVector(ess_tdofs, 0.0, ess_tdofs_count);
   }
  
   y = ky;
   
   pop();
}

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI
