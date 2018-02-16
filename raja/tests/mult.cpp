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
#include "../raja.hpp"
#include "../../laghos_solver.hpp"

namespace mfem {

  // ***************************************************************************
  class RajaPm1APOperator : public RajaOperator{
  private:
    const RajaOperator &Rt;
    const RajaOperator &A;
    const RajaOperator &P;
    mutable RajaVector Px;
    mutable RajaVector APx;
  public:
    /// Construct the RAP operator given R^T, A and P.
    RajaPm1APOperator(const RajaOperator &Rt_, const RajaOperator &A_, const RajaOperator &P_)
      : RajaOperator(Rt_.Width(), P_.Width()), Rt(Rt_), A(A_), P(P_),
        Px(P.Height()), APx(A.Height()) { }
    /// Operator application.
    void Mult(const RajaVector & x, RajaVector & y) const {
      push(SkyBlue);
      P.Mult(x, Px);
      A.Mult(Px, APx);
      Rt.MultTranspose(APx, y);
      pop();
    }
    /// Application of the transpose.
    void MultTranspose(const RajaVector & x, RajaVector & y) const {
      push(SkyBlue);
      Rt.Mult(x, APx);
      A.MultTranspose(APx, Px);
      P.MultTranspose(Px, y);
      pop();
    }
  };

  // ***************************************************************************
  class RajaIdOperator: public RajaOperator {
  public:
    RajaIdOperator(int s = 0) { height = width = s; }
    void Mult(const RajaVector &x, RajaVector &y) const  {
      push(DarkCyan);
      y=x;
      pop();
    }
    void MultTranspose(const RajaVector &x, RajaVector &y) const {
      push(DarkCyan);
      y=x;
      pop();
    }
  };

  // ***************************************************************************
#ifdef __NVCC__
  __global__ void iniK(void){}
#endif

  // ***************************************************************************
  bool multTest(ParMesh *pmesh, const int order, const int max_step){
    assert(order>=1);
    assert(max_step>0);
    
    // Launch first dummy kernel
    // And don't enable API tracing in NVVP
#ifdef __NVCC__
    cuProfilerStart();
    iniK<<<128,1>>>();
    cudaDeviceSynchronize();
#endif

    MPI_Barrier(pmesh->GetComm());
    
    //cuProfilerStart();
    push();

    const int dim = pmesh->Dimension();
    
    const H1_FECollection fec(order, dim);
    RajaFiniteElementSpace fes(pmesh, &fec, 1);
    HYPRE_Int glob_size = fes.GlobalTrueVSize();
    
    if (rconfig::Get().Root())
      cout << "Number of global dofs: " << glob_size << endl;
    
    const int vsize = fes.GetVSize();
    if (rconfig::Get().Root())
      cout << "Number of local dofs: " << vsize << endl;

    push(Ops,Chocolate)
    const RajaOperator &prolong = *fes.GetProlongationOperator();
    const RajaOperator &testP  = prolong;
    const RajaOperator &trialP = prolong;
    const RajaIdOperator Id(vsize);
    RajaPm1APOperator Pm1AP(testP,Id,trialP);
    pop();
    
    push(xy,Magenta);
    RajaVector x(vsize);
    //cout << "x size:" << x.Size() << endl;
    RajaVector y(vsize);
    pop();
    
    push(x=1,Turquoise);
    x=1.0;
    pop();
    
    push(y=2,Turquoise);
    y=2.0;
    pop();


    for(int i=0;i<max_step;i++){
#ifdef __NVCC__
      cudaDeviceSynchronize();
#endif
      push(SkyBlue);
      Pm1AP.Mult(x, y);
      pop();
    }
    
    pop();
    return true;
  }

} // namespace mfem
