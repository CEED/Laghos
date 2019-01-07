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

#ifndef MFEM_LAGHOS_ASSEMBLY_KERNELS
#define MFEM_LAGHOS_ASSEMBLY_KERNELS

// *****************************************************************************
#include "general/okina.hpp"

// *****************************************************************************
#ifdef __NVCC__
//#define kernel __global__
//#define sync __syncthreads()
#define CUDA_BLOCK_SIZE 256
#define exeKernel(id,grid,blck,...) call[id]<<<grid,blck>>>(__VA_ARGS__)
#else
//#define sync
//#define kernel
//#define forall(i,max,body) for(int i=0;i<max;i++){body}
//#define exeKernel(id,grid,blck,...) call[id](__VA_ARGS__)
#endif //__NVCC__

// Offsets *********************************************************************
#define   ijN(i,j,N) (i)+(N)*(j)
#define  ijkN(i,j,k,N) (i)+(N)*((j)+(N)*(k))
#define ijklN(i,j,k,l,N) (i)+(N)*((j)+(N)*((k)+(N)*(l)))

#define    ijNMt(i,j,N,M,t) (t)?((i)+(N)*(j)):((j)+(M)*(i))
#define    ijkNM(i,j,k,N,M) (i)+(N)*((j)+(M)*(k))
#define   _ijkNM(i,j,k,N,M) (j)+(N)*((k)+(M)*(i))
#define   ijklNM(i,j,k,l,N,M) (i)+(N)*((j)+(N)*((k)+(M)*(l)))
#define  _ijklNM(i,j,k,l,N,M) (j)+(N)*((k)+(N)*((l)+(M)*(i)))
#define   ijklmNM(i,j,k,l,m,N,M) (i)+(N)*((j)+(N)*((k)+(M)*((l)+(M)*(m))))
#define __ijklmNM(i,j,qx,qy,e,D,Q,E) (qx)+(Q)*((qy)+(Q)*((e)+(E)*((i)+(D)*j)))

#define   _ijklmNM(i,j,k,l,m,N,M) (j)+(N)*((k)+(N)*((l)+(N)*((m)+(M)*(i))))
#define   ijklmnNM(i,j,k,l,m,n,N,M) (i)+(N)*((j)+(N)*((k)+(M)*((l)+(M)*((m)+(M)*(n)))))
#define __ijxyzeDQE(i,j,qx,qy,qz,e,D,Q,E) (qx)+(Q)*((qy)+(Q)*((qz)+(Q)*((e)+(E)*((i)+(D)*j))))

// *****************************************************************************
//#include "mfem.hpp"
//#include "laghos_abc.hpp"

#ifdef MFEM_USE_MPI

#include <memory>
#include <iostream>
namespace mfem
{

namespace hydrodynamics
{

// *****************************************************************************
class kForcePAOperator : public AbcForcePAOperator
{
private:
   const int dim, nzones;
   QuadratureData *quad_data;
   const ParFiniteElementSpace &h1fes, &l2fes;
   const kFiniteElementSpace &h1k, &l2k;
   const IntegrationRule &integ_rule, &ir1D;
   const int NUM_DOFS_1D, NUM_QUAD_1D;
   const int L2_DOFS_1D, H1_DOFS_1D;
   const int h1sz, l2sz;
   const kDofQuadMaps *l2D2Q, *h1D2Q;
   mutable mfem::Vector gVecL2, gVecH1;
public:
   kForcePAOperator(QuadratureData*,
                    ParFiniteElementSpace&,
                    ParFiniteElementSpace&,
                    const IntegrationRule&);
   void Mult(const Vector&, Vector&) const;
   void MultTranspose(const Vector&, Vector&) const;
};

// *****************************************************************************
class kMassPAOperator : public AbcMassPAOperator
{
private:
   const int dim, nzones;
   QuadratureData *quad_data;
   ParFiniteElementSpace &pfes;
   FiniteElementSpace *fes;
   const IntegrationRule &ir;
   int ess_tdofs_count;
   mfem::Array<int> ess_tdofs;
   ParBilinearForm *paBilinearForm;
   Operator *massOperator;
   mutable mfem::Vector distX;
public:
   kMassPAOperator(QuadratureData*,
                   ParFiniteElementSpace&,
                   const IntegrationRule&);
   virtual void Setup();
   void SetEssentialTrueDofs(mfem::Array<int>&);
   void EliminateRHS(mfem::Vector&);
   virtual void Mult(const mfem::Vector&, mfem::Vector&) const;
   virtual void ComputeDiagonal2D(Vector&) const {};
   virtual void ComputeDiagonal3D(Vector&) const {};
   virtual const Operator *GetProlongation() const
   { return pfes.GetProlongationMatrix(); }
   virtual const Operator *GetRestriction() const
   { return pfes.GetRestrictionMatrix(); }
};
} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_LAGHOS_ASSEMBLY_KERNELS
