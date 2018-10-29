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
#ifndef LAGHOS_CUDA_OPERATOR
#define LAGHOS_CUDA_OPERATOR

namespace mfem
{

// ***************************************************************************
class CudaOperator : public rmemcpy
{
protected:
   int height;
   int width;
public:
   explicit CudaOperator(int s = 0) { height = width = s; }
   CudaOperator(int h, int w) { height = h; width = w; }
   inline int Height() const { return height; }
   inline int Width() const { return width; }
   virtual void Mult(const CudaVector &x, CudaVector &y) const  { assert(false); };
   virtual void MultTranspose(const CudaVector &x, CudaVector &y) const { assert(false); }
   virtual const CudaOperator *GetProlongation() const { assert(false); return NULL; }
   virtual const CudaOperator *GetRestriction() const  { assert(false); return NULL; }
   virtual void RecoverFEMSolution(const CudaVector &X,
                                   const CudaVector &b,
                                   CudaVector &x) {assert(false);}
};


// ***************************************************************************
class CudaTimeDependentOperator : public CudaOperator
{
private:
   double t;
public:
   explicit CudaTimeDependentOperator(int n = 0,
                                      double t_ = 0.0) : CudaOperator(n), t(t_) {}
   void SetTime(const double _t) { t = _t; }
};

// ***************************************************************************
class CudaSolverOperator : public CudaOperator
{
public:
   bool iterative_mode;
   explicit CudaSolverOperator(int s = 0,
                               bool iter_mode = false) :
      CudaOperator(s),
      iterative_mode(iter_mode) { }
   virtual void SetOperator(const CudaOperator &op) = 0;
};

// ***************************************************************************
class CudaRAPOperator : public CudaOperator
{
private:
   const CudaOperator &Rt;
   const CudaOperator &A;
   const CudaOperator &P;
   mutable CudaVector Px;
   mutable CudaVector APx;
public:
   /// Construct the RAP operator given R^T, A and P.
   CudaRAPOperator(const CudaOperator &Rt_, const CudaOperator &A_,
                   const CudaOperator &P_)
      : CudaOperator(Rt_.Width(), P_.Width()), Rt(Rt_), A(A_), P(P_),
        Px(P.Height()), APx(A.Height()) { }
   /// Operator application.
   void Mult(const CudaVector & x, CudaVector & y) const
   {
      P.Mult(x, Px);
      A.Mult(Px, APx);
      Rt.MultTranspose(APx, y);
   }
   /// Application of the transpose.
   void MultTranspose(const CudaVector & x, CudaVector & y) const
   {
      Rt.Mult(x, APx);
      A.MultTranspose(APx, Px);
      P.MultTranspose(Px, y);
   }
};

} // mfem

#endif // LAGHOS_CUDA_OPERATOR
