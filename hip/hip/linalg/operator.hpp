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
#ifndef LAGHOS_HIP_OPERATOR
#define LAGHOS_HIP_OPERATOR

namespace mfem
{

// ***************************************************************************
class HipOperator : public rmemcpy
{
protected:
   int height;
   int width;
public:
   explicit HipOperator(int s = 0) { height = width = s; }
   HipOperator(int h, int w) { height = h; width = w; }
   inline int Height() const { return height; }
   inline int Width() const { return width; }
   virtual void Mult(const HipVector &x, HipVector &y) const  { assert(false); };
   virtual void MultTranspose(const HipVector &x, HipVector &y) const { assert(false); }
   virtual const HipOperator *GetProlongation() const { assert(false); return NULL; }
   virtual const HipOperator *GetRestriction() const  { assert(false); return NULL; }
   virtual void RecoverFEMSolution(const HipVector &X,
                                   const HipVector &b,
                                   HipVector &x) {assert(false);}
};


// ***************************************************************************
class HipTimeDependentOperator : public HipOperator
{
private:
   double t;
public:
   explicit HipTimeDependentOperator(int n = 0,
                                      double t_ = 0.0) : HipOperator(n), t(t_) {}
   void SetTime(const double _t) { t = _t; }
};

// ***************************************************************************
class HipSolverOperator : public HipOperator
{
public:
   bool iterative_mode;
   explicit HipSolverOperator(int s = 0,
                               bool iter_mode = false) :
      HipOperator(s),
      iterative_mode(iter_mode) { }
   virtual void SetOperator(const HipOperator &op) = 0;
};

// ***************************************************************************
class HipRAPOperator : public HipOperator
{
private:
   const HipOperator &Rt;
   const HipOperator &A;
   const HipOperator &P;
   mutable HipVector Px;
   mutable HipVector APx;
public:
   /// Construct the RAP operator given R^T, A and P.
   HipRAPOperator(const HipOperator &Rt_, const HipOperator &A_,
                   const HipOperator &P_)
      : HipOperator(Rt_.Width(), P_.Width()), Rt(Rt_), A(A_), P(P_),
        Px(P.Height()), APx(A.Height()) { }
   /// Operator application.
   void Mult(const HipVector & x, HipVector & y) const
   {
      P.Mult(x, Px);
      A.Mult(Px, APx);
      Rt.MultTranspose(APx, y);
   }
   /// Application of the transpose.
   void MultTranspose(const HipVector & x, HipVector & y) const
   {
      Rt.Mult(x, APx);
      A.MultTranspose(APx, Px);
      P.MultTranspose(Px, y);
   }
};

} // mfem

#endif // LAGHOS_HIP_OPERATOR
