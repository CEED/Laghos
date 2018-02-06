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
#ifndef LAGHOS_RAJA_OPERATOR
#define LAGHOS_RAJA_OPERATOR

namespace mfem {

  // ***************************************************************************
  class RajaOperator {
  protected:
    int height;
    int width;
  public:
    explicit RajaOperator(int s = 0) { height = width = s; }
    RajaOperator(int h, int w) { height = h; width = w; }
    inline int Height() const { return height; }
    inline int NumRows() const { return height; }
    inline int Width() const { return width; }
    inline int NumCols() const { return width; }
    virtual void Mult(const RajaVector &x, RajaVector &y) const
    { mfem_error("Operator::Mult(Vector) is not overloaded!"); };
    virtual void MultTranspose(const RajaVector &x, RajaVector &y) const
    { mfem_error("Operator::MultTranspose() is not overloaded!"); }
    virtual RajaOperator &GetGradient(const RajaVector &x) const {
      mfem_error("Operator::GetGradient() is not overloaded!");
      return const_cast<RajaOperator &>(*this);
    }
    virtual const RajaOperator *GetProlongation() const { return NULL; }
    virtual const RajaOperator *GetRestriction() const  { return NULL; }
    void FormLinearSystem(const Array<int> &ess_tdof_list,
                          RajaVector &x, RajaVector &b,
                          RajaOperator* &A, RajaVector &X, RajaVector &B,
                          int copy_interior = 0);
    virtual void RecoverFEMSolution(const RajaVector &X, const RajaVector &b, RajaVector &x);
    void PrintMatlab(std::ostream & out, int n = 0, int m = 0) const;
    virtual ~RajaOperator() { }
    enum Type {
      ANY_TYPE,         ///< ID for the base class Operator, i.e. any type.
      MFEM_SPARSEMAT,   ///< ID for class SparseMatrix
      Hypre_ParCSR,     ///< ID for class HypreParMatrix.
      PETSC_MATAIJ,     ///< ID for class PetscParMatrix, MATAIJ format.
      PETSC_MATIS,      ///< ID for class PetscParMatrix, MATIS format.
      PETSC_MATSHELL,   ///< ID for class PetscParMatrix, MATSHELL format.
      PETSC_MATNEST,    ///< ID for class PetscParMatrix, MATNEST format.
      PETSC_MATHYPRE,   ///< ID for class PetscParMatrix, MATHYPRE format.
      PETSC_MATGENERIC  ///< ID for class PetscParMatrix, unspecified format.
    };
    Type GetType() const { return ANY_TYPE; }
  };


  //   *************************************************************************
  class RajaTimeDependentOperator : public RajaOperator{
  public:
    enum Type {
      EXPLICIT, 
      IMPLICIT, 
      HOMOGENEOUS
    };
  protected:
    double t;
    Type type; 
  public:
    explicit RajaTimeDependentOperator(int n = 0,
                                       double t_ = 0.0,
                                       Type type_ = EXPLICIT)
      : RajaOperator(n) { t = t_; type = type_; }
   RajaTimeDependentOperator(int h, int w,
                             double t_ = 0.0,
                             Type type_ = EXPLICIT)
      : RajaOperator(h, w) { t = t_; type = type_; }
   virtual double GetTime() const { return t; }
   virtual void SetTime(const double _t) { t = _t; }
   bool isExplicit() const { return (type == EXPLICIT); }
   bool isImplicit() const { return !isExplicit(); }
   bool isHomogeneous() const { return (type == HOMOGENEOUS); }
   virtual void ExplicitMult(const RajaVector &x, RajaVector &y) const   {
      mfem_error("TimeDependentOperator::ExplicitMult() is not overridden!");
   }
   virtual void ImplicitMult(const RajaVector &x, const RajaVector &k, RajaVector &y) const   {
      mfem_error("TimeDependentOperator::ImplicitMult() is not overridden!");
   }
   virtual void Mult(const RajaVector &x, RajaVector &y) const   {
      mfem_error("TimeDependentOperator::Mult() is not overridden!");
   }
   inline virtual void MultTranspose(const RajaVector &x, RajaVector &y) const
   { mfem_error("Operator::MultTranspose() is not overloaded!"); }
   virtual void ImplicitSolve(const double dt, const RajaVector &x, RajaVector &k)
   {
      mfem_error("TimeDependentOperator::ImplicitSolve() is not overridden!");
   }
   virtual RajaOperator& GetImplicitGradient(const RajaVector &x,
                                                   const RajaVector &k,
                                                   double shift) const
   {
      mfem_error("TimeDependentOperator::GetImplicitGradient() is "
                 "not overridden!");
      return const_cast<RajaOperator &>(dynamic_cast<const RajaOperator &>(*this));
   }
    virtual RajaOperator& GetExplicitGradient(const RajaVector &x) const
    {
      mfem_error("TimeDependentOperator::GetExplicitGradient() is "
                 "not overridden!");
      return const_cast<RajaOperator &>
        (dynamic_cast<const RajaOperator &>(*this));
   }
   virtual ~RajaTimeDependentOperator() { }
  };

  
  // ***************************************************************************
  class RajaSolverOperator : public RajaOperator{
  public:
    bool iterative_mode;
    explicit RajaSolverOperator(int s = 0, bool iter_mode = false)
      : RajaOperator(s) { iterative_mode = iter_mode; }
    RajaSolverOperator(int h, int w, bool iter_mode = false)
      : RajaOperator(h, w) { iterative_mode = iter_mode; }
    virtual void SetOperator(const RajaOperator &op) = 0;
  };

  // ***************************************************************************
  /// The operator x -> R*A*P*x.
  class RajaRAPOperator : public RajaOperator{
  private:
    const RajaOperator & Rt;
    const RajaOperator & A;
    const RajaOperator & P;
    mutable RajaVector Px;
    mutable RajaVector APx;

  public:
    /// Construct the RAP operator given R^T, A and P.
    RajaRAPOperator(const RajaOperator &Rt_, const RajaOperator &A_, const RajaOperator &P_)
      : RajaOperator(Rt_.Width(), P_.Width()), Rt(Rt_), A(A_), P(P_),
        Px(P.Height()), APx(A.Height()) { }

   /// Operator application.
   virtual void Mult(const RajaVector & x, RajaVector & y) const
   { P.Mult(x, Px); A.Mult(Px, APx); Rt.MultTranspose(APx, y); }
  
   /// Application of the transpose.
   virtual void MultTranspose(const RajaVector & x, RajaVector & y) const
   { Rt.Mult(x, APx); A.MultTranspose(APx, Px); P.MultTranspose(Px, y); }
};
} // mfem

#endif // LAGHOS_RAJA_OPERATOR
