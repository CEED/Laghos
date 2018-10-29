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
#ifndef LAGHOS_CUDA_SOLVERS
#define LAGHOS_CUDA_SOLVERS

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

namespace mfem
{

// ***************************************************************************
class CudaIterativeSolver : public CudaSolverOperator
{
#ifdef MFEM_USE_MPI
private:
   int dot_prod_type; // 0 - local, 1 - global over 'comm'
   MPI_Comm comm;
#endif
protected:
   const CudaOperator *oper;
   CudaSolverOperator *prec;
   int max_iter, print_level;
   double rel_tol, abs_tol;
   // stats
   mutable int final_iter, converged;
   mutable double final_norm;
   double Dot(const CudaVector &x,
              const CudaVector &y) const
   {
#ifndef MFEM_USE_MPI
      return (x * y);
#else
      if (dot_prod_type == 0)
      {
         return (x * y);
      }
      double local_dot = (x * y);
      double global_dot;
      MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, comm);
      return global_dot;
#endif
   }
   double Norm(const CudaVector &x) const { return sqrt(Dot(x, x)); }
public:
   CudaIterativeSolver(): CudaSolverOperator(0, true)
   {
      oper = NULL;
      prec = NULL;
      max_iter = 10;
      print_level = -1;
      rel_tol = abs_tol = 0.0;
#ifdef MFEM_USE_MPI
      dot_prod_type = 0;
#endif
   }

#ifdef MFEM_USE_MPI
   CudaIterativeSolver(MPI_Comm _comm)
      : CudaSolverOperator(0, true)
   {
      oper = NULL;
      prec = NULL;
      max_iter = 10;
      print_level = -1;
      rel_tol = abs_tol = 0.0;
      dot_prod_type = 1;
      comm = _comm;
   }
#endif

   void SetRelTol(double rtol) { rel_tol = rtol; }
   void SetAbsTol(double atol) { abs_tol = atol; }
   void SetMaxIter(int max_it) { max_iter = max_it; }
   void SetPrintLevel(int print_lvl)
   {
#ifndef MFEM_USE_MPI
      print_level = print_lvl;
#else
      if (dot_prod_type == 0)
      {
         print_level = print_lvl;
      }
      else
      {
         int rank;
         MPI_Comm_rank(comm, &rank);
         if (rank == 0)
         {
            print_level = print_lvl;
         }
      }
#endif
   }
   int GetNumIterations() const { return final_iter; }
   int GetConverged() const { return converged; }
   double GetFinalNorm() const { return final_norm; }
   /// This should be called before SetOperator
   virtual void SetPreconditioner(CudaSolverOperator &pr)
   {
      prec = &pr;
      prec->iterative_mode = false;
   }
   /// Also calls SetOperator for the preconditioner if there is one
   virtual void SetOperator(const CudaOperator &op)
   {
      oper = &op;
      height = op.Height();
      width = op.Width();
      if (prec)
      {
         prec->SetOperator(*oper);
      }
   }
};

// ***************************************************************************
// Conjugate gradient method
// ***************************************************************************
class CudaCGSolver : public CudaIterativeSolver
{
protected:
   mutable CudaVector r, d, z;
   void UpdateVectors()
   {
      r.SetSize(width);
      d.SetSize(width);
      z.SetSize(width);
   }
public:
   CudaCGSolver() { }
#ifdef MFEM_USE_MPI
   CudaCGSolver(MPI_Comm _comm) : CudaIterativeSolver(_comm) { }
#endif
   virtual void SetOperator(const CudaOperator &op)
   {
      CudaIterativeSolver::SetOperator(op);
      UpdateVectors();
   }
   void h_Mult(const CudaVector &b, CudaVector &x) const ;
   virtual void Mult(const CudaVector &b, CudaVector &x) const
   {
      h_Mult(b,x);
   }
};

} // mfem

#endif // LAGHOS_CUDA_SOLVERS
