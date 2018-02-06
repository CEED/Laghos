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
#ifndef LAGHOS_RAJA_SOLVERS
#define LAGHOS_RAJA_SOLVERS

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

namespace mfem {

  class RajaIterativeSolver : public RajaSolverOperator
{
#ifdef MFEM_USE_MPI
private:
   int dot_prod_type; // 0 - local, 1 - global over 'comm'
   MPI_Comm comm;
#endif

protected:
   const RajaOperator *oper;
   RajaSolverOperator *prec;

   int max_iter, print_level;
   double rel_tol, abs_tol;

   // stats
   mutable int final_iter, converged;
   mutable double final_norm;

   double Dot(const RajaVector &x,
              const RajaVector &y) const {
#ifndef MFEM_USE_MPI
      return (x * y);
#else
      if (dot_prod_type == 0){
         return (x * y);
      }
      double local_dot = (x * y);
      double global_dot;
      MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, comm);
      return global_dot;
#endif
   }

   double Norm(const RajaVector &x) const { return sqrt(Dot(x, x)); }

public:
  RajaIterativeSolver();
  
#ifdef MFEM_USE_MPI
   RajaIterativeSolver(MPI_Comm _comm);
#endif

   void SetRelTol(double rtol) { rel_tol = rtol; }
   void SetAbsTol(double atol) { abs_tol = atol; }
   void SetMaxIter(int max_it) { max_iter = max_it; }
   void SetPrintLevel(int print_lvl);

   int GetNumIterations() const { return final_iter; }
   int GetConverged() const { return converged; }
   double GetFinalNorm() const { return final_norm; }

   /// This should be called before SetOperator
  virtual void SetPreconditioner(RajaSolverOperator &pr);

   /// Also calls SetOperator for the preconditioner if there is one
   virtual void SetOperator(const RajaOperator &op);
};


  
/// Conjugate gradient method
  class RajaCGSolver : public RajaIterativeSolver{
  protected:
    mutable RajaVector r, d, z;

    void UpdateVectors()   {
      r.SetSize(width);
      d.SetSize(width);
      z.SetSize(width);
    }

  public:
    RajaCGSolver() { }

#ifdef MFEM_USE_MPI
    RajaCGSolver(MPI_Comm _comm) : RajaIterativeSolver(_comm) { }
#endif

    virtual void SetOperator(const RajaOperator &op) {
      RajaIterativeSolver::SetOperator(op);
      UpdateVectors();
    }

    virtual void Mult(const RajaVector &b, RajaVector &x) const {
      int i;
      double r0, den, nom, nom0, betanom, alpha, beta;

      if (iterative_mode)
      {
        oper->Mult(x, r);
        subtract(b, r, r); // r = b - A x
      }
      else
      {
        r = b;
        x = 0.0;
      }

      if (prec)
      {
        prec->Mult(r, z); // z = B r
        d = z;
      }
      else
      {
        d = r;
      }
      nom0 = nom = Dot(d, r);
      MFEM_ASSERT(IsFinite(nom), "nom = " << nom);

      if (print_level == 1
          || print_level == 3)
      {
        mfem::out << "   Iteration : " << std::setw(3) << 0 << "  (B r, r) = "
                  << nom << (print_level == 3 ? " ...\n" : "\n");
      }

      r0 = std::max(nom*rel_tol*rel_tol,
                    abs_tol*abs_tol);
      if (nom <= r0)
      {
        converged = 1;
        final_iter = 0;
        final_norm = sqrt(nom);
        return;
      }

      oper->Mult(d, z);  // z = A d

      den = Dot(z, d);
      MFEM_ASSERT(IsFinite(den), "den = " << den);

      if (print_level >= 0 && den < 0.0)
      {
        mfem::out << "Negative denominator in step 0 of PCG: " << den << '\n';
      }

      if (den == 0.0)
      {
        converged = 0;
        final_iter = 0;
        final_norm = sqrt(nom);
        return;
      }

      // start iteration
      converged = 0;
      final_iter = max_iter;
      for (i = 1; true; )
      {
        alpha = nom/den;
        add(x,  alpha, d, x);     //  x = x + alpha d
        add(r, -alpha, z, r);     //  r = r - alpha A d

        if (prec)
        {
          prec->Mult(r, z);      //  z = B r
          betanom = Dot(r, z);
        }
        else
        {
          betanom = Dot(r, r);
        }
        MFEM_ASSERT(IsFinite(betanom), "betanom = " << betanom);

        if (print_level == 1)
        {
          mfem::out << "   Iteration : " << std::setw(3) << i << "  (B r, r) = "
                    << betanom << '\n';
        }

        if (betanom < r0)
        {
          if (print_level == 2)
          {
            mfem::out << "Number of PCG iterations: " << i << '\n';
          }
          else if (print_level == 3)
          {
            mfem::out << "   Iteration : " << std::setw(3) << i << "  (B r, r) = "
                      << betanom << '\n';
          }
          converged = 1;
          final_iter = i;
          break;
        }

        if (++i > max_iter)
        {
          break;
        }

        beta = betanom/nom;
        if (prec)
        {
          add(z, beta, d, d);   //  d = z + beta d
        }
        else
        {
          add(r, beta, d, d);
        }
        oper->Mult(d, z);       //  z = A d
        den = Dot(d, z);
        MFEM_ASSERT(IsFinite(den), "den = " << den);
        if (den <= 0.0)
        {
          if (print_level >= 0 && Dot(d, d) > 0.0)
            mfem::out << "PCG: The operator is not positive definite. (Ad, d) = "
                      << den << '\n';
        }
        nom = betanom;
      }
      if (print_level >= 0 && !converged)
      {
         if (print_level != 1)
         {
            if (print_level != 3)
            {
               mfem::out << "   Iteration : " << std::setw(3) << 0 << "  (B r, r) = "
                         << nom0 << " ...\n";
            }
            mfem::out << "   Iteration : " << std::setw(3) << final_iter << "  (B r, r) = "
                      << betanom << '\n';
         }
         mfem::out << "PCG: No convergence!" << '\n';
      }
      if (print_level >= 1 || (print_level >= 0 && !converged))
      {
         mfem::out << "Average reduction factor = "
                   << pow (betanom/nom0, 0.5/final_iter) << '\n';
      }
      final_norm = sqrt(betanom);
   }
};

} // mfem

#endif // LAGHOS_RAJA_SOLVERS
