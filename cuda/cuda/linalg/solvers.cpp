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
#include "../cuda.hpp"

namespace mfem
{

// *************************************************************************
void CudaCGSolver::h_Mult(const CudaVector &b, CudaVector &x) const
{
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

   r0 = std::max(nom*rel_tol*rel_tol,abs_tol*abs_tol);

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

} // mfem
