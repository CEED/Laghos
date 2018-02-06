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
#include "raja.hpp"

namespace mfem {
  
using namespace std;

   
  RajaIterativeSolver::RajaIterativeSolver() : RajaSolverOperator(0, true){
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
  RajaIterativeSolver::RajaIterativeSolver(MPI_Comm _comm)
    : RajaSolverOperator(0, true)
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

void RajaIterativeSolver::SetPrintLevel(int print_lvl){
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

void RajaIterativeSolver::SetPreconditioner(RajaSolverOperator &pr){
   prec = &pr;
   prec->iterative_mode = false;
}

void RajaIterativeSolver::SetOperator(const RajaOperator &op){
   oper = &op;
   height = op.Height();
   width = op.Width();
   if (prec)
   {
      prec->SetOperator(*oper);
   }
}


} // mfem
