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

// ***************************************************************************
CudaTable::CudaTable(const Table &table)
{
   size = table.Size();
   assert(size > 0);
   const int nnz = table.GetI()[size];
   I = new int[size+1];
   J = (int*) operator new (static_cast<size_t>(nnz));
   rHtoH(I,table.GetI(), static_cast<int>(sizeof(int))*(size+1));
   if (nnz>0)
   {
      assert(table.GetJ());
      rHtoD(J,table.GetJ(),static_cast<int>(sizeof(int))*nnz);
   }
}

} // mfem
