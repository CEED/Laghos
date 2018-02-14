// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
#include "../raja.hpp"

namespace mfem {
  //RajaTable::RajaTable():size(0),I(NULL),J(NULL){}
  RajaTable::RajaTable(const Table &table){
    size = table.Size();
    assert(size > 0);
    //printf("[RajaTable] size=%d",size);
    const int nnz = table.GetI()[size];
    I = ::new int[size+1];
    J = (int*) this->operator new(nnz);
#ifdef __NVCC__
    memcpy(I, table.GetI(), sizeof(int)*(size+1));
    //checkCudaErrors(cuMemcpyHtoD((CUdeviceptr)I,table.GetI(),sizeof(int)*(size+1)));
    checkCudaErrors(cuMemcpyHtoD((CUdeviceptr)J,table.GetJ(),sizeof(int)*nnz));
#else
    memcpy(I, table.GetI(), sizeof(int)*(size+1));
    memcpy(J, table.GetJ(), sizeof(int)*nnz);
#endif
  }
  
} // mfem
