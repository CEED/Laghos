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
  
  // ***************************************************************************
  // * RajaConformingProlongationOperator
  // ***************************************************************************
  RajaConformingProlongationOperator::RajaConformingProlongationOperator
  (ParFiniteElementSpace &pfes): RajaOperator(pfes.GetVSize(), pfes.GetTrueVSize()),
                                 external_ldofs(),
                                 gc(pfes.GroupComm()){
    MFEM_VERIFY(pfes.Conforming(), "");
    Array<int> ldofs;
    Table &group_ldof = gc.GroupLDofTable();
    external_ldofs.Reserve(Height()-Width());
    for (int gr = 1; gr < group_ldof.Size(); gr++)
    {
      if (!gc.GetGroupTopology().IAmMaster(gr))
      {
        ldofs.MakeRef(group_ldof.GetRow(gr), group_ldof.RowSize(gr));
        external_ldofs.Append(ldofs);
      }
    }
    external_ldofs.Sort();
    MFEM_ASSERT(external_ldofs.Size() == Height()-Width(), "");
    //gc.PrintInfo();
    //pfes.Dof_TrueDof_Matrix()->PrintCommPkg();
  }

  // ***************************************************************************
  // * Device Mult
  // ***************************************************************************
  void RajaConformingProlongationOperator::d_Mult(const RajaVector &x,
                                                  RajaVector &y) const{
    push();
    MFEM_ASSERT(x.Size() == Width(), "");
    MFEM_ASSERT(y.Size() == Height(), "");
    const double *d_xdata = x.GetData();
    double *d_ydata = y.GetData(); 
    const int m = external_ldofs.Size();
    const int in_layout = 2; // 2 - input is ltdofs array
    push(BcastBegin);
    gc.BcastBegin(const_cast<double*>(d_xdata), in_layout);
    pop();
    push(copy);
    int j = 0;
    for (int i = 0; i < m; i++)
    {
      const int end = external_ldofs[i];
#ifndef __NVCC__
      std::copy(d_xdata+j-i, d_xdata+end-i, d_ydata+j);
#else
      checkCudaErrors(cuMemcpyDtoD((CUdeviceptr)(d_xdata+j-i),
                                   (CUdeviceptr)(d_ydata+j),
                                   (end-j)*sizeof(double)));
#endif
      j = end+1;
    }
#ifndef __NVCC__
    std::copy(d_xdata+j-m, d_xdata+Width(), d_ydata+j);
#else
    assert(false);
#endif
    const int out_layout = 0; // 0 - output is ldofs array
    pop();
    push(BcastEnd);
    gc.BcastEnd(d_ydata, out_layout);
    pop();
    pop();
  }

  // ***************************************************************************
  // * Device MultTranspose
  // ***************************************************************************
  void RajaConformingProlongationOperator::d_MultTranspose(const RajaVector &x,
                                                           RajaVector &y) const{
    MFEM_ASSERT(x.Size() == Height(), "");
    MFEM_ASSERT(y.Size() == Width(), "");
    const double *d_xdata = x.GetData();
    double *d_ydata = y.GetData();
    const int m = external_ldofs.Size();
    push(BcastBegin);
    gc.ReduceBegin(d_xdata);
    pop();
    push(copy);
    int j = 0;
    for (int i = 0; i < m; i++)   {
      const int end = external_ldofs[i];
#ifndef __NVCC__
      std::copy(d_xdata+j, d_xdata+end, d_ydata+j-i);
#else
    assert(false);
#endif
      j = end+1;
    }
#ifndef __NVCC__
    std::copy(d_xdata+j, d_xdata+Height(), d_ydata+j-m);
#else
    assert(false);
#endif
    const int out_layout = 2; // 2 - output is an array on all ltdofs
    pop();
    push(BcastEnd);
    gc.ReduceEnd<double>(d_ydata, out_layout, GroupCommunicator::Sum);
    pop();
    pop();
  }

  // ***************************************************************************
  // * Host Mult
  // ***************************************************************************
  void RajaConformingProlongationOperator::h_Mult(const Vector &x,
                                                  Vector &y) const{
    push();
    MFEM_ASSERT(x.Size() == Width(), "");
    MFEM_ASSERT(y.Size() == Height(), "");
    const double *xdata = x.GetData();
    double *ydata = y.GetData(); 
    const int m = external_ldofs.Size();
    const int in_layout = 2; // 2 - input is ltdofs array
    push(BcastBegin);
    gc.BcastBegin(const_cast<double*>(xdata), in_layout);
    pop();
    push(copy);
    int j = 0;
    for (int i = 0; i < m; i++)
    {
      const int end = external_ldofs[i];
      std::copy(xdata+j-i, xdata+end-i, ydata+j);
      j = end+1;
    }
    std::copy(xdata+j-m, xdata+Width(), ydata+j);
    const int out_layout = 0; // 0 - output is ldofs array
    pop();
    push(BcastEnd);
    gc.BcastEnd(ydata, out_layout);
    pop();
    pop();
  }

  // ***************************************************************************
  // * Host MultTranspose
  // ***************************************************************************
  void RajaConformingProlongationOperator::h_MultTranspose(const Vector &x,
                                                           Vector &y) const{
    MFEM_ASSERT(x.Size() == Height(), "");
    MFEM_ASSERT(y.Size() == Width(), "");
    const double *xdata = x.GetData();
    double *ydata = y.GetData();
    const int m = external_ldofs.Size();
    push(BcastBegin);
    gc.ReduceBegin(xdata);
    pop();
    push(copy);
    int j = 0;
    for (int i = 0; i < m; i++)   {
      const int end = external_ldofs[i];
      std::copy(xdata+j, xdata+end, ydata+j-i);
      j = end+1;
    }
    std::copy(xdata+j, xdata+Height(), ydata+j-m);
    const int out_layout = 2; // 2 - output is an array on all ltdofs
    pop();
    push(BcastEnd);
    gc.ReduceEnd<double>(ydata, out_layout, GroupCommunicator::Sum);
    pop();
    pop();
  }

} // namespace mfem
