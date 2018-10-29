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
#include "../cuda.hpp"

namespace mfem {
  
  // ***************************************************************************
  // * CudaConformingProlongationOperator
  // ***************************************************************************
  CudaConformingProlongationOperator::CudaConformingProlongationOperator
  (ParFiniteElementSpace &pfes): CudaOperator(pfes.GetVSize(), pfes.GetTrueVSize()),
                                 external_ldofs(),
                                 d_external_ldofs(Height()-Width()), // size can be 0 here
                                 gc(new CudaCommD(pfes)),
                                 kMaxTh(0){
    Array<int> ldofs;
    Table &group_ldof = gc->GroupLDofTable();
    external_ldofs.Reserve(Height()-Width());
    for (int gr = 1; gr < group_ldof.Size(); gr++)
    {
      if (!gc->GetGroupTopology().IAmMaster(gr)) 
      {
        ldofs.MakeRef(group_ldof.GetRow(gr), group_ldof.RowSize(gr));
        external_ldofs.Append(ldofs);
      }
    }
    external_ldofs.Sort();
    const int HmW=Height()-Width();
    if (HmW>0)
      d_external_ldofs=external_ldofs;
    assert(external_ldofs.Size() == Height()-Width());
    // *************************************************************************
    const int m = external_ldofs.Size();
    int j = 0;
    for (int i = 0; i < m; i++) {
      const int end = external_ldofs[i];
      const int size = end-j;
      if (size>kMaxTh) kMaxTh=size;
      j = end+1;
    }
  }

  // ***************************************************************************
  // * ~CudaConformingProlongationOperator
  // ***************************************************************************
  CudaConformingProlongationOperator::~CudaConformingProlongationOperator(){
    delete  gc;
  }

  // ***************************************************************************
  // * CUDA Error Status Check
  // ***************************************************************************
  void cuLastCheck(){
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
      exit(fprintf(stderr, "\n\t\033[31;1m[cuLastCheck] failed: %s\033[m\n",
                   cudaGetErrorString(cudaStatus)));
  }

  // ***************************************************************************
  // * k_Mult
  // ***************************************************************************
  static __global__
  void k_Mult(double *y,const double *x,const int *external_ldofs,const int m){
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i>=m) return;
    const int j = (i>0)?external_ldofs[i-1]+1:0;
    const int end = external_ldofs[i];
    for(int k=0;k<(end-j);k+=1)
      y[j+k]=x[j-i+k];
  }
  static __global__
  void k_Mult2(double *y,const double *x,const int *external_ldofs,
               const int m, const int base){
    const int i = base+threadIdx.x;
    const int j = (i>0)?external_ldofs[i-1]+1:0;
    const int end = external_ldofs[i];
    const int k = blockIdx.x;
    if (k>=(end-j)) return;
    y[j+k]=x[j-i+k];
  }

  // ***************************************************************************
  // * Device Mult
  // ***************************************************************************
  void CudaConformingProlongationOperator::d_Mult(const CudaVector &x,
                                                  CudaVector &y) const{
    push(Coral);
    const double *d_xdata = x.GetData();
    const int in_layout = 2; // 2 - input is ltdofs array
    push(d_BcastBegin,Coral);
    gc->d_BcastBegin(const_cast<double*>(d_xdata), in_layout);
    pop();
    push(d_Mult_Work,Coral);
    double *d_ydata = y.GetData(); 
    int j = 0;
    const int m = external_ldofs.Size();    
    if (m>0){
      const int maxXThDim = rconfig::Get().MaxXThreadsDim();
      if (m>maxXThDim){
        const int kTpB=64;
        k_Mult<<<(m+kTpB-1)/kTpB,kTpB>>>(d_ydata,d_xdata,d_external_ldofs,m);
        cuLastCheck();
      }else{
        assert((m/maxXThDim)==0);
        assert(kMaxTh<rconfig::Get().MaxXGridSize());
        for(int of7=0;of7<m/maxXThDim;of7+=1){
          const int base = of7*maxXThDim;
          k_Mult2<<<kMaxTh,maxXThDim>>>(d_ydata,d_xdata,d_external_ldofs,m,base);
          cuLastCheck();
        }
        k_Mult2<<<kMaxTh,m%maxXThDim>>>(d_ydata,d_xdata,d_external_ldofs,m,0);
        cuLastCheck();
      }
      j = external_ldofs[m-1]+1;
    }
    rmemcpy::rDtoD(d_ydata+j,d_xdata+j-m,(Width()+m-j)*sizeof(double));
    pop();
    push(d_BcastEnd,Coral);
    const int out_layout = 0; // 0 - output is ldofs array
    gc->d_BcastEnd(d_ydata, out_layout);
    pop();
    pop();
  }

  
  // ***************************************************************************
  // * k_Mult
  // ***************************************************************************
  static __global__
  void k_MultTranspose(double *y, const double *x,
                       const int *external_ldofs, const int m){
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i>=m) return;
    const int j = (i>0)?external_ldofs[i-1]+1:0;
    const int end = external_ldofs[i];
    for(int k=0;k<(end-j);k+=1)
      y[j-i+k]=x[j+k];
  }
  
  static __global__
  void k_MultTranspose2(double *y, const double *x,
                        const int *external_ldofs,
                        const int m, const int base){
    const int i = base+threadIdx.x;
    const int j = (i>0)?external_ldofs[i-1]+1:0;
    const int end = external_ldofs[i];
    const int k = blockIdx.x;
    if (k>=(end-j)) return;
    y[j-i+k]=x[j+k];
  }
  
  // ***************************************************************************
  // * Device MultTranspose
  // ***************************************************************************
  void CudaConformingProlongationOperator::d_MultTranspose(const CudaVector &x,
                                                           CudaVector &y) const{
    push(Coral);
    const double *d_xdata = x.GetData();
    push(d_ReduceBegin,Coral);
    gc->d_ReduceBegin(d_xdata);
    pop();
    push(d_MultTranspose_Work,Coral);
    double *d_ydata = y.GetData();
    int j = 0;
    const int m = external_ldofs.Size();
    if (m>0){      
      const int maxXThDim = rconfig::Get().MaxXThreadsDim();
      if (m>maxXThDim){
        const int kTpB=64;
        k_MultTranspose<<<(m+kTpB-1)/kTpB,kTpB>>>(d_ydata,d_xdata,d_external_ldofs,m);
        cuLastCheck();
      }else{
        const int TpB = rconfig::Get().MaxXThreadsDim();
        assert(kMaxTh<rconfig::Get().MaxXGridSize());
        for(int of7=0;of7<m/maxXThDim;of7+=1){
        const int base = of7*maxXThDim;
        k_MultTranspose2<<<kMaxTh,maxXThDim>>>(d_ydata,d_xdata,d_external_ldofs,m,base);
        cuLastCheck();
      }
      k_MultTranspose2<<<kMaxTh,m%maxXThDim>>>(d_ydata,d_xdata,d_external_ldofs,m,0);
      cuLastCheck();
      }
      j = external_ldofs[m-1]+1;
    }
    rmemcpy::rDtoD(d_ydata+j-m,d_xdata+j,(Height()-j)*sizeof(double));
    pop();
    push(d_ReduceEnd,Coral);
    const int out_layout = 2; // 2 - output is an array on all ltdofs
    gc->d_ReduceEnd<double>(d_ydata, out_layout, GroupCommunicator::Sum);
    pop();
    pop();
  }

  // ***************************************************************************
  // * Host Mult
  // ***************************************************************************
  void CudaConformingProlongationOperator::h_Mult(const Vector &x,
                                                  Vector &y) const{
    push(Coral);
    const double *xdata = x.GetData();
    double *ydata = y.GetData(); 
    const int m = external_ldofs.Size();
    const int in_layout = 2; // 2 - input is ltdofs array
    push(BcastBegin,Moccasin);
    gc->BcastBegin(const_cast<double*>(xdata), in_layout);
    pop();
    int j = 0;
    for (int i = 0; i < m; i++){
      const int end = external_ldofs[i];
      std::copy(xdata+j-i, xdata+end-i, ydata+j);
      j = end+1;
    }
    std::copy(xdata+j-m, xdata+Width(), ydata+j);
    const int out_layout = 0; // 0 - output is ldofs array
    push(BcastEnd,PeachPuff);
    gc->BcastEnd(ydata, out_layout);
    pop();
    pop();
  }

  // ***************************************************************************
  // * Host MultTranspose
  // ***************************************************************************
  void CudaConformingProlongationOperator::h_MultTranspose(const Vector &x,
                                                           Vector &y) const{
    push(Coral);
    const double *xdata = x.GetData();
    double *ydata = y.GetData();
    const int m = external_ldofs.Size();
    push(ReduceBegin,PapayaWhip);
    gc->ReduceBegin(xdata);
    pop();
    int j = 0;
    for (int i = 0; i < m; i++)   {
      const int end = external_ldofs[i];
      std::copy(xdata+j, xdata+end, ydata+j-i);
      j = end+1;
    }
    std::copy(xdata+j, xdata+Height(), ydata+j-m);
    const int out_layout = 2; // 2 - output is an array on all ltdofs
    push(ReduceEnd,LavenderBlush);
    gc->ReduceEnd<double>(ydata, out_layout, GroupCommunicator::Sum);
    pop();
    pop();
  }

} // namespace mfem
