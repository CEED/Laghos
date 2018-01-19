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
#include "raja.hpp"

namespace mfem {
  
RajaVector::~RajaVector(){
  if (!own) {
    dbg()<<"!own, nothing to do";
   return;
  }
  dbg()<<"delete";
  rdbg("\033[33m[~v");
  rmalloc<double>::_delete(data);
}

// ***************************************************************************
double* RajaVector::alloc(const size_t sz) {
  dbg();
  rdbg("\033[33m[v");
  return (double*) rmalloc<double>::_new(sz);
}

// ***************************************************************************
void RajaVector::SetSize(const size_t sz, const void* ptr) {
  dbg();
  rdbg("\033[33m[size=%d, new sz=%d]\033[m",size,sz);
  own=true;
  size = sz;
  if (!data) { data = alloc(sz); }
#ifdef __NVCC__
  //cudaMemcpyPeer ( void* dst, int  dstDevice, const void* src, int  srcDevice, size_t count ) 
  if (ptr) { checkCudaErrors(cudaMemcpy(data,ptr,bytes(),cudaMemcpyDeviceToDevice));}
  //if (ptr) { cudaMemcpy(data,ptr,bytes(),cudaMemcpyDeviceToDevice);}
#else
  if (ptr) { ::memcpy(data,ptr,bytes());}
#endif
}

// ***************************************************************************
RajaVector::RajaVector(const size_t sz):size(sz),data(alloc(sz)),own(true) {  dbg();
}

RajaVector::RajaVector(const RajaVector& v):
  size(0),data(NULL),own(true) {   dbg();
SetSize(v.Size(), v); }

RajaVector::RajaVector(const RajaVectorRef& ref):
  size(ref.v.size),data(ref.v.data),own(false) {  dbg();
}
  
RajaVector::RajaVector(RajaArray<double>& v):
  size(v.size()),data(v.ptr()),own(false) {  dbg();
}

// Host 2 Device ***************************************************************
RajaVector::RajaVector(const Vector& v):
  size(0),data(NULL),own(false) {  dbg();

  //dbg()<<"Host 2 Device";
#ifdef __NVCC__
  // v comes from the host
  double* d_v= (double*)rmalloc<double>::HoDNew(v.Size());
  checkCudaErrors(cudaMemcpy(d_v,v.GetData(),v.Size()*sizeof(double),cudaMemcpyHostToDevice));
  SetSize(v.Size(), (void*)d_v/*v.GetData()*/);
#else
  SetSize(v.Size(), v.GetData());  
#endif
}

// Device 2 Host ***************************************************************
RajaVector::operator Vector() {
  dbg();
#ifdef __NVCC__
  //dbg()<<"Device 2 Host";
  double *h_data= (double*) ::malloc(bytes());
  checkCudaErrors(cudaMemcpy(h_data,data,bytes(),cudaMemcpyDeviceToHost));
  return Vector(h_data,size);
#else
  return Vector(data,size);
#endif
}
  
RajaVector::operator Vector() const {
  dbg();
#ifdef __NVCC__
  //dbg()<<"Device 2 Host (const)";
  double *h_data= (double*) ::malloc(bytes());
  checkCudaErrors(cudaMemcpy(h_data,data,bytes(),cudaMemcpyDeviceToHost));
  return Vector(h_data,size);
#else
  return Vector(data,size);
#endif
}

// ***************************************************************************
void RajaVector::Print(std::ostream& out, int width) const {
  dbg();
#ifdef __NVCC__
  //dbg()<<"Device 2 Host (const)";
  double *h_data= (double*) ::malloc(bytes());
  checkCudaErrors(cudaMemcpy(h_data,data,bytes(),cudaMemcpyHostToDevice));
#else
  double *h_data=data;
#endif
  for (size_t i=0; i<size; i+=1) 
    printf("\n\t[%ld] %.15e",i,h_data[i]);
}

// ***************************************************************************
RajaVectorRef RajaVector::GetRange(const size_t offset,
                                   const size_t entries) const {
  dbg();
  RajaVectorRef ret;
  RajaVector& v = ret.v;
  v.data = (double*) ((unsigned char*)data + (offset*sizeof(double)));
  v.size = entries;
  v.own = false;
  return ret;
}

// ***************************************************************************
RajaVector& RajaVector::operator=(const RajaVector& v) {
  dbg();
  SetSize(v.Size(),v.data);
  own = false;
  return *this;
}

// ***************************************************************************
RajaVector& RajaVector::operator=(double value) {
  dbg();
  vector_op_eq(size, value, data);
  return *this;
}

// ***************************************************************************
double RajaVector::operator*(const RajaVector& v) const {
  dbg();
  return vector_dot(size, data, v.data);
}

// *****************************************************************************
RajaVector& RajaVector::operator-=(const RajaVector& v) {
  dbg();
  vector_vec_sub(size, data, v.data);
  return *this;
}

// ***************************************************************************
RajaVector& RajaVector::operator+=(const RajaVector& v) {
  vector_vec_add(size, data, v.data);
  return *this;
}

// ***************************************************************************
RajaVector& RajaVector::operator*=(const double d) {
  dbg();
  vector_vec_mul(size, data, d);
  return *this;
}

// ***************************************************************************
RajaVector& RajaVector::Add(const double alpha, const RajaVector& v) {
  dbg();
  vector_axpy(Size(),alpha, data, v.data);
  return *this;
}


// ***************************************************************************
void RajaVector::Neg() {
  dbg();
  vector_neg(Size(),ptr());
}

// *****************************************************************************
void RajaVector::SetSubVector(const RajaArray<int> &ess_tdofs,
                              const double value,
                              const int N) {
  dbg();
  vector_set_subvector_const(N, value, data, ess_tdofs.ptr());
}


// ***************************************************************************
double RajaVector::Min() const {
  dbg();
  return vector_min(Size(),(double*)data);
}

// ***************************************************************************
// from mfem::TCGSolver<mfem::RajaVector>::Mult in linalg/solvers.hpp:224
void add(const RajaVector& v1, const double alpha,
         const RajaVector& v2, RajaVector& out) {
  dbg();
  vector_xpay(out.Size(),alpha,out.ptr(),v1.ptr(),v2.ptr());
}

// *****************************************************************************
void add(const double alpha,
         const RajaVector& v1,
         const double beta,
         const RajaVector& v2,
         RajaVector& out) {
  dbg();
  /* used in templated TRK3SSPSolver, but not here */
  assert(false);
}

// ***************************************************************************
void subtract(const RajaVector& v1,
              const RajaVector& v2,
              RajaVector& out) {
  dbg();
  vector_xsy(out.Size(),out.ptr(),v1.ptr(),v2.ptr());
}

} // mfem
