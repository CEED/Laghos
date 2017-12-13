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
  if (!own) return;
  dbg("\033[33m[~v");
  rUnManage(data);
}

// ***************************************************************************
double* RajaVector::rmalloc(const size_t sz) {
  dbg("\033[33m[v");
  return (double*) rManage(sz/**sizeof(double)*/);
}

// ***************************************************************************
void RajaVector::SetSize(const size_t sz, const void* ptr) {
  //dbg("\033[33m[size=%d, new sz=%d]\033[m",size,sz);
  own=true;
  size = sz;
  if (!data) { data = rmalloc(size); }
  if (ptr) { ::memcpy(data,ptr,bytes());}
}

// ***************************************************************************
RajaVector::RajaVector(const size_t sz):size(sz),data(rmalloc(sz)),own(true) {}

RajaVector::RajaVector(const RajaVector& v):
  size(0),data(NULL),own(true) { SetSize(v.Size(), v); }

RajaVector::RajaVector(const RajaVectorRef& ref):
  size(ref.v.size),data(ref.v.data),own(false) { }

RajaVector::RajaVector(const Vector& v):
  size(0),data(NULL),own(false) { SetSize(v.Size(), v.GetData()); }
//  size(v.Size()),data(v.GetData()),own(false) {}

RajaVector::RajaVector(RajaArray<double>& v):
  size(v.size()),data(v.ptr()),own(false) { /*SetSize(v.size(),v.ptr()); */}

// ***************************************************************************
RajaVector::operator Vector() { return Vector(data,size); }
RajaVector::operator Vector() const { return Vector(data,size); }

// ***************************************************************************
void RajaVector::Print(std::ostream& out, int width) const {
  for (size_t i=0; i<size; i+=1) 
    printf("\n\t[%ld] %.15e",i,data[i]);
}

// ***************************************************************************
RajaVectorRef RajaVector::GetRange(const size_t offset,
                                   const size_t entries) const {
  RajaVectorRef ret;
  RajaVector& v = ret.v;
  v.data = (double*) ((unsigned char*)data + (offset*sizeof(double)));
  v.size = entries;
  v.own = false;
  return ret;
}

// ***************************************************************************
RajaVector& RajaVector::operator=(const RajaVector& v) {
  SetSize(v.Size(),v.data);
  own = false;
  return *this;
}

// ***************************************************************************
RajaVector& RajaVector::operator=(double value) {
  vector_op_eq(size, value, data);
  return *this;
}

// ***************************************************************************
double RajaVector::operator*(const RajaVector& v) const {
  return vector_dot(size, data, v.data);
}

// *****************************************************************************
RajaVector& RajaVector::operator-=(const RajaVector& v) {
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
  vector_vec_mul(size, data, d);
  return *this;
}

// ***************************************************************************
RajaVector& RajaVector::Add(const double alpha, const RajaVector& v) {
  vector_axpy(Size(),alpha, data, v.data);
  return *this;
}


// ***************************************************************************
void RajaVector::Neg() {
  vector_neg(Size(),ptr());
}

// *****************************************************************************
void RajaVector::SetSubVector(const RajaArray<int> &ess_tdofs,
                              const double value,
                              const int N) {
  vector_set_subvector_const(N, value, data, ess_tdofs.ptr());
}


// ***************************************************************************
double RajaVector::Min() const {
  return vector_min(Size(),(double*)data);
}

// ***************************************************************************
// from mfem::TCGSolver<mfem::RajaVector>::Mult in linalg/solvers.hpp:224
void add(const RajaVector& v1, const double alpha,
         const RajaVector& v2, RajaVector& out) {
  vector_xpay(out.Size(),alpha,out.ptr(),v1.ptr(),v2.ptr());
}

// *****************************************************************************
void add(const double alpha,
         const RajaVector& v1,
         const double beta,
         const RajaVector& v2,
         RajaVector& out) {
  /* used in templated TRK3SSPSolver, but not here */
  assert(false);
}

// ***************************************************************************
void subtract(const RajaVector& v1,
              const RajaVector& v2,
              RajaVector& out) {
  vector_xsy(out.Size(),out.ptr(),v1.ptr(),v2.ptr());
}

} // mfem
