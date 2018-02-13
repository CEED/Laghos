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
  
RajaVector::~RajaVector(){
  if (!own) return;
  //rdbg("\033[33m[~v(%d)",size);
  rdbg("\033[33m[~v");
  this->operator delete(data);
}

// ***************************************************************************
double* RajaVector::alloc(const size_t sz) {
  rdbg("\033[33m[v");
  return (double*) this->operator new(sz);
}

// ***************************************************************************
  void RajaVector::SetSize(const size_t sz, const void* ptr) {
  //rdbg("\033[33m[size=%d, new sz=%d]\033[m",size,sz);
  own=true;
  size = sz;
  if (!data) { data = alloc(sz); }
#ifdef __NVCC__
#ifdef __RAJA__
  if (ptr) {
    if (rconfig::Get().Cuda())
      checkCudaErrors(cuMemcpyDtoD((CUdeviceptr)data,(CUdeviceptr)ptr,bytes()));
    else ::memcpy(data,ptr,bytes());
  }
#else // __RAJA__
  if (ptr) {
    assert(data);
    assert(ptr);
    assert(bytes()>0);
    checkCudaErrors(cuMemcpyDtoD((CUdeviceptr)data,(CUdeviceptr)ptr,bytes()));
  }
#endif
#else // __NVCC__
  if (ptr) { ::memcpy(data,ptr,bytes());}
#endif
}

// ***************************************************************************
RajaVector::RajaVector(const size_t sz):size(sz),data(alloc(sz)),own(true) {}
RajaVector::RajaVector(const size_t sz,double value):
  size(sz),data(alloc(sz)),own(true) {
  push(SkyBlue);
  //printf("\033[31m[%d]\033[m",sz);
  *this=value;
  pop();
}

RajaVector::RajaVector(const RajaVector& v):
  size(0),data(NULL),own(true) { SetSize(v.Size(), v); }

RajaVector::RajaVector(const RajaVector *v): size(v->size),data(v->data),own(false) {}
  
RajaVector::RajaVector(RajaArray<double>& v): size(v.size()),data(v.ptr()),own(false) {}

// Host 2 Device ***************************************************************
RajaVector::RajaVector(const Vector& v):
  size(v.Size()),data(NULL),own(true) {
#ifdef __NVCC__
#ifdef __RAJA__
  if (rconfig::Get().Cuda()) {
    checkCudaErrors(cuMemAlloc((CUdeviceptr*)&data, size*sizeof(double)));
    checkCudaErrors(cuMemcpyHtoD((CUdeviceptr)data,v.GetData(),v.Size()*sizeof(double)));
  }else{
    SetSize(v.Size(), v.GetData());  
  }
#else
  //printf("\033[31m[RajaVector()] Host 2 Device\033[m\n");
  checkCudaErrors(cuMemAlloc((CUdeviceptr*)&data, size*sizeof(double)));
  checkCudaErrors(cuMemcpyHtoD((CUdeviceptr)data,v.GetData(),v.Size()*sizeof(double)));
#endif
#else
  SetSize(v.Size(), v.GetData());  
#endif
}

// Device 2 Host ***************************************************************
RajaVector::operator Vector() {
#ifdef __NVCC__
#ifdef __RAJA__
  if (rconfig::Get().Cuda()){
    double *h_data= (double*) ::malloc(bytes());
    checkCudaErrors(cuMemcpyDtoH(h_data,(CUdeviceptr)data,bytes()));
    return Vector(h_data,size);
  }else return Vector(data,size);
#else
  //printf("\033[31m[Vector()] Device 2 Host\033[m\n");
  double *h_data= (double*) ::malloc(bytes());
  checkCudaErrors(cuMemcpyDtoH(h_data,(CUdeviceptr)data,bytes()));
  return Vector(h_data,size);
#endif
#else
  return Vector(data,size);
#endif
}
  
RajaVector::operator Vector() const {
#ifdef __NVCC__
#ifdef __RAJA__
  if (rconfig::Get().Cuda()){
    double *h_data= (double*) ::malloc(bytes());
    checkCudaErrors(cuMemcpyDtoH(h_data,(CUdeviceptr)data,bytes()));
    return Vector(h_data,size);
  }else return Vector(data,size);
#else
  //printf("\033[31m[Vector()const] Device 2 Host\033[m\n");
  double *h_data= (double*) ::malloc(bytes());
  checkCudaErrors(cuMemcpyDtoH(h_data,(CUdeviceptr)data,bytes()));
  return Vector(h_data,size);
#endif
#else
  return Vector(data,size);
#endif
}

// ***************************************************************************
void RajaVector::Print(std::ostream& out, int width) const {
  double *h_data;
#ifdef __NVCC__
  if (rconfig::Get().Cuda()){
    //dbg()<<"Device 2 Host (const)";
    h_data= (double*) ::malloc(bytes());
    checkCudaErrors(cuMemcpyDtoH(h_data,(CUdeviceptr)data,bytes()));
  } else h_data=data;
#else
  h_data=data;
#endif
  for (size_t i=0; i<size; i+=1) 
    printf("\n\t[%ld] %.15e",i,h_data[i]);
}


// *****************************************************************************
static RajaVector ref;
  
// ***************************************************************************
RajaVector* RajaVector::GetRange(const size_t offset,
                                 const size_t entries) const {
  //RajaVector *ref = ::new RajaVector();
  //ref->size = entries;
  //ref->data = (double*) ((unsigned char*)data + (offset*sizeof(double)));
  //ref->own = false;
  ref.size = entries;
  ref.data = (double*) ((unsigned char*)data + (offset*sizeof(double)));
  ref.own = false;
  return &ref;
}

// ***************************************************************************
RajaVector& RajaVector::operator=(const RajaVector& v) {
  SetSize(v.Size(),v.data);
  own = false;
  return *this;
}

// ***************************************************************************
RajaVector& RajaVector::operator=(const Vector& v) {
  size=v.Size();
#ifdef __NVCC__
#ifdef __RAJA__
  if (rconfig::Get().Cuda()) {
    checkCudaErrors(cuMemcpyHtoD((CUdeviceptr)data,v.GetData(),v.Size()*sizeof(double)));
  }else{
    SetSize(v.Size(),v.GetData());  
  }
#else
  checkCudaErrors(cuMemcpyHtoD((CUdeviceptr)data,v.GetData(),v.Size()*sizeof(double)));
#endif
#else
  SetSize(v.Size(),v.GetData());  
#endif
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
RajaVector& RajaVector::operator+=(const Vector& v) {
  double *d_v_data;
#ifdef __NVCC__
#ifdef __RAJA__
  if (rconfig::Get().Cuda()) {
    checkCudaErrors(cuMemAlloc((CUdeviceptr*)&d_v_data, bytes()));
    checkCudaErrors(cuMemcpyHtoD((CUdeviceptr)d_v_data,v.GetData(),bytes()));
    vector_vec_add(size, data, d_v_data);  
  }else{
    vector_vec_add(size, data, v.GetData());  
  }
#else
  checkCudaErrors(cuMemAlloc((CUdeviceptr*)&d_v_data, bytes()));
  checkCudaErrors(cuMemcpyHtoD((CUdeviceptr)d_v_data,v.GetData(),bytes()));
  vector_vec_add(size, data, d_v_data);  
#endif
#else
  vector_vec_add(size, data, v.GetData());  
#endif
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
