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

// ***************************************************************************
static double* rmalloc(const size_t sz) {
  return (double*)::malloc(sz*sizeof(double));
}

// ***************************************************************************
void RajaVector::SetSize(const size_t sz, const void* ptr) {
  size = sz;
  if (!data) { data = rmalloc(size); }
  if (ptr) { ::memcpy(data,ptr,bytes()); }
}

// ***************************************************************************
RajaVector::RajaVector(const size_t sz):size(sz),data(rmalloc(sz)) {}

RajaVector::RajaVector(const RajaVector& v):
  size(0),data(NULL) { SetSize(v.Size(), v); }

RajaVector::RajaVector(const RajaVectorRef& ref):
  size(ref.v.size),data(ref.v.data) { }

RajaVector::RajaVector(const Vector& v):
  size(0),data(NULL) { SetSize(v.Size(), v.GetData()); }

RajaVector::RajaVector(RajaArray<double>& v):
  size(0),data(NULL) { SetSize(v.size(),v.ptr()); }

// ***************************************************************************
RajaVector::operator Vector() { return Vector(data,size); }
RajaVector::operator Vector() const { return Vector(data,size); }

// ***************************************************************************
void RajaVector::Print(std::ostream& out, int width) const {
  for (size_t i=0; i<size; i+=1) {
    printf("\n\t[%ld] %.15e",i,data[i]);
  }
  //Vector(data,size).Print(out, width);
}

// ***************************************************************************
RajaVectorRef RajaVector::GetRange(const size_t offset,
                                   const size_t entries) const {
  RajaVectorRef ret;
  RajaVector& v = ret.v;
  v.data = (double*) ((unsigned char*)data + (offset*sizeof(double)));
  v.size = entries;
  return ret;
}

// ***************************************************************************
RajaVector& RajaVector::operator=(const RajaVector& v) {
  SetSize(v.Size(),v.data);
  return *this;
}

// ***************************************************************************
RajaVector& RajaVector::operator=(double value) {
  const RAJA::RangeSegment range(0, size);
  RAJA::forall<RAJA::seq_exec>(range,[=](RAJA::Index_type i) {
      data[i] = value;
    });
  return *this;
}

// ***************************************************************************
double RajaVector::operator*(const RajaVector& v) const {
  RAJA::ReduceSum<RAJA::seq_reduce, RAJA::Real_type> dot(0.0);
  const RAJA::RangeSegment range(0, size);
  RAJA::forall<RAJA::seq_exec>(range,[=](RAJA::Index_type i) {
      dot += data[i] * v[i];
    });
  return dot;
}

// *****************************************************************************
RajaVector& RajaVector::operator-=(const RajaVector& v) {
  const RAJA::RangeSegment range(0, size);
  RAJA::forall<RAJA::seq_exec>(range,[=](RAJA::Index_type i) {
      data[i] -= v[i];
    });
 return *this;
}

// ***************************************************************************
RajaVector& RajaVector::operator+=(const RajaVector& v) {
  const RAJA::RangeSegment range(0, size);
  RAJA::forall<RAJA::seq_exec>(range,[=](RAJA::Index_type i) {
      data[i] += v[i];
    });
  return *this;
}

// ***************************************************************************
RajaVector& RajaVector::operator*=(const double d) {
  for (size_t i=0; i<size; i+=1)
    data[i]*=d;
  return *this;
}

// ***************************************************************************
RajaVector& RajaVector::Add(const double alpha, const RajaVector& v) {
  const RAJA::RangeSegment range(0, size);
  RAJA::forall<RAJA::seq_exec>(range,[=](RAJA::Index_type i) {
      data[i] += alpha * v[i];
    });
  return *this;
}


// ***************************************************************************
void RajaVector::Neg() {
  const RAJA::RangeSegment range(0, size);
  RAJA::forall<RAJA::seq_exec>(range,[=](RAJA::Index_type i) {
      data[i] *= -1.0;
    });
}

// *****************************************************************************
void RajaVector::SetSubVector(const void* pdofs,
                              const double value,
                              const int N) {
  const RAJA::RangeSegment range(0, N);
  const int* dofs =(const int*)pdofs;
  RAJA::forall<RAJA::seq_exec>(range,[=](RAJA::Index_type i) {
      const int dof_i = dofs[i];
      data[dof_i] = value;
      if (dof_i >= 0) {
        data[dof_i] = value;
      } else {
        data[-dof_i-1] = -value;
      }
    });
}


// ***************************************************************************
double RajaVector::Min() const {
  RAJA::ReduceMin<RAJA::seq_reduce, RAJA::Real_type> min(data[0]);
  const RAJA::RangeSegment range(0, size);
  RAJA::forall<RAJA::seq_exec>(range,[=](RAJA::Index_type i) {
      min.min(data[i]);
    });
  return min;
}

// ***************************************************************************
// from mfem::TCGSolver<mfem::RajaVector>::Mult in linalg/solvers.hpp:224
void add(const RajaVector& v1, const double alpha,
         const RajaVector& v2, RajaVector& out) {
  const RAJA::RangeSegment range(0, out.Size());
  const RAJA::Real_type* x1 = v1.ptr();
  const RAJA::Real_type* x2 = v2.ptr();
  RAJA::Real_type* y=out.ptr();
  RAJA::forall<RAJA::seq_exec>(range,[=](RAJA::Index_type i) {
      y[i] = x1[i] + (alpha * x2[i]);
    });
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
  const RAJA::RangeSegment range(0, out.Size());
  const RAJA::Real_type* x1 = v1.ptr();
  const RAJA::Real_type* x2 = v2.ptr();
  RAJA::Real_type* y=out.ptr();
  RAJA::forall<RAJA::seq_exec>(range,[=](RAJA::Index_type i) {
      y[i] = x1[i]-x2[i];
    });
}

} // mfem
