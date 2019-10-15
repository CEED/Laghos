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
#include "../hip.hpp"

namespace mfem
{

HipVector::~HipVector()
{
   if (!own) { return; }
   rmalloc::operator delete (data);
}

// ***************************************************************************
double* HipVector::alloc(const size_t sz)
{
   return (double*) rmalloc::operator new (sz);
}

// ***************************************************************************
void HipVector::SetSize(const size_t sz, const void* ptr)
{
   own=true;
   size = sz;
   if (!data) { data = alloc(sz); }
   if (ptr) { rDtoD(data,ptr,bytes()); }
}

// ***************************************************************************
HipVector::HipVector(const size_t sz):size(sz),data(alloc(sz)),own(true) {}
HipVector::HipVector(const size_t sz,double value):
   size(sz),data(alloc(sz)),own(true)
{
   *this=value;
}

HipVector::HipVector(const HipVector& v):
   size(0),data(NULL),own(true) { SetSize(v.Size(), v); }

HipVector::HipVector(const HipVector *v):size(v->size),data(v->data),
   own(false) {}

HipVector::HipVector(HipArray<double>& v):size(v.size()),data(v.ptr()),
   own(false) {}

// Host 2 Device ***************************************************************
HipVector::HipVector(const Vector& v):size(v.Size()),data(alloc(size)),
   own(true)
{
   assert(v.GetData());
   rmemcpy::rHtoD(data,v.GetData(),size*sizeof(double));
}

// Device 2 Host ***************************************************************
HipVector::operator Vector()
{
   if (!rconfig::Get().Hip()) { return Vector(data,size); }
   double *h_data= (double*) ::malloc(bytes());
   rmemcpy::rDtoH(h_data,data,bytes());
   Vector mfem_vector(h_data,size);
   mfem_vector.MakeDataOwner();
   return mfem_vector;
}

HipVector::operator Vector() const
{
   if (!rconfig::Get().Hip()) { return Vector(data,size); }
   double *h_data= (double*) ::malloc(bytes());
   rmemcpy::rDtoH(h_data,data,bytes());
   Vector mfem_vector(h_data,size);
   mfem_vector.MakeDataOwner();
   return mfem_vector;
}

// ***************************************************************************
void HipVector::Print(std::ostream& out, int width) const
{
   double *h_data = (double*) ::malloc(bytes());
   rmemcpy::rDtoH(h_data,data,bytes());
   for (size_t i=0; i<size; i+=1)
   {
      printf("\n\t[%ld] %.15e",i,h_data[i]);
   }
   free(h_data);
}

// ***************************************************************************
HipVector* HipVector::GetRange(const size_t offset,
                                 const size_t entries) const
{
   static HipVector ref;
   ref.size = entries;
   ref.data = (double*) ((unsigned char*)data + (offset*sizeof(double)));
   ref.own = false;
   return &ref;
}

// ***************************************************************************
HipVector& HipVector::operator=(const HipVector& v)
{
   SetSize(v.Size(),v.data);
   own = false;
   return *this;
}

// ***************************************************************************
HipVector& HipVector::operator=(const Vector& v)
{
   size=v.Size();
   if (!rconfig::Get().Hip()) { SetSize(size,v.GetData()); }
   else { rHtoD(data,v.GetData(),bytes()); }
   own = false;
   return *this;
}

// ***************************************************************************
HipVector& HipVector::operator=(double value)
{
   vector_op_eq(size, value, data);
   return *this;
}

// ***************************************************************************
double HipVector::operator*(const HipVector& v) const
{
   return vector_dot(size, data, v.data);
}

// *****************************************************************************
HipVector& HipVector::operator-=(const HipVector& v)
{
   vector_vec_sub(size, data, v.data);
   return *this;
}

// ***************************************************************************
HipVector& HipVector::operator+=(const HipVector& v)
{
   vector_vec_add(size, data, v.data);
   return *this;
}

// ***************************************************************************
HipVector& HipVector::operator+=(const Vector& v)
{
   double *d_v_data;
   assert(v.GetData());
   if (!rconfig::Get().Hip()) { d_v_data=v.GetData(); }
   else { rmemcpy::rHtoD(d_v_data = alloc(size),v.GetData(),bytes()); }
   vector_vec_add(size, data, d_v_data);
   return *this;
}

// ***************************************************************************
HipVector& HipVector::operator*=(const double d)
{
   vector_vec_mul(size, data, d);
   return *this;
}

// ***************************************************************************
HipVector& HipVector::Add(const double alpha, const HipVector& v)
{
   vector_axpy(Size(),alpha, data, v.data);
   return *this;
}

// ***************************************************************************
void HipVector::Neg()
{
   vector_neg(Size(),ptr());
}

// *****************************************************************************
void HipVector::SetSubVector(const HipArray<int> &ess_tdofs,
                              const double value,
                              const int N)
{
   vector_set_subvector_const(N, value, data, ess_tdofs.ptr());
}


// ***************************************************************************
double HipVector::Min() const
{
   return vector_min(Size(),(double*)data);
}

// ***************************************************************************
void add(const HipVector& v1, const double alpha,
         const HipVector& v2, HipVector& out)
{
   vector_xpay(out.Size(),alpha,out.ptr(),v1.ptr(),v2.ptr());
}

// *****************************************************************************
void add(const double alpha,
         const HipVector& v1,
         const double beta,
         const HipVector& v2,
         HipVector& out) { assert(false); }

// ***************************************************************************
void subtract(const HipVector& v1,
              const HipVector& v2,
              HipVector& out)
{
   vector_xsy(out.Size(),out.ptr(),v1.ptr(),v2.ptr());
}

} // mfem
