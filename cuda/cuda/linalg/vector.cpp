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

CudaVector::~CudaVector()
{
   if (!own) { return; }
   rmalloc::operator delete (data);
}

// ***************************************************************************
double* CudaVector::alloc(const int sz)
{
   return (double*) rmalloc::operator new (static_cast<size_t>(sz));
}

// ***************************************************************************
void CudaVector::SetSize(const int sz, const void* ptr)
{
   own=true;
   size = sz;
   if (!data) { data = alloc(sz); }
   if (ptr) { rDtoD(data,ptr,bytes()); }
}

// ***************************************************************************
CudaVector::CudaVector(const int sz):size(sz),data(alloc(sz)),own(true) {}
CudaVector::CudaVector(const int sz,double value):
   size(sz),data(alloc(sz)),own(true)
{
   *this=value;
}

CudaVector::CudaVector(const CudaVector& v):
   size(0),data(NULL),own(true) { SetSize(v.Size(), v); }

CudaVector::CudaVector(const CudaVector *v):size(v->size),data(v->data),
   own(false) {}

CudaVector::CudaVector(CudaArray<double>& v):size(v.size()),data(v.ptr()),
   own(false) {}

// Host 2 Device ***************************************************************
CudaVector::CudaVector(const Vector& v):size(v.Size()),data(alloc(size)),
   own(true)
{
   assert(v.GetData());
   rmemcpy::rHtoD(data,v.GetData(),size*static_cast<int>(sizeof(double)));
}

// Device 2 Host ***************************************************************
CudaVector::operator Vector()
{
   if (!rconfig::Get().Cuda()) { return Vector(data,size); }
   double *h_data= (double*) ::malloc(static_cast<size_t>(bytes()));
   rmemcpy::rDtoH(h_data,data,bytes());
   Vector mfem_vector(h_data,size);
   mfem_vector.MakeDataOwner();
   return mfem_vector;
}

CudaVector::operator Vector() const
{
   if (!rconfig::Get().Cuda()) { return Vector(data,size); }
   double *h_data= (double*) ::malloc(static_cast<size_t>(bytes()));
   rmemcpy::rDtoH(h_data,data,bytes());
   Vector mfem_vector(h_data,size);
   mfem_vector.MakeDataOwner();
   return mfem_vector;
}

// ***************************************************************************
void CudaVector::Print(std::ostream& out, int width) const
{
   double *h_data = (double*) ::malloc(static_cast<size_t>(bytes()));
   rmemcpy::rDtoH(h_data,data,bytes());
   for (int i=0; i<size; i+=1)
   {
      printf("\n\t[%d] %.15e", i, h_data[i]);
   }
   free(h_data);
}

// ***************************************************************************
CudaVector* CudaVector::GetRange(const int offset,
                                 const int entries) const
{
   static CudaVector ref;
   ref.size = entries;
   ref.data = (double*) ((unsigned char*)data + (offset*sizeof(double)));
   ref.own = false;
   return &ref;
}

// ***************************************************************************
CudaVector& CudaVector::operator=(const CudaVector& v)
{
   SetSize(v.Size(),v.data);
   own = false;
   return *this;
}

// ***************************************************************************
CudaVector& CudaVector::operator=(const Vector& v)
{
   size=v.Size();
   if (!rconfig::Get().Cuda()) { SetSize(size,v.GetData()); }
   else { rHtoD(data,v.GetData(),bytes()); }
   own = false;
   return *this;
}

// ***************************************************************************
CudaVector& CudaVector::operator=(double value)
{
   vector_op_eq(size, value, data);
   return *this;
}

// ***************************************************************************
double CudaVector::operator*(const CudaVector& v) const
{
   return vector_dot(size, data, v.data);
}

// *****************************************************************************
CudaVector& CudaVector::operator-=(const CudaVector& v)
{
   vector_vec_sub(size, data, v.data);
   return *this;
}

// ***************************************************************************
CudaVector& CudaVector::operator+=(const CudaVector& v)
{
   vector_vec_add(size, data, v.data);
   return *this;
}

// ***************************************************************************
CudaVector& CudaVector::operator+=(const Vector& v)
{
   double *d_v_data;
   assert(v.GetData());
   if (!rconfig::Get().Cuda()) { d_v_data=v.GetData(); }
   else { rmemcpy::rHtoD(d_v_data = alloc(size),v.GetData(),bytes()); }
   vector_vec_add(size, data, d_v_data);
   return *this;
}

// ***************************************************************************
CudaVector& CudaVector::operator*=(const double d)
{
   vector_vec_mul(size, data, d);
   return *this;
}

// ***************************************************************************
CudaVector& CudaVector::Add(const double alpha, const CudaVector& v)
{
   vector_axpy(Size(),alpha, data, v.data);
   return *this;
}

// ***************************************************************************
void CudaVector::Neg()
{
   vector_neg(Size(),ptr());
}

// *****************************************************************************
void CudaVector::SetSubVector(const CudaArray<int> &ess_tdofs,
                              const double value,
                              const int N)
{
   vector_set_subvector_const(N, value, data, ess_tdofs.ptr());
}


// ***************************************************************************
double CudaVector::Min() const
{
   return vector_min(Size(),(double*)data);
}

// ***************************************************************************
void add(const CudaVector& v1, const double alpha,
         const CudaVector& v2, CudaVector& out)
{
   vector_xpay(out.Size(),alpha,out.ptr(),v1.ptr(),v2.ptr());
}

// *****************************************************************************
void add(const double alpha,
         const CudaVector& v1,
         const double beta,
         const CudaVector& v2,
         CudaVector& out) { assert(false); }

// ***************************************************************************
void subtract(const CudaVector& v1,
              const CudaVector& v2,
              CudaVector& out)
{
   vector_xsy(out.Size(),out.ptr(),v1.ptr(),v2.ptr());
}

} // mfem
