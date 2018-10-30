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
#ifndef LAGHOS_CUDA_VECTOR
#define LAGHOS_CUDA_VECTOR

namespace mfem
{

class CudaVector : public rmalloc<double>
{
private:
   size_t size = 0;
   double* data = NULL;
   bool own = true;
public:
   CudaVector(): size(0),data(NULL),own(true) {}
   CudaVector(const CudaVector&);
   CudaVector(const CudaVector*);
   CudaVector(const size_t);
   CudaVector(const size_t,double);
   CudaVector(const Vector& v);
   CudaVector(CudaArray<double>& v);
   operator Vector();
   operator Vector() const;
   double* alloc(const size_t);
   inline double* ptr() const { return data;}
   inline double* GetData() const { return data;}
   inline operator double* () { return data; }
   inline operator const double* () const { return data; }
   void Print(std::ostream& = std::cout, int = 8) const;
   void SetSize(const size_t,const void* =NULL);
   inline size_t Size() const { return size; }
   inline size_t bytes() const { return size*sizeof(double); }
   double operator* (const CudaVector& v) const;
   CudaVector& operator = (const CudaVector& v);
   CudaVector& operator = (const Vector& v);
   CudaVector& operator = (double value);
   CudaVector& operator -= (const CudaVector& v);
   CudaVector& operator += (const CudaVector& v);
   CudaVector& operator += (const Vector& v);
   CudaVector& operator *=(const double d);
   CudaVector& Add(const double a, const CudaVector& Va);
   void Neg();
   CudaVector* GetRange(const size_t, const size_t) const;
   void SetSubVector(const CudaArray<int> &, const double, const int);
   double Min() const;
   ~CudaVector();
};

// ***************************************************************************
void add(const CudaVector&,const double,const CudaVector&,CudaVector&);
void add(const CudaVector&,const CudaVector&,CudaVector&);
void add(const double,const CudaVector&,const double,const CudaVector&,
         CudaVector&);
void subtract(const CudaVector&,const CudaVector&,CudaVector&);

}

#endif // LAGHOS_CUDA_VECTOR
