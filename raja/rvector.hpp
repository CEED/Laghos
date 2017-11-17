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
#ifndef MFEM_RAJA_VECTOR
#define MFEM_RAJA_VECTOR

namespace mfem {

struct RajaVectorRef;

class RajaVector {
 private:
  size_t size;
  double* data;
 public:
  RajaVector(): size(0),data(NULL) {}
  RajaVector(const RajaVector&);
  RajaVector(const RajaVectorRef&);
  RajaVector(const size_t);
  RajaVector(const Vector& v);
  RajaVector(RajaArray<double>& v);
  operator Vector();
  operator Vector() const;
  inline double* ptr() const { return data;}
  inline operator double* () { return data; }
  inline operator const double* () const { return data; }
  void Print(std::ostream& = std::cout, int = 8) const;
  void SetSize(const size_t,const void* =NULL);
  inline size_t Size() const { return size; }
  inline size_t bytes() const { return size*sizeof(double); }
  double operator* (const RajaVector& v) const;
  RajaVector& operator = (const RajaVector& v);
  RajaVector& operator = (double value);
  RajaVector& operator -= (const RajaVector& v);
  RajaVector& operator += (const RajaVector& v);
  RajaVector& Add(const double a, const RajaVector& Va);
  void Neg();
  RajaVectorRef GetRange(const size_t, const size_t) const;
  void SetSubVector(const void*, const double, const int);
  double Min() const;
  inline virtual ~RajaVector() {}
};

struct RajaVectorRef { RajaVector v; };

// ***************************************************************************
void add(const RajaVector&,const double,const RajaVector&,RajaVector&);
void add(const RajaVector&,const RajaVector&,RajaVector&);
void add(const double,const RajaVector&,const double,const RajaVector&,
         RajaVector&);
void subtract(const RajaVector&,const RajaVector&,RajaVector&);

}

#endif // MFEM_RAJA_VECTOR
