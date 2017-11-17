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
#ifndef OKINA_RAJA_ARRAY_HEADER
#define OKINA_RAJA_ARRAY_HEADER

namespace mfem {

template <class T, bool xyz = true> class RajaArray;

// Partial Specializations for xyz==TRUE *************************************
template <class T> class RajaArray<T,true> {
 private:
  T* data = NULL;
  size_t d[4];
 public:
  RajaArray():d{0,0,0,0} {}
  RajaArray(const size_t x) {allocate(x);}
  RajaArray(const size_t x,const size_t y) {allocate(x,y);}
  inline T* ptr() { return data; }
  inline const T* ptr() const { return data; }
  inline operator T* () { return data; }
  inline operator const T* () const { return data; }
  inline size_t size() const { return d[0]*d[1]*d[2]*d[3]; }
  inline size_t bytes() const { return size()*sizeof(T); }
  void allocate(const size_t X, const size_t Y =1,
                const size_t Z =1, const size_t D =1,
                const bool transposed = false) {
    d[0]=X; d[1]=Y; d[2]=Z; d[3]=D;
    data=(T*)::malloc(bytes());
  }
  inline T& operator[](const size_t x) { return data[x]; }
  inline T& operator()(const size_t x, const size_t y) {
    return data[x + d[0]*y];
  }
  inline T& operator()(const size_t x, const size_t y, const size_t z) {
    return data[x + d[0]*(y + d[1]*z)];
  }
};

// Partial Specializations for xyz==FALSE ************************************
template <class T> class RajaArray<T,false> {
 private:
  static const int DIM = 4;
  T* data = NULL;
  size_t d[DIM];
 public:
  RajaArray():d{0,0,0,0} {}
  RajaArray(const size_t d0) {allocate(d0);}
  inline T* ptr() { return data; }
  inline const T* ptr() const { return data; }
  inline operator T* () { return data; }
  inline operator const T* () const { return data; }
  inline size_t size() const { return d[0]*d[1]*d[2]*d[3]; }
  inline size_t bytes() const { return size()*sizeof(T); }
  void allocate(const size_t X, const size_t Y =1,
                const size_t Z =1, const size_t D =1,
                const bool transposed = false) {
    d[0]=X; d[1]=Y; d[2]=Z; d[3]=D;
    data=(T*)::malloc(bytes());
#define xsw(a,b) a^=b^=a^=b
    if (transposed) { xsw(d[0],d[1]); }
    for (size_t i=1,b=d[0]; i<DIM; xsw(d[i],b),++i) {
      d[i]*=d[i-1];
    }
    d[0]=1;
    if (transposed) { xsw(d[0],d[1]); }
  }
  inline T& operator[](const size_t x) { return data[x]; }
  inline T& operator()(const size_t x, const size_t y) {
    return data[d[0]*x + d[1]*y];
  }
  inline T& operator()(const size_t x, const size_t y, const size_t z) {
    return data[d[0]*x + d[1]*y + d[2]*z];
  }
};

} // mfem

#endif // OKINA_RAJA_ARRAY_HEADER

