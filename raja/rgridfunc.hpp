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
#ifndef MFEM_RAJA_GRIDFUNCTION
#define MFEM_RAJA_GRIDFUNCTION

namespace mfem {

class RajaGridFunction : public RajaVector {
 public:
  const RajaFiniteElementSpace& fes;
 public:
  RajaGridFunction(const RajaFiniteElementSpace& f):
    RajaVector(f.GetVSize()),fes(f) {}
  RajaGridFunction(const RajaFiniteElementSpace& f,const RajaVectorRef ref):
    RajaVector(ref), fes(f) {}
  void ToQuad(const IntegrationRule&,RajaVector&);

  RajaGridFunction& operator=(const RajaVector& v) {
    RajaVector::operator=(v);
    return *this;
  }
};

} // mfem

#endif
