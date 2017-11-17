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
#ifndef MFEM_RAJA_BILINEARFORM
#define MFEM_RAJA_BILINEARFORM

namespace mfem {

// ***************************************************************************
// * RajaIntegratorType
// ***************************************************************************
enum RajaIntegratorType {
  DomainIntegrator       = 0,
  BoundaryIntegrator     = 1,
  InteriorFaceIntegrator = 2,
  BoundaryFaceIntegrator = 3,
};

class RajaIntegrator;

// ***************************************************************************
// * RajaBilinearForm
// ***************************************************************************
class RajaBilinearForm : public Operator {
  friend class RajaIntegrator;
 protected:
  typedef std::vector<RajaIntegrator*> IntegratorVector;
  mutable Mesh* mesh;
  mutable RajaFiniteElementSpace* trialFes;
  mutable RajaFiniteElementSpace* testFes;
  IntegratorVector integrators;
  mutable RajaVector localX, localY;
 public:
  RajaBilinearForm(RajaFiniteElementSpace*);
  Mesh& GetMesh() const { return *mesh; }
  RajaFiniteElementSpace& GetTrialFESpace() const { return *trialFes;}
  RajaFiniteElementSpace& GetTestFESpace() const { return *testFes;}
  // *************************************************************************
  void AddDomainIntegrator(RajaIntegrator*);
  void AddBoundaryIntegrator(RajaIntegrator*);
  void AddInteriorFaceIntegrator(RajaIntegrator*);
  void AddBoundaryFaceIntegrator(RajaIntegrator*);
  void AddIntegrator(RajaIntegrator*, const RajaIntegratorType);
  // *************************************************************************
  virtual void Assemble();
  void FormLinearSystem(const Array<int>& constraintList,
                        RajaVector& x, RajaVector& b,
                        Operator*& Aout,
                        RajaVector& X, RajaVector& B,
                        int copy_interior = 0);

  void FormOperator(const Array<int>& constraintList, Operator*& Aout);

  void InitRHS(const Array<int>& constraintList,
               RajaVector& x, RajaVector& b,
               Operator* Aout,
               RajaVector& X, RajaVector& B,
               int copy_interior = 0);

  virtual void Mult(const RajaVector& x, RajaVector& y) const;
  virtual void MultTranspose(const RajaVector& x, RajaVector& y) const;
  void RecoverFEMSolution(const RajaVector&, const RajaVector&, RajaVector&);
};


// ***************************************************************************
// * Constrained Operator
// ***************************************************************************
class RajaConstrainedOperator : public Operator {
 protected:
  Operator* A;
  bool own_A;
  RajaArray<int> constraintList;
  int constraintIndices;
  mutable RajaVector z, w;
 public:
  RajaConstrainedOperator(Operator* A_,
                          const Array<int>& constraintList_,
                          bool own_A_ = false);
  void Setup(Operator* A_,
             const Array<int>& constraintList_,
             bool own_A_ = false);
  void EliminateRHS(const RajaVector& x, RajaVector& b) const;
  virtual void Mult(const RajaVector& x, RajaVector& y) const;
  virtual ~RajaConstrainedOperator() {}
};

} // mfem

#endif // MFEM_RAJA_BILINEARFORM
