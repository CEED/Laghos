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

#ifndef MFEM_LAGHOS_ABSTRACTS_HPP
#define MFEM_LAGHOS_ABSTRACTS_HPP

namespace mfem
{

namespace hydrodynamics
{
   
// Abstract base class AbcForcePAOperator **************************************
class AbcForcePAOperator : public Operator{
public:
   AbcForcePAOperator(int size):Operator(size){}
   AbcForcePAOperator(PLayout &layout):Operator(layout){}
   virtual void Mult(const Vector&, Vector&) const =0;
   virtual void MultTranspose(const Vector&, Vector&) const =0;
};
   
// Abstract base class AbcMassPAOperator ***************************************
class AbcMassPAOperator : public Operator{
public:
   AbcMassPAOperator(int size):Operator(size){}
   AbcMassPAOperator(PLayout &layout):Operator(layout){}
   virtual void Setup() =0;
   virtual void ComputeDiagonal2D(Vector&) const =0;
   virtual void ComputeDiagonal3D(Vector&) const =0;
   virtual void Mult(const Vector&, Vector&) const =0;
   virtual const Operator *GetProlongation() const =0;
   virtual const Operator *GetRestriction() const =0;
};

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_LAGHOS_ABSTRACTS_HPP
