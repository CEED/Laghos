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
#ifndef MFEM_RAJA_BILININTEG
#define MFEM_RAJA_BILININTEG

namespace mfem {

// ***************************************************************************
// * RajaGeometry
// ***************************************************************************
class RajaGeometry {
 public:
  RajaArray<double> meshNodes;
  RajaArray<double> J, invJ, detJ;
  static RajaGeometry Get(RajaFiniteElementSpace&,const IntegrationRule&);
};

// ***************************************************************************
// * RajaDofQuadMaps
// ***************************************************************************
class RajaDofQuadMaps {
 private:
  static std::map<std::string, RajaDofQuadMaps> AllDofQuadMaps;
  std::string hash;
 public:
  RajaArray<double, false> dofToQuad, dofToQuadD; // B
  RajaArray<double, false> quadToDof, quadToDofD; // B^T
  RajaArray<double> quadWeights;
 public:
  static RajaDofQuadMaps& Get(const RajaFiniteElementSpace&,
                              const IntegrationRule&,
                              const bool = false);
  static RajaDofQuadMaps& Get(const RajaFiniteElementSpace&,
                              const RajaFiniteElementSpace&,
                              const IntegrationRule&,
                              const bool = false);
  static RajaDofQuadMaps& Get(const FiniteElement&,
                              const FiniteElement&,
                              const IntegrationRule&,
                              const bool = false);
  static RajaDofQuadMaps& GetTensorMaps(const FiniteElement&,
                                        const FiniteElement&,
                                        const IntegrationRule&,
                                        const bool = false);
  static RajaDofQuadMaps GetD2QTensorMaps(const FiniteElement&,
                                          const IntegrationRule&,
                                          const bool = false);
  static RajaDofQuadMaps& GetSimplexMaps(const FiniteElement&,
                                         const IntegrationRule&,
                                         const bool = false);
  static RajaDofQuadMaps& GetSimplexMaps(const FiniteElement&,
                                         const FiniteElement&,
                                         const IntegrationRule&,
                                         const bool = false);
  static RajaDofQuadMaps GetD2QSimplexMaps(const FiniteElement&,
      const IntegrationRule&,
      const bool = false);
};

// ***************************************************************************
// * Base Integrator
// ***************************************************************************
class RajaIntegrator {
 protected:
  Mesh* mesh = NULL;
  RajaFiniteElementSpace* trialFESpace = NULL;
  RajaFiniteElementSpace* testFESpace = NULL;
  RajaIntegratorType itype;
  const IntegrationRule* ir = NULL;
  RajaDofQuadMaps maps;
  RajaDofQuadMaps mapsTranspose;
 private:
 public:
  virtual std::string GetName() = 0;
  void SetIntegrationRule(const IntegrationRule& ir_);
  const IntegrationRule& GetIntegrationRule() const;
  virtual void SetupIntegrationRule() = 0;
  virtual void SetupIntegrator(RajaBilinearForm& bform_,
                               const RajaIntegratorType itype_);
  virtual void Setup() = 0;
  virtual void Assemble() = 0;
  virtual void MultAdd(RajaVector& x, RajaVector& y) = 0;
  virtual void MultTransposeAdd(RajaVector&, RajaVector&) {assert(false);}
  RajaGeometry GetGeometry();
};

// ***************************************************************************
// * Mass Integrator
// ***************************************************************************
class RajaMassIntegrator : public RajaIntegrator {
 private:
  double coeff;
  RajaVector op;
 public:
  RajaMassIntegrator(const double c = 1.0):coeff(c) {}
  virtual ~RajaMassIntegrator() {}
  virtual std::string GetName() {return "MassIntegrator";}
  virtual void SetupIntegrationRule();
  virtual void Setup() {}
  virtual void Assemble();
  void SetOperator(RajaVector& v);
  virtual void MultAdd(RajaVector& x, RajaVector& y);
};

} // mfem

#endif
