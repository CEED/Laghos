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
#ifndef LAGHOS_HIP_BILININTEG
#define LAGHOS_HIP_BILININTEG

namespace mfem
{

// ***************************************************************************
// * HipGeometry
// ***************************************************************************
class HipGeometry
{
public:
   ~HipGeometry();
   HipArray<int> eMap;
   HipArray<double> meshNodes;
   HipArray<double> J, invJ, detJ;
   static HipGeometry* Get(HipFiniteElementSpace&,
                            const IntegrationRule&);
   static HipGeometry* Get(HipFiniteElementSpace&,
                            const IntegrationRule&,
                            const HipVector&);
   static void ReorderByVDim(GridFunction& nodes);
   static void ReorderByNodes(GridFunction& nodes);
};

// ***************************************************************************
// * HipDofQuadMaps
// ***************************************************************************
class HipDofQuadMaps
{
private:
   std::string hash;
public:
   HipArray<double, false> dofToQuad, dofToQuadD; // B
   HipArray<double, false> quadToDof, quadToDofD; // B^T
   HipArray<double> quadWeights;
public:
   ~HipDofQuadMaps();
   static void delHipDofQuadMaps();
   static HipDofQuadMaps* Get(const HipFiniteElementSpace&,
                               const IntegrationRule&,
                               const bool = false);
   static HipDofQuadMaps* Get(const HipFiniteElementSpace&,
                               const HipFiniteElementSpace&,
                               const IntegrationRule&,
                               const bool = false);
   static HipDofQuadMaps* Get(const FiniteElement&,
                               const FiniteElement&,
                               const IntegrationRule&,
                               const bool = false);
   static HipDofQuadMaps* GetTensorMaps(const FiniteElement&,
                                         const FiniteElement&,
                                         const IntegrationRule&,
                                         const bool = false);
   static HipDofQuadMaps* GetD2QTensorMaps(const FiniteElement&,
                                            const IntegrationRule&,
                                            const bool = false);
   static HipDofQuadMaps* GetSimplexMaps(const FiniteElement&,
                                          const IntegrationRule&,
                                          const bool = false);
   static HipDofQuadMaps* GetSimplexMaps(const FiniteElement&,
                                          const FiniteElement&,
                                          const IntegrationRule&,
                                          const bool = false);
   static HipDofQuadMaps* GetD2QSimplexMaps(const FiniteElement&,
                                             const IntegrationRule&,
                                             const bool = false);
};

// ***************************************************************************
// * Base Integrator
// ***************************************************************************
class HipIntegrator
{
protected:
   Mesh* mesh = NULL;
   HipFiniteElementSpace* trialFESpace = NULL;
   HipFiniteElementSpace* testFESpace = NULL;
   HipIntegratorType itype;
   const IntegrationRule* ir = NULL;
   HipDofQuadMaps* maps;
   HipDofQuadMaps* mapsTranspose;
private:
public:
   virtual std::string GetName() = 0;
   void SetIntegrationRule(const IntegrationRule& ir_);
   const IntegrationRule& GetIntegrationRule() const;
   virtual void SetupIntegrationRule() = 0;
   virtual void SetupIntegrator(HipBilinearForm& bform_,
                                const HipIntegratorType itype_);
   virtual void Setup() = 0;
   virtual void Assemble() = 0;
   virtual void MultAdd(HipVector& x, HipVector& y) = 0;
   virtual void MultTransposeAdd(HipVector&, HipVector&) {assert(false);}
   HipGeometry* GetGeometry();
};

// ***************************************************************************
// * Mass Integrator
// ***************************************************************************
class HipMassIntegrator : public HipIntegrator
{
private:
   HipVector op;
public:
   HipMassIntegrator() {}
   virtual ~HipMassIntegrator() {}
   virtual std::string GetName() {return "MassIntegrator";}
   virtual void SetupIntegrationRule();
   virtual void Setup() {}
   virtual void Assemble();
   void SetOperator(HipVector& v);
   virtual void MultAdd(HipVector& x, HipVector& y);
};

} // mfem

#endif // LAGHOS_HIP_BILININTEG
