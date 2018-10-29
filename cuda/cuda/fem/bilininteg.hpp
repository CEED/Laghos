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
#ifndef LAGHOS_CUDA_BILININTEG
#define LAGHOS_CUDA_BILININTEG

namespace mfem
{

// ***************************************************************************
// * CudaGeometry
// ***************************************************************************
class CudaGeometry
{
public:
   ~CudaGeometry();
   CudaArray<int> eMap;
   CudaArray<double> meshNodes;
   CudaArray<double> J, invJ, detJ;
   static CudaGeometry* Get(CudaFiniteElementSpace&,
                            const IntegrationRule&);
   static CudaGeometry* Get(CudaFiniteElementSpace&,
                            const IntegrationRule&,
                            const CudaVector&);
   static void ReorderByVDim(GridFunction& nodes);
   static void ReorderByNodes(GridFunction& nodes);
};

// ***************************************************************************
// * CudaDofQuadMaps
// ***************************************************************************
class CudaDofQuadMaps
{
private:
   std::string hash;
public:
   CudaArray<double, false> dofToQuad, dofToQuadD; // B
   CudaArray<double, false> quadToDof, quadToDofD; // B^T
   CudaArray<double> quadWeights;
public:
   ~CudaDofQuadMaps();
   static void delCudaDofQuadMaps();
   static CudaDofQuadMaps* Get(const CudaFiniteElementSpace&,
                               const IntegrationRule&,
                               const bool = false);
   static CudaDofQuadMaps* Get(const CudaFiniteElementSpace&,
                               const CudaFiniteElementSpace&,
                               const IntegrationRule&,
                               const bool = false);
   static CudaDofQuadMaps* Get(const FiniteElement&,
                               const FiniteElement&,
                               const IntegrationRule&,
                               const bool = false);
   static CudaDofQuadMaps* GetTensorMaps(const FiniteElement&,
                                         const FiniteElement&,
                                         const IntegrationRule&,
                                         const bool = false);
   static CudaDofQuadMaps* GetD2QTensorMaps(const FiniteElement&,
                                            const IntegrationRule&,
                                            const bool = false);
   static CudaDofQuadMaps* GetSimplexMaps(const FiniteElement&,
                                          const IntegrationRule&,
                                          const bool = false);
   static CudaDofQuadMaps* GetSimplexMaps(const FiniteElement&,
                                          const FiniteElement&,
                                          const IntegrationRule&,
                                          const bool = false);
   static CudaDofQuadMaps* GetD2QSimplexMaps(const FiniteElement&,
                                             const IntegrationRule&,
                                             const bool = false);
};

// ***************************************************************************
// * Base Integrator
// ***************************************************************************
class CudaIntegrator
{
protected:
   Mesh* mesh = NULL;
   CudaFiniteElementSpace* trialFESpace = NULL;
   CudaFiniteElementSpace* testFESpace = NULL;
   CudaIntegratorType itype;
   const IntegrationRule* ir = NULL;
   CudaDofQuadMaps* maps;
   CudaDofQuadMaps* mapsTranspose;
private:
public:
   virtual std::string GetName() = 0;
   void SetIntegrationRule(const IntegrationRule& ir_);
   const IntegrationRule& GetIntegrationRule() const;
   virtual void SetupIntegrationRule() = 0;
   virtual void SetupIntegrator(CudaBilinearForm& bform_,
                                const CudaIntegratorType itype_);
   virtual void Setup() = 0;
   virtual void Assemble() = 0;
   virtual void MultAdd(CudaVector& x, CudaVector& y) = 0;
   virtual void MultTransposeAdd(CudaVector&, CudaVector&) {assert(false);}
   CudaGeometry* GetGeometry();
};

// ***************************************************************************
// * Mass Integrator
// ***************************************************************************
class CudaMassIntegrator : public CudaIntegrator
{
private:
   CudaVector op;
public:
   CudaMassIntegrator() {}
   virtual ~CudaMassIntegrator() {}
   virtual std::string GetName() {return "MassIntegrator";}
   virtual void SetupIntegrationRule();
   virtual void Setup() {}
   virtual void Assemble();
   void SetOperator(CudaVector& v);
   virtual void MultAdd(CudaVector& x, CudaVector& y);
};

} // mfem

#endif // LAGHOS_CUDA_BILININTEG
