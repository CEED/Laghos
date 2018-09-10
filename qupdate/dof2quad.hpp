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

#ifndef MFEM_LAGHOS_QUPDATE_DOF2QUAD
#define MFEM_LAGHOS_QUPDATE_DOF2QUAD

namespace mfem
{

namespace hydrodynamics {
   
// ***************************************************************************
// * qDofQuadMaps
// ***************************************************************************
class qDofQuadMaps
{
private:
   std::string hash;
public:
   qarray<double, false> dofToQuad, dofToQuadD; // B
   qarray<double, false> quadToDof, quadToDofD; // B^T
   qarray<double> quadWeights;
public:
   ~qDofQuadMaps();
   static void delqDofQuadMaps();
   static qDofQuadMaps* Get(const mfem::FiniteElementSpace&,
                            const mfem::IntegrationRule&,
                            const bool = false);
   static qDofQuadMaps* Get(const mfem::FiniteElementSpace&,
                            const mfem::FiniteElementSpace&,
                            const mfem::IntegrationRule&,
                            const bool = false);
   static qDofQuadMaps* Get(const mfem::FiniteElement&,
                            const mfem::FiniteElement&,
                            const mfem::IntegrationRule&,
                            const bool = false);
   static qDofQuadMaps* GetTensorMaps(const mfem::FiniteElement&,
                                      const mfem::FiniteElement&,
                                      const mfem::IntegrationRule&,
                                      const bool = false);
   static qDofQuadMaps* GetD2QTensorMaps(const mfem::FiniteElement&,
                                         const mfem::IntegrationRule&,
                                         const bool = false);
   static qDofQuadMaps* GetSimplexMaps(const mfem::FiniteElement&,
                                       const mfem::IntegrationRule&,
                                       const bool = false);
   static qDofQuadMaps* GetSimplexMaps(const mfem::FiniteElement&,
                                       const mfem::FiniteElement&,
                                       const mfem::IntegrationRule&,
                                       const bool = false);
   static qDofQuadMaps* GetD2QSimplexMaps(const mfem::FiniteElement&,
                                          const mfem::IntegrationRule&,
                                          const bool = false);
};

} // namespace hydrodynamics
   
} // namespace mfem

#endif // MFEM_LAGHOS_QUPDATE_DOF2QUAD
