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
#ifndef LAGHOS_RAJA_KERNELS
#define LAGHOS_RAJA_KERNELS

#define restrict __restrict


// ***************************************************************************
void rGridFuncToQuad(const int dim,
                     const int NUM_VDIM,
                     const int NUM_DOFS_1D,
                     const int NUM_QUAD_1D,
                     const int numElements,
                     const double* dofToQuad,
                     const int* l2gMap,
                     const double* gf,
                     double* restrict out);

// mapping *******************************************************************
void rSetSubVector(const int entries,
                   const int* indices,
                   const double* in,
                   double* restrict out);

void rMapSubVector(const int entries,
                   const int* indices,
                   const double* in,
                   double* restrict out);

void rExtractSubVector(const int entries,
                       const int* indices,
                       const double* in,
                       double* restrict out);

// kQuadratureData ***********************************************************
void rInitQuadratureData(const int NUM_QUAD,
                         const int numElements,
                         const double* rho0,
                         const double* detJ,
                         const double* quadWeights,
                         double* restrict rho0DetJ0w);

void rUpdateQuadratureData(const double GAMMA,
                           const double H0,
                           const double CFL,
                           const bool USE_VISCOSITY,
                           const int NUM_DIM,
                           const int NUM_QUAD,
                           const int NUM_QUAD_1D,
                           const int NUM_DOFS_1D,
                           const int numElements,
                           const double* dofToQuad,
                           const double* dofToQuadD,
                           const double* quadWeights,
                           const double* v,
                           const double* e,
                           const double* rho0DetJ0w,
                           const double* invJ0,
                           const double* J,
                           const double* invJ,
                           const double* detJ,
                           double* restrict stressJinvT,
                           double* restrict dtEst);

// kForce ********************************************************************
void rForceMult(const int NUM_DIM,
                const int NUM_DOFS_1D,
                const int NUM_QUAD_1D,
                const int L2_DOFS_1D,
                const int H1_DOFS_1D,
                const int nzones,
                const double* L2DofToQuad,
                const double* H1QuadToDof,
                const double* H1QuadToDofD,
                const double* stressJinvT,
                const double* e,
                double* restrict v);

void rForceMultTranspose(const int NUM_DIM,
                         const int NUM_DOFS_1D,
                         const int NUM_QUAD_1D,
                         const int L2_DOFS_1D,
                         const int H1_DOFS_1D,
                         const int nzones,
                         const double* L2QuadToDof,
                         const double* H1DofToQuad,
                         const double* H1DofToQuadD,
                         const double* stressJinvT,
                         const double* v,
                         double* restrict e);

// ***************************************************************************
void rIniGeom(const int dim,
              const int nDofs,
              const int nQuads,
              const int nzones,
              const double* dofToQuadD,
              const double* nodes,
              double* restrict J,
              double* restrict invJ,
              double* restrict detJ);

// ***************************************************************************
void rGlobalToLocal(const int NUM_VDIM,
                    const bool VDIM_ORDERING,
                    const int globalEntries,
                    const int localEntries,
                    const int* offsets,
                    const int* indices,
                    const double* globalX,
                    double* restrict localX);

void rLocalToGlobal(const int NUM_VDIM,
                    const bool VDIM_ORDERING,
                    const int globalEntries,
                    const int localEntries,
                    const int* offsets,
                    const int* indices,
                    const double* localX,
                    double* restrict globalX);

// ***************************************************************************
void rMassAssemble(const int dim,
                   const int NUM_QUAD,
                   const int numElements,
                   const double COEFF,
                   const double* quadWeights,
                   const double* J,
                   double* restrict oper);

// ***************************************************************************
void rMassMultAdd(const int dim,
                  const int NUM_DOFS_1D,
                  const int NUM_QUAD_1D,
                  const int numElements,
                  const double* dofToQuad,
                  const double* dofToQuadD,
                  const double* quadToDof,
                  const double* quadToDofD,
                  const double* op,
                  const double* x,
                  double* restrict y);

#endif // LAGHOS_RAJAC_KERNELS
