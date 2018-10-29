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
#ifndef LAGHOS_CUDA_KERNELS
#define LAGHOS_CUDA_KERNELS

// *****************************************************************************
#define restrict __restrict__

// **** BLAS1 ******************************************************************
void vector_neg(const int, double* restrict);
void vector_op_eq(const int, const double, double* restrict);
void vector_xpay(const int, const double,
                 double* restrict, const double* restrict,
                 const double* restrict);
void vector_xsy(const int, double* restrict,
                const double* restrict, const double* restrict);
void vector_axpy(const int, const double, double* restrict,
                 const double* restrict);
void vector_map_dofs(const int, double* restrict,
                     const double* restrict, const int* restrict);
void vector_clear_dofs(const int, double* restrict, const int* restrict);
void vector_vec_sub(const int, double* restrict, const double* restrict);
void vector_vec_add(const int, double* restrict, const double* restrict);
void vector_vec_mul(const int, double* restrict, const double);
void vector_set_subvector(const int, double* restrict, const double* restrict,
                          const int* restrict);
void vector_get_subvector(const int, double* restrict, const double* restrict,
                          const int* restrict);
void vector_set_subvector_const(const int, const double, double* restrict,
                                const int* restrict);
double vector_dot(const int, const double* restrict, const double* restrict);
double vector_min(const int, const double* restrict);

// *****************************************************************************
void reduceMin(int, const double*, double*);
void reduceSum(int, const double*, const double*, double*);

// *****************************************************************************
void rGridFuncToQuad(const int, const int, const int,
                     const int, const int,
                     const double* restrict, const int* restrict,
                     const double* restrict, double* restrict);

void rGridFuncToQuadS(const int, const int, const int,
                      const int, const int,
                      const double* restrict, const int* restrict,
                      const double* restrict, double* restrict);

// mapping *********************************************************************
void rSetSubVector(const int, const int* restrict,
                   const double* restrict, double* restrict);

void rMapSubVector(const int, const int* restrict,
                   const double* restrict, double* restrict);

void rExtractSubVector(const int ries, const int* restrict,
                       const double* restrict, double* restrict);

// kQuadratureData *************************************************************
void rInitQuadratureData(const int, const int,
                         const double* restrict, const double* restrict,
                         const double* restrict, double* restrict);

void rUpdateQuadratureData(const double, const double, const double,
                           const bool, const int, const int, const int,
                           const int, const int,
                           const double* restrict, const double* restrict,
                           const double* restrict, const double* restrict,
                           const double* restrict, const double* restrict,
                           const double* restrict, const double* restrict,
                           const double* restrict, const double* restrict,
                           double* restrict, double* restrict);
void rUpdateQuadratureDataS(const double, const double, const double,
                            const bool, const int, const int, const int,
                            const int, const int,
                            const double* restrict, const double* restrict,
                            const double* restrict, const double* restrict,
                            const double* restrict, const double* restrict,
                            const double* restrict, const double* restrict,
                            const double* restrict, const double* restrict,
                            double* restrict, double* restrict);

// kForce **********************************************************************
void rForceMult(const int, const int, const int, const int, const int,
                const int,
                const double* restrict, const double* restrict,
                const double* restrict, const double* restrict,
                const double* restrict, double* restrict);
void rForceMultS(const int, const int, const int, const int, const int,
                 const int, const double* restrict, const double* restrict,
                 const double* restrict, const double* restrict,
                 const double* restrict, double* restrict);

void rForceMultTranspose(const int, const int, const int, const int,
                         const int, const int,
                         const double* restrict, const double* restrict,
                         const double* restrict, const double* restrict,
                         const double* restrict, double* restrict);
void rForceMultTransposeS(const int, const int, const int,
                          const int, const int, const int,
                          const double* restrict, const double* restrict,
                          const double* restrict, const double* restrict,
                          const double* restrict, double* restrict);

// *****************************************************************************
void rNodeCopyByVDim(const int, const int, const int, const int,
                     const int* restrict, const double* restrict,
                     double* restrict);

// *****************************************************************************
void rIniGeom(const int, const int, const int, const int,
              const double* restrict, const double* restrict,
              double* restrict, double* restrict, double* restrict);

// *****************************************************************************
void rGlobalToLocal(const int, const bool, const int, const int,
                    const int* restrict, const int* restrict,
                    const double* restrict, double* restrict);

void rLocalToGlobal(const int, const bool, const int,
                    const int, const int* restrict, const int* restrict,
                    const double* restrict, double* restrict);

// *****************************************************************************
void rMassMultAdd(const int, const int, const int, const int,
                  const double* restrict, const double* restrict,
                  const double* restrict, const double* restrict,
                  const double* restrict, const double* restrict,
                  double* restrict);

void rMassMultAddS(const int, const int, const int, const int,
                   const double* restrict, const double* restrict,
                   const double* restrict, const double* restrict,
                   const double* restrict, const double* restrict,
                   double* restrict);

#endif // LAGHOS_CUDA_KERNELS
