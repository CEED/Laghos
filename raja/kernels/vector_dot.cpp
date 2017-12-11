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
#include "defines.hpp"
double vector_dot(const int N,
                  const double* __restrict vec1,
                  const double* __restrict vec2) {
  /*RAJA::ReduceSum<RAJA::seq_reduce, RAJA::Real_type> dot(0.0);
  const RAJA::RangeSegment range(0, size);
  RAJA::forall<RAJA::seq_exec>(range,[=](RAJA::Index_type i) {
    dot += data[i] * v[i];
  });
  return dot;*/
  #warning dot
  double r_red = 0.0;
  forall(N,[&]_device_(int i){
    r_red += vec1[i] * vec2[i];
    });
  return r_red;
}
