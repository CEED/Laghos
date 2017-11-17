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
extern "C" void vector_clear_dofs(const int entries,
                                  double* __restrict v0,
                                  const int* v1) {
  for (int i = 0; i < entries; ++i) {
    v0[v1[i]] = 0.0;
  }
}
